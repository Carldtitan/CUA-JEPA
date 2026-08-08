from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from playwright.sync_api import Browser, BrowserContext, Page, Route

from cua_jepa.actions import Action, choose_distinct_actions, enumerate_actions, execute_action
from cua_jepa.qa import (
    changed_pixel_fraction,
    image_metrics,
    is_usable_screen,
    png_to_lossless_webp,
    stable_screenshot,
)


STABILITY_CSS = """
*, *::before, *::after {
  animation: none !important;
  transition: none !important;
  scroll-behavior: auto !important;
  caret-color: transparent !important;
}
"""


@dataclass(frozen=True)
class BranchArtifact:
    action: dict[str, Any]
    element_hint: str | None
    after_webp: bytes
    after_sha256: str
    changed_pixel_fraction: float
    state_diff_paths: tuple[str, ...]
    state_diff_bytes: int


@dataclass(frozen=True)
class BundleArtifact:
    bundle_id: str
    app: str
    split: str
    seed: int
    source_task_id: str
    current_webp: bytes
    current_sha256: str
    warmup_actions: tuple[dict[str, Any], ...]
    branches: tuple[BranchArtifact, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "bundle_id": self.bundle_id,
            "app": self.app,
            "split": self.split,
            "seed": self.seed,
            "source_task_id": self.source_task_id,
            "current_file": "current.webp",
            "current_sha256": self.current_sha256,
            "warmup_actions": list(self.warmup_actions),
            "branches": [
                {
                    "branch_index": index,
                    "action": branch.action,
                    "after_file": f"after_{index}.webp",
                    "after_sha256": branch.after_sha256,
                    "changed_pixel_fraction": branch.changed_pixel_fraction,
                    "qa": {
                        "element_hint": branch.element_hint,
                        "state_diff_paths": list(branch.state_diff_paths),
                        "state_diff_bytes": branch.state_diff_bytes,
                    },
                }
                for index, branch in enumerate(self.branches)
            ],
        }


class BundleRejected(RuntimeError):
    pass


class BundleCollector:
    def __init__(
        self,
        browser: Browser,
        base_url: str,
        app: str,
        split: str,
        viewport_width: int,
        viewport_height: int,
        actions_per_bundle: int = 4,
        minimum_changed_fraction: float = 0.0001,
        request_timeout_seconds: int = 15,
    ) -> None:
        self.browser = browser
        self.base_url = base_url.rstrip("/")
        self.app = app
        self.split = split
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.actions_per_bundle = actions_per_bundle
        self.minimum_changed_fraction = minimum_changed_fraction
        self.request_timeout_seconds = request_timeout_seconds
        self.http = requests.Session()

    def _state_url(self, endpoint: str, sid: str) -> str:
        return f"{self.base_url}/{endpoint}?sid={sid}"

    def _inject_state(self, sid: str, state: dict[str, Any]) -> None:
        response = self.http.post(
            self._state_url("post", sid),
            json={"action": "set", "state": state},
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()

    def _state_diff(self, sid: str) -> dict[str, Any]:
        response = self.http.get(
            self._state_url("go", sid), timeout=self.request_timeout_seconds
        )
        response.raise_for_status()
        value = response.json().get("state_diff")
        return value if isinstance(value, dict) else {}

    def _placeholder_svg(self, url: str) -> bytes:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        color = f"#{digest[:6]}"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128">'
            f'<rect width="128" height="128" fill="{color}"/>'
            "</svg>"
        ).encode("utf-8")

    def _route(self, route: Route) -> None:
        request = route.request
        parsed = urlparse(request.url)
        if parsed.hostname in {"127.0.0.1", "localhost"}:
            route.continue_()
        elif request.resource_type == "image":
            route.fulfill(
                status=200,
                content_type="image/svg+xml",
                body=self._placeholder_svg(request.url),
            )
        else:
            route.abort()

    def _new_page(self, sid: str, state: dict[str, Any]) -> tuple[BrowserContext, Page]:
        self._inject_state(sid, state)
        context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1,
            locale="en-US",
            timezone_id="UTC",
            reduced_motion="reduce",
        )
        context.route("**/*", self._route)
        page = context.new_page()
        page.goto(f"{self.base_url}/?sid={sid}", wait_until="domcontentloaded", timeout=30_000)
        page.add_style_tag(content=STABILITY_CSS)
        page.wait_for_timeout(500)
        return context, page

    def _settle(self, page: Page) -> bytes:
        try:
            page.add_style_tag(content=STABILITY_CSS)
        except Exception:
            pass
        return stable_screenshot(page)

    def _replay_warmups(self, page: Page, warmups: list[Action]) -> None:
        for action in warmups:
            execute_action(page, action)
            page.wait_for_timeout(300)

    def _choose_warmups(self, page: Page, seed: int) -> list[Action]:
        rng = random.Random(seed)
        desired = rng.choice((0, 1, 1, 2))
        warmups: list[Action] = []
        for index in range(desired):
            candidates = enumerate_actions(page, seed + index)
            clicks = [candidate for candidate in candidates if candidate.kind == "click"]
            if not clicks:
                break
            action = clicks[rng.randrange(len(clicks))]
            execute_action(page, action)
            page.wait_for_timeout(350)
            warmups.append(action)
        return warmups

    def generate(
        self,
        bundle_id: str,
        seed: int,
        source_task_id: str,
        initial_state: dict[str, Any],
    ) -> BundleArtifact:
        base_sid = f"base-{uuid.uuid4()}"
        base_context, base_page = self._new_page(base_sid, initial_state)
        try:
            warmups = self._choose_warmups(base_page, seed)
            current_png = self._settle(base_page)
            current_metrics = image_metrics(current_png)
            if not is_usable_screen(current_metrics):
                raise BundleRejected(
                    "base_screen_unusable:"
                    f"{current_metrics.width}x{current_metrics.height}:"
                    f"luminance_stddev={current_metrics.luminance_stddev:.3f}"
                )

            candidates = enumerate_actions(base_page, seed + 1009, bundle_id[-8:])
            selected = choose_distinct_actions(candidates, self.actions_per_bundle, seed + 2017)
            if len(selected) != self.actions_per_bundle:
                raise BundleRejected("insufficient_actions")
        finally:
            base_context.close()

        branch_artifacts: list[BranchArtifact] = []
        branch_start_hashes: list[str] = []
        for branch_index, action in enumerate(selected):
            sid = f"branch-{branch_index}-{uuid.uuid4()}"
            context, page = self._new_page(sid, initial_state)
            try:
                self._replay_warmups(page, warmups)
                branch_start_png = self._settle(page)
                branch_start_hash = image_metrics(branch_start_png).sha256
                branch_start_hashes.append(branch_start_hash)
                if branch_start_png != current_png:
                    raise BundleRejected("nonidentical_branch_start")

                execute_action(page, action)
                page.wait_for_timeout(400)
                after_png = self._settle(page)
                after_metrics = image_metrics(after_png)
                if not is_usable_screen(after_metrics):
                    raise BundleRejected(
                        "after_screen_unusable:"
                        f"{after_metrics.width}x{after_metrics.height}:"
                        f"luminance_stddev={after_metrics.luminance_stddev:.3f}"
                    )
                changed = changed_pixel_fraction(current_png, after_png)
                state_diff = self._state_diff(sid)
                if changed < self.minimum_changed_fraction and not state_diff:
                    raise BundleRejected("action_had_no_observable_effect")

                branch_artifacts.append(
                    BranchArtifact(
                        action=action.as_dict(self.viewport_width, self.viewport_height),
                        element_hint=action.element_hint,
                        after_webp=png_to_lossless_webp(after_png),
                        after_sha256=after_metrics.sha256,
                        changed_pixel_fraction=round(changed, 8),
                        state_diff_paths=tuple(sorted(state_diff.keys())),
                        state_diff_bytes=len(
                            json.dumps(state_diff, separators=(",", ":")).encode("utf-8")
                        ),
                    )
                )
            finally:
                context.close()

        if len(set(branch_start_hashes)) != 1:
            raise BundleRejected("branch_start_hash_mismatch")
        after_hashes = {branch.after_sha256 for branch in branch_artifacts}
        if len(after_hashes) != self.actions_per_bundle:
            raise BundleRejected("duplicate_branch_futures")

        return BundleArtifact(
            bundle_id=bundle_id,
            app=self.app,
            split=self.split,
            seed=seed,
            source_task_id=source_task_id,
            current_webp=png_to_lossless_webp(current_png),
            current_sha256=current_metrics.sha256,
            warmup_actions=tuple(
                action.as_dict(self.viewport_width, self.viewport_height) for action in warmups
            ),
            branches=tuple(branch_artifacts),
        )


def with_attempts(
    operation: Callable[[int], BundleArtifact], maximum_attempts: int
) -> tuple[BundleArtifact, list[str]]:
    failures: list[str] = []
    for attempt in range(maximum_attempts):
        try:
            return operation(attempt), failures
        except BundleRejected as exc:
            failures.append(str(exc))
    raise BundleRejected(json.dumps({"attempt_failures": failures}))
