from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from playwright.sync_api import Browser, BrowserContext, Page, Route

from cua_jepa.actions import (
    Action,
    action_from_dict,
    choose_distinct_actions,
    enumerate_actions,
    execute_action,
    typed_text_is_present,
)
from cua_jepa.qa import (
    changed_pixel_fraction,
    image_metrics,
    is_usable_screen,
    png_to_lossless_webp,
    stable_screenshot,
)
from cua_jepa.state_variants import initial_path_for_app


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
    branch_start_render_sha256: str
    branch_start_changed_pixel_fraction: float
    after_webp: bytes
    after_sha256: str
    after_render_sha256: str
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
    current_render_sha256: str
    warmup_actions: tuple[dict[str, Any], ...]
    action_candidate_failures: dict[str, int]
    branches: tuple[BranchArtifact, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "bundle_id": self.bundle_id,
            "app": self.app,
            "split": self.split,
            "seed": self.seed,
            "source_task_id": self.source_task_id,
            "current_file": "current.webp",
            "current_sha256": self.current_sha256,
            "warmup_actions": list(self.warmup_actions),
            "qa": {
                "current_render_sha256": self.current_render_sha256,
                "action_candidate_failures": self.action_candidate_failures,
            },
            "branches": [
                {
                    "branch_index": index,
                    "action": branch.action,
                    "after_file": f"after_{index}.webp",
                    "after_sha256": branch.after_sha256,
                    "changed_pixel_fraction": branch.changed_pixel_fraction,
                    "qa": {
                        "element_hint": branch.element_hint,
                        "branch_start_render_sha256": (branch.branch_start_render_sha256),
                        "branch_start_changed_pixel_fraction": (
                            branch.branch_start_changed_pixel_fraction
                        ),
                        "after_render_sha256": branch.after_render_sha256,
                        "state_diff_paths": list(branch.state_diff_paths),
                        "state_diff_bytes": branch.state_diff_bytes,
                    },
                }
                for index, branch in enumerate(self.branches)
            ],
        }


class BundleRejected(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        message = f"{code}:{detail}" if detail else code
        super().__init__(message)


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
        maximum_action_candidates: int = 12,
        maximum_branch_reset_attempts: int = 3,
        maximum_reset_changed_fraction: float = 0.00005,
        allow_warmups: bool = True,
        minimum_changed_fraction: float = 0.0001,
        maximum_changed_fraction: float = 0.95,
        request_timeout_seconds: int = 15,
    ) -> None:
        self.browser = browser
        self.base_url = base_url.rstrip("/")
        self.app = app
        self.split = split
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.actions_per_bundle = actions_per_bundle
        self.maximum_action_candidates = maximum_action_candidates
        self.maximum_branch_reset_attempts = maximum_branch_reset_attempts
        self.maximum_reset_changed_fraction = maximum_reset_changed_fraction
        self.allow_warmups = allow_warmups
        self.minimum_changed_fraction = minimum_changed_fraction
        self.maximum_changed_fraction = maximum_changed_fraction
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
        response = self.http.get(self._state_url("go", sid), timeout=self.request_timeout_seconds)
        response.raise_for_status()
        value = response.json().get("state_diff")
        return value if isinstance(value, dict) else {}

    def _current_state(self, sid: str) -> dict[str, Any]:
        response = self.http.get(
            self._state_url("state", sid), timeout=self.request_timeout_seconds
        )
        response.raise_for_status()
        value = response.json().get("stored_state")
        if not isinstance(value, dict):
            raise BundleRejected("post_warmup_state_unavailable")
        return value

    @staticmethod
    def _url_with_sid(url: str, sid: str) -> str:
        parsed = urlparse(url)
        query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "sid"]
        query.append(("sid", sid))
        return urlunparse(parsed._replace(query=urlencode(query)))

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

    def _new_page(
        self,
        sid: str,
        state: dict[str, Any],
        start_url: str | None = None,
        render_seed: int = 0,
    ) -> tuple[BrowserContext, Page]:
        self._inject_state(sid, state)
        context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1,
            locale="en-US",
            timezone_id="UTC",
            reduced_motion="reduce",
        )
        context.route("**/*", self._route)
        context.add_init_script(
            script=f"""
                (() => {{
                    let value = {render_seed & 0xFFFFFFFF};
                    Math.random = () => {{
                        value = (1664525 * value + 1013904223) >>> 0;
                        return value / 4294967296;
                    }};
                }})();
            """
        )
        page = context.new_page()
        page.clock.set_fixed_time("2026-08-08T20:00:00Z")
        target_url = self._url_with_sid(start_url or f"{self.base_url}/", sid)
        page.goto(target_url, wait_until="domcontentloaded", timeout=30_000)
        page.add_style_tag(content=STABILITY_CSS)
        page.wait_for_timeout(500)
        return context, page

    def _settle(self, page: Page) -> bytes:
        try:
            page.add_style_tag(content=STABILITY_CSS)
        except Exception:
            pass
        page.mouse.move(self.viewport_width - 1, self.viewport_height - 1)
        page.evaluate(
            """() => {
                const active = document.activeElement;
                if (active && typeof active.blur === 'function') active.blur();
            }"""
        )
        return stable_screenshot(page)

    def _choose_warmups(self, page: Page, seed: int) -> list[Action]:
        rng = random.Random(seed)
        desired = rng.choice((0, 1, 1, 2))
        warmups: list[Action] = []
        for index in range(desired):
            candidates = enumerate_actions(page, seed + index)
            clicks = [
                candidate
                for candidate in candidates
                if candidate.kind == "click" and candidate.warmup_safe
            ]
            if not clicks:
                break
            action = clicks[rng.randrange(len(clicks))]
            execute_action(page, action)
            page.wait_for_timeout(350)
            destination_metrics = image_metrics(self._settle(page))
            destination_actions = enumerate_actions(page, seed + index + 10_000)
            if (
                not is_usable_screen(destination_metrics)
                or len(destination_actions) < self.actions_per_bundle
            ):
                page.go_back(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(350)
                break
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
        initial_path = initial_path_for_app(self.app, initial_state, seed)
        base_context, base_page = self._new_page(
            base_sid,
            initial_state,
            start_url=f"{self.base_url}{initial_path}",
            render_seed=seed,
        )
        try:
            warmups = self._choose_warmups(base_page, seed) if self.allow_warmups else []
            current_png = self._settle(base_page)
            current_metrics = image_metrics(current_png)
            if not is_usable_screen(current_metrics):
                raise BundleRejected(
                    "base_screen_unusable",
                    f"{current_metrics.width}x{current_metrics.height}:"
                    f"luminance_stddev={current_metrics.luminance_stddev:.3f}",
                )

            candidates = enumerate_actions(base_page, seed + 1009, bundle_id[-8:])
            selected = choose_distinct_actions(
                candidates, self.maximum_action_candidates, seed + 2017
            )
            if len(selected) < self.actions_per_bundle:
                raise BundleRejected("insufficient_actions")
            branch_initial_state = self._current_state(base_sid)
            branch_start_url = base_page.url
        finally:
            base_context.close()

        branch_artifacts: list[BranchArtifact] = []
        action_candidate_failures: Counter[str] = Counter()
        after_hashes: set[str] = set()
        for candidate_index, action in enumerate(selected):
            reset_succeeded = False
            last_reset_difference = 1.0
            for _ in range(self.maximum_branch_reset_attempts):
                sid = f"branch-{candidate_index}-{uuid.uuid4()}"
                context, page = self._new_page(
                    sid,
                    branch_initial_state,
                    start_url=branch_start_url,
                    render_seed=seed,
                )
                try:
                    branch_start_png = self._settle(page)
                    branch_start_hash = image_metrics(branch_start_png).sha256
                    difference = changed_pixel_fraction(current_png, branch_start_png)
                    last_reset_difference = difference
                    if difference > self.maximum_reset_changed_fraction:
                        action_candidate_failures["branch_reset_retry"] += 1
                        continue
                    reset_succeeded = True

                    execute_action(page, action)
                    page.wait_for_timeout(400)
                    after_png = self._settle(page)
                    after_metrics = image_metrics(after_png)
                    if not is_usable_screen(after_metrics):
                        action_candidate_failures["after_screen_unusable"] += 1
                        break
                    changed = changed_pixel_fraction(current_png, after_png)
                    state_diff = self._state_diff(sid)
                    if changed < self.minimum_changed_fraction:
                        action_candidate_failures["action_had_no_observable_effect"] += 1
                        break
                    if changed > self.maximum_changed_fraction:
                        action_candidate_failures["action_changed_too_much"] += 1
                        break

                    after_webp = png_to_lossless_webp(after_png)
                    after_sha256 = hashlib.sha256(after_webp).hexdigest()
                    if after_sha256 in after_hashes:
                        action_candidate_failures["duplicate_branch_future"] += 1
                        break
                    after_hashes.add(after_sha256)
                    branch_artifacts.append(
                        BranchArtifact(
                            action=action.as_dict(self.viewport_width, self.viewport_height),
                            element_hint=action.element_hint,
                            branch_start_render_sha256=branch_start_hash,
                            branch_start_changed_pixel_fraction=round(difference, 8),
                            after_webp=after_webp,
                            after_sha256=after_sha256,
                            after_render_sha256=after_metrics.sha256,
                            changed_pixel_fraction=round(changed, 8),
                            state_diff_paths=tuple(sorted(state_diff.keys())),
                            state_diff_bytes=len(
                                json.dumps(state_diff, separators=(",", ":")).encode("utf-8")
                            ),
                        )
                    )
                    break
                finally:
                    context.close()
            if not reset_succeeded:
                raise BundleRejected(
                    "nonidentical_branch_start",
                    f"changed_pixel_fraction={last_reset_difference:.8f}",
                )
            if len(branch_artifacts) == self.actions_per_bundle:
                break

        if len(branch_artifacts) != self.actions_per_bundle:
            raise BundleRejected(
                "insufficient_effective_actions",
                json.dumps(dict(action_candidate_failures), sort_keys=True),
            )

        current_webp = png_to_lossless_webp(current_png)
        return BundleArtifact(
            bundle_id=bundle_id,
            app=self.app,
            split=self.split,
            seed=seed,
            source_task_id=source_task_id,
            current_webp=current_webp,
            current_sha256=hashlib.sha256(current_webp).hexdigest(),
            current_render_sha256=current_metrics.sha256,
            warmup_actions=tuple(
                action.as_dict(self.viewport_width, self.viewport_height) for action in warmups
            ),
            action_candidate_failures=dict(action_candidate_failures),
            branches=tuple(branch_artifacts),
        )

    def regenerate_type_branch(
        self,
        record: dict[str, Any],
        initial_state: dict[str, Any],
        branch_index: int,
        text_for_hint: Callable[[str | None], str],
        stored_current_webp: bytes,
    ) -> BranchArtifact:
        """Replay one stored bundle and replace only one typing branch."""
        stored_branch = record["branches"][branch_index]
        if stored_branch["action"].get("kind") != "type":
            raise ValueError(f"Branch {branch_index} is not a typing action")
        seed = int(record["seed"])
        replay_tolerance = max(self.maximum_reset_changed_fraction, 0.0001)
        base_sid = f"typing-base-{uuid.uuid4()}"
        initial_path = initial_path_for_app(self.app, initial_state, seed)
        base_context, base_page = self._new_page(
            base_sid,
            initial_state,
            start_url=f"{self.base_url}{initial_path}",
            render_seed=seed,
        )
        try:
            for stored_warmup in record.get("warmup_actions", []):
                execute_action(base_page, action_from_dict(stored_warmup))
                base_page.wait_for_timeout(350)
                self._settle(base_page)
            current_png = self._settle(base_page)
            current_webp = png_to_lossless_webp(current_png)
            current_sha = hashlib.sha256(current_webp).hexdigest()
            if current_sha != record["current_sha256"]:
                replay_difference = changed_pixel_fraction(stored_current_webp, current_png)
                if replay_difference > replay_tolerance:
                    raise BundleRejected(
                        "stored_current_replay_mismatch",
                        f"changed_pixel_fraction={replay_difference:.8f};"
                        f"{current_sha} != {record['current_sha256']}",
                    )
            branch_initial_state = self._current_state(base_sid)
            branch_start_url = base_page.url
            typing_targets = [
                candidate
                for candidate in enumerate_actions(
                    base_page,
                    seed + 8191 + branch_index,
                    record["bundle_id"][-8:],
                )
                if candidate.kind == "type"
            ]
            if not typing_targets:
                raise BundleRejected("no_unobscured_typing_target")
        finally:
            base_context.close()

        last_reset_difference = 1.0
        last_candidate_failure = "no_candidate_attempted"
        maximum_attempts = max(self.maximum_branch_reset_attempts, len(typing_targets))
        for attempt in range(maximum_attempts):
            target = typing_targets[attempt % len(typing_targets)]
            action = Action(
                kind="type",
                x=target.x,
                y=target.y,
                text=text_for_hint(target.element_hint),
                element_hint=target.element_hint,
            )
            sid = f"typing-{branch_index}-{attempt}-{uuid.uuid4()}"
            context, page = self._new_page(
                sid,
                branch_initial_state,
                start_url=branch_start_url,
                render_seed=seed,
            )
            try:
                branch_start_png = self._settle(page)
                branch_start_metrics = image_metrics(branch_start_png)
                difference = changed_pixel_fraction(stored_current_webp, branch_start_png)
                last_reset_difference = difference
                if difference > replay_tolerance:
                    last_candidate_failure = f"reset_changed_pixel_fraction={difference:.8f}"
                    continue
                execute_action(page, action)
                if not typed_text_is_present(page, action.text or ""):
                    last_candidate_failure = "typed_text_not_present_immediately"
                    continue
                page.wait_for_timeout(400)
                after_png = self._settle(page)
                if not typed_text_is_present(page, action.text or ""):
                    last_candidate_failure = "typed_text_not_present_after_settle"
                    continue
                after_metrics = image_metrics(after_png)
                if not is_usable_screen(after_metrics):
                    last_candidate_failure = "typing_after_screen_unusable"
                    continue
                changed = changed_pixel_fraction(stored_current_webp, after_png)
                if changed < self.minimum_changed_fraction:
                    last_candidate_failure = f"changed_pixel_fraction_too_low={changed:.8f}"
                    continue
                if changed > self.maximum_changed_fraction:
                    last_candidate_failure = f"changed_pixel_fraction_too_high={changed:.8f}"
                    continue
                after_webp = png_to_lossless_webp(after_png)
                after_sha = hashlib.sha256(after_webp).hexdigest()
                other_hashes = {
                    branch["after_sha256"]
                    for index, branch in enumerate(record["branches"])
                    if index != branch_index
                }
                if after_sha in other_hashes:
                    last_candidate_failure = "typing_duplicate_branch_future"
                    continue
                state_diff = self._state_diff(sid)
                return BranchArtifact(
                    action=action.as_dict(self.viewport_width, self.viewport_height),
                    element_hint=action.element_hint,
                    branch_start_render_sha256=branch_start_metrics.sha256,
                    branch_start_changed_pixel_fraction=round(difference, 8),
                    after_webp=after_webp,
                    after_sha256=after_sha,
                    after_render_sha256=after_metrics.sha256,
                    changed_pixel_fraction=round(changed, 8),
                    state_diff_paths=tuple(sorted(state_diff.keys())),
                    state_diff_bytes=len(
                        json.dumps(state_diff, separators=(",", ":")).encode("utf-8")
                    ),
                )
            finally:
                context.close()
        raise BundleRejected(
            "typing_regeneration_attempts_exhausted",
            f"last_reset_changed_pixel_fraction={last_reset_difference:.8f};"
            f"last_candidate_failure={last_candidate_failure}",
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
    raise BundleRejected("maximum_attempts_exhausted", json.dumps({"attempt_failures": failures}))
