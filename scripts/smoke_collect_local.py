from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from cua_jepa.collector import BundleCollector, BundleRejected
from cua_jepa.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--app", default="gmail_mock")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.json"))
    parser.add_argument("--catalog", type=Path, default=Path(".cache/state_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("data/pilot/local-smoke"))
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))[args.app]
    args.output.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        collector = BundleCollector(
            browser=browser,
            base_url=args.url,
            app=args.app,
            split=args.split,
            viewport_width=config.viewport.width,
            viewport_height=config.viewport.height,
            actions_per_bundle=config.actions_per_bundle,
            maximum_action_candidates=(
                config.quality.maximum_action_candidates_per_bundle
            ),
            maximum_reset_changed_fraction=(
                config.quality.maximum_reset_changed_pixel_fraction
            ),
            minimum_changed_fraction=config.quality.minimum_changed_pixel_fraction,
        )
        artifact = None
        for attempt in range(config.quality.maximum_generation_attempts_per_bundle):
            state_entry = catalog[attempt % len(catalog)]
            try:
                artifact = collector.generate(
                    bundle_id=f"local-smoke-{attempt:02d}",
                    seed=config.seed + args.seed_offset + attempt,
                    source_task_id=state_entry["source_task_id"],
                    initial_state=state_entry["state"],
                )
                break
            except BundleRejected as exc:
                failures.append(str(exc))
        browser.close()

    if artifact is None:
        raise RuntimeError(f"Smoke collection failed: {failures}")

    (args.output / "current.webp").write_bytes(artifact.current_webp)
    for index, branch in enumerate(artifact.branches):
        (args.output / f"after_{index}.webp").write_bytes(branch.after_webp)
    metadata = artifact.metadata()
    metadata["failed_attempts"] = failures
    (args.output / "bundle.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
