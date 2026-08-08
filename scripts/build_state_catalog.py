from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cua_jepa.config import load_config


def extract_literal_states(source: str) -> list[dict[str, Any]]:
    """Extract top-level literal dictionaries assigned to state-like variables."""
    tree = ast.parse(source)
    states: list[dict[str, Any]] = []
    for node in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
        if not name or value is None or not (name == "state" or name.endswith("_state")):
            continue
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and candidate:
            states.append(candidate)
    return states


def build_catalog(tasks_root: Path, selected_apps: set[str]) -> tuple[dict[str, list], Counter]:
    catalog: dict[str, list] = {app: [] for app in sorted(selected_apps)}
    stats: Counter = Counter()
    for task_path in tasks_root.glob("*/task.json"):
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["invalid_task_json"] += 1
            continue
        app = task.get("app_type")
        if app not in selected_apps:
            continue
        setup_path = task_path.with_name("initial_setup.py")
        if not setup_path.exists():
            stats[f"{app}:missing_setup"] += 1
            continue
        try:
            states = extract_literal_states(setup_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            stats[f"{app}:invalid_setup"] += 1
            continue
        if not states:
            stats[f"{app}:no_literal_state"] += 1
            continue
        for state_index, state in enumerate(states):
            catalog[app].append(
                {
                    "source_task_id": task.get("id", task_path.parent.name),
                    "state_index": state_index,
                    "state": state,
                }
            )
            stats[f"{app}:states"] += 1
    return catalog, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.json"))
    parser.add_argument("--output", type=Path, default=Path(".cache/state_catalog.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    apps = {app for split in config.splits.values() for app in split.apps}
    catalog, stats = build_catalog(args.tasks_root, apps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")

    print(f"Wrote {args.output}")
    for app in sorted(apps):
        print(f"{app}: {len(catalog[app])} literal states")
    for key, value in sorted(stats.items()):
        if not key.endswith(":states"):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

