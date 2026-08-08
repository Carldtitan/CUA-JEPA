from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cua_jepa.config import load_config


APP_REQUIRED_STATE_KEYS: dict[str, frozenset[str]] = {
    "github_mock": frozenset({"repos", "issues", "pullRequests"}),
    "gitlab_mock": frozenset({"projects", "issues", "mergeRequests"}),
    "gmail_mock": frozenset({"emails", "labels", "drafts"}),
    "google_docs_mock": frozenset({"documents", "comments", "users"}),
    "google_sheets_mock": frozenset({"sheets", "selectedCell", "title"}),
    "jira_mock": frozenset({"projects", "issues", "sprints"}),
    "outlook_web_mock": frozenset({"messages", "folders", "calendars"}),
    "salesforce_mock": frozenset({"accounts", "opportunities", "leads"}),
    "shopify_admin_mock": frozenset({"orders", "products", "store"}),
    "slack_mock": frozenset({"channels", "messages", "workspace"}),
    "stripe_dashboard_mock": frozenset({"payments", "business", "balance"}),
    "trello_mock": frozenset({"boards", "lists", "cards"}),
}


def state_matches_app_schema(app: str, state: dict[str, Any]) -> bool:
    required = APP_REQUIRED_STATE_KEYS.get(app)
    if required is None:
        raise KeyError(f"No state-schema guard configured for {app}")
    return required.issubset(state)


def extract_literal_states(source: str, app: str | None = None) -> list[dict[str, Any]]:
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
        if (
            isinstance(candidate, dict)
            and candidate
            and (app is None or state_matches_app_schema(app, candidate))
        ):
            states.append(candidate)
    return states


def filter_catalog(
    catalog: dict[str, list[dict[str, Any]]], selected_apps: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    stats: Counter = Counter()
    for app in sorted(selected_apps):
        entries = catalog.get(app, [])
        kept = [entry for entry in entries if state_matches_app_schema(app, entry["state"])]
        filtered[app] = kept
        stats[f"{app}:states"] = len(kept)
        stats[f"{app}:rejected_foreign_states"] = len(entries) - len(kept)
    return filtered, stats


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
            states = extract_literal_states(
                setup_path.read_text(encoding="utf-8"), app=app
            )
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-root", type=Path)
    source.add_argument("--input-catalog", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_v1.json"))
    parser.add_argument("--output", type=Path, default=Path(".cache/state_catalog.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    apps = {app for split in config.splits.values() for app in split.apps}
    if args.input_catalog:
        raw_catalog = json.loads(args.input_catalog.read_text(encoding="utf-8"))
        catalog, stats = filter_catalog(raw_catalog, apps)
    else:
        catalog, stats = build_catalog(args.tasks_root, apps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")

    print(f"Wrote {args.output}")
    for app in sorted(apps):
        print(f"{app}: {len(catalog[app])} literal states")
    for key, value in sorted(stats.items()):
        if not key.endswith(":states") and value:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

