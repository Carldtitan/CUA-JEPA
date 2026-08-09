from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def copy_or_link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "hardlink"
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Conflicting existing file: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble an audited base dataset with regenerated app shards"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-app", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve()
    replacement = args.replacement.resolve()
    output = args.output.resolve()
    replacement_apps = set(args.replace_app)
    if not base.is_dir() or not replacement.is_dir():
        raise RuntimeError("Base and replacement roots must exist")

    replacement_shards: Counter[str] = Counter()
    for path in replacement.rglob("*.tar"):
        relative = path.relative_to(replacement)
        if len(relative.parts) != 3:
            raise RuntimeError(f"Unexpected replacement shard path: {relative}")
        replacement_shards[relative.parts[1]] += 1
    missing_apps = replacement_apps - set(replacement_shards)
    if missing_apps:
        raise RuntimeError(f"Missing replacement shards for: {sorted(missing_apps)}")

    methods: Counter[str] = Counter()
    reused_shards: Counter[str] = Counter()
    for source in base.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(base)
        if len(relative.parts) != 3:
            continue
        app = relative.parts[1]
        if app in replacement_apps:
            continue
        methods[copy_or_link(source, output / relative)] += 1
        if source.suffix == ".tar":
            reused_shards[app] += 1

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_root": str(base),
        "replacement_root": str(replacement),
        "output_root": str(output),
        "replacement_apps": sorted(replacement_apps),
        "replacement_shards": dict(sorted(replacement_shards.items())),
        "reused_shards": dict(sorted(reused_shards.items())),
        "file_materialization": dict(methods),
    }
    shard_manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.rglob("*.json")
    ]
    progress = {
        "run_id": output.parent.name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "assembled": True,
        "completed_shards": len(shard_manifests),
        "accepted_bundles": sum(
            manifest["accepted_bundles"] for manifest in shard_manifests
        ),
        "transitions": sum(manifest["transitions"] for manifest in shard_manifests),
        "local_bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
    }
    (output.parent / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    report["progress"] = progress
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
