from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


EXPECTED_SPLITS = {
    "train": {
        "gmail_mock": 1000,
        "google_docs_mock": 1000,
        "google_sheets_mock": 1000,
        "gitlab_mock": 1000,
        "shopify_admin_mock": 1000,
        "salesforce_mock": 1000,
        "github_mock": 1000,
        "stripe_dashboard_mock": 1000,
    },
    "validation": {"slack_mock": 250, "jira_mock": 250},
    "test": {"outlook_web_mock": 250, "trello_mock": 250},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def action_signature(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def audit(
    root: Path,
    *,
    allow_filtered_counts: bool = False,
    require_unique_current: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    shard_paths = sorted(root.rglob("*.tar"))
    split_app_bundles: Counter[tuple[str, str]] = Counter()
    split_app_shards: Counter[tuple[str, str]] = Counter()
    action_kinds: Counter[str] = Counter()
    app_action_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    element_hints: dict[str, Counter[str]] = defaultdict(Counter)
    source_task_ids: dict[str, set[str]] = defaultdict(set)
    app_current_hash_counts: dict[str, Counter[str]] = defaultdict(Counter)
    app_after_hash_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bundle_ids: set[str] = set()
    current_hashes: dict[str, set[str]] = defaultdict(set)
    current_render_hashes: dict[str, set[str]] = defaultdict(set)
    after_hashes: dict[str, set[str]] = defaultdict(set)
    all_current_hash_counts: Counter[str] = Counter()
    all_after_hash_counts: Counter[str] = Counter()
    changed_fractions: list[float] = []
    reset_fractions: list[float] = []
    app_changed_fractions: dict[str, list[float]] = defaultdict(list)
    image_count = 0
    bundles = 0
    transitions = 0
    duplicate_action_bundles = 0
    duplicate_outcome_bundles = 0
    duplicate_outcomes_within_bundles = 0
    state_changing_transitions = 0
    generation_attempts = 0
    generation_failures: Counter[str] = Counter()
    app_generation_attempts: Counter[str] = Counter()
    app_generation_failures: dict[str, Counter[str]] = defaultdict(Counter)
    total_bytes = 0

    for tar_path in shard_paths:
        relative = tar_path.relative_to(root)
        if len(relative.parts) != 3:
            errors.append(f"unexpected shard path: {relative}")
            continue
        path_split, path_app, _ = relative.parts
        json_path = tar_path.with_suffix(".json")
        sha_path = tar_path.with_suffix(".sha256")
        if not json_path.exists() or not sha_path.exists():
            errors.append(f"missing sidecar for {relative}")
            continue

        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        actual_tar_sha = sha256_file(tar_path)
        sidecar_parts = sha_path.read_text(encoding="ascii").strip().split()
        sidecar_sha = sidecar_parts[0] if sidecar_parts else ""
        if actual_tar_sha != manifest.get("sha256") or actual_tar_sha != sidecar_sha:
            errors.append(f"shard checksum mismatch: {relative}")
        if sidecar_parts[-1] != tar_path.name:
            errors.append(f"wrong sidecar filename: {relative}")
        if manifest.get("bytes") != tar_path.stat().st_size:
            errors.append(f"wrong byte count: {relative}")
        total_bytes += tar_path.stat().st_size

        spec = manifest.get("spec", {})
        split = spec.get("split")
        app = spec.get("app")
        if (split, app) != (path_split, path_app):
            errors.append(f"path/manifest identity mismatch: {relative}")
        expected_bundles = spec.get("bundle_count")
        if manifest.get("accepted_bundles") != expected_bundles:
            errors.append(f"incomplete accepted bundle count: {relative}")
        if manifest.get("transitions") != expected_bundles * 4:
            errors.append(f"incomplete transition count: {relative}")
        split_app_shards[(split, app)] += 1
        generation_attempts += manifest.get("attempts", 0)
        app_generation_attempts[app] += manifest.get("attempts", 0)
        generation_failures.update(manifest.get("failures", {}))
        app_generation_failures[app].update(manifest.get("failures", {}))

        with tarfile.open(tar_path, mode="r") as archive:
            members = archive.getmembers()
            if any(not safe_member_name(member.name) for member in members):
                errors.append(f"unsafe tar member: {relative}")
            if len(members) != expected_bundles * 6 + 1:
                errors.append(f"unexpected member count: {relative} ({len(members)})")
            by_name = {member.name: member for member in members}
            embedded_member = by_name.get("manifest.json")
            if embedded_member is None:
                errors.append(f"missing embedded manifest: {relative}")
                continue
            embedded = json.load(archive.extractfile(embedded_member))
            comparable = {key: value for key, value in manifest.items() if key not in {"sha256", "bytes"}}
            if embedded != comparable:
                errors.append(f"embedded/sidecar manifest mismatch: {relative}")

            shard_bundle_ids: list[str] = []
            for member in members:
                if not member.name.endswith("/bundle.json"):
                    continue
                record = json.load(archive.extractfile(member))
                bundle_id = record.get("bundle_id")
                shard_bundle_ids.append(bundle_id)
                if bundle_id in bundle_ids:
                    errors.append(f"duplicate bundle id: {bundle_id}")
                bundle_ids.add(bundle_id)
                bundles += 1
                split_app_bundles[(split, app)] += 1
                if (record.get("split"), record.get("app")) != (split, app):
                    errors.append(f"bundle identity mismatch: {bundle_id}")
                source_task_ids[app].add(record.get("source_task_id", ""))

                current_name = f"{bundle_id}/{record['current_file']}"
                current_member = by_name.get(current_name)
                if current_member is None:
                    errors.append(f"missing current image: {bundle_id}")
                    continue
                current_bytes = archive.extractfile(current_member).read()
                current_sha = sha256_bytes(current_bytes)
                if current_sha != record.get("current_sha256"):
                    errors.append(f"current checksum mismatch: {bundle_id}")
                image_count += 1
                all_current_hash_counts[current_sha] += 1
                app_current_hash_counts[app][current_sha] += 1
                current_hashes[split].add(current_sha)
                render_sha = record.get("qa", {}).get("current_render_sha256", "")
                current_render_hashes[split].add(render_sha)
                try:
                    with Image.open(io.BytesIO(current_bytes)) as image:
                        if image.format != "WEBP" or image.size != (1280, 720):
                            errors.append(f"invalid current image: {bundle_id} {image.format} {image.size}")
                        image.verify()
                except Exception as exc:
                    errors.append(f"unreadable current image: {bundle_id}: {exc}")

                branches = record.get("branches", [])
                if len(branches) != 4:
                    errors.append(f"wrong branch count: {bundle_id}")
                signatures = [action_signature(branch.get("action", {})) for branch in branches]
                if len(set(signatures)) != len(signatures):
                    duplicate_action_bundles += 1
                    errors.append(f"duplicate actions in bundle: {bundle_id}")

                bundle_after_hashes: list[str] = []

                for index, branch in enumerate(branches):
                    transitions += 1
                    if branch.get("branch_index") != index:
                        errors.append(f"wrong branch index: {bundle_id}/{index}")
                    action = branch.get("action", {})
                    kind = action.get("kind")
                    action_kinds[kind] += 1
                    app_action_kinds[app][kind] += 1
                    if kind not in {"click", "scroll", "type"}:
                        errors.append(f"invalid action kind: {bundle_id}/{index}")
                    if kind in {"click", "type"}:
                        x = action.get("x")
                        y = action.get("y")
                        xn = action.get("x_normalized")
                        yn = action.get("y_normalized")
                        if not (0 <= x < 1280 and 0 <= y < 720 and 0 <= xn <= 1 and 0 <= yn <= 1):
                            errors.append(f"invalid action coordinates: {bundle_id}/{index}")
                        elif abs(x / 1280 - xn) > 1e-6 or abs(y / 720 - yn) > 1e-6:
                            errors.append(f"bad normalized coordinates: {bundle_id}/{index}")
                    if kind == "type" and not action.get("text"):
                        errors.append(f"empty type action: {bundle_id}/{index}")
                    if kind == "scroll" and not action.get("delta_y"):
                        errors.append(f"zero scroll action: {bundle_id}/{index}")

                    fraction = branch.get("changed_pixel_fraction")
                    reset_fraction = branch.get("qa", {}).get("branch_start_changed_pixel_fraction")
                    if not (0.0001 <= fraction <= 0.95):
                        errors.append(f"changed fraction out of bounds: {bundle_id}/{index}: {fraction}")
                    if not (0 <= reset_fraction <= 0.00005):
                        errors.append(f"reset fraction out of bounds: {bundle_id}/{index}: {reset_fraction}")
                    if branch.get("qa", {}).get("branch_start_render_sha256") != render_sha and reset_fraction == 0:
                        errors.append(f"reset hash mismatch without pixel drift: {bundle_id}/{index}")
                    changed_fractions.append(fraction)
                    reset_fractions.append(reset_fraction)
                    app_changed_fractions[app].append(fraction)
                    state_paths = branch.get("qa", {}).get("state_diff_paths", [])
                    state_changing_transitions += bool(state_paths)
                    element_hints[app][branch.get("qa", {}).get("element_hint", "")] += 1

                    after_name = f"{bundle_id}/{branch['after_file']}"
                    after_member = by_name.get(after_name)
                    if after_member is None:
                        errors.append(f"missing after image: {bundle_id}/{index}")
                        continue
                    after_bytes = archive.extractfile(after_member).read()
                    after_sha = sha256_bytes(after_bytes)
                    if after_sha != branch.get("after_sha256"):
                        errors.append(f"after checksum mismatch: {bundle_id}/{index}")
                    image_count += 1
                    all_after_hash_counts[after_sha] += 1
                    app_after_hash_counts[app][after_sha] += 1
                    bundle_after_hashes.append(after_sha)
                    after_hashes[split].add(after_sha)
                    try:
                        with Image.open(io.BytesIO(after_bytes)) as image:
                            if image.format != "WEBP" or image.size != (1280, 720):
                                errors.append(f"invalid after image: {bundle_id}/{index} {image.format} {image.size}")
                            image.verify()
                    except Exception as exc:
                        errors.append(f"unreadable after image: {bundle_id}/{index}: {exc}")

                duplicate_outcomes = len(bundle_after_hashes) - len(set(bundle_after_hashes))
                if duplicate_outcomes:
                    duplicate_outcome_bundles += 1
                    duplicate_outcomes_within_bundles += duplicate_outcomes

            if shard_bundle_ids != manifest.get("bundle_ids"):
                errors.append(f"bundle id order mismatch: {relative}")

    expected_pairs = {
        (split, app): count
        for split, apps in EXPECTED_SPLITS.items()
        for app, count in apps.items()
    }
    if allow_filtered_counts:
        if set(split_app_bundles) != set(expected_pairs):
            errors.append(f"split/app coverage mismatch: {dict(split_app_bundles)}")
        for pair, count in split_app_bundles.items():
            if not 0 < count <= expected_pairs[pair]:
                errors.append(f"invalid filtered bundle count: {pair}={count}")
    elif dict(split_app_bundles) != expected_pairs:
        errors.append(f"split/app bundle spread mismatch: {dict(split_app_bundles)}")
    expected_shards = {pair: count // 25 for pair, count in expected_pairs.items()}
    if dict(split_app_shards) != expected_shards:
        errors.append(f"split/app shard spread mismatch: {dict(split_app_shards)}")
    if allow_filtered_counts:
        if transitions != bundles * 4 or image_count != bundles * 5:
            errors.append(f"wrong filtered counts: {bundles=} {transitions=} {image_count=}")
    elif bundles != 9000 or transitions != 36000 or image_count != 45000:
        errors.append(f"wrong global counts: {bundles=} {transitions=} {image_count=}")
    if set(action_kinds) != {"click", "scroll", "type"}:
        errors.append(f"missing action kind: {dict(action_kinds)}")

    split_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    leakage: dict[str, dict[str, int]] = {}
    for left, right in split_pairs:
        leakage[f"{left}__{right}"] = {
            "current_webp_hash_overlap": len(current_hashes[left] & current_hashes[right]),
            "current_render_hash_overlap": len(current_render_hashes[left] & current_render_hashes[right]),
            "after_webp_hash_overlap": len(after_hashes[left] & after_hashes[right]),
        }
    if any(value for pair in leakage.values() for value in pair.values()):
        errors.append(f"exact screenshot leakage between splits: {leakage}")

    duplicate_current_images = sum(count - 1 for count in all_current_hash_counts.values() if count > 1)
    duplicate_after_images = sum(count - 1 for count in all_after_hash_counts.values() if count > 1)
    if duplicate_current_images:
        message = f"{duplicate_current_images} duplicate current screenshots within splits"
        if require_unique_current:
            errors.append(message)
        else:
            warnings.append(message)

    per_app = {}
    for app in sorted(app_action_kinds):
        fractions = app_changed_fractions[app]
        per_app[app] = {
            "bundles": sum(count for (split, name), count in split_app_bundles.items() if name == app),
            "transitions": sum(app_action_kinds[app].values()),
            "action_kinds": dict(app_action_kinds[app]),
            "generation_attempts": app_generation_attempts[app],
            "generation_acceptance_rate": (
                sum(count for (split, name), count in split_app_bundles.items() if name == app)
                / app_generation_attempts[app]
            ),
            "generation_failures": dict(app_generation_failures[app]),
            "unique_source_task_ids": len(source_task_ids[app]),
            "unique_current_images": len(app_current_hash_counts[app]),
            "duplicate_current_images": sum(
                count - 1 for count in app_current_hash_counts[app].values() if count > 1
            ),
            "unique_after_images": len(app_after_hash_counts[app]),
            "unique_element_hints": len(element_hints[app]),
            "top_element_hints": element_hints[app].most_common(10),
            "changed_pixel_fraction": {
                "min": min(fractions),
                "p50": percentile(fractions, 0.5),
                "p95": percentile(fractions, 0.95),
                "max": max(fractions),
            },
        }

    return {
        "passed": not errors,
        "root": str(root),
        "shards": len(shard_paths),
        "bundles": bundles,
        "transitions": transitions,
        "images": image_count,
        "tar_bytes": total_bytes,
        "split_app_bundles": {
            f"{split}/{app}": count for (split, app), count in sorted(split_app_bundles.items())
        },
        "action_kinds": dict(action_kinds),
        "generation_attempts": generation_attempts,
        "generation_acceptance_rate": bundles / generation_attempts,
        "generation_failures": dict(generation_failures),
        "duplicate_action_bundles": duplicate_action_bundles,
        "duplicate_outcome_bundles": duplicate_outcome_bundles,
        "duplicate_outcomes_within_bundles": duplicate_outcomes_within_bundles,
        "unique_current_images": len(all_current_hash_counts),
        "duplicate_current_images": duplicate_current_images,
        "unique_after_images": len(all_after_hash_counts),
        "duplicate_after_images": duplicate_after_images,
        "state_changing_transitions": state_changing_transitions,
        "changed_pixel_fraction": {
            "min": min(changed_fractions, default=0),
            "p05": percentile(changed_fractions, 0.05),
            "p50": percentile(changed_fractions, 0.5),
            "p95": percentile(changed_fractions, 0.95),
            "max": max(changed_fractions, default=0),
        },
        "reset_changed_pixel_fraction": {
            "min": min(reset_fractions, default=0),
            "p50": percentile(reset_fractions, 0.5),
            "p95": percentile(reset_fractions, 0.95),
            "max": max(reset_fractions, default=0),
            "nonzero": sum(value > 0 for value in reset_fractions),
        },
        "split_leakage": leakage,
        "per_app": per_app,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deeply audit a CUA-JEPA synthetic dataset")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-filtered-counts", action="store_true")
    parser.add_argument("--require-unique-current", action="store_true")
    args = parser.parse_args()
    report = audit(
        args.root,
        allow_filtered_counts=args.allow_filtered_counts,
        require_unique_current=args.require_unique_current,
    )
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
