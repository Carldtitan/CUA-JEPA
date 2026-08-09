from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle_records(
    archive: tarfile.TarFile,
) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith("/bundle.json"):
            payload = archive.extractfile(member)
            if payload is None:
                raise RuntimeError(f"Cannot read {member.name}")
            records.append((member.name, json.load(payload)))
    return records


def filtered_manifest(
    original: dict[str, Any],
    records: list[dict[str, Any]],
    removed_bundle_ids: list[str],
) -> dict[str, Any]:
    branches = [branch for record in records for branch in record["branches"]]
    action_kinds = Counter(branch["action"]["kind"] for branch in branches)
    changed = [branch["changed_pixel_fraction"] for branch in branches]
    reset = [
        branch["qa"]["branch_start_changed_pixel_fraction"] for branch in branches
    ]
    manifest = {
        key: value for key, value in original.items() if key not in {"sha256", "bytes"}
    }
    manifest["spec"] = dict(original["spec"])
    manifest["spec"]["bundle_count"] = len(records)
    manifest["accepted_bundles"] = len(records)
    manifest["transitions"] = len(branches)
    manifest["bundle_ids"] = [record["bundle_id"] for record in records]
    manifest["action_kinds"] = dict(action_kinds)
    manifest["quality"] = {
        "minimum_changed_pixel_fraction": min(changed),
        "maximum_changed_pixel_fraction": max(changed),
        "maximum_reset_changed_pixel_fraction": max(reset),
        "duplicate_action_bundles": sum(
            len(
                {
                    json.dumps(branch["action"], sort_keys=True)
                    for branch in record["branches"]
                }
            )
            != len(record["branches"])
            for record in records
        ),
    }
    manifest["filtering"] = {
        "method": "keep_first_bundle_per_exact_current_sha256",
        "input_bundles": original["accepted_bundles"],
        "removed_bundles": len(removed_bundle_ids),
    }
    return manifest


def copy_bundle(
    source: tarfile.TarFile,
    destination: tarfile.TarFile,
    bundle_member_name: str,
    record: dict[str, Any],
) -> None:
    prefix = str(PurePosixPath(bundle_member_name).parent)
    names = [
        f"{prefix}/{record['current_file']}",
        *(f"{prefix}/{branch['after_file']}" for branch in record["branches"]),
        bundle_member_name,
    ]
    for name in names:
        payload = source.extractfile(name)
        if payload is None:
            raise RuntimeError(f"Missing tar member: {name}")
        add_bytes(destination, name, payload.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove bundles with duplicate exact starting screenshots"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    temporary_root = output_root.with_name(f".{output_root.name}.building")
    if not input_root.is_dir():
        raise RuntimeError(f"Input dataset does not exist: {input_root}")
    if output_root.exists() or temporary_root.exists():
        raise RuntimeError(
            f"Refusing to overwrite an existing output: {output_root} or {temporary_root}"
        )

    temporary_root.mkdir(parents=True)
    seen_current_hashes: set[str] = set()
    removed_bundle_ids: list[str] = []
    kept_bundles = 0
    kept_transitions = 0
    shard_count = 0
    split_app_bundles: Counter[str] = Counter()

    for source_tar in sorted(input_root.rglob("*.tar")):
        relative = source_tar.relative_to(input_root)
        destination_tar = temporary_root / relative
        destination_tar.parent.mkdir(parents=True, exist_ok=True)
        source_manifest_path = source_tar.with_suffix(".json")
        original_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )

        with tarfile.open(source_tar, mode="r") as source:
            records = read_bundle_records(source)
            kept: list[tuple[str, dict[str, Any]]] = []
            removed_from_shard: list[str] = []
            for member_name, record in records:
                current_hash = record["current_sha256"]
                if current_hash in seen_current_hashes:
                    removed_bundle_ids.append(record["bundle_id"])
                    removed_from_shard.append(record["bundle_id"])
                    continue
                seen_current_hashes.add(current_hash)
                kept.append((member_name, record))

            if not kept:
                raise RuntimeError(f"Filtering would empty an entire shard: {relative}")
            manifest = filtered_manifest(
                original_manifest,
                [record for _, record in kept],
                removed_from_shard,
            )
            with tarfile.open(destination_tar, mode="w") as destination:
                for member_name, record in kept:
                    copy_bundle(source, destination, member_name, record)
                add_bytes(
                    destination,
                    "manifest.json",
                    json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
                )

        digest = file_sha256(destination_tar)
        manifest["sha256"] = digest
        manifest["bytes"] = destination_tar.stat().st_size
        destination_tar.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        destination_tar.with_suffix(".sha256").write_text(
            f"{digest}  {destination_tar.name}\n", encoding="ascii"
        )
        shard_count += 1
        kept_bundles += len(kept)
        kept_transitions += len(kept) * 4
        split_app_bundles[f"{relative.parts[0]}/{relative.parts[1]}"] += len(kept)

    os.replace(temporary_root, output_root)
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "created_at": created_at,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "method": "keep_first_bundle_per_exact_current_sha256",
        "input_bundles": kept_bundles + len(removed_bundle_ids),
        "kept_bundles": kept_bundles,
        "removed_bundles": len(removed_bundle_ids),
        "transitions": kept_transitions,
        "shards": shard_count,
        "unique_current_images": len(seen_current_hashes),
        "split_app_bundles": dict(sorted(split_app_bundles.items())),
        "removed_bundle_ids": removed_bundle_ids,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    (output_root.parent / "deduplication_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    progress = {
        "run_id": output_root.parent.name,
        "updated_at": created_at,
        "assembled": True,
        "completed_shards": shard_count,
        "accepted_bundles": kept_bundles,
        "transitions": kept_transitions,
        "local_bytes": sum(
            path.stat().st_size for path in output_root.rglob("*") if path.is_file()
        ),
    }
    (output_root.parent / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
