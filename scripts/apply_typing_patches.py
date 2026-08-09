from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
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


def rebuild_manifest(
    original: dict[str, Any], records: list[dict[str, Any]], patch_count: int
) -> dict[str, Any]:
    branches = [branch for record in records for branch in record["branches"]]
    changed = [branch["changed_pixel_fraction"] for branch in branches]
    reset = [branch["qa"]["branch_start_changed_pixel_fraction"] for branch in branches]
    result = {key: value for key, value in original.items() if key not in {"sha256", "bytes"}}
    result["action_kinds"] = dict(Counter(branch["action"]["kind"] for branch in branches))
    result["quality"] = {
        "minimum_changed_pixel_fraction": min(changed),
        "maximum_changed_pixel_fraction": max(changed),
        "maximum_reset_changed_pixel_fraction": max(reset),
        "duplicate_action_bundles": sum(
            len({json.dumps(branch["action"], sort_keys=True) for branch in record["branches"]})
            != len(record["branches"])
            for record in records
        ),
    }
    result["typing_regeneration"] = {
        "version": 1,
        "regenerated_branches": patch_count,
        "preserved_non_typing_branches": len(branches) - patch_count,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply regenerated typing branches")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_root = args.input.resolve()
    patch_root = args.patches.resolve()
    output_root = args.output.resolve()
    temporary_root = output_root.with_name(f".{output_root.name}.building")
    if output_root.exists() or temporary_root.exists():
        raise RuntimeError("Refusing to overwrite existing output")

    patch_paths: dict[tuple[str, str, str], Path] = {}
    for path in patch_root.rglob("*.typing.tar"):
        relative = path.relative_to(patch_root)
        if len(relative.parts) != 3:
            continue
        original_name = relative.name.replace(".typing.tar", ".tar")
        patch_paths[(relative.parts[0], relative.parts[1], original_name)] = path

    source_tars = sorted(input_root.rglob("*.tar"))
    if len(patch_paths) != len(source_tars):
        raise RuntimeError(
            f"Expected one patch per source shard: {len(patch_paths)} != {len(source_tars)}"
        )
    temporary_root.mkdir(parents=True)
    total_patches = 0
    for source_tar in source_tars:
        relative = source_tar.relative_to(input_root)
        patch_tar_path = patch_paths.get(tuple(relative.parts))
        if patch_tar_path is None:
            raise RuntimeError(f"Missing patch for {relative}")
        destination_tar = temporary_root / relative
        destination_tar.parent.mkdir(parents=True, exist_ok=True)
        original_manifest = json.loads(source_tar.with_suffix(".json").read_text(encoding="utf-8"))
        with tarfile.open(patch_tar_path, mode="r") as patch_tar:
            patch_manifest = json.load(patch_tar.extractfile("patches.json"))
            patch_by_key = {
                (item["bundle_id"], item["branch_index"]): item
                for item in patch_manifest["patches"]
            }
            patch_images = {
                f"{item['bundle_id']}/{item['after_file']}": patch_tar.extractfile(
                    f"{item['bundle_id']}/{item['after_file']}"
                ).read()
                for item in patch_manifest["patches"]
            }
        total_patches += len(patch_by_key)

        updated_records: dict[str, bytes] = {}
        parsed_records: list[dict[str, Any]] = []
        with tarfile.open(source_tar, mode="r") as source:
            for member in source.getmembers():
                if member.isfile() and member.name.endswith("/bundle.json"):
                    record = json.load(source.extractfile(member))
                    for branch_index, branch in enumerate(record["branches"]):
                        patch = patch_by_key.get((record["bundle_id"], branch_index))
                        if patch is not None:
                            record["branches"][branch_index] = {
                                key: value for key, value in patch.items() if key != "bundle_id"
                            }
                    parsed_records.append(record)
                    updated_records[member.name] = json.dumps(record, separators=(",", ":")).encode(
                        "utf-8"
                    )
            manifest = rebuild_manifest(original_manifest, parsed_records, len(patch_by_key))
            with tarfile.open(destination_tar, mode="w") as destination:
                for member in source.getmembers():
                    if not member.isfile() or member.name == "manifest.json":
                        continue
                    if member.name in patch_images:
                        value = patch_images[member.name]
                    elif member.name in updated_records:
                        value = updated_records[member.name]
                    else:
                        value = source.extractfile(member).read()
                    add_bytes(destination, member.name, value)
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

    os.replace(temporary_root, output_root)
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "created_at": created_at,
        "input_root": str(input_root),
        "patch_root": str(patch_root),
        "output_root": str(output_root),
        "shards": len(source_tars),
        "regenerated_typing_branches": total_patches,
    }
    (output_root.parent / "typing_regeneration_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
