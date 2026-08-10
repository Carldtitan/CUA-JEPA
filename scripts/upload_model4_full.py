from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parents[1]
DATA_ROOT = ROOT / "data" / "synthetic" / "clean-20260808-v7" / "full"
REMOTE_ROOT = "/model4-full"
VOLUME_NAME = "cua-jepa-synthetic-v1"
EXPECTED_FILES = {"train": 320, "validation": 20}


def _split_files(split: str) -> list[Path]:
    files = sorted((DATA_ROOT / split).rglob("*.tar"))
    expected = EXPECTED_FILES[split]
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} {split} tar files, found {len(files)}")
    return files


def main() -> None:
    audit = json.loads(
        (DATA_ROOT.parent / "audit_report.json").read_text(encoding="utf-8")
    )
    if not audit.get("passed") or audit.get("transitions") != 34_444:
        raise RuntimeError("The final dataset audit is missing or failed")
    if any(
        count
        for pair in audit.get("split_leakage", {}).values()
        for count in pair.values()
    ):
        raise RuntimeError("The audited dataset contains split leakage")
    train_files = _split_files("train")
    validation_files = _split_files("validation")
    for split, files in (("train", train_files), ("validation", validation_files)):
        if any(path.relative_to(DATA_ROOT).parts[0] != split for path in files):
            raise RuntimeError(f"A wrong split entered the {split} upload")

    volume = modal.Volume.from_name(VOLUME_NAME)
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(DATA_ROOT / "train", f"{REMOTE_ROOT}/train")
        batch.put_directory(
            DATA_ROOT / "validation", f"{REMOTE_ROOT}/validation"
        )
    print(
        f"Uploaded {len(train_files)} train and {len(validation_files)} validation "
        f"tar files to {VOLUME_NAME}:{REMOTE_ROOT}"
    )


if __name__ == "__main__":
    main()
