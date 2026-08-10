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
    for split in ("train", "validation"):
        app_directories = sorted(
            path for path in (DATA_ROOT / split).iterdir() if path.is_dir()
        )
        for index, app_directory in enumerate(app_directories, start=1):
            app_files = sorted(app_directory.glob("*.tar"))
            print(
                f"Uploading {split}/{app_directory.name}: {len(app_files)} files "
                f"({index}/{len(app_directories)})",
                flush=True,
            )
            with volume.batch_upload(force=True) as batch:
                batch.put_directory(
                    app_directory,
                    f"{REMOTE_ROOT}/{split}/{app_directory.name}",
                )
            print(f"Committed {split}/{app_directory.name}", flush=True)

    remote_paths = [
        entry.path.lstrip("/")
        for entry in volume.listdir(REMOTE_ROOT, recursive=True)
    ]
    remote_train = sum(
        path.startswith("model4-full/train/") and path.endswith(".tar")
        for path in remote_paths
    )
    remote_validation = sum(
        path.startswith("model4-full/validation/") and path.endswith(".tar")
        for path in remote_paths
    )
    remote_test = sum("/test/" in f"/{path}" for path in remote_paths)
    if (remote_train, remote_validation, remote_test) != (320, 20, 0):
        raise RuntimeError(
            "Remote verification failed: "
            f"{remote_train} train, {remote_validation} validation, "
            f"{remote_test} test tar files"
        )
    print(
        f"Uploaded {len(train_files)} train and {len(validation_files)} validation "
        f"tar files to {VOLUME_NAME}:{REMOTE_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
