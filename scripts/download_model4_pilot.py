from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parents[1]
DEFAULT_RUN_ID = "model4-stage2-20260810T001315Z"


def main(run_id: str, force: bool = False) -> None:
    remote_root = f"/{run_id}"
    local_root = ROOT / "artifacts" / run_id
    volume = modal.Volume.from_name("cua-jepa-training-v1")
    local_root.mkdir(parents=True, exist_ok=True)
    entries = volume.listdir(remote_root, recursive=True)
    downloaded = 0
    skipped = 0
    for entry in entries:
        if entry.type == 2:  # directory
            continue
        relative = Path(entry.path).relative_to(run_id)
        destination = local_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not force and destination.is_file() and destination.stat().st_size == entry.size:
            skipped += 1
            continue
        temporary = destination.with_name(f"{destination.name}.part")
        if temporary.exists():
            temporary.unlink()
        with temporary.open("wb") as handle:
            volume.read_file_into_fileobj(f"/{entry.path}", handle)
        if temporary.stat().st_size != entry.size:
            raise RuntimeError(
                f"Incomplete download for {entry.path}: "
                f"{temporary.stat().st_size} of {entry.size} bytes"
            )
        temporary.replace(destination)
        downloaded += 1
    print(
        f"Downloaded {downloaded} files and kept {skipped} complete files "
        f"in {local_root}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    main(arguments.run_id, arguments.force)
