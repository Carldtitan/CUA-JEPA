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


def main(run_id: str) -> None:
    remote_root = f"/{run_id}"
    local_root = ROOT / "artifacts" / run_id
    volume = modal.Volume.from_name("cua-jepa-training-v1")
    local_root.mkdir(parents=True, exist_ok=True)
    entries = volume.listdir(remote_root, recursive=True)
    downloaded = 0
    for entry in entries:
        if entry.type == 2:  # directory
            continue
        relative = Path(entry.path).relative_to(run_id)
        destination = local_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            volume.read_file_into_fileobj(f"/{entry.path}", handle)
        downloaded += 1
    print(f"Downloaded {downloaded} files to {local_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    arguments = parser.parse_args()
    main(arguments.run_id)
