from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parents[1]
RUN_ID = "model4-stage2-20260810T001315Z"
REMOTE_ROOT = f"/{RUN_ID}"
LOCAL_ROOT = ROOT / "artifacts" / RUN_ID


def main() -> None:
    volume = modal.Volume.from_name("cua-jepa-training-v1")
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    entries = volume.listdir(REMOTE_ROOT, recursive=True)
    downloaded = 0
    for entry in entries:
        if entry.type == 2:  # directory
            continue
        relative = Path(entry.path).relative_to(RUN_ID)
        destination = LOCAL_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            volume.read_file_into_fileobj(f"/{entry.path}", handle)
        downloaded += 1
    print(f"Downloaded {downloaded} files to {LOCAL_ROOT}")


if __name__ == "__main__":
    main()
