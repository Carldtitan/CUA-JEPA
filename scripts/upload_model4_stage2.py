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
LOCAL_STAGE = ROOT / "data" / "pilot" / "model4-stage2"
REMOTE_STAGE = "/model4-stage2"
VOLUME_NAME = "cua-jepa-synthetic-v1"


def main() -> None:
    files = list(LOCAL_STAGE.rglob("*.tar"))
    if len(files) != 25:
        raise RuntimeError(f"Expected 25 staged tar files, found {len(files)}")
    volume = modal.Volume.from_name(VOLUME_NAME)
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_STAGE, REMOTE_STAGE)
    print(f"Uploaded {len(files)} tar files to {VOLUME_NAME}:{REMOTE_STAGE}")


if __name__ == "__main__":
    main()
