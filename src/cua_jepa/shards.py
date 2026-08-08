from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from cua_jepa.collector import BundleArtifact
from cua_jepa.config import DatasetConfig


@dataclass(frozen=True)
class ShardSpec:
    phase: str
    split: str
    app: str
    shard_index: int
    start_index: int
    bundle_count: int

    @property
    def name(self) -> str:
        return f"{self.split}-{self.app}-shard-{self.shard_index:04d}.tar"


def build_full_specs(config: DatasetConfig) -> list[ShardSpec]:
    specs: list[ShardSpec] = []
    for split_name, split in config.splits.items():
        base, remainder = divmod(split.bundles, len(split.apps))
        for app_index, app in enumerate(split.apps):
            app_bundles = base + (1 if app_index < remainder else 0)
            shard_count = math.ceil(app_bundles / config.bundles_per_shard)
            for shard_index in range(shard_count):
                start = shard_index * config.bundles_per_shard
                count = min(config.bundles_per_shard, app_bundles - start)
                specs.append(
                    ShardSpec(
                        phase="full",
                        split=split_name,
                        app=app,
                        shard_index=shard_index,
                        start_index=start,
                        bundle_count=count,
                    )
                )
    return specs


def build_pilot_specs(config: DatasetConfig) -> list[ShardSpec]:
    app_splits = [
        (split_name, app)
        for split_name, split in config.splits.items()
        for app in split.apps
    ]
    base, remainder = divmod(config.pilot_bundles, len(app_splits))
    return [
        ShardSpec(
            phase="pilot",
            split=split,
            app=app,
            shard_index=0,
            start_index=0,
            bundle_count=base + (1 if index < remainder else 0),
        )
        for index, (split, app) in enumerate(app_splits)
    ]


def deterministic_index(seed: int, app: str, candidate_index: int, length: int) -> int:
    digest = hashlib.sha256(f"{seed}:{app}:{candidate_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % length


def add_bytes(tar: tarfile.TarFile, path: str, value: bytes) -> None:
    info = tarfile.TarInfo(path)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(value))


def add_bundle(tar: tarfile.TarFile, artifact: BundleArtifact) -> dict[str, Any]:
    prefix = artifact.bundle_id
    add_bytes(tar, f"{prefix}/current.webp", artifact.current_webp)
    for index, branch in enumerate(artifact.branches):
        add_bytes(tar, f"{prefix}/after_{index}.webp", branch.after_webp)
    metadata = artifact.metadata()
    add_bytes(
        tar,
        f"{prefix}/bundle.json",
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
    )
    return metadata


def finalize_shard(
    temporary_path: Path,
    final_path: Path,
    spec: ShardSpec,
    metadata: Iterable[dict[str, Any]],
    failures: Counter[str],
    attempts: int,
) -> dict[str, Any]:
    records = list(metadata)
    manifest = {
        "schema_version": 1,
        "spec": asdict(spec),
        "accepted_bundles": len(records),
        "transitions": sum(len(record["branches"]) for record in records),
        "attempts": attempts,
        "failures": dict(failures),
        "bundle_ids": [record["bundle_id"] for record in records],
    }
    with tarfile.open(temporary_path, mode="a") as tar:
        add_bytes(
            tar,
            "manifest.json",
            json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
        )
    os.replace(temporary_path, final_path)

    digest = hashlib.sha256()
    with final_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest["sha256"] = digest.hexdigest()
    manifest["bytes"] = final_path.stat().st_size
    final_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    final_path.with_suffix(".sha256").write_text(
        f"{manifest['sha256']}  {final_path.name}\n", encoding="ascii"
    )
    return manifest

