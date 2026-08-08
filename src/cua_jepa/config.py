from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int
    device_scale_factor: int


@dataclass(frozen=True)
class QualityConfig:
    minimum_bundle_acceptance_rate: float
    minimum_identical_reset_rate: float
    maximum_reset_changed_pixel_fraction: float
    minimum_changed_pixel_fraction: float
    maximum_changed_pixel_fraction: float
    maximum_generation_attempts_per_bundle: int
    replay_fraction: float


@dataclass(frozen=True)
class SplitConfig:
    bundles: int
    apps: tuple[str, ...]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    seed: int
    viewport: Viewport
    actions_per_bundle: int
    bundles_per_shard: int
    pilot_bundles: int
    maximum_local_bytes: int
    maximum_runtime_hours: int
    maximum_remote_workers: int
    splits: dict[str, SplitConfig]
    quality: QualityConfig

    @property
    def total_bundles(self) -> int:
        return sum(split.bundles for split in self.splits.values())

    @property
    def total_transitions(self) -> int:
        return self.total_bundles * self.actions_per_bundle


def load_config(path: str | Path) -> DatasetConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetConfig(
        dataset_name=raw["dataset_name"],
        seed=int(raw["seed"]),
        viewport=Viewport(**raw["viewport"]),
        actions_per_bundle=int(raw["actions_per_bundle"]),
        bundles_per_shard=int(raw["bundles_per_shard"]),
        pilot_bundles=int(raw["pilot_bundles"]),
        maximum_local_bytes=int(raw["maximum_local_bytes"]),
        maximum_runtime_hours=int(raw["maximum_runtime_hours"]),
        maximum_remote_workers=int(raw["maximum_remote_workers"]),
        splits={
            name: SplitConfig(bundles=int(value["bundles"]), apps=tuple(value["apps"]))
            for name, value in raw["splits"].items()
        },
        quality=QualityConfig(**raw["quality"]),
    )
