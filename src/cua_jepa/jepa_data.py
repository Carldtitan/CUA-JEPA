from __future__ import annotations

import io
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class TransitionSample:
    bundle_id: str
    app: str
    split: str
    branch_index: int
    action: dict[str, Any]
    current_webp: bytes
    future_webp: bytes
    changed_pixel_fraction: float

    def current_image(self) -> Image.Image:
        return _decode_webp(self.current_webp)

    def future_image(self) -> Image.Image:
        return _decode_webp(self.future_webp)


def _decode_webp(value: bytes) -> Image.Image:
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB")


def _image_hashes(samples: Iterable[TransitionSample]) -> set[str]:
    hashes: set[str] = set()
    for sample in samples:
        hashes.add(hashlib.sha256(sample.current_webp).hexdigest())
        hashes.add(hashlib.sha256(sample.future_webp).hexdigest())
    return hashes


def validate_dataset_assignments(
    train_samples: list[TransitionSample],
    validation_samples: list[TransitionSample],
    expected_train_transitions: int = 0,
    expected_validation_transitions: int = 0,
) -> dict[str, Any]:
    """Reject split mistakes and exact data leakage before training."""

    train_splits = {sample.split for sample in train_samples}
    validation_splits = {sample.split for sample in validation_samples}
    if train_splits != {"train"}:
        raise RuntimeError(f"Training data contains wrong splits: {sorted(train_splits)}")
    if validation_splits != {"validation"}:
        raise RuntimeError(f"Validation data contains wrong splits: {sorted(validation_splits)}")
    if expected_train_transitions and len(train_samples) != expected_train_transitions:
        raise RuntimeError(
            f"Expected {expected_train_transitions} training transitions, "
            f"loaded {len(train_samples)}"
        )
    if expected_validation_transitions and (
        len(validation_samples) != expected_validation_transitions
    ):
        raise RuntimeError(
            f"Expected {expected_validation_transitions} validation transitions, "
            f"loaded {len(validation_samples)}"
        )
    train_bundles = {sample.bundle_id for sample in train_samples}
    validation_bundles = {sample.bundle_id for sample in validation_samples}
    bundle_overlap = train_bundles & validation_bundles
    if bundle_overlap:
        raise RuntimeError(f"Training and validation share {len(bundle_overlap)} bundle IDs")
    image_overlap = _image_hashes(train_samples) & _image_hashes(validation_samples)
    if image_overlap:
        raise RuntimeError(f"Training and validation share {len(image_overlap)} exact screenshots")
    return {
        "train_split": "train",
        "validation_split": "validation",
        "train_transitions": len(train_samples),
        "validation_transitions": len(validation_samples),
        "train_bundles": len(train_bundles),
        "validation_bundles": len(validation_bundles),
        "bundle_overlap": 0,
        "exact_screenshot_overlap": 0,
    }


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Tar member is not a regular file: {name}")
    return handle.read()


def load_transition_tar(path: str | Path, limit: int | None = None) -> list[TransitionSample]:
    """Load independent same-state branch transitions from one audited dataset tar."""

    tar_path = Path(path)
    samples: list[TransitionSample] = []
    with tarfile.open(tar_path, mode="r") as archive:
        bundle_members = sorted(
            (member for member in archive.getmembers() if member.name.endswith("/bundle.json")),
            key=lambda member: member.name,
        )
        for bundle_member in bundle_members:
            bundle_handle = archive.extractfile(bundle_member)
            if bundle_handle is None:
                raise ValueError(f"Unable to read {bundle_member.name} from {tar_path}")
            bundle = json.load(bundle_handle)
            root = str(PurePosixPath(bundle_member.name).parent)
            current = _read_member(archive, f"{root}/{bundle['current_file']}")
            for branch in bundle["branches"]:
                samples.append(
                    TransitionSample(
                        bundle_id=str(bundle["bundle_id"]),
                        app=str(bundle["app"]),
                        split=str(bundle["split"]),
                        branch_index=int(branch["branch_index"]),
                        action=dict(branch["action"]),
                        current_webp=current,
                        future_webp=_read_member(archive, f"{root}/{branch['after_file']}"),
                        changed_pixel_fraction=float(branch["changed_pixel_fraction"]),
                    )
                )
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def load_transition_tars(
    paths: Iterable[str | Path], limit: int | None = None
) -> list[TransitionSample]:
    samples: list[TransitionSample] = []
    for path in paths:
        remaining = None if limit is None else limit - len(samples)
        if remaining is not None and remaining <= 0:
            break
        samples.extend(load_transition_tar(path, limit=remaining))
    return samples


def group_by_bundle(samples: Iterable[TransitionSample]) -> dict[str, list[TransitionSample]]:
    groups: dict[str, list[TransitionSample]] = {}
    for sample in samples:
        groups.setdefault(sample.bundle_id, []).append(sample)
    for branches in groups.values():
        branches.sort(key=lambda sample: sample.branch_index)
    return groups


def balanced_bundle_groups(
    samples: Iterable[TransitionSample], max_bundles: int
) -> list[list[TransitionSample]]:
    """Select a fixed round-robin bundle sample across applications."""

    groups = [branches for branches in group_by_bundle(samples).values() if len(branches) == 4]
    if max_bundles <= 0 or max_bundles >= len(groups):
        return groups

    by_app: dict[str, list[list[TransitionSample]]] = {}
    for branches in groups:
        by_app.setdefault(branches[0].app, []).append(branches)

    selected: list[list[TransitionSample]] = []
    positions = {app: 0 for app in by_app}
    apps = sorted(by_app)
    while len(selected) < max_bundles:
        added = False
        for app in apps:
            position = positions[app]
            if position >= len(by_app[app]):
                continue
            selected.append(by_app[app][position])
            positions[app] += 1
            added = True
            if len(selected) == max_bundles:
                break
        if not added:
            break
    return selected
