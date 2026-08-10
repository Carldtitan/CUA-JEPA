from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


APP_NAME = "cua-jepa-agentnet-sft"
VOLUME_NAME = "cua-jepa-sft-v1"
DATASET_ROOT = Path("/sft/agentnet-v1")
AGENTNET_REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"
HF_ROOT = f"https://huggingface.co/datasets/xlangai/AgentNet/resolve/{AGENTNET_REVISION}"
TRAJECTORY_FILES = ("agentnet_ubuntu_5k.jsonl", "agentnet_win_mac_18k.jsonl")
ARCHIVES = {
    "agentnet_ubuntu_5k.jsonl": (
        "ubuntu_images",
        [f"images.z{index:02d}" for index in range(1, 14)] + ["images.zip"],
    ),
    "agentnet_win_mac_18k.jsonl": (
        "win_mac_images",
        [f"images.z{index:02d}" for index in range(1, 24)] + ["images.zip"],
    ),
}
FINAL_ARCHIVE_SIZES = {
    "agentnet_ubuntu_5k.jsonl": 3_727_419_649,
    "agentnet_win_mac_18k.jsonl": 855_815_467,
}

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "p7zip-full", "zip")
        .pip_install("pillow>=10,<13", "requests>=2.31,<3")
        .add_local_python_source("cua_jepa", copy=True)
    )
else:
    image = modal.Image.debian_slim()

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_jsonl(url: str):
    import requests

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _curl_download(
    name: str, url: str, destination: Path, expected_size: int | None = None
) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    if path.is_file() and (expected_size is None or path.stat().st_size == expected_size):
        return name
    subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1048576",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
            "--output",
            str(path),
            url,
        ],
        check=True,
    )
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Wrong size for {name}: {path.stat().st_size}, expected {expected_size}"
        )
    return name


def _parallel_download(
    files: list[tuple[str, str, int | None]], destination: Path
) -> list[str]:
    completed = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_curl_download, name, url, destination, expected_size): name
            for name, url, expected_size in files
        }
        for future in as_completed(futures):
            name = future.result()
            completed.append(name)
            print(f"Downloaded {len(completed)}/{len(files)}: {name}", flush=True)
    return sorted(completed)


@app.function(image=image, cpu=1, memory=1024, timeout=10 * 60)
def smoke_test_split_archive() -> dict:
    root = Path("/tmp/split-archive-smoke")
    source = root / "source"
    archive_root = root / "archive"
    extracted = root / "extracted"
    source.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"image-{index}.png").write_bytes(bytes([index + 1]) * 100_000)
    subprocess.run(
        [
            "zip",
            "-q",
            "-0",
            "-s",
            "64k",
            str(archive_root / "images.zip"),
            *[str(path) for path in sorted(source.iterdir())],
        ],
        check=True,
    )
    _extract_selected(archive_root / "images.zip", ["image-1.png"], extracted)
    selected = list(extracted.rglob("image-1.png"))
    unwanted = list(extracted.rglob("image-0.png")) + list(extracted.rglob("image-2.png"))
    if len(selected) != 1 or unwanted or selected[0].read_bytes() != bytes([2]) * 100_000:
        raise RuntimeError("Split archive selective extraction failed")
    download_root = root / "downloads"
    _parallel_download(
        [
            ("README.md", f"{HF_ROOT}/README.md", None),
            ("LICENSE.txt", f"{HF_ROOT}/LICENSE.txt", None),
        ],
        download_root,
    )
    downloaded = sorted(path.name for path in download_root.iterdir() if path.is_file())
    if "README.md" not in downloaded or "LICENSE.txt" not in downloaded:
        raise RuntimeError(f"Multi-file download names are wrong: {downloaded}")
    return {
        "passed": True,
        "archive_parts": sorted(path.name for path in archive_root.iterdir()),
        "selected_files": [path.name for path in selected],
        "unwanted_files": [path.name for path in unwanted],
        "downloaded_files": downloaded,
    }


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={"/sft": volume},
)
def plan_agentnet_sft(train_count: int = 2_000, validation_count: int = 250) -> dict:
    from cua_jepa.sft_data import candidate_examples, dataset_audit, split_examples

    started = _utc_now()
    metadata = {
        str(item["task_id"]): item
        for item in _stream_jsonl(f"{HF_ROOT}/meta_data_merged.jsonl")
    }
    candidates = []
    source_counts = {}
    for source_file in TRAJECTORY_FILES:
        source_candidates = candidate_examples(
            _stream_jsonl(f"{HF_ROOT}/{source_file}"), metadata, source_file
        )
        source_counts[source_file] = len(source_candidates)
        candidates.extend(source_candidates)
        print(f"Parsed {source_file}: {len(source_candidates)} usable examples", flush=True)

    train, validation = split_examples(
        candidates,
        train_count=train_count,
        validation_count=validation_count,
        seed=20260810,
    )
    records = [item.to_dict() for item in train + validation]
    root = DATASET_ROOT
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "selection.jsonl"
    temporary = plan_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, plan_path)
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    audit = dataset_audit(train, validation)
    audit.update(
        {
            "created_at_utc": started,
            "agentnet_revision": AGENTNET_REVISION,
            "selection_sha256": plan_sha256,
            "sampling_policy": {
                "train_system_balance": "equal",
                "train_click_share_cap": 0.65,
                "validation_system_balance": "equal",
                "validation_action_mix": "source-like",
                "max_examples_per_task": 3,
            },
            "candidate_examples": len(candidates),
            "candidate_examples_by_source": source_counts,
            "candidate_examples_by_system": dict(
                sorted(Counter(item.system for item in candidates).items())
            ),
            "candidate_examples_by_action": dict(
                sorted(Counter(item.action_kind for item in candidates).items())
            ),
            "selected_by_source": dict(Counter(item.source_file for item in train + validation)),
            "status": "planned",
        }
    )
    _write_json(root / "selection_audit.json", audit)
    volume.commit()
    return audit


def _download_archives(source_file: str, destination: Path) -> Path:
    directory, names = ARCHIVES[source_file]
    files = [
        (
            name,
            f"{HF_ROOT}/{directory}/{name}",
            FINAL_ARCHIVE_SIZES[source_file] if name == "images.zip" else 5_368_709_120,
        )
        for name in names
    ]
    _parallel_download(files, destination)
    missing = [name for name in names if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Archive download is incomplete: {missing}")
    return destination / "images.zip"


def _extract_selected(archive: Path, names: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    include_file = archive.parent / "selected-images.txt"
    include_file.write_text("\n".join(sorted(set(names))) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "7z",
            "x",
            str(archive),
            f"-o{destination}",
            f"-ir@{include_file}",
            "-y",
        ],
        check=True,
    )


@app.function(
    image=image,
    cpu=8,
    memory=16_384,
    ephemeral_disk=512 * 1024,
    timeout=8 * 60 * 60,
    volumes={"/sft": volume},
)
def ingest_agentnet_images() -> dict:
    from PIL import Image

    root = DATASET_ROOT
    selection_path = root / "selection.jsonl"
    audit_path = root / "selection_audit.json"
    if not selection_path.is_file() or not audit_path.is_file():
        raise RuntimeError("Run plan_agentnet_sft before image ingestion")
    records = [json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines()]
    expected_sha256 = json.loads(audit_path.read_text(encoding="utf-8"))["selection_sha256"]
    if hashlib.sha256(selection_path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("The SFT selection file changed after its audit")

    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        stored_name = hashlib.sha256(record["example_id"].encode()).hexdigest()[:24] + ".webp"
        record["stored_image"] = f"images/{stored_name}"

    scratch = Path("/tmp/agentnet")
    for source_file in TRAJECTORY_FILES:
        source_records = [record for record in records if record["source_file"] == source_file]
        pending_records = [
            record
            for record in source_records
            if not (root / record["stored_image"]).is_file()
        ]
        if not pending_records:
            print(f"All {len(source_records)} {source_file} images already exist", flush=True)
            continue
        selected_names = [record["image_file"] for record in pending_records]
        source_scratch = scratch / source_file.removesuffix(".jsonl")
        archive = _download_archives(source_file, source_scratch)
        extracted = scratch / f"extracted-{source_file.removesuffix('.jsonl')}"
        _extract_selected(archive, selected_names, extracted)
        source_images: dict[str, Path] = {}
        selected_name_set = set(selected_names)
        for path in extracted.rglob("*"):
            if path.is_file() and path.name in selected_name_set:
                source_images[path.name] = path
        missing = [
            record["example_id"]
            for record in pending_records
            if record["image_file"] not in source_images
        ]
        if missing:
            raise RuntimeError(
                f"The archive did not contain {len(missing)} selected images: {missing[:10]}"
            )
        for index, record in enumerate(pending_records, start=1):
            destination = root / record["stored_image"]
            with Image.open(source_images[record["image_file"]]) as opened:
                opened.convert("RGB").save(destination, format="WEBP", quality=90, method=6)
            with Image.open(destination) as verified:
                verified.verify()
            if index % 250 == 0:
                volume.commit()
                print(
                    f"Stored {index}/{len(pending_records)} images from {source_file}",
                    flush=True,
                )
        volume.commit()
        print(f"Stored all {len(source_records)} images from {source_file}", flush=True)
        shutil.rmtree(source_scratch, ignore_errors=True)
        shutil.rmtree(extracted, ignore_errors=True)

    image_manifest = []
    total_bytes = 0
    missing = []
    for record in records:
        destination = root / record["stored_image"]
        if not destination.is_file():
            missing.append(record["example_id"])
            continue
        image_bytes = destination.read_bytes()
        with Image.open(destination) as opened:
            width, height = opened.size
            opened.verify()
        total_bytes += len(image_bytes)
        image_manifest.append(
            {
                "example_id": record["example_id"],
                "stored_image": record["stored_image"],
                "width": width,
                "height": height,
                "bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
    if missing:
        raise RuntimeError(f"Missing {len(missing)} stored images: {missing[:10]}")

    image_sha_by_example = {
        item["example_id"]: item["sha256"] for item in image_manifest
    }
    train_image_hashes = {
        image_sha_by_example[record["example_id"]]
        for record in records
        if record["split"] == "train"
    }
    validation_image_hashes = {
        image_sha_by_example[record["example_id"]]
        for record in records
        if record["split"] == "validation"
    }
    exact_image_overlap = train_image_hashes & validation_image_hashes
    if exact_image_overlap:
        raise RuntimeError(
            f"Train and validation contain {len(exact_image_overlap)} exact screenshot hashes"
        )

    dataset_path = root / "dataset.jsonl"
    temporary = dataset_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, dataset_path)
    _write_json(root / "image_manifest.json", image_manifest)
    license_text = next(_stream_text(f"{HF_ROOT}/LICENSE.txt"))
    (root / "AGENTNET_LICENSE.txt").write_text(license_text, encoding="utf-8")

    final_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    final_audit.update(
        {
            "completed_at_utc": _utc_now(),
            "status": "ready",
            "stored_images": len(image_manifest),
            "stored_image_bytes": total_bytes,
            "unique_stored_image_hashes": len(set(image_sha_by_example.values())),
            "duplicate_stored_images": len(records) - len(set(image_sha_by_example.values())),
            "exact_train_validation_image_hash_overlap": len(exact_image_overlap),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "missing_images": 0,
        }
    )
    _write_json(root / "dataset_audit.json", final_audit)
    volume.commit()
    shutil.rmtree(scratch, ignore_errors=True)
    return final_audit


def _stream_text(url: str):
    import requests

    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        yield response.text


@app.local_entrypoint()
def main(mode: str = "plan") -> None:
    if mode == "smoke":
        result = smoke_test_split_archive.remote()
    elif mode == "plan":
        result = plan_agentnet_sft.remote()
    elif mode == "ingest":
        result = ingest_agentnet_images.remote()
    else:
        raise ValueError("mode must be 'smoke', 'plan', or 'ingest'")
    print(json.dumps(result, indent=2, sort_keys=True))
