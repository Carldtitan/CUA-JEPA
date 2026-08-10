from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
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

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("aria2", "p7zip-full")
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
    destination.mkdir(parents=True, exist_ok=True)
    urls = [f"{HF_ROOT}/{directory}/{name}" for name in names]
    command = [
        "aria2c",
        "--continue=true",
        "--check-integrity=true",
        "--max-concurrent-downloads=4",
        "--split=8",
        "--max-connection-per-server=8",
        "--min-split-size=64M",
        "--file-allocation=none",
        f"--dir={destination}",
        *urls,
    ]
    subprocess.run(command, check=True)
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
    timeout=4 * 60 * 60,
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

    scratch = Path("/tmp/agentnet")
    extracted_roots: dict[str, Path] = {}
    for source_file in TRAJECTORY_FILES:
        selected_names = [
            record["image_file"] for record in records if record["source_file"] == source_file
        ]
        if not selected_names:
            continue
        archive = _download_archives(source_file, scratch / source_file.removesuffix(".jsonl"))
        extracted = scratch / f"extracted-{source_file.removesuffix('.jsonl')}"
        _extract_selected(archive, selected_names, extracted)
        extracted_roots[source_file] = extracted

    source_images: dict[tuple[str, str], Path] = {}
    for source_file, extracted in extracted_roots.items():
        selected_names = {
            record["image_file"] for record in records if record["source_file"] == source_file
        }
        for path in extracted.rglob("*"):
            if path.is_file() and path.name in selected_names:
                source_images[(source_file, path.name)] = path
    missing = [
        record["example_id"]
        for record in records
        if (record["source_file"], record["image_file"]) not in source_images
    ]
    if missing:
        raise RuntimeError(f"The archive did not contain {len(missing)} selected images: {missing[:10]}")

    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    image_manifest = []
    total_bytes = 0
    for index, record in enumerate(records, start=1):
        source = source_images[(record["source_file"], record["image_file"])]
        stored_name = hashlib.sha256(record["example_id"].encode()).hexdigest()[:24] + ".webp"
        destination = images_root / stored_name
        with Image.open(source) as opened:
            image_value = opened.convert("RGB")
            width, height = image_value.size
            image_value.save(destination, format="WEBP", quality=90, method=6)
        image_bytes = destination.read_bytes()
        total_bytes += len(image_bytes)
        record["stored_image"] = f"images/{stored_name}"
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
        if index % 500 == 0:
            volume.commit()
            print(f"Stored and verified {index}/{len(records)} images", flush=True)

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
    if mode == "plan":
        result = plan_agentnet_sft.remote()
    elif mode == "ingest":
        result = ingest_agentnet_images.remote()
    else:
        raise ValueError("mode must be 'plan' or 'ingest'")
    print(json.dumps(result, indent=2, sort_keys=True))
