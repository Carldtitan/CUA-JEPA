from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    # Avoid intermittent pycares failures against Modal's blob-upload endpoint.
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "configs" / "model4_jepa_pilot.json"
DATA_ROOT = ROOT / "data" / "synthetic" / "clean-20260808-v7" / "full"
TRAIN_TAR = (
    DATA_ROOT
    / "train"
    / "github_mock"
    / "train-github_mock-shard-0000.tar"
)
VALIDATION_TAR = (
    DATA_ROOT
    / "validation"
    / "jira_mock"
    / "validation-jira_mock-shard-0000.tar"
)


def _diverse_pilot_tars(split: str, per_app: int, extras: int) -> tuple[Path, ...]:
    selected: list[Path] = []
    app_directories = sorted(path for path in (DATA_ROOT / split).iterdir() if path.is_dir())
    for app_directory in app_directories:
        selected.extend(sorted(app_directory.glob("*.tar"))[:per_app])
    for app_directory in app_directories[:extras]:
        candidates = sorted(app_directory.glob("*.tar"))
        if len(candidates) > per_app:
            selected.append(candidates[per_app])
    return tuple(selected)


if modal.is_local():
    PILOT_TRAIN_TARS = _diverse_pilot_tars("train", per_app=2, extras=4)
    PILOT_VALIDATION_TARS = _diverse_pilot_tars("validation", per_app=2, extras=1)
else:
    PILOT_TRAIN_TARS = ()
    PILOT_VALIDATION_TARS = ()

APP_NAME = "cua-jepa-model4-pilot"
OUTPUT_VOLUME_NAME = "cua-jepa-training-v1"
HF_CACHE_VOLUME_NAME = "hf-cache"
DATA_VOLUME_NAME = "cua-jepa-synthetic-v1"


def _require_inputs() -> None:
    required = (CONFIG_PATH, TRAIN_TAR, VALIDATION_TAR)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing Model 4 pilot inputs:\n" + "\n".join(map(str, missing)))


if modal.is_local():
    _require_inputs()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "accelerate>=1.13,<2",
            "numpy>=2,<3",
            "peft>=0.19,<1",
            "pillow>=10,<13",
            "safetensors>=0.5,<1",
            "torch>=2.8,<3",
            "transformers>=4.57,<5",
        )
        .add_local_python_source("cua_jepa", copy=True)
        .add_local_file(CONFIG_PATH, "/opt/cua-jepa/model4_jepa_pilot.json", copy=True)
        .add_local_file(TRAIN_TAR, "/opt/cua-jepa/train.tar", copy=True)
        .add_local_file(VALIDATION_TAR, "/opt/cua-jepa/validation.tar", copy=True)
    )
else:
    image = modal.Image.debian_slim()

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def check_processor() -> dict:
    from transformers import AutoImageProcessor

    from cua_jepa.jepa_data import load_transition_tar

    sample = load_transition_tar("/opt/cua-jepa/train.tar", limit=1)[0]
    processor = AutoImageProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", use_fast=False
    )
    processor.max_pixels = 262_144
    processor.min_pixels = 65_536
    values = processor(images=[sample.current_image(), sample.future_image()], return_tensors="pt")
    return {
        "pixel_values_shape": list(values["pixel_values"].shape),
        "image_grid_thw": values["image_grid_thw"].tolist(),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/training": output_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/dataset": data_volume,
    },
)
def run_model4_pilot(mode: str = "smoke") -> dict:
    from cua_jepa.train_jepa import JEPATrainConfig, train_model4_jepa

    config = JEPATrainConfig.from_json("/opt/cua-jepa/model4_jepa_pilot.json")
    if mode == "smoke":
        config.max_steps = 2
        config.max_train_transitions = 8
        config.max_validation_transitions = 8
        config.evaluation_bundles = 1
        config.validation_evaluation_bundles = 0
        config.log_every = 1
    elif mode == "stage2":
        config.max_steps = 2_000
        config.max_train_transitions = 2_000
        config.max_validation_transitions = 500
        config.evaluation_bundles = 25
        config.validation_evaluation_bundles = 0
        config.log_every = 100
    elif mode == "quick":
        config.max_steps = 100
        config.max_train_transitions = 100
        config.max_validation_transitions = 500
        config.evaluation_bundles = 8
        config.validation_evaluation_bundles = 0
        config.log_every = 10
    elif mode in {"pure", "separation"}:
        config.max_steps = 500
        config.max_train_transitions = 2_000
        config.max_validation_transitions = 500
        config.evaluation_bundles = 25
        config.validation_evaluation_bundles = 0
        config.action_separation_weight = 0.0 if mode == "pure" else 0.25
        config.log_every = 25
    elif mode != "pilot":
        raise ValueError(
            "mode must be 'smoke', 'pilot', 'quick', 'pure', 'separation', or 'stage2'"
        )
    run_id = datetime.now(timezone.utc).strftime(f"model4-{mode}-%Y%m%dT%H%M%SZ")
    output_dir = Path("/training") / run_id
    train_paths = ["/opt/cua-jepa/train.tar"]
    validation_paths = ["/opt/cua-jepa/validation.tar"]
    if mode in {"quick", "pure", "separation", "stage2"}:
        train_paths = [
            str(path) for path in sorted(Path("/dataset/model4-stage2/train").glob("*.tar"))
        ]
        validation_paths = [
            str(path)
            for path in sorted(Path("/dataset/model4-stage2/validation").glob("*.tar"))
        ]
    metrics = train_model4_jepa(
        train_tar_paths=train_paths,
        validation_tar_paths=validation_paths,
        output_dir=output_dir,
        config=config,
    )
    output_volume.commit()
    metrics["run_id"] = run_id
    metrics["modal_output_dir"] = str(output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@app.local_entrypoint()
def main(mode: str = "deps") -> None:
    if mode == "deps":
        print(json.dumps(check_processor.remote(), indent=2))
        return
    metrics = run_model4_pilot.remote(mode)
    print(json.dumps(metrics, indent=2))
