from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parent
AUDIT_REPORT = (
    ROOT
    / "data"
    / "synthetic"
    / "clean-20260808-v7"
    / "audit_report.json"
)
APP_NAME = "cua-jepa-model6"
TRAINING_VOLUME_NAME = "cua-jepa-training-v1"
DATA_VOLUME_NAME = "cua-jepa-synthetic-v1"
SFT_VOLUME_NAME = "cua-jepa-sft-v1"
HF_CACHE_VOLUME_NAME = "hf-cache"

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "accelerate>=1.13,<2",
            "numpy>=2,<3",
            "peft>=0.19,<1",
            "pillow>=10,<13",
            "safetensors>=0.5,<1",
            "sentence-transformers>=5,<6",
            "torch>=2.8,<3",
            "torchvision>=0.23,<1",
            "transformers>=4.57,<5",
        )
        .add_local_python_source("cua_jepa", copy=True)
    )
else:
    image = modal.Image.debian_slim()

app = modal.App(APP_NAME)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME)
sft_volume = modal.Volume.from_name(SFT_VOLUME_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _audit_sha256() -> str:
    return hashlib.sha256(AUDIT_REPORT.read_bytes()).hexdigest()


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=75 * 60,
    scaledown_window=60,
    volumes={
        "/training": training_volume,
        "/dataset": data_volume,
        "/sft": sft_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_model6_dynamics(
    mode: str,
    seed: int,
    git_commit: str,
    source_dataset_audit_sha256: str,
    cost_limit_usd: float,
) -> dict:
    from cua_jepa.train_vjepa2 import VJEPA2PilotConfig, train_vjepa2_gui_pilot

    if mode not in {"smoke", "pilot", "full"}:
        raise ValueError("Model 6 dynamics mode must be 'smoke', 'pilot', or 'full'")
    config = VJEPA2PilotConfig(
        architecture_name="model6",
        seed=seed,
        feature_cache_dir="/training/vjepa2-feature-cache",
        memory_gib=24.0,
        action_separation_weight=0.25,
        model6_inverse_loss_weight=0.25,
        approved_cost_limit_usd=min(cost_limit_usd, 5.0),
    )
    if mode == "smoke":
        config.max_train_transitions = 8
        config.max_validation_transitions = 8
        config.max_steps = 2
        config.encoder_bundle_batch_size = 2
        config.train_evaluation_bundles = 2
        config.validation_evaluation_bundles = 2
        config.evaluation_steps = (0, 1, 2)
        config.log_every = 1
        config.max_runtime_seconds = 20 * 60
        config.approved_cost_limit_usd = min(cost_limit_usd, 0.75)
    elif mode == "pilot":
        config.max_train_transitions = 2_000
        config.max_validation_transitions = 500
        config.max_steps = 500
        config.encoder_bundle_batch_size = 8
        config.train_evaluation_bundles = 25
        config.validation_evaluation_bundles = 125
        config.evaluation_steps = (0, 100, 250, 500)
        config.log_every = 25
        config.max_runtime_seconds = 45 * 60
        config.approved_cost_limit_usd = min(cost_limit_usd, 2.0)
    else:
        config.max_train_transitions = 30_668
        config.max_validation_transitions = 2_000
        config.max_steps = 7_667
        config.encoder_bundle_batch_size = 8
        config.train_evaluation_bundles = 25
        config.validation_evaluation_bundles = 500
        config.evaluation_steps = (0, 100, 250, 500, 1_000, 2_000, 4_000, 7_667)
        config.log_every = 100
        config.max_runtime_seconds = 65 * 60

    train_paths = [
        str(path) for path in sorted(Path("/dataset/model4-full/train").rglob("*.tar"))
    ]
    validation_paths = [
        str(path)
        for path in sorted(Path("/dataset/model4-full/validation").rglob("*.tar"))
    ]
    if len(train_paths) != 320 or len(validation_paths) != 20:
        raise RuntimeError(
            "Model 6 requires 320 training archives and 20 validation archives"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"model6-dynamics-{mode}-seed{seed}-{timestamp}"
    output_dir = Path("/training") / run_id
    metadata = {
        "run_id": run_id,
        "run_mode": f"model6_dynamics_{mode}",
        "model_variant": "model6",
        "git_commit": git_commit,
        "dataset_id": "clean-20260808-v7",
        "source_dataset_audit_sha256": source_dataset_audit_sha256,
        "modal_app_name": APP_NAME,
        "modal_app_id": app.app_id or os.environ.get("MODAL_APP_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        result = train_vjepa2_gui_pilot(
            train_paths,
            validation_paths,
            output_dir,
            config,
            metadata,
            training_volume.commit,
        )
    except BaseException as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stop_reason.json").write_text(
            json.dumps(
                {
                    "reason": "exception",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        training_volume.commit()
        raise
    result["run_id"] = run_id
    result["modal_output_dir"] = str(output_dir)
    return result


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    seed: int = 20260813,
    cost_limit_usd: float = 20.0,
) -> None:
    result = run_model6_dynamics.remote(
        mode,
        seed,
        _git_commit(),
        _audit_sha256(),
        cost_limit_usd,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
