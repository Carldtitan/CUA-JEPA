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
    # Avoid intermittent pycares failures against Modal's blob-upload endpoint.
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "configs" / "model4_jepa_pilot.json"
DATA_ROOT = ROOT / "data" / "synthetic" / "clean-20260808-v7" / "full"
TRAIN_TAR = DATA_ROOT / "train" / "github_mock" / "train-github_mock-shard-0000.tar"
VALIDATION_TAR = DATA_ROOT / "validation" / "jira_mock" / "validation-jira_mock-shard-0000.tar"
AUDIT_REPORT = DATA_ROOT.parent / "audit_report.json"


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

APP_NAME = "cua-jepa-model4-training"
OUTPUT_VOLUME_NAME = "cua-jepa-training-v1"
HF_CACHE_VOLUME_NAME = "hf-cache"
DATA_VOLUME_NAME = "cua-jepa-synthetic-v1"


def _require_inputs() -> None:
    required = (CONFIG_PATH, TRAIN_TAR, VALIDATION_TAR, AUDIT_REPORT)
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
            "torchvision>=0.23,<1",
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
        "Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        use_fast=False,
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
    timeout=30 * 60,
    scaledown_window=60,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/dataset": data_volume,
    },
)
def check_vjepa2_encoder() -> dict:
    import torch
    from transformers import AutoModel, AutoVideoProcessor

    from cua_jepa.jepa_data import load_transition_tar
    from cua_jepa.train_vjepa2 import gui_image_video

    model_id = "facebook/vjepa2-vitl-fpc64-256"
    paths = sorted(Path("/dataset/model4-stage2/train").glob("*.tar"))
    if not paths:
        raise RuntimeError("The Model 4 stage-2 data volume is empty")
    branches = load_transition_tar(paths[0], limit=4)
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
    model.requires_grad_(False).eval().to("cuda")

    def encode_many(images) -> tuple[torch.Tensor, dict[str, list[int]]]:
        inputs = processor(
            [gui_image_video(image) for image in images],
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        input_shapes = {key: list(value.shape) for key, value in inputs.items()}
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            output = model(**inputs, skip_predictor=True)
        return output.last_hidden_state.float().cpu(), input_shapes

    currents, input_shapes = encode_many([branches[0].current_image(), branches[0].current_image()])
    futures, _ = encode_many([branch.future_image() for branch in branches])
    current_first = currents[0]
    current_second = currents[1]
    pair_distances: list[float] = []
    for first in range(4):
        for second in range(first + 1, 4):
            pair_distances.append(
                float(torch.nn.functional.mse_loss(futures[first], futures[second]).item())
            )
    return {
        "model_id": model_id,
        "processor_input_shapes": input_shapes,
        "hidden_shape": list(currents.shape),
        "dtype": str(current_first.dtype),
        "repeat_max_difference": float((current_first - current_second).abs().max().item()),
        "future_pair_mse_min": min(pair_distances),
        "future_pair_mse_mean": sum(pair_distances) / len(pair_distances),
        "future_pair_mse_max": max(pair_distances),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16_384,
    timeout=4 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/training": output_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/dataset": data_volume,
    },
)
def run_model4_training(
    mode: str = "smoke",
    seed: int = 0,
    git_commit: str = "unknown",
    source_dataset_audit_sha256: str = "unknown",
) -> dict:
    from cua_jepa.train_jepa import JEPATrainConfig, train_model4_jepa

    config = JEPATrainConfig.from_json("/opt/cua-jepa/model4_jepa_pilot.json")
    if seed:
        config.seed = seed
    if mode == "smoke":
        config.max_steps = 2
        config.max_train_transitions = 8
        config.max_validation_transitions = 8
        config.evaluation_bundles = 1
        config.validation_evaluation_bundles = 0
        config.log_every = 1
    elif mode == "lora_smoke":
        config.train_qwen_lora = True
        config.action_separation_weight = 0.25
        config.max_steps = 2
        config.max_train_transitions = 8
        config.max_validation_transitions = 8
        config.expected_train_transitions = 8
        config.expected_validation_transitions = 8
        config.evaluation_bundles = 1
        config.validation_evaluation_bundles = 2
        config.monitor_evaluation_bundles = 2
        config.evaluation_steps = [0, 1, 2]
        config.evaluation_every = 0
        config.checkpoint_every = 1
        config.collapse_check_every = 1
        config.collapse_check_start_step = 1
        config.collapse_patience = 10
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
    elif mode == "model4_full":
        config.train_qwen_lora = True
        config.action_separation_weight = 0.25
        config.max_steps = 7_667
        config.max_train_transitions = 30_668
        config.max_validation_transitions = 2_000
        config.expected_train_transitions = 30_668
        config.expected_validation_transitions = 2_000
        config.evaluation_bundles = 25
        config.validation_evaluation_bundles = 0
        config.monitor_evaluation_bundles = 125
        config.evaluation_steps = [0, 100, 250, 500]
        config.evaluation_every = 500
        config.checkpoint_every = 500
        config.collapse_check_every = 500
        config.collapse_check_bundles = 25
        config.collapse_check_start_step = 1_000
        config.log_every = 25
    elif mode != "pilot":
        raise ValueError(
            "mode must be 'smoke', 'lora_smoke', 'pilot', 'quick', 'pure', "
            "'separation', 'stage2', or 'model4_full'"
        )
    seed_label = f"-seed{config.seed}"
    run_id = datetime.now(timezone.utc).strftime(f"model4-{mode}{seed_label}-%Y%m%dT%H%M%SZ")
    output_dir = Path("/training") / run_id
    train_paths = ["/opt/cua-jepa/train.tar"]
    validation_paths = ["/opt/cua-jepa/validation.tar"]
    if mode in {"quick", "pure", "separation", "stage2", "lora_smoke"}:
        train_paths = [
            str(path) for path in sorted(Path("/dataset/model4-stage2/train").glob("*.tar"))
        ]
        validation_paths = [
            str(path) for path in sorted(Path("/dataset/model4-stage2/validation").glob("*.tar"))
        ]
    elif mode == "model4_full":
        train_paths = [
            str(path) for path in sorted(Path("/dataset/model4-full/train").rglob("*.tar"))
        ]
        validation_paths = [
            str(path) for path in sorted(Path("/dataset/model4-full/validation").rglob("*.tar"))
        ]
        if len(train_paths) != 320 or len(validation_paths) != 20:
            raise RuntimeError(
                "Full Model 4 data is incomplete: "
                f"{len(train_paths)} train tar files and "
                f"{len(validation_paths)} validation tar files"
            )
        if any("/test/" in path for path in train_paths + validation_paths):
            raise RuntimeError("The final test split entered a training path")
    run_metadata = {
        "run_id": run_id,
        "run_mode": mode,
        "git_commit": git_commit,
        "dataset_id": "clean-20260808-v7",
        "source_dataset_audit_sha256": source_dataset_audit_sha256,
        "modal_app_name": APP_NAME,
        "modal_app_id": app.app_id or os.environ.get("MODAL_APP_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        metrics = train_model4_jepa(
            train_tar_paths=train_paths,
            validation_tar_paths=validation_paths,
            output_dir=output_dir,
            config=config,
            run_metadata=run_metadata,
            persistence_callback=output_volume.commit,
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
        output_volume.commit()
        raise
    metrics["run_id"] = run_id
    metrics["modal_output_dir"] = str(output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=75 * 60,
    scaledown_window=60,
    volumes={
        "/training": output_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/dataset": data_volume,
    },
)
def run_vjepa2_gui_pilot(
    mode: str = "smoke",
    seed: int = 0,
    git_commit: str = "unknown",
    source_dataset_audit_sha256: str = "unknown",
) -> dict:
    from cua_jepa.train_vjepa2 import VJEPA2PilotConfig, train_vjepa2_gui_pilot

    if mode not in {
        "smoke",
        "action_token_smoke",
        "pure",
        "separation",
        "scaled_separation",
        "scaled_action_token",
        "tiled_smoke",
        "tiled_separation",
        "independent_tiled_smoke",
        "independent_tiled_separation",
    }:
        raise ValueError("Unsupported V-JEPA 2 GUI pilot mode")
    config = VJEPA2PilotConfig()
    config.memory_gib = 24.0
    if seed:
        config.seed = seed
    if mode in {
        "smoke",
        "action_token_smoke",
        "tiled_smoke",
        "independent_tiled_smoke",
    }:
        config.max_steps = 2
        config.max_train_transitions = 8
        config.max_validation_transitions = 8
        config.encoder_bundle_batch_size = 2
        config.train_evaluation_bundles = 2
        config.validation_evaluation_bundles = 2
        config.evaluation_steps = (0, 1, 2)
        config.log_every = 1
        config.approved_cost_limit_usd = 0.50
        config.max_runtime_seconds = 20 * 60
        if mode == "action_token_smoke":
            config.predictor_architecture = "action_token_spatial"
        elif mode == "tiled_smoke":
            config.screen_views = "two_tiles"
            config.predictor_architecture = "tiled_adaln_spatial"
        elif mode == "independent_tiled_smoke":
            config.screen_views = "two_tiles"
            config.predictor_architecture = "independent_tiled_adaln_spatial"
    elif mode == "pure":
        config.action_separation_weight = 0.0
    elif mode == "separation":
        config.action_separation_weight = 0.25
    else:
        config.action_separation_weight = 0.25
        config.max_train_transitions = 8_000
        config.max_steps = 2_000
        config.encoder_bundle_batch_size = 8
        config.evaluation_steps = (0, 100, 250, 500, 1_000, 2_000)
        config.log_every = 50
        if mode == "scaled_action_token":
            config.predictor_architecture = "action_token_spatial"
        elif mode == "tiled_separation":
            config.screen_views = "two_tiles"
            config.predictor_architecture = "tiled_adaln_spatial"
            config.encoder_bundle_batch_size = 4
        elif mode == "independent_tiled_separation":
            config.screen_views = "two_tiles"
            config.predictor_architecture = "independent_tiled_adaln_spatial"
            config.encoder_bundle_batch_size = 4

    if mode in {
        "scaled_separation",
        "scaled_action_token",
        "tiled_separation",
        "independent_tiled_separation",
    }:
        train_paths = [
            str(path) for path in sorted(Path("/dataset/model4-full/train").rglob("*.tar"))
        ]
        if len(train_paths) != 320:
            raise RuntimeError(f"Expected 320 full training tar files, found {len(train_paths)}")
    else:
        train_paths = [
            str(path) for path in sorted(Path("/dataset/model4-stage2/train").glob("*.tar"))
        ]
    validation_paths = [
        str(path) for path in sorted(Path("/dataset/model4-full/validation").rglob("*.tar"))
    ]
    if not train_paths:
        raise RuntimeError("The Model 4 stage-2 training data is empty")
    if len(validation_paths) != 20:
        raise RuntimeError(f"Expected 20 full validation tar files, found {len(validation_paths)}")
    if any("/test/" in path for path in train_paths + validation_paths):
        raise RuntimeError("The test split entered a V-JEPA 2 training path")

    seed_label = f"-seed{config.seed}"
    run_id = datetime.now(timezone.utc).strftime(f"vjepa2-gui-{mode}{seed_label}-%Y%m%dT%H%M%SZ")
    output_dir = Path("/training") / run_id
    run_metadata = {
        "run_id": run_id,
        "run_mode": f"vjepa2_gui_{mode}",
        "git_commit": git_commit,
        "dataset_id": "clean-20260808-v7",
        "source_dataset_audit_sha256": source_dataset_audit_sha256,
        "modal_app_name": APP_NAME,
        "modal_app_id": app.app_id or os.environ.get("MODAL_APP_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        metrics = train_vjepa2_gui_pilot(
            train_tar_paths=train_paths,
            validation_tar_paths=validation_paths,
            output_dir=output_dir,
            config=config,
            run_metadata=run_metadata,
            persistence_callback=output_volume.commit,
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
        output_volume.commit()
        raise
    metrics["run_id"] = run_id
    metrics["modal_output_dir"] = str(output_dir)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


@app.local_entrypoint()
def main(mode: str = "deps", seed: int = 0) -> None:
    if mode == "deps":
        print(json.dumps(check_processor.remote(), indent=2))
        return
    if mode == "vjepa2_encoder_smoke":
        print(json.dumps(check_vjepa2_encoder.remote(), indent=2))
        return
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    source_dataset_audit_sha256 = hashlib.sha256(AUDIT_REPORT.read_bytes()).hexdigest()
    if mode in {
        "vjepa2_gui_smoke",
        "vjepa2_gui_action_token_smoke",
        "vjepa2_gui_pure",
        "vjepa2_gui_separation",
        "vjepa2_gui_scaled_separation",
        "vjepa2_gui_scaled_action_token",
        "vjepa2_gui_tiled_smoke",
        "vjepa2_gui_tiled_separation",
        "vjepa2_gui_independent_tiled_smoke",
        "vjepa2_gui_independent_tiled_separation",
    }:
        vjepa2_mode = mode.removeprefix("vjepa2_gui_")
        metrics = run_vjepa2_gui_pilot.remote(
            vjepa2_mode,
            seed,
            git_commit,
            source_dataset_audit_sha256,
        )
        print(json.dumps(metrics, indent=2))
        return
    metrics = run_model4_training.remote(mode, seed, git_commit, source_dataset_audit_sha256)
    print(json.dumps(metrics, indent=2))
