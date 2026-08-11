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


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=4 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/training": training_volume,
        "/sft": sft_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def generate_model6_candidates(
    mode: str,
    seed: int,
    git_commit: str,
    cost_limit_usd: float,
    resume_run_id: str = "",
) -> dict:
    import time

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from cua_jepa.model6_data import (
        build_candidate_record,
        candidate_generation_metrics,
        validate_candidate_resume_prefix,
    )
    from cua_jepa.observability import estimated_modal_cost, utc_now
    from cua_jepa.train_sft import SFTTrainConfig, _load_dataset, make_generation_inputs

    if mode not in {"candidate_smoke", "candidates"}:
        raise ValueError("Candidate mode must be 'candidate_smoke' or 'candidates'")
    started = time.perf_counter()
    config = SFTTrainConfig(seed=seed, max_new_tokens=64)
    dataset_root = Path("/sft/agentnet-v1")
    train, validation, _, dataset_audit = _load_dataset(dataset_root)
    if mode == "candidate_smoke":
        records = train[:2] + validation[:2]
        phase_cost_limit = min(cost_limit_usd, 1.0)
    else:
        records = train + validation
        phase_cost_limit = min(cost_limit_usd, 8.0)
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=False,
    )
    processor.image_processor.max_pixels = config.max_pixels
    processor.image_processor.min_pixels = config.min_pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False).eval().to("cuda")
    model.config.use_cache = True
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Raw Qwen is not completely frozen")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = resume_run_id or f"model6-{mode}-raw-qwen-seed{seed}-{timestamp}"
    output_dir = Path("/training") / run_id
    candidate_path = output_dir / "candidates.jsonl"
    timing_path = output_dir / "generation_timing.jsonl"
    if resume_run_id:
        if mode != "candidates" or not candidate_path.is_file():
            raise RuntimeError("Model 6 can resume only a saved full candidate run")
        saved_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if (
            int(saved_config["seed"]) != seed
            or saved_config["dataset_sha256"] != dataset_audit["dataset_sha256"]
        ):
            raise RuntimeError("Model 6 resume configuration does not match this run")
        outputs = [
            json.loads(line)
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
        ]
        completed = validate_candidate_resume_prefix(records, outputs)
        saved_timings = [
            json.loads(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(saved_timings) != completed:
            raise RuntimeError("Model 6 resume timing count does not match candidate count")
        total_greedy_seconds = sum(value["greedy_seconds"] for value in saved_timings)
        total_sample_seconds = sum(value["sample_batch_seconds"] for value in saved_timings)
        with (output_dir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "resumed_at_utc": utc_now(),
                        "completed_examples": completed,
                        "git_commit": git_commit,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_variant": "model6",
                    "qwen_policy": "raw_base",
                    "qwen_model_id": config.model_id,
                    "qwen_model_revision": config.model_revision,
                    "seed": seed,
                    "mode": mode,
                    "maximum_candidates": 4,
                    "sample_count": 7,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "git_commit": git_commit,
                    "dataset_sha256": dataset_audit["dataset_sha256"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs = []
        completed = 0
        total_greedy_seconds = 0.0
        total_sample_seconds = 0.0

    for index, record in enumerate(records[completed:], start=completed + 1):
        elapsed = time.perf_counter() - started
        cost = estimated_modal_cost(
            elapsed,
            0.000222,
            4.0,
            0.00003942,
            24.0,
            0.00000667,
        )
        if cost["total_usd"] >= phase_cost_limit:
            raise RuntimeError("Model 6 candidate generation reached its cost limit")
        image_path = dataset_root / record["stored_image"]
        with Image.open(image_path) as opened:
            width, height = opened.size
        inputs = make_generation_inputs(processor, record, image_path, torch.device("cuda"))
        input_length = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        greedy_started = time.perf_counter()
        with torch.inference_mode():
            greedy_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        torch.cuda.synchronize()
        greedy_seconds = time.perf_counter() - greedy_started
        greedy_text = processor.batch_decode(
            greedy_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        torch.manual_seed(
            int.from_bytes(hashlib.sha256(f"{seed}:{record['example_id']}".encode()).digest()[:8])
            % (2**63 - 1)
        )
        torch.cuda.synchronize()
        sample_started = time.perf_counter()
        with torch.inference_mode():
            sampled_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=7,
                use_cache=True,
            )
        torch.cuda.synchronize()
        sample_seconds = time.perf_counter() - sample_started
        sampled_texts = processor.batch_decode(
            sampled_ids[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        candidate = build_candidate_record(
            record,
            greedy_text,
            sampled_texts,
            width,
            height,
            inject_training_oracle=record["split"] == "train",
        )
        with candidate_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
        timing = {
            "recorded_at_utc": utc_now(),
            "example_id": record["example_id"],
            "index": index,
            "greedy_seconds": greedy_seconds,
            "sample_batch_seconds": sample_seconds,
            "candidate_count": candidate["candidate_count"],
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with timing_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(timing, sort_keys=True) + "\n")
        outputs.append(candidate)
        total_greedy_seconds += greedy_seconds
        total_sample_seconds += sample_seconds
        del inputs, greedy_ids, sampled_ids
        if index == 1 or index % 25 == 0 or index == len(records):
            print(
                json.dumps(
                    {
                        "event": "model6_candidate_generation",
                        "completed": index,
                        "total": len(records),
                        "elapsed_seconds": time.perf_counter() - started,
                        "latest_candidate_count": candidate["candidate_count"],
                        "estimated_cost_usd": cost,
                    }
                ),
                flush=True,
            )
        if index % 100 == 0:
            training_volume.commit()

    elapsed = time.perf_counter() - started
    final_cost = estimated_modal_cost(
        elapsed,
        0.000222,
        4.0,
        0.00003942,
        24.0,
        0.00000667,
    )
    split_metrics = {
        split: candidate_generation_metrics(
            [record for record in outputs if record["split"] == split]
        )
        for split in sorted({record["split"] for record in outputs})
    }
    metrics = {
        "run_id": run_id,
        "model_variant": "model6",
        "qwen_frozen": True,
        "examples": len(outputs),
        "elapsed_seconds": elapsed,
        "mean_greedy_seconds": total_greedy_seconds / len(outputs),
        "mean_sample_batch_seconds": total_sample_seconds / len(outputs),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "estimated_modal_cost_usd": final_cost,
        "by_split": split_metrics,
        "completed_at_utc": utc_now(),
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    training_volume.commit()
    return {**metrics, "modal_output_dir": str(output_dir)}


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=2 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/training": training_volume,
        "/sft": sft_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def prepare_model6_agentnet_features(candidate_run_id: str) -> dict:
    import time

    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoVideoProcessor

    from cua_jepa.observability import utc_now
    from cua_jepa.train_sft import sha256_file
    from cua_jepa.train_vjepa2 import encode_screen_batch

    started = time.perf_counter()
    candidate_path = Path("/training") / candidate_run_id / "candidates.jsonl"
    if not candidate_path.is_file():
        raise RuntimeError(f"Model 6 candidates are missing: {candidate_path}")
    records = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise RuntimeError("Model 6 candidate data is empty")
    cache_key = hashlib.sha256(
        (
            sha256_file(candidate_path)
            + "|facebook/vjepa2-vitl-fpc64-256"
            + "|b3c1679b7c34d3255ef3547f27c7b226aefab26f"
            + "|sentence-transformers/all-MiniLM-L6-v2"
            + "|c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        ).encode()
    ).hexdigest()
    cache_dir = Path("/training/model6-agentnet-feature-cache")
    cache_path = cache_dir / f"{cache_key}.pt"
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cached.get("features", {})) != len(records):
            raise RuntimeError("Model 6 AgentNet feature cache has the wrong size")
        return {
            "cache_hit": True,
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "examples": len(records),
            "timing": cached.get("timing"),
        }

    device = torch.device("cuda")
    vjepa_revision = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
    vjepa_processor = AutoVideoProcessor.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256", revision=vjepa_revision
    )
    vjepa = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        revision=vjepa_revision,
        dtype=torch.bfloat16,
    )
    vjepa.requires_grad_(False).eval().to(device)
    goal_revision = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    goal_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision=goal_revision,
        device="cuda",
    )
    goal_encoder.requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in vjepa.parameters()) or any(
        parameter.requires_grad for parameter in goal_encoder.parameters()
    ):
        raise RuntimeError("A Model 6 feature encoder is trainable")

    features = {}
    dataset_root = Path("/sft/agentnet-v1")
    batch_size = 8
    total_vjepa_seconds = 0.0
    total_goal_seconds = 0.0
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        images = []
        for record in batch:
            with Image.open(dataset_root / record["stored_image"]) as opened:
                images.append(opened.convert("RGB"))
        torch.cuda.synchronize()
        vjepa_started = time.perf_counter()
        current = encode_screen_batch(images, vjepa_processor, vjepa, device)
        torch.cuda.synchronize()
        total_vjepa_seconds += time.perf_counter() - vjepa_started
        goal_started = time.perf_counter()
        goals = goal_encoder.encode(
            [record["instruction"] for record in batch],
            batch_size=len(batch),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).detach().to(device="cpu", dtype=torch.float16)
        torch.cuda.synchronize()
        total_goal_seconds += time.perf_counter() - goal_started
        for index, record in enumerate(batch):
            features[record["example_id"]] = {
                "current": current[index],
                "goal": goals[index],
            }
        if offset == 0 or offset + len(batch) == len(records) or offset % 200 == 0:
            print(
                json.dumps(
                    {
                        "event": "model6_feature_preparation",
                        "completed": offset + len(batch),
                        "total": len(records),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".incomplete")
    timing = {
        "vjepa_total_seconds": total_vjepa_seconds,
        "vjepa_mean_ms_per_screen": total_vjepa_seconds / len(records) * 1000.0,
        "goal_encoder_total_seconds": total_goal_seconds,
        "goal_encoder_mean_ms_per_task": total_goal_seconds / len(records) * 1000.0,
        "batch_size": batch_size,
    }
    torch.save(
        {
            "cache_key": cache_key,
            "candidate_run_id": candidate_run_id,
            "vjepa_revision": vjepa_revision,
            "goal_model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "goal_model_revision": goal_revision,
            "timing": timing,
            "features": features,
        },
        temporary,
    )
    temporary.replace(cache_path)
    training_volume.commit()
    return {
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "examples": len(records),
        "timing": timing,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "completed_at_utc": utc_now(),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=3 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/training": training_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_model6_scorer(
    mode: str,
    seed: int,
    dynamics_run_id: str,
    candidate_run_id: str,
    feature_cache_path: str,
    git_commit: str,
) -> dict:
    import torch

    from cua_jepa.model6_runtime import load_model6_dynamics
    from cua_jepa.train_model6_scorer import Model6ScorerConfig, train_model6_scorers

    if mode not in {"scorer_smoke", "scorer"}:
        raise ValueError("Scorer mode must be 'scorer_smoke' or 'scorer'")
    dynamics_path = (
        Path("/training") / dynamics_run_id / "model6_dynamics_heads.pt"
    )
    candidate_path = Path("/training") / candidate_run_id / "candidates.jsonl"
    for path in (dynamics_path, candidate_path, Path(feature_cache_path)):
        if not path.is_file():
            raise RuntimeError(f"Model 6 scorer input is missing: {path}")
    candidates = [
        json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    train = [record for record in candidates if record["split"] == "train"]
    validation = [record for record in candidates if record["split"] == "validation"]
    cached = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    features = cached["features"]
    if set(features) != {record["example_id"] for record in candidates}:
        raise RuntimeError("Model 6 scorer features do not match the candidates")
    config = Model6ScorerConfig(seed=seed)
    if mode == "scorer_smoke":
        config.maximum_epochs = 2
        config.early_stopping_patience = 1
        config.development_examples = 1
        config.log_every = 1
        train = train[:2]
        validation = validation[:2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"model6-{mode}-seed{seed}-{timestamp}"
    output_dir = Path("/training") / run_id
    device = torch.device("cuda")
    dynamics = load_model6_dynamics(dynamics_path, device)
    result = train_model6_scorers(
        train,
        validation,
        features,
        dynamics,
        output_dir,
        config,
        device,
        training_volume.commit,
    )
    result.update(
        {
            "run_id": run_id,
            "modal_output_dir": str(output_dir),
            "dynamics_run_id": dynamics_run_id,
            "candidate_run_id": candidate_run_id,
            "feature_cache_path": feature_cache_path,
            "git_commit": git_commit,
        }
    )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    training_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    seed: int = 20260813,
    cost_limit_usd: float = 20.0,
    dynamics_run_id: str = "",
    candidate_run_id: str = "",
    resume_run_id: str = "",
) -> None:
    if mode in {"candidate_smoke", "candidates"}:
        result = generate_model6_candidates.remote(
            mode,
            seed,
            _git_commit(),
            cost_limit_usd,
            resume_run_id,
        )
    elif mode in {"scorer_smoke", "scorer"}:
        if not dynamics_run_id or not candidate_run_id:
            raise ValueError("Scorer mode requires dynamics and candidate run IDs")
        prepared = prepare_model6_agentnet_features.remote(candidate_run_id)
        result = run_model6_scorer.remote(
            mode,
            seed,
            dynamics_run_id,
            candidate_run_id,
            prepared["cache_path"],
            _git_commit(),
        )
        result["feature_preparation"] = prepared
    else:
        result = run_model6_dynamics.remote(
            mode,
            seed,
            _git_commit(),
            _audit_sha256(),
            cost_limit_usd,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
