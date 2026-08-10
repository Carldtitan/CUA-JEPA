from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from cua_jepa.jepa_data import TransitionSample, group_by_bundle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), allow_nan=False) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tar_file_manifest(paths: Iterable[str | Path], split: str) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    manifest_digest = hashlib.sha256()
    for path_value in sorted(map(Path, paths)):
        size = path_value.stat().st_size
        sha256 = file_sha256(path_value)
        parts = path_value.as_posix().split("/")
        display_path = "/".join(parts[-3:]) if len(parts) >= 3 else path_value.name
        row = {"path": display_path, "bytes": size, "sha256": sha256}
        files.append(row)
        manifest_digest.update(f"{display_path}\0{size}\0{sha256}\n".encode("utf-8"))
    return {
        "split": split,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "manifest_sha256": manifest_digest.hexdigest(),
        "files": files,
    }


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def changed_pixel_bucket(value: float) -> str:
    if value < 0.01:
        return "0-1%"
    if value < 0.05:
        return "1-5%"
    if value < 0.20:
        return "5-20%"
    if value < 0.50:
        return "20-50%"
    return "50-100%"


def safe_action_record(action: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(action.get("kind", "unknown"))
    record: dict[str, Any] = {"kind": kind}
    for key in (
        "x",
        "y",
        "x_normalized",
        "y_normalized",
        "delta_x",
        "delta_y",
        "amount",
        "direction",
    ):
        value = action.get(key)
        if isinstance(value, (int, float, str)):
            record[key] = value
    if kind == "type":
        text = str(action.get("text", ""))
        record["text_length"] = len(text)
        record["text_category"] = (
            "empty"
            if not text
            else "numeric"
            if text.isnumeric()
            else "alphanumeric"
            if text.isalnum()
            else "phrase"
        )
    return record


def dataset_profile(samples: Iterable[TransitionSample]) -> dict[str, Any]:
    values = list(samples)
    action_counts = Counter(str(sample.action.get("kind", "unknown")) for sample in values)
    changed_values = [float(sample.changed_pixel_fraction) for sample in values]
    changed_buckets = Counter(changed_pixel_bucket(value) for value in changed_values)
    groups = group_by_bundle(values)
    return {
        "transitions": len(values),
        "bundles": len(groups),
        "applications": sorted({sample.app for sample in values}),
        "application_transition_counts": dict(sorted(Counter(s.app for s in values).items())),
        "action_counts": dict(sorted(action_counts.items())),
        "changed_pixel_fraction": {
            "minimum": min(changed_values, default=0.0),
            "p25": _quantile(changed_values, 0.25),
            "median": _quantile(changed_values, 0.50),
            "p75": _quantile(changed_values, 0.75),
            "maximum": max(changed_values, default=0.0),
            "mean": sum(changed_values) / max(len(changed_values), 1),
            "buckets": dict(sorted(changed_buckets.items())),
        },
        "ordered_bundle_ids": list(groups),
    }


def named_tensors_sha256(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(values, key=lambda item: item[0]):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in parameters:
            total += float(parameter.detach().float().square().sum().item())
    return math.sqrt(total)


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def nonfinite_gradient_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    count = 0
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                count += int((~torch.isfinite(parameter.grad)).sum().item())
    return count


def runtime_manifest(
    config: Mapping[str, Any], run_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "numpy", "peft", "pillow", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    modal_values = {
        key: os.environ.get(key)
        for key in (
            "MODAL_APP_ID",
            "MODAL_ENVIRONMENT",
            "MODAL_FUNCTION_ID",
            "MODAL_IS_REMOTE",
            "MODAL_TASK_ID",
        )
        if os.environ.get(key)
    }
    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return {
        "created_at_utc": utc_now(),
        "config": dict(config),
        "run_metadata": dict(run_metadata or {}),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "modal": modal_values,
    }


def estimated_modal_cost(
    elapsed_seconds: float,
    gpu_cost_per_second: float,
    cpu_cores: float,
    cpu_cost_per_core_second: float,
    memory_gib: float,
    memory_cost_per_gib_second: float,
) -> dict[str, float]:
    gpu = elapsed_seconds * gpu_cost_per_second
    cpu = elapsed_seconds * cpu_cores * cpu_cost_per_core_second
    memory = elapsed_seconds * memory_gib * memory_cost_per_gib_second
    return {
        "gpu_usd": gpu,
        "cpu_usd": cpu,
        "memory_usd": memory,
        "total_usd": gpu + cpu + memory,
    }


def _strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON value {value} in {path}")
        ),
    )


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(
                json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"Non-standard JSON value {value}")
                    ),
                )
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"Required JSONL file is empty: {path}")
    return rows


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def validate_run_artifacts(
    run_directory: str | Path, expected_steps: int | None = None
) -> dict[str, Any]:
    root = Path(run_directory)
    required_files = {
        "run_manifest.json",
        "dataset_audit.json",
        "initialization_audit.json",
        "train.jsonl",
        "evaluation_checkpoints.jsonl",
        "evaluation_bundles.jsonl",
        "resource_usage.jsonl",
        "final_metrics.json",
        "metrics.json",
        "jepa_heads.pt",
        "stop_reason.json",
        "bundle_order.json",
    }
    missing_files = sorted(name for name in required_files if not (root / name).is_file())
    if missing_files:
        raise ValueError(f"Run is missing required files: {missing_files}")

    manifest = _strict_json(root / "run_manifest.json")
    dataset = _strict_json(root / "dataset_audit.json")
    initialization = _strict_json(root / "initialization_audit.json")
    final_metrics = _strict_json(root / "final_metrics.json")
    stop_reason = _strict_json(root / "stop_reason.json")
    train_rows = _strict_jsonl(root / "train.jsonl")
    evaluation_rows = _strict_jsonl(root / "evaluation_checkpoints.jsonl")
    bundle_rows = _strict_jsonl(root / "evaluation_bundles.jsonl")
    resource_rows = _strict_jsonl(root / "resource_usage.jsonl")

    _require_keys(
        manifest,
        {"created_at_utc", "completed_at_utc", "config", "run_metadata", "cuda"},
        "run manifest",
    )
    _require_keys(
        manifest["run_metadata"],
        {
            "run_id",
            "run_mode",
            "git_commit",
            "dataset_id",
            "source_dataset_audit_sha256",
            "modal_app_name",
            "modal_app_id",
            "modal_task_id",
        },
        "run metadata",
    )
    for key in (
        "git_commit",
        "source_dataset_audit_sha256",
        "modal_app_id",
        "modal_task_id",
    ):
        if manifest["run_metadata"][key] in (None, "", "unknown"):
            raise ValueError(f"Run metadata has no valid {key}")
    _require_keys(
        dataset,
        {
            "train_transitions",
            "validation_transitions",
            "bundle_overlap",
            "exact_screenshot_overlap",
            "train_tar_manifest",
            "validation_tar_manifest",
            "test_paths_present",
        },
        "dataset audit",
    )
    if dataset["bundle_overlap"] or dataset["exact_screenshot_overlap"]:
        raise ValueError("Dataset audit reports split overlap")
    if dataset["test_paths_present"]:
        raise ValueError("Dataset audit reports a test path")
    _require_keys(
        initialization,
        {
            "base_qwen_trainable_parameter_count",
            "online_lora_parameter_count",
            "target_lora_trainable_parameter_count",
            "optimizer_parameter_count",
            "initial_online_target_max_difference",
            "initial_online_lora_sha256",
            "initial_target_lora_sha256",
            "initial_action_encoder_sha256",
            "initial_predictor_sha256",
        },
        "initialization audit",
    )
    if initialization["base_qwen_trainable_parameter_count"] != 0:
        raise ValueError("Base Qwen parameters were trainable")
    if initialization["target_lora_trainable_parameter_count"] != 0:
        raise ValueError("Target LoRA parameters were trainable")
    has_lora = initialization["online_lora_parameter_count"] > 0
    if has_lora:
        if initialization["initial_online_target_max_difference"] != 0.0:
            raise ValueError("Online and target LoRA did not start identically")
        if (
            not initialization["initial_online_lora_sha256"]
            or initialization["initial_online_lora_sha256"]
            != initialization["initial_target_lora_sha256"]
        ):
            raise ValueError("Online and target LoRA hashes did not start identically")
    elif any(
        initialization[key] is not None
        for key in (
            "initial_online_target_max_difference",
            "initial_online_lora_sha256",
            "initial_target_lora_sha256",
        )
    ):
        raise ValueError("Frozen-encoder run reports unexpected LoRA initialization data")

    training_fields = {
        "step",
        "loss",
        "regression_loss",
        "changed_region_loss",
        "global_loss",
        "variance_loss",
        "covariance_loss",
        "relation_loss",
        "action_separation_loss",
        "learning_rate",
        "total_gradient_norm_before_clip",
        "online_lora_gradient_norm",
        "action_encoder_gradient_norm",
        "predictor_gradient_norm",
        "online_lora_parameter_norm",
        "online_target_max_difference",
        "current_cuda_memory_gib",
        "peak_cuda_memory_gib",
        "updates_per_second",
        "estimated_remaining_seconds",
        "estimated_modal_cost_usd",
    }
    metric_train_rows = [row for row in train_rows if "loss" in row]
    if not metric_train_rows:
        raise ValueError("Training log contains no metric rows")
    for index, row in enumerate(metric_train_rows):
        _require_keys(row, training_fields, f"training row {index}")

    evaluation_fields = {
        "step",
        "evaluation",
        "four_way_accuracy",
        "shuffled_minus_correct",
        "prediction_to_target_separation_ratio",
        "target_latent_variance",
        "predicted_latent_variance",
        "by_app",
        "by_action_kind",
        "by_changed_pixel_bucket",
    }
    for index, row in enumerate(evaluation_rows):
        _require_keys(row, evaluation_fields, f"evaluation row {index}")
    evaluation_names = {row["evaluation"] for row in evaluation_rows}
    required_evaluations = {
        "initial_train",
        "initial_validation_monitor",
        "train_monitor",
        "validation_monitor",
        "final_train",
        "final_validation",
    }
    if not required_evaluations <= evaluation_names:
        raise ValueError(
            f"Missing evaluation types: {sorted(required_evaluations - evaluation_names)}"
        )
    for index, row in enumerate(bundle_rows):
        _require_keys(
            row,
            {
                "step",
                "evaluation",
                "bundle_id",
                "application",
                "split",
                "actions",
                "target_latent_variance",
                "predicted_latent_variance",
            },
            f"bundle row {index}",
        )
        if len(row["actions"]) != 4:
            raise ValueError(f"Bundle row {index} does not contain four actions")
    _require_keys(resource_rows[-1], {"step", "estimated_modal_cost_usd"}, "resource row")
    _require_keys(final_metrics, {"steps", "timing", "stop_reason"}, "final metrics")
    _require_keys(stop_reason, {"reason", "steps", "requested_steps"}, "stop reason")
    if expected_steps is not None and final_metrics["steps"] != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} completed steps, found {final_metrics['steps']}"
        )

    checkpoint_directories = sorted((root / "checkpoints").glob("step-*"))
    if not checkpoint_directories:
        raise ValueError("Run has no checkpoint directories")
    for checkpoint in checkpoint_directories:
        training_state_path = checkpoint / "training_state.pt"
        if not training_state_path.is_file():
            raise ValueError(f"Checkpoint has no training state: {checkpoint}")
        _strict_json(checkpoint / "checkpoint_metadata.json")
        try:
            checkpoint_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
        except Exception as error:
            raise ValueError(f"Checkpoint cannot be loaded: {training_state_path}") from error
        _require_keys(
            checkpoint_state,
            {
                "step",
                "epoch",
                "next_bundle_position",
                "epoch_bundle_ids",
                "action_encoder",
                "predictor",
                "optimizer",
                "python_random_state",
                "torch_random_state",
                "cuda_random_states",
                "config",
            },
            f"checkpoint {checkpoint.name}",
        )
        if not checkpoint_state["optimizer"].get("state"):
            raise ValueError(f"Checkpoint optimizer state is empty: {checkpoint}")
    try:
        torch.load(root / "jepa_heads.pt", map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("JEPA head weights cannot be loaded") from error
    from safetensors import safe_open

    if has_lora:
        for adapter_directory in (
            root / "qwen_vision_online_lora",
            root / "qwen_vision_target_lora",
        ):
            adapter_files = list(adapter_directory.rglob("*.safetensors"))
            if not adapter_files:
                raise ValueError(f"Adapter weights are missing from {adapter_directory}")
            for adapter_file in adapter_files:
                try:
                    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
                        if not list(handle.keys()):
                            raise ValueError("Adapter contains no tensors")
                except Exception as error:
                    raise ValueError(f"Adapter cannot be loaded: {adapter_file}") from error

    return {
        "passed": True,
        "run_directory": str(root),
        "steps": final_metrics["steps"],
        "training_records": len(metric_train_rows),
        "evaluation_records": len(evaluation_rows),
        "bundle_records": len(bundle_rows),
        "resource_records": len(resource_rows),
        "checkpoint_count": len(checkpoint_directories),
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }
