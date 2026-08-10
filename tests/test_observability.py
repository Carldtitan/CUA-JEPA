from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from cua_jepa.jepa_data import TransitionSample
from cua_jepa.observability import (
    append_jsonl,
    dataset_profile,
    estimated_modal_cost,
    gradient_l2_norm,
    named_tensors_sha256,
    safe_action_record,
    validate_run_artifacts,
    write_json,
)


def _sample(index: int, action: dict) -> TransitionSample:
    return TransitionSample(
        bundle_id="bundle-1",
        app="app-1",
        split="train",
        branch_index=index,
        action=action,
        current_webp=b"current",
        future_webp=f"future-{index}".encode(),
        changed_pixel_fraction=0.01 * (index + 1),
    )


def test_dataset_profile_and_safe_actions_do_not_store_typing_text() -> None:
    samples = [
        _sample(0, {"kind": "type", "text": "private words", "x": 10, "y": 20}),
        _sample(1, {"kind": "click", "x": 30, "y": 40}),
        _sample(2, {"kind": "scroll", "delta_y": 100}),
        _sample(3, {"kind": "click", "x": 50, "y": 60}),
    ]
    profile = dataset_profile(samples)
    assert profile["bundles"] == 1
    assert profile["action_counts"] == {"click": 2, "scroll": 1, "type": 1}
    safe = safe_action_record(samples[0].action)
    assert "text" not in safe
    assert safe["text_length"] == 13
    assert safe["text_category"] == "phrase"


def test_parameter_hash_gradient_norm_and_cost_are_measured() -> None:
    layer = nn.Linear(2, 1, bias=False)
    initial_hash = named_tensors_sha256(layer.state_dict().items())
    layer(torch.ones(1, 2)).sum().backward()
    assert gradient_l2_norm(layer.parameters()) > 0
    with torch.no_grad():
        layer.weight.add_(1)
    assert named_tensors_sha256(layer.state_dict().items()) != initial_hash
    cost = estimated_modal_cost(10, 1.0, 2, 0.5, 4, 0.25)
    assert cost == {"gpu_usd": 10.0, "cpu_usd": 10.0, "memory_usd": 10.0, "total_usd": 30.0}


def _build_fake_run(root: Path) -> None:
    write_json(
        root / "run_manifest.json",
        {
            "created_at_utc": "start",
            "completed_at_utc": "end",
            "config": {},
            "run_metadata": {
                "run_id": "run",
                "run_mode": "smoke",
                "git_commit": "commit",
                "dataset_id": "dataset",
                "source_dataset_audit_sha256": "audit",
                "modal_app_name": "app",
                "modal_app_id": "ap-123",
                "modal_task_id": "ta-123",
            },
            "cuda": {},
        },
    )
    write_json(
        root / "dataset_audit.json",
        {
            "train_transitions": 8,
            "validation_transitions": 8,
            "bundle_overlap": 0,
            "exact_screenshot_overlap": 0,
            "train_tar_manifest": {},
            "validation_tar_manifest": {},
            "test_paths_present": False,
        },
    )
    write_json(
        root / "initialization_audit.json",
        {
            "base_qwen_trainable_parameter_count": 0,
            "online_lora_parameter_count": 1,
            "target_lora_trainable_parameter_count": 0,
            "optimizer_parameter_count": 3,
            "initial_online_target_max_difference": 0.0,
            "initial_online_lora_sha256": "a",
            "initial_target_lora_sha256": "a",
            "initial_action_encoder_sha256": "b",
            "initial_predictor_sha256": "c",
        },
    )
    training = {
        "step": 2,
        "loss": 1.0,
        "regression_loss": 1.0,
        "changed_region_loss": 1.0,
        "global_loss": 1.0,
        "variance_loss": 1.0,
        "covariance_loss": 1.0,
        "relation_loss": 1.0,
        "action_separation_loss": 1.0,
        "learning_rate": 0.1,
        "total_gradient_norm_before_clip": 1.0,
        "online_lora_gradient_norm": 1.0,
        "action_encoder_gradient_norm": 1.0,
        "predictor_gradient_norm": 1.0,
        "online_lora_parameter_norm": 1.0,
        "online_target_max_difference": 0.1,
        "current_cuda_memory_gib": 1.0,
        "peak_cuda_memory_gib": 2.0,
        "updates_per_second": 1.0,
        "estimated_remaining_seconds": 0.0,
        "estimated_modal_cost_usd": {"total_usd": 0.1},
    }
    append_jsonl(root / "train.jsonl", training)
    evaluation_base = {
        "step": 0,
        "four_way_accuracy": 0.25,
        "shuffled_minus_correct": 0.0,
        "prediction_to_target_separation_ratio": 0.1,
        "target_latent_variance": 0.2,
        "predicted_latent_variance": 0.1,
        "by_app": {},
        "by_action_kind": {},
        "by_changed_pixel_bucket": {},
    }
    for name in (
        "initial_train",
        "initial_validation_monitor",
        "final_train",
        "final_validation",
    ):
        append_jsonl(
            root / "evaluation_checkpoints.jsonl",
            {**evaluation_base, "evaluation": name},
        )
    append_jsonl(
        root / "evaluation_bundles.jsonl",
        {
            "step": 0,
            "evaluation": "initial_train",
            "bundle_id": "b",
            "application": "app",
            "split": "train",
            "actions": [{}, {}, {}, {}],
            "target_latent_variance": 0.2,
            "predicted_latent_variance": 0.1,
        },
    )
    append_jsonl(
        root / "resource_usage.jsonl",
        {"step": 2, "estimated_modal_cost_usd": {"total_usd": 0.1}},
    )
    write_json(root / "final_metrics.json", {"steps": 2, "timing": {}, "stop_reason": "maximum_steps_completed"})
    write_json(root / "metrics.json", {})
    write_json(root / "stop_reason.json", {"reason": "maximum_steps_completed", "steps": 2, "requested_steps": 2})
    write_json(root / "bundle_order.json", {"train": ["b"], "validation": ["v"]})
    torch.save({}, root / "jepa_heads.pt")
    checkpoint = root / "checkpoints" / "step-000002"
    checkpoint.mkdir(parents=True)
    torch.save({}, checkpoint / "training_state.pt")
    write_json(checkpoint / "checkpoint_metadata.json", {})
    for adapter in ("qwen_vision_online_lora", "qwen_vision_target_lora"):
        directory = root / adapter
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"weights")


def test_artifact_validator_accepts_complete_run_and_rejects_missing_metric(
    tmp_path: Path,
) -> None:
    _build_fake_run(tmp_path)
    report = validate_run_artifacts(tmp_path, expected_steps=2)
    assert report["passed"]
    assert report["checkpoint_count"] == 1

    rows = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    del row["online_lora_gradient_norm"]
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="online_lora_gradient_norm"):
        validate_run_artifacts(tmp_path, expected_steps=2)
