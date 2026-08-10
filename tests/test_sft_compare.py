import json

import pytest

from cua_jepa.sft_compare import (
    EXPECTED_MODEL4_JEPA_ADAPTER_SHA256,
    audit_controlled_run_pair,
    audit_controlled_run_triplet,
    paired_bootstrap,
)


def _record(example_id: str, score: float, action: str = "click") -> dict:
    return {"example_id": example_id, "score": score, "target_action": action}


def REDACTED() -> None:
    model2 = [_record(str(index), 0.0, "click" if index % 2 else "write") for index in range(20)]
    model4 = [_record(str(index), 1.0, "click" if index % 2 else "write") for index in range(20)]
    result = paired_bootstrap(model2, model4, samples=200, seed=4)
    assert result["model4_minus_model2_mean_score"] == 1.0
    assert result["bootstrap_95_percent_interval"] == [1.0, 1.0]
    assert result["model4_wins"] == 20
    assert result["model2_wins"] == 0
    assert result["by_action_mean_delta"] == {"click": 1.0, "write": 1.0}


def test_paired_bootstrap_rejects_different_examples() -> None:
    with pytest.raises(ValueError, match="do not match"):
        paired_bootstrap([_record("one", 0.0)], [_record("two", 1.0)], samples=10)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_controlled_pair_audit_checks_all_initial_conditions(tmp_path) -> None:
    model2 = tmp_path / "model2"
    model4 = tmp_path / "model4"
    model2.mkdir()
    model4.mkdir()
    common_files = {
        "config.json": {"seed": 1, "max_pixels": 100},
        "dataset_audit.json": {"dataset_sha256": "data"},
        "validation_example_ids.json": ["v1"],
        "training_order.json": [{"example_id": "t1"}],
        "runtime_audit.json": {
            "training_code_sha256": "train-code",
            "evaluation_code_sha256": "eval-code",
            "gpu_name": "L4",
        },
    }
    for name, value in common_files.items():
        _write_json(model2 / name, value)
        _write_json(model4 / name, value)
    _write_json(model2 / "source_jepa_audit.json", {"source": None})
    _write_json(
        model4 / "source_jepa_audit.json",
        {
            "source": "action-conditioned JEPA",
            "adapter_model_sha256": EXPECTED_MODEL4_JEPA_ADAPTER_SHA256,
            "uses_action_separation": True,
            "objective": "latent regression plus action separation",
        },
    )
    base_init = {
        "base_model": "qwen",
        "base_revision": "revision",
        "base_trainable_parameters": 0,
        "vision_lora_parameters": 10,
        "language_lora_parameters": 20,
        "initial_language_lora_sha256": "same-language",
    }
    _write_json(
        model2 / "initialization_audit.json",
        {
            **base_init,
            "variant": "model2",
            "source_jepa_adapter_sha256": None,
            "initial_vision_lora_sha256": "fresh-vision",
        },
    )
    _write_json(
        model4 / "initialization_audit.json",
        {
            **base_init,
            "variant": "model4",
            "source_jepa_adapter_sha256": EXPECTED_MODEL4_JEPA_ADAPTER_SHA256,
            "initial_vision_lora_sha256": "jepa-vision",
        },
    )
    result = audit_controlled_run_pair(model2, model4)
    assert result["passed"] is True
    assert result["training_examples"] == 1
    _write_json(model4 / "validation_example_ids.json", ["different"])
    with pytest.raises(ValueError, match="same_validation_ids"):
        audit_controlled_run_pair(model2, model4)


def test_controlled_triplet_audit_checks_model3_action_control(tmp_path) -> None:
    directories = {name: tmp_path / name for name in ("model2", "model3", "model4")}
    for directory in directories.values():
        directory.mkdir()
    common_files = {
        "config.json": {"seed": 1, "max_pixels": 100},
        "dataset_audit.json": {"dataset_sha256": "sft-data"},
        "validation_example_ids.json": ["v1"],
        "training_order.json": [{"example_id": "t1"}],
        "runtime_audit.json": {
            "training_code_sha256": "train-code",
            "evaluation_code_sha256": "eval-code",
        },
    }
    for directory in directories.values():
        for filename, value in common_files.items():
            _write_json(directory / filename, value)

    base_init = {
        "base_model": "qwen",
        "base_revision": "revision",
        "base_trainable_parameters": 0,
        "vision_lora_parameters": 10,
        "language_lora_parameters": 20,
        "initial_language_lora_sha256": "same-language",
    }
    for name, directory in directories.items():
        _write_json(
            directory / "initialization_audit.json",
            {
                **base_init,
                "variant": name,
                "source_jepa_adapter_sha256": None if name == "model2" else f"{name}-jepa",
            },
        )
    _write_json(directories["model2"] / "source_jepa_audit.json", {"source": None})
    source_common = {
        "dataset_audit_sha256": "jepa-data",
        "bundle_order_sha256": "jepa-order",
        "steps": 7_667,
        "training_config": {"model_id": "qwen", "max_steps": 7_667},
    }
    _write_json(
        directories["model3"] / "source_jepa_audit.json",
        {
            **source_common,
            "source": "no-action JEPA control",
            "training_action_assignment": "no_action",
            "uses_correct_action_information": False,
        },
    )
    _write_json(
        directories["model4"] / "source_jepa_audit.json",
        {
            **source_common,
            "source": "action-conditioned JEPA",
            "training_action_assignment": "correct",
            "uses_correct_action_information": True,
        },
    )
    result = audit_controlled_run_triplet(**{
        f"{name}_dir": directory for name, directory in directories.items()
    })
    assert result["passed"] is True
    assert result["checks"]["model3_has_no_correct_actions"] is True

    source3 = json.loads(
        (directories["model3"] / "source_jepa_audit.json").read_text(encoding="utf-8")
    )
    source3["training_action_assignment"] = "correct"
    _write_json(directories["model3"] / "source_jepa_audit.json", source3)
    with pytest.raises(ValueError, match="model3_has_no_correct_actions"):
        audit_controlled_run_triplet(**{
            f"{name}_dir": directory for name, directory in directories.items()
        })
