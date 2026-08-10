import json

import pytest

from cua_jepa.sft_compare import (
    EXPECTED_MODEL4_JEPA_ADAPTER_SHA256,
    audit_controlled_run_pair,
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
    }
    for name, value in common_files.items():
        _write_json(model2 / name, value)
        _write_json(model4 / name, value)
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
