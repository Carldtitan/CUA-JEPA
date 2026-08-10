"""Paired uncertainty analysis for Model 2 and Model 4 predictions."""

from __future__ import annotations

import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_MODEL4_JEPA_ADAPTER_SHA256 = (
    "177ec69820030e10190d30ddb62ba4a78cd88e77ef7e3738dd33dd15cf64096d"
)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Required run file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_controlled_run_pair(model2_dir: str | Path, model4_dir: str | Path) -> dict[str, Any]:
    """Reject a Model 2 versus Model 4 comparison if its controls differ."""
    model2_dir = Path(model2_dir)
    model4_dir = Path(model4_dir)
    config2 = _read_json(model2_dir / "config.json")
    config4 = _read_json(model4_dir / "config.json")
    audit2 = _read_json(model2_dir / "dataset_audit.json")
    audit4 = _read_json(model4_dir / "dataset_audit.json")
    init2 = _read_json(model2_dir / "initialization_audit.json")
    init4 = _read_json(model4_dir / "initialization_audit.json")
    runtime2 = _read_json(model2_dir / "runtime_audit.json")
    runtime4 = _read_json(model4_dir / "runtime_audit.json")
    source2 = _read_json(model2_dir / "source_jepa_audit.json")
    source4 = _read_json(model4_dir / "source_jepa_audit.json")
    validation2 = _read_json(model2_dir / "validation_example_ids.json")
    validation4 = _read_json(model4_dir / "validation_example_ids.json")
    order2 = _read_json(model2_dir / "training_order.json")
    order4 = _read_json(model4_dir / "training_order.json")

    checks = {
        "same_config": config2 == config4,
        "same_runtime": runtime2 == runtime4,
        "same_dataset_sha256": audit2.get("dataset_sha256") == audit4.get("dataset_sha256"),
        "same_validation_ids": validation2 == validation4,
        "same_training_order": order2 == order4,
        "correct_variants": init2.get("variant") == "model2" and init4.get("variant") == "model4",
        "same_base_model": init2.get("base_model") == init4.get("base_model"),
        "same_base_revision": init2.get("base_revision") == init4.get("base_revision"),
        "base_weights_frozen": init2.get("base_trainable_parameters") == 0
        and init4.get("base_trainable_parameters") == 0,
        "same_vision_lora_parameters": init2.get("vision_lora_parameters")
        == init4.get("vision_lora_parameters"),
        "same_language_lora_parameters": init2.get("language_lora_parameters")
        == init4.get("language_lora_parameters"),
        "same_language_initialization": init2.get("initial_language_lora_sha256")
        == init4.get("initial_language_lora_sha256"),
        "model2_has_no_jepa_source": init2.get("source_jepa_adapter_sha256") is None,
        "model4_has_approved_jepa_source": init4.get("source_jepa_adapter_sha256")
        == EXPECTED_MODEL4_JEPA_ADAPTER_SHA256,
        "model2_source_audit_is_empty": source2.get("source") is None,
        "model4_source_audit_matches_adapter": source4.get("adapter_model_sha256")
        == init4.get("source_jepa_adapter_sha256"),
        "model4_source_uses_action_separation": source4.get("uses_action_separation") is True,
        "vision_initializations_differ": init2.get("initial_vision_lora_sha256")
        != init4.get("initial_vision_lora_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"The controlled SFT pair failed checks: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "dataset_sha256": audit2.get("dataset_sha256"),
        "validation_examples": len(validation2),
        "training_examples": len(order2),
        "model4_jepa_adapter_sha256": init4.get("source_jepa_adapter_sha256"),
        "training_code_sha256": runtime2.get("training_code_sha256"),
        "evaluation_code_sha256": runtime2.get("evaluation_code_sha256"),
        "model4_source_objective": source4.get("objective"),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile from no values")
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    model2_records: Iterable[Mapping[str, Any]],
    model4_records: Iterable[Mapping[str, Any]],
    samples: int = 10_000,
    seed: int = 20260810,
) -> dict[str, Any]:
    model2 = {str(value["example_id"]): value for value in model2_records}
    model4 = {str(value["example_id"]): value for value in model4_records}
    if set(model2) != set(model4):
        missing_from_model2 = sorted(set(model4) - set(model2))
        missing_from_model4 = sorted(set(model2) - set(model4))
        raise ValueError(
            "Prediction example IDs do not match: "
            f"missing from Model 2 {missing_from_model2[:5]}, "
            f"missing from Model 4 {missing_from_model4[:5]}"
        )
    example_ids = sorted(model2)
    if not example_ids:
        raise ValueError("No paired predictions were provided")
    deltas = [
        float(model4[example_id]["score"]) - float(model2[example_id]["score"])
        for example_id in example_ids
    ]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrap_values = []
    for _ in range(samples):
        bootstrap_values.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    bootstrap_values.sort()

    by_action: dict[str, list[float]] = defaultdict(list)
    for example_id, delta in zip(example_ids, deltas, strict=True):
        by_action[str(model2[example_id]["target_action"])].append(delta)
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - wins - losses
    return {
        "examples": len(deltas),
        "model4_minus_model2_mean_score": observed,
        "bootstrap_95_percent_interval": [
            _percentile(bootstrap_values, 0.025),
            _percentile(bootstrap_values, 0.975),
        ],
        "bootstrap_probability_model4_is_better": sum(value > 0 for value in bootstrap_values)
        / len(bootstrap_values),
        "model4_wins": wins,
        "ties": ties,
        "model2_wins": losses,
        "by_action_mean_delta": {
            action: sum(values) / len(values) for action, values in sorted(by_action.items())
        },
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
