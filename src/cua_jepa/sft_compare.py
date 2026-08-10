"""Paired uncertainty analysis for Model 2 and Model 4 predictions."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable, Mapping


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
