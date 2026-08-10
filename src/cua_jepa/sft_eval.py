"""Parse and score canonical computer-use actions."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping


COORDINATE_ACTIONS = {"click", "double_click", "right_click", "moveTo", "dragTo"}


def parse_action_prediction(text: str) -> dict[str, Any] | None:
    """Read the first JSON object from a model response."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("action"), str):
            return value
    return None


def _normalized_position(action: Mapping[str, Any], width: int, height: int) -> tuple[float, float]:
    x = float(action["x"])
    y = float(action["y"])
    if abs(x) > 1.0:
        x /= max(width, 1)
    if abs(y) > 1.0:
        y /= max(height, 1)
    return x, y


def score_action(
    prediction: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    width: int,
    height: int,
    coordinate_threshold: float = 0.1,
) -> dict[str, float | str | bool | None]:
    target_kind = str(target.get("action", ""))
    prediction_kind = str((prediction or {}).get("action", ""))
    type_correct = prediction_kind == target_kind
    result: dict[str, float | str | bool | None] = {
        "target_action": target_kind,
        "predicted_action": prediction_kind or None,
        "parsed": prediction is not None,
        "type_correct": type_correct,
        "score": 0.0,
        "coordinate_distance": None,
    }
    if not type_correct or prediction is None:
        return result

    if target_kind in COORDINATE_ACTIONS:
        try:
            predicted_x, predicted_y = _normalized_position(prediction, width, height)
            target_x, target_y = _normalized_position(target, width, height)
        except (KeyError, TypeError, ValueError, OverflowError):
            return result
        distance = math.dist((predicted_x, predicted_y), (target_x, target_y))
        result["coordinate_distance"] = distance
        result["score"] = float(distance <= coordinate_threshold)
        return result

    if target_kind == "write":
        predicted_text = str(prediction.get("text", ""))
        target_text = str(target.get("text", ""))
        result["score"] = SequenceMatcher(None, predicted_text, target_text).ratio()
        result["exact_content"] = predicted_text == target_text
        return result

    if target_kind in {"press", "hotkey"}:
        predicted_keys = [str(value).lower() for value in prediction.get("keys", [])]
        target_keys = [str(value).lower() for value in target.get("keys", [])]
        result["score"] = float(predicted_keys == target_keys)
        result["exact_content"] = predicted_keys == target_keys
        return result

    if target_kind == "scroll":
        try:
            predicted_amount = float(prediction["amount"])
            target_amount = float(target["amount"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return result
        same_direction = predicted_amount == target_amount == 0 or predicted_amount * target_amount > 0
        if same_direction:
            result["score"] = min(abs(predicted_amount), abs(target_amount)) / max(
                abs(predicted_amount), abs(target_amount), 1e-8
            )
        result["direction_correct"] = same_direction
        return result

    result["score"] = 1.0
    return result


def summarize_scores(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(records)
    if not values:
        raise ValueError("No evaluation records were provided")

    def mean(key: str, items: list[Mapping[str, Any]]) -> float:
        return sum(float(item.get(key, 0.0)) for item in items) / len(items)

    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_system: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in values:
        by_action[str(value["target_action"])].append(value)
        by_system[str(value.get("system", "unknown"))].append(value)
    action_summaries = {
        action: {
            "examples": len(items),
            "action_type_accuracy": mean("type_correct", items),
            "mean_action_score": mean("score", items),
        }
        for action, items in sorted(by_action.items())
    }
    coordinate_values = [
        value for value in values if str(value["target_action"]) in COORDINATE_ACTIONS
    ]
    measured_distances = [
        float(value["coordinate_distance"])
        for value in coordinate_values
        if value.get("coordinate_distance") is not None
    ]
    coordinate_hit_rates = {
        f"hit_rate_at_{str(threshold).replace('.', '_')}": sum(
            float(value.get("coordinate_distance") is not None)
            * float(float(value.get("coordinate_distance") or 0.0) <= threshold)
            for value in coordinate_values
        )
        / len(coordinate_values)
        if coordinate_values
        else 0.0
        for threshold in (0.02, 0.05, 0.1)
    }
    non_coordinate_values = [
        value for value in values if str(value["target_action"]) not in COORDINATE_ACTIONS
    ]
    return {
        "examples": len(values),
        "parse_rate": mean("parsed", values),
        "action_type_accuracy": mean("type_correct", values),
        "mean_action_score": mean("score", values),
        "macro_action_score": sum(
            value["mean_action_score"] for value in action_summaries.values()
        )
        / len(action_summaries),
        "exact_success_rate": sum(float(value.get("score", 0.0)) == 1.0 for value in values)
        / len(values),
        "coordinate": {
            "examples": len(coordinate_values),
            "hit_rate": mean("score", coordinate_values) if coordinate_values else 0.0,
            **coordinate_hit_rates,
            "mean_distance_when_action_type_correct": (
                sum(measured_distances) / len(measured_distances) if measured_distances else None
            ),
        },
        "non_coordinate": {
            "examples": len(non_coordinate_values),
            "mean_action_score": (
                mean("score", non_coordinate_values) if non_coordinate_values else 0.0
            ),
        },
        "by_action": action_summaries,
        "by_system": {
            system: {
                "examples": len(items),
                "action_type_accuracy": mean("type_correct", items),
                "mean_action_score": mean("score", items),
            }
            for system, items in sorted(by_system.items())
        },
    }
