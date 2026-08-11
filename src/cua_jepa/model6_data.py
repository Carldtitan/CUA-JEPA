"""Candidate data and split controls for Model 6."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable

from cua_jepa.sft_eval import parse_action_prediction, score_action


SUPPORTED_DYNAMICS_ACTIONS = {"click", "write", "scroll"}


def _safe_number(value: Any) -> tuple[float, bool]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (number, True) if math.isfinite(number) else (0.0, False)


def policy_action_to_dynamics(action: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Convert a Qwen policy action to the smaller synthetic dynamics action space."""

    kind = str(action.get("action", ""))
    supported = kind in SUPPORTED_DYNAMICS_ACTIONS
    if kind in {"click", "double_click", "right_click", "moveTo", "dragTo"}:
        x, valid_x = _safe_number(action.get("x", 0.0))
        y, valid_y = _safe_number(action.get("y", 0.0))
        return {
            "kind": "click",
            "x_normalized": x,
            "y_normalized": y,
        }, supported and valid_x and valid_y
    if kind == "write":
        text = action.get("text", "")
        return {
            "kind": "type",
            "text": text if isinstance(text, str) else "",
            "x_normalized": 0.0,
            "y_normalized": 0.0,
        }, supported and isinstance(text, str)
    if kind == "scroll":
        amount, valid_amount = _safe_number(action.get("amount", 0.0))
        return {
            "kind": "scroll",
            "delta_y": amount,
        }, supported and valid_amount
    if kind in {"press", "hotkey"}:
        return {
            "kind": "press",
            "text": "+".join(str(value) for value in action.get("keys", [])),
        }, supported
    return {"kind": "unknown"}, False


def _canonical_candidate(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_candidate_record(
    record: dict[str, Any],
    greedy_text: str,
    sampled_texts: list[str],
    width: int,
    height: int,
    inject_training_oracle: bool,
    maximum_candidates: int = 4,
) -> dict[str, Any]:
    """Parse, deduplicate, label, and audit one Qwen candidate set."""

    raw = [("greedy", greedy_text)] + [("sampled", value) for value in sampled_texts]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    parse_failures = 0
    duplicates = 0
    for source, text in raw:
        action = parse_action_prediction(text)
        if action is None:
            parse_failures += 1
            continue
        key = _canonical_candidate(action)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        scored = score_action(action, record["action"], width, height)
        candidates.append(
            {
                "source": source,
                "raw_text": text,
                "action": action,
                "action_score": float(scored["score"]),
                "type_correct": bool(scored["type_correct"]),
                "dynamics_action": policy_action_to_dynamics(action)[0],
                "dynamics_action_supported": policy_action_to_dynamics(action)[1],
            }
        )
        if len(candidates) == maximum_candidates:
            break

    oracle_injected = False
    if inject_training_oracle and not any(item["action_score"] > 0.0 for item in candidates):
        oracle = {
            "source": "training_oracle",
            "raw_text": record["target"],
            "action": record["action"],
            "action_score": 1.0,
            "type_correct": True,
            "dynamics_action": policy_action_to_dynamics(record["action"])[0],
            "dynamics_action_supported": policy_action_to_dynamics(record["action"])[1],
        }
        if len(candidates) >= maximum_candidates:
            candidates[-1] = oracle
        else:
            candidates.append(oracle)
        oracle_injected = True

    return {
        "example_id": record["example_id"],
        "split": record["split"],
        "task_id": record["task_id"],
        "instruction": record["instruction"],
        "history": record.get("history", []),
        "system": record.get("system", "unknown"),
        "domain": record.get("domain", "unknown"),
        "stored_image": record["stored_image"],
        "target_action": record["action"],
        "width": width,
        "height": height,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "parse_failures": parse_failures,
        "duplicate_candidates": duplicates,
        "oracle_injected": oracle_injected,
    }


def split_scorer_training_records(
    records: Iterable[dict[str, Any]],
    development_examples: int = 200,
    seed: int = 20260813,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create an exact task-disjoint fit and development split."""

    values = list(records)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in values:
        by_task[str(record["task_id"])].append(record)
    ordered_tasks = sorted(
        by_task,
        key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).digest(),
    )
    selected_tasks: set[str] = set()
    selected_count = 0
    remaining = development_examples
    while remaining:
        match = next(
            (
                task
                for task in ordered_tasks
                if task not in selected_tasks and len(by_task[task]) <= remaining
            ),
            None,
        )
        if match is None:
            raise RuntimeError("Cannot create the exact task-disjoint development split")
        selected_tasks.add(match)
        count = len(by_task[match])
        selected_count += count
        remaining -= count
    development = [record for record in values if str(record["task_id"]) in selected_tasks]
    fit = [record for record in values if str(record["task_id"]) not in selected_tasks]
    if selected_count != development_examples or len(fit) + len(development) != len(values):
        raise RuntimeError("Model 6 scorer split has the wrong size")
    if {record["task_id"] for record in fit} & {
        record["task_id"] for record in development
    }:
        raise RuntimeError("Model 6 scorer task IDs overlap")
    return fit, development


def candidate_generation_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    if not values:
        raise ValueError("No Model 6 candidate records were provided")
    greedy_scores = []
    oracle_scores = []
    candidate_counts = []
    for record in values:
        candidates = record["candidates"]
        greedy = next(
            (item for item in candidates if item["source"] == "greedy"),
            None,
        )
        greedy_scores.append(float(greedy["action_score"]) if greedy else 0.0)
        oracle_scores.append(max((float(item["action_score"]) for item in candidates), default=0.0))
        candidate_counts.append(len(candidates))
    count = len(values)
    return {
        "examples": count,
        "mean_candidate_count": sum(candidate_counts) / count,
        "four_candidate_rate": sum(value == 4 for value in candidate_counts) / count,
        "greedy_exact_success": sum(value == 1.0 for value in greedy_scores) / count,
        "candidate_recall_at_4": sum(value == 1.0 for value in oracle_scores) / count,
        "candidate_positive_rate": sum(value > 0.0 for value in oracle_scores) / count,
        "parse_failures": sum(int(record["parse_failures"]) for record in values),
        "duplicates": sum(int(record["duplicate_candidates"]) for record in values),
        "oracle_injections": sum(bool(record["oracle_injected"]) for record in values),
    }


def validate_candidate_resume_prefix(
    source_records: list[dict[str, Any]],
    completed_records: list[dict[str, Any]],
) -> int:
    """Verify that saved candidate records are an exact source-data prefix."""

    if len(completed_records) > len(source_records):
        raise ValueError("Saved Model 6 candidates exceed the source data")
    expected = [record["example_id"] for record in source_records[: len(completed_records)]]
    actual = [record["example_id"] for record in completed_records]
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("Saved Model 6 candidates are not an exact unique prefix")
    return len(completed_records)
