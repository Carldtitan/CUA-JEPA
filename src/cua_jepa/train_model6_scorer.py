"""Train and evaluate the Model 6 goal-conditioned reranker."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.utils import clip_grad_norm_

from cua_jepa.model6 import (
    ActionOnlyScorer,
    GoalConditionedFutureScorer,
    scorer_training_loss,
)
from cua_jepa.model6_data import policy_action_to_dynamics, split_scorer_training_records
from cua_jepa.model6_runtime import predict_candidate_futures
from cua_jepa.observability import append_jsonl, utc_now, write_json


@dataclass
class Model6ScorerConfig:
    seed: int = 20260813
    maximum_epochs: int = 10
    early_stopping_patience: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    rejection_weight: float = 0.5
    gradient_clip: float = 1.0
    development_examples: int = 200
    log_every: int = 100
    maximum_runtime_seconds: int = 2 * 60 * 60


def binary_auc(labels: list[int], scores: list[float]) -> float | None:
    """Calculate AUROC with correct handling for tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    wins = 0.0
    for first, first_label in zip(scores, labels, strict=True):
        if not first_label:
            continue
        for second, second_label in zip(scores, labels, strict=True):
            if second_label:
                continue
            wins += float(first > second) + 0.5 * float(first == second)
    return wins / (positives * negatives)


def _selected_record(
    record: dict[str, Any],
    logits: torch.Tensor,
    system: str,
) -> dict[str, Any]:
    index = int(logits.argmax().item())
    candidate = record["candidates"][index]
    target_supported = policy_action_to_dynamics(record["target_action"])[1]
    return {
        "example_id": record["example_id"],
        "task_id": record["task_id"],
        "system_name": system,
        "selected_index": index,
        "selected_action_score": float(candidate["action_score"]),
        "selected_exact": float(candidate["action_score"]) == 1.0,
        "candidate_count": len(record["candidates"]),
        "oracle_action_score": max(
            float(value["action_score"]) for value in record["candidates"]
        ),
        "target_action": str(record["target_action"].get("action", "unknown")),
        "target_dynamics_supported": target_supported,
        "operating_system": record.get("system", "unknown"),
        "domain": record.get("domain", "unknown"),
        "candidate_logits": [float(value) for value in logits.detach().cpu().tolist()],
        "candidate_labels": [
            float(value["action_score"]) for value in record["candidates"]
        ],
    }


def summarize_selection(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No Model 6 selection records were provided")

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(values)
        return {
            "examples": count,
            "exact_success_rate": sum(value["selected_exact"] for value in values) / count,
            "mean_action_score": sum(value["selected_action_score"] for value in values)
            / count,
            "oracle_exact_recall": sum(value["oracle_action_score"] == 1.0 for value in values)
            / count,
            "mean_regret": sum(
                value["oracle_action_score"] - value["selected_action_score"] for value in values
            )
            / count,
        }

    labels = []
    logits = []
    for record in records:
        labels.extend(int(value > 0.0) for value in record["candidate_labels"])
        logits.extend(record["candidate_logits"])
    supported = [value for value in records if value["target_dynamics_supported"]]
    return {
        **summarize(records),
        "bad_action_rejection_auc": binary_auc(labels, logits),
        "supported_action_subset": summarize(supported) if supported else None,
    }


def summarize_qwen_greedy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the unchanged Qwen policy from the same candidate records."""

    selected = []
    for record in records:
        greedy = next(
            (item for item in record["candidates"] if item["source"] == "greedy"),
            None,
        )
        score = float(greedy["action_score"]) if greedy is not None else 0.0
        selected.append(
            {
                "score": score,
                "oracle": max(
                    (float(item["action_score"]) for item in record["candidates"]),
                    default=0.0,
                ),
                "supported": policy_action_to_dynamics(record["target_action"])[1],
            }
        )

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not values:
            return None
        count = len(values)
        return {
            "examples": count,
            "exact_success_rate": sum(value["score"] == 1.0 for value in values) / count,
            "mean_action_score": sum(value["score"] for value in values) / count,
            "oracle_exact_recall": sum(value["oracle"] == 1.0 for value in values) / count,
            "mean_regret": sum(value["oracle"] - value["score"] for value in values)
            / count,
        }

    return {
        **(summarize(selected) or {}),
        "supported_action_subset": summarize(
            [value for value in selected if value["supported"]]
        ),
    }


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"measurements": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index] * 1000.0

    return {
        "measurements": len(ordered),
        "mean_ms": sum(ordered) / len(ordered) * 1000.0,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
    }


def evaluate_model6_scorers(
    records: list[dict[str, Any]],
    features: dict[str, dict[str, torch.Tensor]],
    dynamics: dict[str, torch.nn.Module | dict[str, Any]],
    future_scorer: GoalConditionedFutureScorer,
    action_scorer: ActionOnlyScorer,
    device: torch.device,
    measure_latency: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    future_scorer.eval()
    action_scorer.eval()
    outputs: list[dict[str, Any]] = []
    dynamics_latencies: list[float] = []
    scorer_latencies: list[float] = []
    with torch.inference_mode():
        for record in records:
            if not record["candidates"]:
                continue
            feature = features[record["example_id"]]
            actions = [value["dynamics_action"] for value in record["candidates"]]
            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            dynamics_started = time.perf_counter()
            futures, action_embeddings = predict_candidate_futures(
                feature["current"],
                actions,
                int(record["width"]),
                int(record["height"]),
                dynamics,
                device,
            )
            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            if measure_latency:
                dynamics_latencies.append(time.perf_counter() - dynamics_started)
            goal = feature["goal"].to(device=device, dtype=torch.float32)
            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            scorer_started = time.perf_counter()
            future_logits = future_scorer(futures, goal)
            action_logits = action_scorer(action_embeddings, goal)
            shuffled_logits = future_scorer(futures.roll(1, dims=0), goal)
            if measure_latency and device.type == "cuda":
                torch.cuda.synchronize(device)
            if measure_latency:
                scorer_latencies.append(time.perf_counter() - scorer_started)
            outputs.extend(
                [
                    _selected_record(record, future_logits, "model6_future"),
                    _selected_record(record, action_logits, "action_only"),
                    _selected_record(record, shuffled_logits, "shuffled_future"),
                ]
            )
    metrics = {
        system: summarize_selection(
            [record for record in outputs if record["system_name"] == system]
        )
        for system in ("model6_future", "action_only", "shuffled_future")
    }
    metrics["qwen_greedy"] = summarize_qwen_greedy(records)
    if measure_latency:
        metrics["runtime_latency"] = {
            "batched_jepa_prediction": _latency_summary(dynamics_latencies),
            "all_three_scorer_controls": _latency_summary(scorer_latencies),
        }
    future_scorer.train()
    action_scorer.train()
    return metrics, outputs


def train_model6_scorers(
    candidate_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    features: dict[str, dict[str, torch.Tensor]],
    dynamics: dict[str, torch.nn.Module | dict[str, Any]],
    output_dir: str | Path,
    config: Model6ScorerConfig,
    device: torch.device,
    persist_outputs: Callable[[], None] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    fit, development = split_scorer_training_records(
        candidate_records,
        config.development_examples,
        config.seed,
    )
    future_scorer = GoalConditionedFutureScorer().to(device)
    action_scorer = ActionOnlyScorer().to(device)
    trainable = list(future_scorer.parameters()) + list(action_scorer.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_future = None
    best_action = None
    best_future_score = -1.0
    best_action_score = -1.0
    epochs_without_future_gain = 0
    train_path = output / "train.jsonl"
    step = 0
    for epoch in range(1, config.maximum_epochs + 1):
        order = list(fit)
        random.Random(config.seed + epoch).shuffle(order)
        for record in order:
            if time.perf_counter() - started >= config.maximum_runtime_seconds:
                raise RuntimeError("Model 6 scorer reached its runtime limit")
            if not record["candidates"]:
                continue
            feature = features[record["example_id"]]
            actions = [value["dynamics_action"] for value in record["candidates"]]
            with torch.no_grad():
                futures, action_embeddings = predict_candidate_futures(
                    feature["current"],
                    actions,
                    int(record["width"]),
                    int(record["height"]),
                    dynamics,
                    device,
                )
            goal = feature["goal"].to(device=device, dtype=torch.float32)
            labels = torch.tensor(
                [float(value["action_score"]) for value in record["candidates"]],
                device=device,
            )
            future_logits = future_scorer(futures, goal)
            action_logits = action_scorer(action_embeddings, goal)
            future_loss, future_parts = scorer_training_loss(
                future_logits, labels, config.rejection_weight
            )
            action_loss, action_parts = scorer_training_loss(
                action_logits, labels, config.rejection_weight
            )
            loss = future_loss + action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(clip_grad_norm_(trainable, config.gradient_clip).item())
            optimizer.step()
            step += 1
            if step == 1 or step % config.log_every == 0:
                append_jsonl(
                    train_path,
                    {
                        "recorded_at_utc": utc_now(),
                        "epoch": epoch,
                        "step": step,
                        "example_id": record["example_id"],
                        "loss": float(loss.detach().item()),
                        "future_ranking_loss": float(future_parts["ranking"].detach().item()),
                        "future_rejection_loss": float(
                            future_parts["rejection"].detach().item()
                        ),
                        "action_ranking_loss": float(action_parts["ranking"].detach().item()),
                        "action_rejection_loss": float(
                            action_parts["rejection"].detach().item()
                        ),
                        "gradient_norm_before_clip": gradient_norm,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
        development_metrics, _ = evaluate_model6_scorers(
            development,
            features,
            dynamics,
            future_scorer,
            action_scorer,
            device,
        )
        append_jsonl(
            output / "development.jsonl",
            {
                "recorded_at_utc": utc_now(),
                "epoch": epoch,
                "step": step,
                "metrics": development_metrics,
            },
        )
        future_score = development_metrics["model6_future"]["mean_action_score"]
        action_score = development_metrics["action_only"]["mean_action_score"]
        if future_score > best_future_score:
            best_future_score = future_score
            best_future = copy.deepcopy(future_scorer.state_dict())
            epochs_without_future_gain = 0
        else:
            epochs_without_future_gain += 1
        if action_score > best_action_score:
            best_action_score = action_score
            best_action = copy.deepcopy(action_scorer.state_dict())
        if epochs_without_future_gain >= config.early_stopping_patience:
            break
    if best_future is None or best_action is None:
        raise RuntimeError("Model 6 scorer did not produce a checkpoint")
    future_scorer.load_state_dict(best_future)
    action_scorer.load_state_dict(best_action)
    final_metrics, predictions = evaluate_model6_scorers(
        validation_records,
        features,
        dynamics,
        future_scorer,
        action_scorer,
        device,
        measure_latency=True,
    )
    torch.save(
        {
            "config": asdict(config),
            "future_scorer": future_scorer.state_dict(),
            "action_scorer": action_scorer.state_dict(),
        },
        output / "model6_scorers.pt",
    )
    for prediction in predictions:
        append_jsonl(output / "validation_predictions.jsonl", prediction)
    result = {
        "model_variant": "model6",
        "steps": step,
        "epochs": epoch,
        "fit_examples": len(fit),
        "development_examples": len(development),
        "validation_examples": len(validation_records),
        "best_development_future_score": best_future_score,
        "best_development_action_score": best_action_score,
        "final_validation": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "completed_at_utc": utc_now(),
    }
    write_json(output / "final_metrics.json", result)
    if persist_outputs is not None:
        persist_outputs()
    return result
