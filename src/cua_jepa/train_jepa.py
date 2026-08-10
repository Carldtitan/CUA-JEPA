from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_

from cua_jepa.jepa_data import TransitionSample, group_by_bundle, load_transition_tars
from cua_jepa.jepa_model import (
    ActionConditionedPredictor,
    ActionEncoder,
    action_separation_loss,
    action_spatial_features,
    actions_to_tensors,
    bundle_anti_collapse_losses,
    change_patch_weights,
    latent_delta,
    latent_delta_loss,
    latent_prediction_loss,
    reconstruct_future_latent,
)


@dataclass
class JEPATrainConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    model_revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    seed: int = 20260809
    max_pixels: int = 262_144
    min_pixels: int = 65_536
    lora_rank: int = 8
    lora_alpha: int = 16
    predictor_dim: int = 384
    predictor_layers: int = 2
    predictor_heads: int = 8
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    train_qwen_lora: bool = False
    target_ema_decay: float = 0.996
    require_fresh_start: bool = True
    expected_train_transitions: int = 0
    expected_validation_transitions: int = 0
    changed_patch_weight: float = 1.0
    unchanged_patch_weight: float = 0.05
    changed_token_threshold: float = 0.01
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    max_train_transitions: int = 100
    max_validation_transitions: int = 100
    evaluation_bundles: int = 8
    validation_evaluation_bundles: int = 0
    action_separation_weight: float = 0.0
    action_separation_temperature: float = 0.1
    changed_region_loss_weight: float = 0.8
    global_loss_weight: float = 0.2
    delta_direction_weight: float = 0.75
    delta_magnitude_weight: float = 0.25
    variance_regularization_weight: float = 0.05
    covariance_regularization_weight: float = 0.05
    relation_regularization_weight: float = 0.1
    collapse_check_every: int = 100
    collapse_check_bundles: int = 8
    collapse_check_start_step: int = 200
    collapse_patience: int = 2
    collapse_accuracy_tolerance: float = 0.02
    collapse_min_separation_ratio: float = 0.03
    collapse_min_shuffled_gap: float = 0.0001
    stop_on_collapse: bool = True
    log_every: int = 5

    @classmethod
    def from_json(cls, path: str | Path) -> "JEPATrainConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def copy_online_adapter_to_target(model: torch.nn.Module, decay: float = 0.0) -> None:
    parameters = dict(model.named_parameters())
    copied = 0
    with torch.no_grad():
        for name, target in parameters.items():
            if ".target." not in name:
                continue
            online_name = name.replace(".target.", ".online.")
            online = parameters.get(online_name)
            if online is None:
                raise RuntimeError(f"Missing online EMA source for {name}")
            target.mul_(decay).add_(online, alpha=1.0 - decay)
            copied += 1
    if copied == 0:
        raise RuntimeError("No target LoRA parameters were found for the EMA update")


def _set_adapter(model: torch.nn.Module, name: str) -> None:
    if getattr(model, "peft_config", None):
        model.set_adapter(name)
        for parameter_name, parameter in model.named_parameters():
            if ".target." in parameter_name:
                parameter.requires_grad_(False)
            elif ".online." in parameter_name:
                parameter.requires_grad_(name == "online")
            else:
                parameter.requires_grad_(False)


def adapter_pair_max_difference(model: torch.nn.Module) -> float:
    parameters = dict(model.named_parameters())
    differences: list[float] = []
    with torch.no_grad():
        for name, target in parameters.items():
            if ".target." not in name:
                continue
            online = parameters.get(name.replace(".target.", ".online."))
            if online is None:
                raise RuntimeError(f"Missing online adapter pair for {name}")
            differences.append(float((target - online).abs().max().item()))
    if not differences:
        raise RuntimeError("No online/target LoRA pairs were found")
    return max(differences)


def _image_hashes(samples: Iterable[TransitionSample]) -> set[str]:
    hashes: set[str] = set()
    for sample in samples:
        hashes.add(hashlib.sha256(sample.current_webp).hexdigest())
        hashes.add(hashlib.sha256(sample.future_webp).hexdigest())
    return hashes


def validate_dataset_assignments(
    train_samples: list[TransitionSample],
    validation_samples: list[TransitionSample],
    expected_train_transitions: int = 0,
    expected_validation_transitions: int = 0,
) -> dict[str, Any]:
    train_splits = {sample.split for sample in train_samples}
    validation_splits = {sample.split for sample in validation_samples}
    if train_splits != {"train"}:
        raise RuntimeError(f"Training data contains wrong splits: {sorted(train_splits)}")
    if validation_splits != {"validation"}:
        raise RuntimeError(
            f"Validation data contains wrong splits: {sorted(validation_splits)}"
        )
    if expected_train_transitions and len(train_samples) != expected_train_transitions:
        raise RuntimeError(
            f"Expected {expected_train_transitions} training transitions, "
            f"loaded {len(train_samples)}"
        )
    if expected_validation_transitions and (
        len(validation_samples) != expected_validation_transitions
    ):
        raise RuntimeError(
            f"Expected {expected_validation_transitions} validation transitions, "
            f"loaded {len(validation_samples)}"
        )
    train_bundles = {sample.bundle_id for sample in train_samples}
    validation_bundles = {sample.bundle_id for sample in validation_samples}
    bundle_overlap = train_bundles & validation_bundles
    if bundle_overlap:
        raise RuntimeError(
            f"Training and validation share {len(bundle_overlap)} bundle IDs"
        )
    image_overlap = _image_hashes(train_samples) & _image_hashes(validation_samples)
    if image_overlap:
        raise RuntimeError(
            f"Training and validation share {len(image_overlap)} exact screenshots"
        )
    return {
        "train_split": "train",
        "validation_split": "validation",
        "train_transitions": len(train_samples),
        "validation_transitions": len(validation_samples),
        "train_bundles": len(train_bundles),
        "validation_bundles": len(validation_bundles),
        "bundle_overlap": 0,
        "exact_screenshot_overlap": 0,
    }


def approved_vision_parameters(
    vision: torch.nn.Module, train_qwen_lora: bool
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    online = [
        parameter
        for name, parameter in vision.named_parameters()
        if ".online." in name and parameter.requires_grad
    ]
    target = [
        parameter for name, parameter in vision.named_parameters() if ".target." in name
    ]
    base = [
        parameter
        for name, parameter in vision.named_parameters()
        if ".online." not in name and ".target." not in name
    ]
    if train_qwen_lora and not online:
        raise RuntimeError("No trainable Qwen online vision LoRA parameters were found")
    if not train_qwen_lora and online:
        raise RuntimeError("Qwen LoRA parameters exist in a frozen-Qwen run")
    if any(parameter.requires_grad for parameter in target):
        raise RuntimeError("Target LoRA parameters must never receive gradients")
    if any(parameter.requires_grad for parameter in base):
        raise RuntimeError("Base Qwen parameters must remain frozen")
    return online, target, base


def _vision_config(vision: torch.nn.Module):
    base_model = getattr(vision, "base_model", None)
    wrapped_model = getattr(base_model, "model", None)
    if wrapped_model is not None and hasattr(wrapped_model, "config"):
        return wrapped_model.config
    return vision.config


def _load_qwen_vision(config: JEPATrainConfig, device: torch.device):
    from transformers import AutoImageProcessor, Qwen3VLForConditionalGeneration

    processor = AutoImageProcessor.from_pretrained(
        config.model_id, revision=config.model_revision, use_fast=False
    )
    processor.max_pixels = config.max_pixels
    processor.min_pixels = config.min_pixels
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    vision = full_model.model.visual
    del full_model
    gc.collect()
    vision.to(device)

    if not config.train_qwen_lora:
        vision.requires_grad_(False)
        vision.eval()
        return processor, vision

    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)$",
    )
    vision = get_peft_model(vision, lora, adapter_name="online")
    vision.add_adapter("target", lora)
    copy_online_adapter_to_target(vision, decay=0.0)
    _set_adapter(vision, "online")
    if adapter_pair_max_difference(vision) != 0.0:
        raise RuntimeError("Online and target LoRA adapters did not start identically")
    return processor, vision


def _encode(vision, adapter: str, pixels: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    _set_adapter(vision, adapter)
    output = vision(pixels, grid_thw=grid, return_dict=True)
    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output
    if isinstance(output, tuple):
        expected_dim = int(_vision_config(vision).out_hidden_size)
        candidates = [
            value
            for value in output
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[-1] == expected_dim
        ]
        if len(candidates) == 1:
            # PEFT 0.19 flattens Qwen's ModelOutput but does not preserve the
            # dataclass field ordering. Select the merged 2048-D tokens by
            # their declared output dimension rather than a fragile index.
            return candidates[0]
        shapes = [tuple(value.shape) if torch.is_tensor(value) else type(value).__name__ for value in output]
        raise TypeError(f"Could not identify Qwen pooled tokens in tuple: {shapes}")
    raise TypeError(f"Unexpected Qwen vision output type: {type(output)!r}")


def _action_embedding(
    action_encoder: ActionEncoder, actions: list[dict[str, Any]], device: torch.device
) -> torch.Tensor:
    return action_encoder(*actions_to_tensors(actions, device=device))


@torch.no_grad()
def evaluate_action_sensitivity(
    samples: Iterable[TransitionSample],
    processor,
    vision,
    action_encoder: ActionEncoder,
    predictor: ActionConditionedPredictor,
    device: torch.device,
    max_bundles: int,
) -> dict[str, Any]:
    groups = [branches for branches in group_by_bundle(samples).values() if len(branches) == 4]
    if max_bundles > 0:
        groups = groups[:max_bundles]
    if not groups:
        return {"bundles": 0.0, "four_way_accuracy": 0.0}
    vision.eval()
    action_encoder.eval()
    predictor.eval()
    correct_ranks = 0
    correct_distances: list[float] = []
    shuffled_distances: list[float] = []
    target_pair_distances: list[float] = []
    target_pair_distances_unweighted: list[float] = []
    prediction_pair_distances: list[float] = []
    delta_distances: list[float] = []
    app_results: dict[str, dict[str, float]] = {}
    action_results: dict[str, dict[str, float]] = {}
    for branches in groups:
        current_image = branches[0].current_image()
        context_inputs = processor(images=[current_image], return_tensors="pt")
        context_tokens = _encode(
            vision,
            "online",
            context_inputs["pixel_values"].to(device),
            context_inputs["image_grid_thw"].to(device),
        )
        targets: list[torch.Tensor] = []
        target_weights: list[torch.Tensor] = []
        for branch in branches:
            future_image = branch.future_image()
            future_inputs = processor(images=[future_image], return_tensors="pt")
            targets.append(
                _encode(
                    vision,
                    "target",
                    future_inputs["pixel_values"].to(device),
                    future_inputs["image_grid_thw"].to(device),
                )
            )
            target_weights.append(
                change_patch_weights(
                    current_image,
                    future_image,
                    future_inputs["image_grid_thw"][0],
                    device,
                )
            )
        _set_adapter(vision, "online")
        actions = [branch.action for branch in branches]
        embeddings = _action_embedding(action_encoder, actions, device)
        spatial_actions = action_spatial_features(
            actions, context_inputs["image_grid_thw"][0], device
        )
        predicted_deltas = predictor(context_tokens, embeddings, spatial_actions)
        predictions = reconstruct_future_latent(context_tokens, predicted_deltas)
        target_tensor = torch.stack(targets)
        target_delta = latent_delta(context_tokens, target_tensor)
        weight_tensor = torch.stack(target_weights)
        per_bundle_delta_loss = latent_delta_loss(
            predicted_deltas, target_delta, weight_tensor
        )
        delta_distances.extend([float(per_bundle_delta_loss.item())] * 4)
        for first in range(4):
            for second in range(first + 1, 4):
                pair_weights = torch.maximum(target_weights[first], target_weights[second])
                target_pair_distances.append(
                    float(
                        latent_prediction_loss(
                            targets[first], targets[second], pair_weights
                        ).item()
                    )
                )
                target_pair_distances_unweighted.append(
                    float(latent_prediction_loss(targets[first], targets[second]).item())
                )
                prediction_pair_distances.append(
                    float(
                        latent_prediction_loss(
                            predictions[first], predictions[second], pair_weights
                        ).item()
                    )
                )
        for index, prediction in enumerate(predictions):
            distances = [
                float(latent_prediction_loss(prediction, target, target_weights[target_index]).item())
                for target_index, target in enumerate(targets)
            ]
            is_correct = int(min(range(4), key=distances.__getitem__) == index)
            correct_ranks += is_correct
            correct_distances.append(distances[index])
            shuffled_prediction = predictions[(index + 1) % 4]
            shuffled_distances.append(
                float(
                    latent_prediction_loss(
                        shuffled_prediction, targets[index], target_weights[index]
                    ).item()
                )
            )
            app = branches[index].app
            action_kind = str(branches[index].action.get("kind", "unknown"))
            for key, results in ((app, app_results), (action_kind, action_results)):
                row = results.setdefault(
                    key, {"count": 0.0, "correct": 0.0, "correct_distance": 0.0}
                )
                row["count"] += 1.0
                row["correct"] += is_correct
                row["correct_distance"] += distances[index]
    _set_adapter(vision, "online")
    if any(parameter.requires_grad for parameter in vision.parameters()):
        vision.train()
    else:
        vision.eval()
    action_encoder.train()
    predictor.train()
    count = len(groups) * 4
    target_pair_mean = sum(target_pair_distances) / len(target_pair_distances)
    prediction_pair_mean = sum(prediction_pair_distances) / len(prediction_pair_distances)
    def finalize_breakdown(
        values: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        return {
            key: {
                "count": row["count"],
                "four_way_accuracy": row["correct"] / row["count"],
                "correct_action_distance": row["correct_distance"] / row["count"],
            }
            for key, row in sorted(values.items())
        }

    return {
        "bundles": float(len(groups)),
        "four_way_accuracy": correct_ranks / count,
        "delta_prediction_loss": sum(delta_distances) / count,
        "correct_action_distance": sum(correct_distances) / count,
        "shuffled_action_distance": sum(shuffled_distances) / count,
        "shuffled_minus_correct": (sum(shuffled_distances) - sum(correct_distances)) / count,
        "target_pair_distance_mean": target_pair_mean,
        "target_pair_distance_min": min(target_pair_distances),
        "target_pair_distance_max": max(target_pair_distances),
        "target_pair_distance_unweighted_mean": sum(target_pair_distances_unweighted)
        / len(target_pair_distances_unweighted),
        "prediction_action_distance_mean": prediction_pair_mean,
        "prediction_action_distance_min": min(prediction_pair_distances),
        "prediction_action_distance_max": max(prediction_pair_distances),
        "prediction_to_target_separation_ratio": prediction_pair_mean
        / max(target_pair_mean, 1e-8),
        "by_app": finalize_breakdown(app_results),
        "by_action_kind": finalize_breakdown(action_results),
    }


def train_model4_jepa(
    train_tar_paths: list[str | Path],
    validation_tar_paths: list[str | Path],
    output_dir: str | Path,
    config: JEPATrainConfig,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Model 4 JEPA training requires a CUDA GPU")
    device = torch.device("cuda")
    output_path = Path(output_dir)
    if config.require_fresh_start and output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"Fresh-start output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    train_samples = load_transition_tars(train_tar_paths, limit=config.max_train_transitions)
    validation_samples = load_transition_tars(
        validation_tar_paths, limit=config.max_validation_transitions
    )
    if not train_samples or not validation_samples:
        raise RuntimeError("Both training and validation transitions are required")
    dataset_audit = validate_dataset_assignments(
        train_samples,
        validation_samples,
        expected_train_transitions=config.expected_train_transitions,
        expected_validation_transitions=config.expected_validation_transitions,
    )
    train_bundles = [
        branches for branches in group_by_bundle(train_samples).values() if len(branches) == 4
    ]
    if not train_bundles:
        raise RuntimeError("Training requires complete four-branch bundles")

    started = time.perf_counter()
    processor, vision = _load_qwen_vision(config, device)
    latent_dim = int(_vision_config(vision).out_hidden_size)
    action_encoder = ActionEncoder(output_dim=config.predictor_dim).to(device)
    predictor = ActionConditionedPredictor(
        latent_dim=latent_dim,
        hidden_dim=config.predictor_dim,
        action_dim=config.predictor_dim,
        layers=config.predictor_layers,
        heads=config.predictor_heads,
    ).to(device)

    online_parameters, target_parameters, base_parameters = approved_vision_parameters(
        vision, config.train_qwen_lora
    )
    head_parameters = list(action_encoder.parameters()) + list(predictor.parameters())
    trainable = online_parameters + head_parameters
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != {id(parameter) for parameter in trainable}:
        raise RuntimeError("Optimizer parameter scope does not match the approved parameters")
    if optimizer_parameter_ids & {id(parameter) for parameter in base_parameters}:
        raise RuntimeError("Optimizer contains frozen base Qwen parameters")
    if optimizer_parameter_ids & {id(parameter) for parameter in target_parameters}:
        raise RuntimeError("Optimizer contains target LoRA parameters")
    initialization_audit = {
        "fresh_start": True,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "train_qwen_lora": config.train_qwen_lora,
        "base_qwen_parameter_count": sum(p.numel() for p in base_parameters),
        "base_qwen_trainable_parameter_count": sum(
            p.numel() for p in base_parameters if p.requires_grad
        ),
        "online_lora_parameter_count": sum(p.numel() for p in online_parameters),
        "target_lora_parameter_count": sum(p.numel() for p in target_parameters),
        "target_lora_trainable_parameter_count": sum(
            p.numel() for p in target_parameters if p.requires_grad
        ),
        "head_parameter_count": sum(p.numel() for p in head_parameters),
        "optimizer_parameter_count": sum(p.numel() for p in trainable),
        "initial_online_target_max_difference": (
            adapter_pair_max_difference(vision) if config.train_qwen_lora else None
        ),
    }
    (output_path / "initialization_audit.json").write_text(
        json.dumps(initialization_audit, indent=2), encoding="utf-8"
    )
    optimizer.zero_grad(set_to_none=True)

    initial_train = evaluate_action_sensitivity(
        train_samples,
        processor,
        vision,
        action_encoder,
        predictor,
        device,
        config.evaluation_bundles,
    )
    initial_validation = evaluate_action_sensitivity(
        validation_samples,
        processor,
        vision,
        action_encoder,
        predictor,
        device,
        config.validation_evaluation_bundles,
    )

    step = 0
    losses: list[float] = []
    regression_losses: list[float] = []
    changed_region_losses: list[float] = []
    global_losses: list[float] = []
    variance_losses: list[float] = []
    covariance_losses: list[float] = []
    relation_losses: list[float] = []
    separation_losses: list[float] = []
    collapse_checks: list[dict[str, Any]] = []
    consecutive_collapse_checks = 0
    stopped_for_collapse = False
    log_path = output_path / "train.jsonl"
    while step < config.max_steps:
        epoch_bundles = list(train_bundles)
        random.Random(config.seed + step).shuffle(epoch_bundles)
        for branches in epoch_bundles:
            current = branches[0].current_image()
            current_inputs = processor(images=[current], return_tensors="pt")
            current_grid = current_inputs["image_grid_thw"].to(device)
            current_pixels = current_inputs["pixel_values"].to(device)
            if not config.train_qwen_lora:
                with torch.no_grad():
                    current_tokens = _encode(vision, "online", current_pixels, current_grid)
            else:
                current_tokens = _encode(vision, "online", current_pixels, current_grid)
            future_tokens: list[torch.Tensor] = []
            weights: list[torch.Tensor] = []
            with torch.no_grad():
                for branch in branches:
                    future = branch.future_image()
                    future_inputs = processor(images=[future], return_tensors="pt")
                    future_grid = future_inputs["image_grid_thw"].to(device)
                    future_tokens.append(
                        _encode(
                            vision,
                            "target",
                            future_inputs["pixel_values"].to(device),
                            future_grid,
                        )
                    )
                    weights.append(
                        change_patch_weights(
                            current,
                            future,
                            future_grid[0],
                            device,
                            changed_weight=config.changed_patch_weight,
                            unchanged_weight=config.unchanged_patch_weight,
                            changed_token_threshold=config.changed_token_threshold,
                        )
                    )
            _set_adapter(vision, "online")
            actions = [branch.action for branch in branches]
            action_embeddings = _action_embedding(action_encoder, actions, device)
            spatial_actions = action_spatial_features(actions, current_grid[0], device)
            predicted_deltas = predictor(
                current_tokens, action_embeddings, spatial_actions
            )
            targets = torch.stack(future_tokens)
            target_weights = torch.stack(weights)
            target_deltas = latent_delta(current_tokens, targets)
            changed_region_loss = latent_delta_loss(
                predicted_deltas,
                target_deltas,
                target_weights,
                direction_weight=config.delta_direction_weight,
                magnitude_weight=config.delta_magnitude_weight,
            )
            global_loss = latent_delta_loss(
                predicted_deltas,
                target_deltas,
                direction_weight=config.delta_direction_weight,
                magnitude_weight=config.delta_magnitude_weight,
            )
            regression_loss = (
                config.changed_region_loss_weight * changed_region_loss
                + config.global_loss_weight * global_loss
            )
            regularizers = bundle_anti_collapse_losses(
                predicted_deltas, target_deltas, target_weights
            )
            predictions = reconstruct_future_latent(current_tokens, predicted_deltas)
            if config.action_separation_weight > 0:
                separation_loss = action_separation_loss(
                    predictions,
                    targets,
                    target_weights,
                    temperature=config.action_separation_temperature,
                )
            else:
                separation_loss = regression_loss.new_zeros(())
            loss = (
                regression_loss
                + config.variance_regularization_weight * regularizers["variance"]
                + config.covariance_regularization_weight * regularizers["covariance"]
                + config.relation_regularization_weight * regularizers["relation"]
                + config.action_separation_weight * separation_loss
            )
            (loss / config.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().item()))
            regression_losses.append(float(regression_loss.detach().item()))
            changed_region_losses.append(float(changed_region_loss.detach().item()))
            global_losses.append(float(global_loss.detach().item()))
            variance_losses.append(float(regularizers["variance"].detach().item()))
            covariance_losses.append(float(regularizers["covariance"].detach().item()))
            relation_losses.append(float(regularizers["relation"].detach().item()))
            separation_losses.append(float(separation_loss.detach().item()))
            step += 1
            if step % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if config.train_qwen_lora:
                    copy_online_adapter_to_target(
                        vision, decay=config.target_ema_decay
                    )
                    _set_adapter(vision, "online")
            if step == 1 or step % config.log_every == 0:
                record = {
                    "step": step,
                    "loss": losses[-1],
                    "regression_loss": regression_losses[-1],
                    "changed_region_loss": changed_region_losses[-1],
                    "global_loss": global_losses[-1],
                    "variance_loss": variance_losses[-1],
                    "covariance_loss": covariance_losses[-1],
                    "relation_loss": relation_losses[-1],
                    "action_separation_loss": separation_losses[-1],
                    "mean_recent_loss": sum(losses[-config.log_every :])
                    / min(len(losses), config.log_every),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if (
                config.collapse_check_every > 0
                and step >= config.collapse_check_start_step
                and step % config.collapse_check_every == 0
            ):
                collapse_metrics = evaluate_action_sensitivity(
                    validation_samples,
                    processor,
                    vision,
                    action_encoder,
                    predictor,
                    device,
                    config.collapse_check_bundles,
                )
                collapsed = (
                    collapse_metrics["four_way_accuracy"]
                    <= 0.25 + config.collapse_accuracy_tolerance
                    and collapse_metrics["prediction_to_target_separation_ratio"]
                    < config.collapse_min_separation_ratio
                    and collapse_metrics["shuffled_minus_correct"]
                    < config.collapse_min_shuffled_gap
                )
                consecutive_collapse_checks = (
                    consecutive_collapse_checks + 1 if collapsed else 0
                )
                check_record: dict[str, Any] = {
                    "step": step,
                    "event": "collapse_check",
                    "collapsed": collapsed,
                    "consecutive_collapse_checks": consecutive_collapse_checks,
                    "metrics": collapse_metrics,
                }
                collapse_checks.append(check_record)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(check_record) + "\n")
                print(json.dumps(check_record), flush=True)
                if (
                    config.stop_on_collapse
                    and consecutive_collapse_checks >= config.collapse_patience
                ):
                    stopped_for_collapse = True
                    break
            if step >= config.max_steps:
                break
        if stopped_for_collapse:
            break

    if step % config.gradient_accumulation_steps:
        clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if config.train_qwen_lora:
            copy_online_adapter_to_target(vision, decay=config.target_ema_decay)
            _set_adapter(vision, "online")

    final_train = evaluate_action_sensitivity(
        train_samples,
        processor,
        vision,
        action_encoder,
        predictor,
        device,
        config.evaluation_bundles,
    )
    final_validation = evaluate_action_sensitivity(
        validation_samples,
        processor,
        vision,
        action_encoder,
        predictor,
        device,
        config.validation_evaluation_bundles,
    )
    elapsed = time.perf_counter() - started
    if online_parameters:
        adapter_path = output_path / "qwen_vision_online_lora"
        vision.save_pretrained(
            adapter_path, selected_adapters=["online"], safe_serialization=True
        )
    torch.save(
        {
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": asdict(config),
        },
        output_path / "jepa_heads.pt",
    )
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "dataset_audit": dataset_audit,
        "initialization_audit": initialization_audit,
        "train_transitions": len(train_samples),
        "train_bundles": len(train_bundles),
        "validation_transitions": len(validation_samples),
        "steps": step,
        "elapsed_seconds": elapsed,
        "steps_per_second": step / elapsed,
        "mean_loss": sum(losses) / len(losses),
        "mean_regression_loss": sum(regression_losses) / len(regression_losses),
        "mean_changed_region_loss": sum(changed_region_losses)
        / len(changed_region_losses),
        "mean_global_loss": sum(global_losses) / len(global_losses),
        "mean_variance_loss": sum(variance_losses) / len(variance_losses),
        "mean_covariance_loss": sum(covariance_losses) / len(covariance_losses),
        "mean_relation_loss": sum(relation_losses) / len(relation_losses),
        "mean_action_separation_loss": sum(separation_losses) / len(separation_losses),
        "first_10_loss": sum(losses[:10]) / min(10, len(losses)),
        "last_10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "collapse_checks": collapse_checks,
        "stopped_for_collapse": stopped_for_collapse,
        "trainable_online_lora_parameters": sum(p.numel() for p in online_parameters),
        "target_lora_parameters": sum(p.numel() for p in target_parameters),
        "final_online_target_max_difference": (
            adapter_pair_max_difference(vision) if config.train_qwen_lora else None
        ),
        "trainable_head_parameters": sum(p.numel() for p in head_parameters),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
