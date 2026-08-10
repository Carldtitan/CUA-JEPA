from __future__ import annotations

import gc
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
    change_patch_weights,
    latent_prediction_loss,
)


@dataclass
class JEPATrainConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
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
    ema_decay: float = 0.996
    freeze_online_encoder: bool = True
    freeze_target_encoder: bool = True
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
    log_every: int = 5

    @classmethod
    def from_json(cls, path: str | Path) -> "JEPATrainConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def copy_online_adapter_to_target(model: torch.nn.Module, decay: float = 0.0) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, target in parameters.items():
            if ".target." not in name:
                continue
            online_name = name.replace(".target.", ".online.")
            online = parameters.get(online_name)
            if online is None:
                raise RuntimeError(f"Missing online EMA source for {name}")
            target.mul_(decay).add_(online, alpha=1.0 - decay)


def _set_adapter(model: torch.nn.Module, name: str) -> None:
    if getattr(model, "peft_config", None):
        model.set_adapter(name)


def _vision_config(vision: torch.nn.Module):
    base_model = getattr(vision, "base_model", None)
    wrapped_model = getattr(base_model, "model", None)
    if wrapped_model is not None and hasattr(wrapped_model, "config"):
        return wrapped_model.config
    return vision.config


def _load_qwen_vision(config: JEPATrainConfig, device: torch.device):
    from transformers import AutoImageProcessor, Qwen3VLForConditionalGeneration

    processor = AutoImageProcessor.from_pretrained(config.model_id, use_fast=False)
    processor.max_pixels = config.max_pixels
    processor.min_pixels = config.min_pixels
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    vision = full_model.model.visual
    del full_model
    gc.collect()
    vision.to(device)

    if config.freeze_online_encoder:
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
) -> dict[str, float]:
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
        predictions = [
            predictor(
                context_tokens,
                embeddings[index : index + 1],
                spatial_actions[index : index + 1],
            )
            for index in range(4)
        ]
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
            correct_ranks += int(min(range(4), key=distances.__getitem__) == index)
            correct_distances.append(distances[index])
            shuffled_prediction = predictions[(index + 1) % 4]
            shuffled_distances.append(
                float(
                    latent_prediction_loss(
                        shuffled_prediction, targets[index], target_weights[index]
                    ).item()
                )
            )
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
    return {
        "bundles": float(len(groups)),
        "four_way_accuracy": correct_ranks / count,
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
    output_path.mkdir(parents=True, exist_ok=True)
    train_samples = load_transition_tars(train_tar_paths, limit=config.max_train_transitions)
    validation_samples = load_transition_tars(
        validation_tar_paths, limit=config.max_validation_transitions
    )
    if not train_samples or not validation_samples:
        raise RuntimeError("Both training and validation transitions are required")
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

    online_parameters = [
        parameter
        for name, parameter in vision.named_parameters()
        if ".online." in name and parameter.requires_grad
    ]
    if not config.freeze_online_encoder and not online_parameters:
        raise RuntimeError("No trainable Qwen online vision LoRA parameters were found")
    head_parameters = list(action_encoder.parameters()) + list(predictor.parameters())
    trainable = online_parameters + head_parameters
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
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
    separation_losses: list[float] = []
    log_path = output_path / "train.jsonl"
    while step < config.max_steps:
        epoch_bundles = list(train_bundles)
        random.Random(config.seed + step).shuffle(epoch_bundles)
        for branches in epoch_bundles:
            current = branches[0].current_image()
            current_inputs = processor(images=[current], return_tensors="pt")
            current_grid = current_inputs["image_grid_thw"].to(device)
            current_pixels = current_inputs["pixel_values"].to(device)
            if config.freeze_online_encoder:
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
            predictions = predictor(current_tokens, action_embeddings, spatial_actions)
            targets = torch.stack(future_tokens)
            target_weights = torch.stack(weights)
            regression_loss = latent_prediction_loss(
                predictions, targets, target_weights
            )
            if config.action_separation_weight > 0:
                separation_loss = action_separation_loss(
                    predictions,
                    targets,
                    target_weights,
                    temperature=config.action_separation_temperature,
                )
            else:
                separation_loss = regression_loss.new_zeros(())
            loss = regression_loss + config.action_separation_weight * separation_loss
            (loss / config.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().item()))
            regression_losses.append(float(regression_loss.detach().item()))
            separation_losses.append(float(separation_loss.detach().item()))
            step += 1
            if step % config.gradient_accumulation_steps == 0:
                clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if not config.freeze_online_encoder and not config.freeze_target_encoder:
                    copy_online_adapter_to_target(vision, decay=config.ema_decay)
            if step == 1 or step % config.log_every == 0:
                record = {
                    "step": step,
                    "loss": losses[-1],
                    "regression_loss": regression_losses[-1],
                    "action_separation_loss": separation_losses[-1],
                    "mean_recent_loss": sum(losses[-config.log_every :])
                    / min(len(losses), config.log_every),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if step >= config.max_steps:
                break

    if step % config.gradient_accumulation_steps:
        clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not config.freeze_online_encoder and not config.freeze_target_encoder:
            copy_online_adapter_to_target(vision, decay=config.ema_decay)

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
        "train_transitions": len(train_samples),
        "train_bundles": len(train_bundles),
        "validation_transitions": len(validation_samples),
        "steps": step,
        "elapsed_seconds": elapsed,
        "steps_per_second": step / elapsed,
        "mean_loss": sum(losses) / len(losses),
        "mean_regression_loss": sum(regression_losses) / len(regression_losses),
        "mean_action_separation_loss": sum(separation_losses) / len(separation_losses),
        "first_10_loss": sum(losses[:10]) / min(10, len(losses)),
        "last_10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "trainable_online_lora_parameters": sum(p.numel() for p in online_parameters),
        "trainable_head_parameters": sum(p.numel() for p in head_parameters),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
