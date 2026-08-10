from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_

from cua_jepa.jepa_data import (
    TransitionSample,
    balanced_bundle_groups,
    load_transition_tars,
    validate_dataset_assignments,
)
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
from cua_jepa.observability import (
    append_jsonl,
    changed_pixel_bucket,
    dataset_profile,
    estimated_modal_cost,
    gradient_l2_norm,
    named_tensors_sha256,
    nonfinite_gradient_count,
    parameter_l2_norm,
    runtime_manifest,
    safe_action_record,
    tar_file_manifest,
    utc_now,
    write_json,
)


@dataclass
class VJEPA2PilotConfig:
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    model_revision: str = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
    seed: int = 20260811
    max_train_transitions: int = 2_000
    max_validation_transitions: int = 500
    max_steps: int = 500
    encoder_bundle_batch_size: int = 4
    predictor_dim: int = 384
    predictor_layers: int = 6
    predictor_heads: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    action_separation_weight: float = 0.25
    action_separation_temperature: float = 0.1
    changed_region_loss_weight: float = 0.8
    global_loss_weight: float = 0.2
    delta_direction_weight: float = 0.75
    delta_magnitude_weight: float = 0.25
    variance_regularization_weight: float = 0.05
    covariance_regularization_weight: float = 0.05
    relation_regularization_weight: float = 0.1
    changed_patch_weight: float = 1.0
    unchanged_patch_weight: float = 0.05
    changed_token_threshold: float = 0.01
    evaluation_steps: tuple[int, ...] = (0, 100, 250, 500)
    train_evaluation_bundles: int = 25
    validation_evaluation_bundles: int = 125
    log_every: int = 25
    gradient_clip: float = 1.0
    approved_cost_limit_usd: float = 2.0
    max_runtime_seconds: int = 60 * 60
    gpu_cost_per_second: float = 0.000222
    cpu_cores: float = 4.0
    cpu_cost_per_core_second: float = 0.0000131
    memory_gib: float = 16.0
    memory_cost_per_gib_second: float = 0.00000222


@dataclass
class EncodedBundle:
    bundle_id: str
    app: str
    split: str
    actions: list[dict[str, Any]]
    changed_pixel_fractions: list[float]
    current: torch.Tensor
    targets: torch.Tensor
    target_weights: torch.Tensor


def _action_embedding(
    action_encoder: ActionEncoder,
    actions: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    kinds, numeric, text_bytes, text_lengths = actions_to_tensors(actions, device)
    return action_encoder(kinds, numeric, text_bytes, text_lengths)


def letterbox_gui_image(image: Image.Image, size: int = 256) -> Image.Image:
    """Fit a complete GUI screenshot into a square without cropping it."""

    source = image.convert("RGB")
    scale = min(size / source.width, size / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    result = Image.new("RGB", (size, size), (124, 116, 104))
    result.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return result


def gui_image_video(image: Image.Image) -> torch.Tensor:
    array = np.asarray(letterbox_gui_image(image), dtype=np.uint8).copy()
    frame = torch.from_numpy(array).permute(2, 0, 1)
    return torch.stack((frame, frame), dim=0)


def encode_screen_batch(images, processor, encoder, device: torch.device) -> torch.Tensor:
    inputs = processor(
        [gui_image_video(image) for image in images],
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    with torch.inference_mode():
        output = encoder(**inputs, skip_predictor=True)
    features = output.last_hidden_state
    if features.ndim != 3 or features.shape[1:] != (256, 1024):
        raise RuntimeError(f"Unexpected V-JEPA 2 feature shape: {tuple(features.shape)}")
    return features.detach().to(device="cpu", dtype=torch.bfloat16)


def encode_bundles(
    groups: list[list[TransitionSample]],
    processor,
    encoder,
    device: torch.device,
    config: VJEPA2PilotConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[EncodedBundle]:
    encoded: list[EncodedBundle] = []
    grid = torch.tensor([1, 32, 32])
    batch_size = max(config.encoder_bundle_batch_size, 1)
    for offset in range(0, len(groups), batch_size):
        chunk = groups[offset : offset + batch_size]
        decoded: list[tuple[Any, list[Any]]] = []
        images = []
        for branches in chunk:
            current = branches[0].current_image()
            futures = [branch.future_image() for branch in branches]
            decoded.append((current, futures))
            images.extend([current, *futures])
        features = encode_screen_batch(images, processor, encoder, device)
        for index, branches in enumerate(chunk):
            start = index * 5
            current_image, future_images = decoded[index]
            current_mask_image = letterbox_gui_image(current_image)
            weights = torch.stack(
                [
                    change_patch_weights(
                        current_mask_image,
                        letterbox_gui_image(future),
                        grid,
                        torch.device("cpu"),
                        changed_weight=config.changed_patch_weight,
                        unchanged_weight=config.unchanged_patch_weight,
                        changed_token_threshold=config.changed_token_threshold,
                    )
                    for future in future_images
                ]
            )
            encoded.append(
                EncodedBundle(
                    bundle_id=branches[0].bundle_id,
                    app=branches[0].app,
                    split=branches[0].split,
                    actions=[dict(branch.action) for branch in branches],
                    changed_pixel_fractions=[
                        float(branch.changed_pixel_fraction) for branch in branches
                    ],
                    current=features[start],
                    targets=features[start + 1 : start + 5],
                    target_weights=weights,
                )
            )
        if progress is not None:
            progress(
                {
                    "event": "feature_encoding",
                    "completed_bundles": len(encoded),
                    "total_bundles": len(groups),
                }
            )
    return encoded


def select_balanced_encoded_bundles(
    bundles: Iterable[EncodedBundle], max_bundles: int
) -> list[EncodedBundle]:
    values = list(bundles)
    if max_bundles <= 0 or max_bundles >= len(values):
        return values
    by_app: dict[str, list[EncodedBundle]] = {}
    for bundle in values:
        by_app.setdefault(bundle.app, []).append(bundle)
    selected: list[EncodedBundle] = []
    positions = {app: 0 for app in by_app}
    while len(selected) < max_bundles:
        added = False
        for app in sorted(by_app):
            position = positions[app]
            if position >= len(by_app[app]):
                continue
            selected.append(by_app[app][position])
            positions[app] += 1
            added = True
            if len(selected) == max_bundles:
                break
        if not added:
            break
    return selected


def balanced_training_epoch(bundles: Iterable[EncodedBundle], seed: int) -> list[EncodedBundle]:
    """Create one shuffled epoch with equal sampling from each application."""

    by_app: dict[str, list[EncodedBundle]] = {}
    for bundle in bundles:
        by_app.setdefault(bundle.app, []).append(bundle)
    if not by_app:
        return []
    rng = random.Random(seed)
    for values in by_app.values():
        rng.shuffle(values)
    maximum = max(len(values) for values in by_app.values())
    order: list[EncodedBundle] = []
    for index in range(maximum):
        apps = sorted(by_app)
        rng.shuffle(apps)
        for app_name in apps:
            values = by_app[app_name]
            order.append(values[index % len(values)])
    return order


def bundle_bootstrap_ci95(
    bundle_accuracies: list[float], seed: int, draws: int = 1_000
) -> list[float]:
    if not bundle_accuracies:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    count = len(bundle_accuracies)
    for _ in range(draws):
        means.append(sum(bundle_accuracies[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    return [means[int(draws * 0.025)], means[min(int(draws * 0.975), draws - 1)]]


def _finalize_breakdown(
    values: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        key: {
            "count": row["count"],
            "four_way_accuracy": row["correct"] / row["count"],
        }
        for key, row in sorted(values.items())
    }


def evaluate_encoded_bundles(
    bundles: Iterable[EncodedBundle],
    action_encoder: ActionEncoder,
    predictor: ActionConditionedPredictor,
    device: torch.device,
    max_bundles: int,
    seed: int,
) -> dict[str, Any]:
    selected = select_balanced_encoded_bundles(bundles, max_bundles)
    action_encoder.eval()
    predictor.eval()
    correct = 0
    shuffled_correct = 0
    correct_distances: list[float] = []
    shuffled_distances: list[float] = []
    target_pair_distances: list[float] = []
    prediction_pair_distances: list[float] = []
    target_variances: list[float] = []
    prediction_variances: list[float] = []
    app_results: dict[str, dict[str, float]] = {}
    action_results: dict[str, dict[str, float]] = {}
    changed_results: dict[str, dict[str, float]] = {}
    bundle_accuracies: list[float] = []
    records: list[dict[str, Any]] = []
    grid = torch.tensor([1, 32, 32])
    with torch.inference_mode():
        for bundle in selected:
            current = bundle.current.to(device=device, dtype=torch.float32)
            targets = bundle.targets.to(device=device, dtype=torch.float32)
            weights = bundle.target_weights.to(device=device, dtype=torch.float32)
            embeddings = _action_embedding(action_encoder, bundle.actions, device)
            spatial = action_spatial_features(bundle.actions, grid, device)
            predicted_deltas = predictor(current, embeddings, spatial)
            predictions = reconstruct_future_latent(current, predicted_deltas)
            shared_weights = weights.max(dim=0).values
            target_variances.append(float(targets.var(dim=0).mean().item()))
            prediction_variances.append(float(predictions.var(dim=0).mean().item()))
            for first in range(4):
                for second in range(first + 1, 4):
                    target_pair_distances.append(
                        float(
                            latent_prediction_loss(
                                targets[first], targets[second], shared_weights
                            ).item()
                        )
                    )
                    prediction_pair_distances.append(
                        float(
                            latent_prediction_loss(
                                predictions[first], predictions[second], shared_weights
                            ).item()
                        )
                    )
            action_rows = []
            bundle_correct = 0
            for index, prediction in enumerate(predictions):
                distances = [
                    float(latent_prediction_loss(prediction, target, shared_weights).item())
                    for target in targets
                ]
                predicted_index = min(range(4), key=distances.__getitem__)
                is_correct = int(predicted_index == index)
                correct += is_correct
                bundle_correct += is_correct
                correct_distances.append(distances[index])
                shuffled_prediction = predictions[(index + 1) % 4]
                shuffled_candidates = [
                    float(
                        latent_prediction_loss(shuffled_prediction, target, shared_weights).item()
                    )
                    for target in targets
                ]
                shuffled_index = min(range(4), key=shuffled_candidates.__getitem__)
                shuffled_is_correct = int(shuffled_index == index)
                shuffled_correct += shuffled_is_correct
                shuffled_distances.append(shuffled_candidates[index])
                action_kind = str(bundle.actions[index].get("kind", "unknown"))
                changed = bundle.changed_pixel_fractions[index]
                change_bucket = changed_pixel_bucket(changed)
                for key, destination in (
                    (bundle.app, app_results),
                    (action_kind, action_results),
                    (change_bucket, changed_results),
                    ("large_change" if changed >= 0.2 else "small_change", changed_results),
                ):
                    row = destination.setdefault(key, {"count": 0.0, "correct": 0.0})
                    row["count"] += 1.0
                    row["correct"] += is_correct
                action_rows.append(
                    {
                        "branch_index": index,
                        "action": safe_action_record(bundle.actions[index]),
                        "changed_pixel_fraction": changed,
                        "correct": bool(is_correct),
                        "predicted_target_branch_index": predicted_index,
                        "shuffled_correct": bool(shuffled_is_correct),
                        "shuffled_predicted_target_branch_index": shuffled_index,
                    }
                )
            bundle_accuracies.append(bundle_correct / 4)
            records.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "application": bundle.app,
                    "split": bundle.split,
                    "actions": action_rows,
                }
            )
    count = len(selected) * 4
    accuracy = correct / count
    shuffled_accuracy = shuffled_correct / count
    target_pair_mean = sum(target_pair_distances) / len(target_pair_distances)
    prediction_pair_mean = sum(prediction_pair_distances) / len(prediction_pair_distances)
    action_encoder.train()
    predictor.train()
    return {
        "bundles": len(selected),
        "four_way_accuracy": accuracy,
        "bundle_bootstrap_ci95": bundle_bootstrap_ci95(bundle_accuracies, seed),
        "shuffled_four_way_accuracy": shuffled_accuracy,
        "action_accuracy_drop": accuracy - shuffled_accuracy,
        "correct_action_distance": sum(correct_distances) / count,
        "shuffled_action_distance": sum(shuffled_distances) / count,
        "shuffled_minus_correct": (sum(shuffled_distances) - sum(correct_distances)) / count,
        "target_pair_distance_mean": target_pair_mean,
        "prediction_action_distance_mean": prediction_pair_mean,
        "prediction_to_target_separation_ratio": prediction_pair_mean / max(target_pair_mean, 1e-8),
        "target_latent_variance": sum(target_variances) / len(target_variances),
        "predicted_latent_variance": sum(prediction_variances) / len(prediction_variances),
        "by_app": _finalize_breakdown(app_results),
        "by_action_kind": _finalize_breakdown(action_results),
        "by_changed_pixel_bucket": _finalize_breakdown(changed_results),
        "per_bundle_records": records,
    }


def pilot_success(metrics: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    click = metrics["by_action_kind"].get("click", {}).get("four_way_accuracy", 0.0)
    large = metrics["by_changed_pixel_bucket"].get("large_change", {}).get("four_way_accuracy", 0.0)
    minimum_app = min(
        (row["four_way_accuracy"] for row in metrics["by_app"].values()),
        default=0.0,
    )
    checks = {
        "overall_at_least_40_percent": metrics["four_way_accuracy"] >= 0.40,
        "click_at_least_38_percent": click >= 0.38,
        "large_change_at_least_30_percent": large >= 0.30,
        "action_drop_at_least_10_points": metrics["action_accuracy_drop"] >= 0.10,
        "every_app_at_least_30_percent": minimum_app >= 0.30,
        "ci_lower_bound_above_chance": metrics["bundle_bootstrap_ci95"][0] > 0.25,
    }
    return all(checks.values()), checks


def validate_vjepa2_pilot_artifacts(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    required = {
        "config.json",
        "dataset_audit.json",
        "run_manifest.json",
        "train.jsonl",
        "evaluation_checkpoints.jsonl",
        "evaluation_bundles.jsonl",
        "final_metrics.json",
        "vjepa2_gui_heads.pt",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError(f"V-JEPA 2 pilot artifacts are missing: {missing}")
    metrics = json.loads((root / "final_metrics.json").read_text(encoding="utf-8"))
    json.dumps(metrics, allow_nan=False)
    if metrics["steps"] <= 0:
        raise ValueError("V-JEPA 2 pilot saved no completed training steps")
    weights = torch.load(root / "vjepa2_gui_heads.pt", map_location="cpu", weights_only=False)
    if not weights.get("action_encoder") or not weights.get("predictor"):
        raise ValueError("V-JEPA 2 pilot head weights are empty")
    evaluation_rows = [
        json.loads(line)
        for line in (root / "evaluation_checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    names = {row["evaluation"] for row in evaluation_rows}
    if not {"initial_validation", "final_validation"} <= names:
        raise ValueError("V-JEPA 2 pilot evaluations are incomplete")
    return {
        "passed": True,
        "steps": metrics["steps"],
        "evaluation_records": len(evaluation_rows),
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
    }


def train_vjepa2_gui_pilot(
    train_tar_paths: list[str],
    validation_tar_paths: list[str],
    output_dir: str | Path,
    config: VJEPA2PilotConfig,
    run_metadata: dict[str, Any] | None = None,
    persistence_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from transformers import AutoModel, AutoVideoProcessor

    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The V-JEPA 2 GUI pilot requires a CUDA GPU")
    device = torch.device("cuda")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"V-JEPA 2 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(config))
    if config.max_train_transitions % 4 or config.max_validation_transitions % 4:
        raise ValueError("Transition limits must contain complete four-branch bundles")
    all_train_samples = load_transition_tars(train_tar_paths)
    all_validation_samples = load_transition_tars(validation_tar_paths)
    train_groups = balanced_bundle_groups(all_train_samples, config.max_train_transitions // 4)
    validation_groups = balanced_bundle_groups(
        all_validation_samples, config.max_validation_transitions // 4
    )
    train_samples = [sample for branches in train_groups for sample in branches]
    validation_samples = [sample for branches in validation_groups for sample in branches]
    assignment_audit = validate_dataset_assignments(train_samples, validation_samples)
    dataset_audit = {
        **assignment_audit,
        "train": dataset_profile(train_samples),
        "validation": dataset_profile(validation_samples),
        "train_tar_manifest": tar_file_manifest(train_tar_paths, "train"),
        "validation_tar_manifest": tar_file_manifest(validation_tar_paths, "validation"),
    }
    write_json(output / "dataset_audit.json", dataset_audit)
    if len(train_groups) * 4 != len(train_samples):
        raise RuntimeError("Training data contains incomplete four-branch bundles")
    if len(validation_groups) * 4 != len(validation_samples):
        raise RuntimeError("Validation data contains incomplete four-branch bundles")

    manifest = runtime_manifest(asdict(config), run_metadata)
    manifest["encoder_frozen"] = True
    write_json(output / "run_manifest.json", manifest)
    processor = AutoVideoProcessor.from_pretrained(config.model_id, revision=config.model_revision)
    encoder = AutoModel.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
    )
    encoder.requires_grad_(False).eval().to(device)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("The V-JEPA 2 encoder is not completely frozen")

    def report(value: dict[str, Any]) -> None:
        value = {"recorded_at_utc": utc_now(), **value}
        print(json.dumps(value, allow_nan=False), flush=True)

    report({"event": "phase", "phase": "encode_train"})
    encoded_train = encode_bundles(train_groups, processor, encoder, device, config, report)
    report({"event": "phase", "phase": "encode_validation"})
    encoded_validation = encode_bundles(
        validation_groups, processor, encoder, device, config, report
    )
    del encoder, processor
    gc.collect()
    torch.cuda.empty_cache()

    action_encoder = ActionEncoder(config.predictor_dim).to(device)
    predictor = ActionConditionedPredictor(
        latent_dim=1024,
        hidden_dim=config.predictor_dim,
        action_dim=config.predictor_dim,
        layers=config.predictor_layers,
        heads=config.predictor_heads,
    ).to(device)
    trainable = list(action_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_hashes = {
        "action_encoder": named_tensors_sha256(action_encoder.state_dict().items()),
        "predictor": named_tensors_sha256(predictor.state_dict().items()),
    }
    write_json(
        output / "initialization_audit.json",
        {
            "encoder_trainable_parameters": 0,
            "action_encoder_parameters": sum(
                parameter.numel() for parameter in action_encoder.parameters()
            ),
            "predictor_parameters": sum(parameter.numel() for parameter in predictor.parameters()),
            "initial_hashes": initial_hashes,
        },
    )
    evaluation_path = output / "evaluation_checkpoints.jsonl"
    bundle_path = output / "evaluation_bundles.jsonl"

    def evaluate(name: str, step: int, values, maximum: int) -> dict[str, Any]:
        result = evaluate_encoded_bundles(
            values, action_encoder, predictor, device, maximum, config.seed + step
        )
        records = result.pop("per_bundle_records")
        row = {"recorded_at_utc": utc_now(), "step": step, "evaluation": name, **result}
        append_jsonl(evaluation_path, row)
        for record in records:
            append_jsonl(bundle_path, {"step": step, "evaluation": name, **record})
        print(json.dumps({"event": "evaluation", **row}, allow_nan=False), flush=True)
        return result

    initial_train = evaluate("initial_train", 0, encoded_train, config.train_evaluation_bundles)
    initial_validation = evaluate(
        "initial_validation",
        0,
        encoded_validation,
        config.validation_evaluation_bundles,
    )
    losses: list[float] = []
    train_path = output / "train.jsonl"
    grid = torch.tensor([1, 32, 32])
    step = 0
    epoch = 0
    while step < config.max_steps:
        epoch += 1
        order = balanced_training_epoch(encoded_train, config.seed + epoch)
        for bundle in order:
            if step >= config.max_steps:
                break
            elapsed = time.perf_counter() - started
            cost = estimated_modal_cost(
                elapsed,
                config.gpu_cost_per_second,
                config.cpu_cores,
                config.cpu_cost_per_core_second,
                config.memory_gib,
                config.memory_cost_per_gib_second,
            )
            if elapsed >= config.max_runtime_seconds:
                raise RuntimeError("V-JEPA 2 pilot reached its runtime limit")
            if cost["total_usd"] >= config.approved_cost_limit_usd:
                raise RuntimeError("V-JEPA 2 pilot reached its cost limit")
            current = bundle.current.to(device=device, dtype=torch.float32)
            targets = bundle.targets.to(device=device, dtype=torch.float32)
            weights = bundle.target_weights.to(device=device, dtype=torch.float32)
            embeddings = _action_embedding(action_encoder, bundle.actions, device)
            spatial = action_spatial_features(bundle.actions, grid, device)
            predicted_deltas = predictor(current, embeddings, spatial)
            target_deltas = latent_delta(current, targets)
            changed_loss = latent_delta_loss(
                predicted_deltas,
                target_deltas,
                weights,
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
                config.changed_region_loss_weight * changed_loss
                + config.global_loss_weight * global_loss
            )
            regularizers = bundle_anti_collapse_losses(predicted_deltas, target_deltas, weights)
            predictions = reconstruct_future_latent(current, predicted_deltas)
            separation_loss = action_separation_loss(
                predictions,
                targets,
                weights,
                temperature=config.action_separation_temperature,
            )
            loss = (
                regression_loss
                + config.variance_regularization_weight * regularizers["variance"]
                + config.covariance_regularization_weight * regularizers["covariance"]
                + config.relation_regularization_weight * regularizers["relation"]
                + config.action_separation_weight * separation_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("V-JEPA 2 pilot produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(clip_grad_norm_(trainable, config.gradient_clip).item())
            nonfinite_gradients = nonfinite_gradient_count(trainable)
            if nonfinite_gradients:
                raise RuntimeError(
                    f"V-JEPA 2 pilot produced {nonfinite_gradients} non-finite gradients"
                )
            optimizer.step()
            step += 1
            losses.append(float(loss.detach().item()))
            if step == 1 or step % config.log_every == 0:
                record = {
                    "recorded_at_utc": utc_now(),
                    "step": step,
                    "epoch": epoch,
                    "bundle_id": bundle.bundle_id,
                    "application": bundle.app,
                    "loss": losses[-1],
                    "mean_recent_loss": sum(losses[-config.log_every :])
                    / min(len(losses), config.log_every),
                    "regression_loss": float(regression_loss.detach().item()),
                    "changed_region_loss": float(changed_loss.detach().item()),
                    "global_loss": float(global_loss.detach().item()),
                    "action_separation_loss": float(separation_loss.detach().item()),
                    "gradient_norm_before_clip": gradient_norm,
                    "action_encoder_gradient_norm": gradient_l2_norm(action_encoder.parameters()),
                    "predictor_gradient_norm": gradient_l2_norm(predictor.parameters()),
                    "action_encoder_parameter_norm": parameter_l2_norm(action_encoder.parameters()),
                    "predictor_parameter_norm": parameter_l2_norm(predictor.parameters()),
                    "elapsed_seconds": time.perf_counter() - started,
                    "estimated_modal_cost_usd": cost,
                }
                append_jsonl(train_path, record)
                print(json.dumps(record, allow_nan=False), flush=True)
            if step in config.evaluation_steps and step < config.max_steps:
                evaluate(
                    "train_monitor",
                    step,
                    encoded_train,
                    config.train_evaluation_bundles,
                )
                evaluate(
                    "validation_monitor",
                    step,
                    encoded_validation,
                    config.validation_evaluation_bundles,
                )

    final_train = evaluate("final_train", step, encoded_train, config.train_evaluation_bundles)
    final_validation = evaluate(
        "final_validation",
        step,
        encoded_validation,
        config.validation_evaluation_bundles,
    )
    passed, checks = pilot_success(final_validation)
    final_hashes = {
        "action_encoder": named_tensors_sha256(action_encoder.state_dict().items()),
        "predictor": named_tensors_sha256(predictor.state_dict().items()),
    }
    if final_hashes == initial_hashes:
        raise RuntimeError("V-JEPA 2 pilot head weights did not change")
    torch.save(
        {
            "config": asdict(config),
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "initial_hashes": initial_hashes,
            "final_hashes": final_hashes,
        },
        output / "vjepa2_gui_heads.pt",
    )
    elapsed = time.perf_counter() - started
    final_cost = estimated_modal_cost(
        elapsed,
        config.gpu_cost_per_second,
        config.cpu_cores,
        config.cpu_cost_per_core_second,
        config.memory_gib,
        config.memory_cost_per_gib_second,
    )
    metrics = {
        "steps": step,
        "elapsed_seconds": elapsed,
        "estimated_modal_cost_usd": final_cost,
        "mean_loss": sum(losses) / len(losses),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "success_gate_passed": passed,
        "success_gate_checks": checks,
        "encoder_frozen": True,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    manifest["completed_at_utc"] = utc_now()
    manifest["completed_steps"] = step
    write_json(output / "run_manifest.json", manifest)
    write_json(output / "final_metrics.json", metrics)
    validation = validate_vjepa2_pilot_artifacts(output)
    metrics["artifact_validation"] = validation
    write_json(output / "final_metrics.json", metrics)
    if persistence_callback is not None:
        persistence_callback()
    return metrics
