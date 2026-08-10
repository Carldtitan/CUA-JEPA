from pathlib import Path

import torch
from PIL import Image, ImageDraw

from cua_jepa.train_vjepa2 import (
    EncodedBundle,
    action_prediction_targets,
    balanced_training_epoch,
    bundle_bootstrap_ci95,
    encoded_feature_cache_key,
    fusion_gate_metrics,
    gui_image_video,
    letterbox_action_coordinates,
    letterbox_gui_image,
    pilot_success,
    prediction_space,
    prediction_target_weights,
    qwen_feature_cache_key,
    qwen_tokens_to_fixed_grid,
    select_balanced_encoded_bundles,
    two_tile_action_coordinates,
    two_tile_gui_images,
    two_tile_screen_positions,
    validate_vjepa2_pilot_artifacts,
    validate_encoded_feature_cache,
    VJEPA2PilotConfig,
)

from cua_jepa.jepa_model import QwenVJEPAFusionPredictor


def _bundle(bundle_id: str, app: str) -> EncodedBundle:
    return EncodedBundle(
        bundle_id=bundle_id,
        app=app,
        split="validation",
        actions=[{"kind": "click"}] * 4,
        spatial_actions=[{"kind": "click"}] * 4,
        changed_pixel_fractions=[0.3] * 4,
        current=torch.zeros(256, 1024),
        targets=torch.zeros(4, 256, 1024),
        target_weights=torch.ones(4, 256),
    )


def test_letterbox_keeps_both_edges_of_wide_gui() -> None:
    image = Image.new("RGB", (128, 72), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 7, 71), fill="red")
    draw.rectangle((120, 0, 127, 71), fill="blue")
    result = letterbox_gui_image(image)
    assert result.size == (256, 256)
    assert result.getpixel((2, 128))[0] > 200
    assert result.getpixel((253, 128))[2] > 200
    video = gui_image_video(image)
    assert video.shape == (2, 3, 256, 256)
    assert torch.equal(video[0], video[1])


def test_letterbox_moves_pointer_coordinates_with_gui_content() -> None:
    top_left = letterbox_action_coordinates(
        {"kind": "click", "x_normalized": 0.0, "y_normalized": 0.0}, 1280, 720
    )
    bottom_right = letterbox_action_coordinates(
        {"kind": "click", "x_normalized": 1.0, "y_normalized": 1.0}, 1280, 720
    )
    assert top_left["x_normalized"] == 0.0
    assert top_left["y_normalized"] == 56 / 256
    assert bottom_right["x_normalized"] == 1.0
    assert bottom_right["y_normalized"] == 200 / 256


def test_two_tiles_keep_both_screen_edges_at_full_height() -> None:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 39, 719), fill="red")
    draw.rectangle((1240, 0, 1279, 719), fill="blue")
    left, right = two_tile_gui_images(image)
    assert left.size == (256, 256)
    assert right.size == (256, 256)
    assert left.getpixel((2, 128))[0] > 200
    assert right.getpixel((253, 128))[2] > 200


def test_two_tile_pointer_is_active_only_where_visible() -> None:
    left_action, right_action = two_tile_action_coordinates(
        {"kind": "click", "x_normalized": 0.1, "y_normalized": 0.75}, 1280, 720
    )
    assert left_action["spatial_active"] is True
    assert right_action["spatial_active"] is False
    assert left_action["y_normalized"] == 0.75
    overlap_left, overlap_right = two_tile_action_coordinates(
        {"kind": "click", "x_normalized": 0.5, "y_normalized": 0.5}, 1280, 720
    )
    assert overlap_left["spatial_active"] is True
    assert overlap_right["spatial_active"] is True


def test_two_tile_positions_cover_complete_screen() -> None:
    positions = two_tile_screen_positions(1280, 720)
    assert positions.shape == (512, 3)
    assert positions[:, 0].min() > 0.0
    assert positions[:, 0].max() < 1.0
    assert set(positions[:, 2].tolist()) == {0.0, 1.0}


def test_select_balanced_encoded_bundles_round_robins_apps() -> None:
    bundles = [_bundle(f"{app}-{index}", app) for app in ("jira", "slack") for index in range(3)]
    selected = select_balanced_encoded_bundles(bundles, 5)
    assert [bundle.app for bundle in selected] == [
        "jira",
        "slack",
        "jira",
        "slack",
        "jira",
    ]


def test_balanced_training_epoch_oversamples_small_app_pool() -> None:
    bundles = [_bundle(f"jira-{index}", "jira") for index in range(3)]
    bundles += [_bundle("slack-0", "slack")]
    selected = balanced_training_epoch(bundles, seed=5)
    assert len(selected) == 6
    assert sum(bundle.app == "jira" for bundle in selected) == 3
    assert sum(bundle.app == "slack" for bundle in selected) == 3


def test_bundle_bootstrap_is_deterministic_and_bounded() -> None:
    first = bundle_bootstrap_ci95([0.0, 0.25, 0.5, 1.0], seed=7, draws=200)
    second = bundle_bootstrap_ci95([0.0, 0.25, 0.5, 1.0], seed=7, draws=200)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_counterfactual_targets_remove_common_future_content() -> None:
    current = torch.randn(6, 8)
    common_future = torch.randn(6, 8)
    branch_changes = torch.randn(4, 6, 8) * 0.01
    futures = common_future.unsqueeze(0) + branch_changes
    residuals = action_prediction_targets(current, futures, "counterfactual_residual")
    assert residuals.shape == futures.shape
    assert torch.allclose(residuals.mean(dim=0), torch.zeros_like(current), atol=1e-6)
    shifted = futures + torch.randn(6, 8).unsqueeze(0)
    shifted_residuals = action_prediction_targets(
        current, shifted, "counterfactual_residual"
    )
    # Normalization changes the exact values, but a common unnormalized copy path
    # can no longer appear directly in the centered target.
    assert torch.allclose(
        shifted_residuals.mean(dim=0), torch.zeros_like(current), atol=1e-6
    )


def test_counterfactual_prediction_space_does_not_copy_current_screen() -> None:
    current = torch.randn(6, 8)
    futures = torch.randn(4, 6, 8)
    predicted = torch.randn(4, 6, 8)
    student, target = prediction_space(
        current, futures, predicted, "counterfactual_residual"
    )
    assert student.data_ptr() == predicted.data_ptr()
    assert torch.allclose(target.mean(dim=0), torch.zeros_like(current), atol=1e-6)


def test_counterfactual_targets_use_one_union_change_mask() -> None:
    weights = torch.tensor(
        [[1.0, 0.05], [0.05, 1.0], [0.05, 0.05], [0.05, 0.05]]
    )
    effective = prediction_target_weights(weights, "counterfactual_residual")
    assert torch.equal(effective, torch.ones_like(weights))


def test_frozen_feature_cache_key_changes_with_screen_view() -> None:
    audit = {
        "train_tar_manifest": {"manifest_sha256": "train"},
        "validation_tar_manifest": {"manifest_sha256": "validation"},
    }
    letterbox = VJEPA2PilotConfig(screen_views="letterbox")
    tiles = VJEPA2PilotConfig(screen_views="two_tiles")
    assert encoded_feature_cache_key(letterbox, audit) != encoded_feature_cache_key(tiles, audit)


def test_frozen_feature_cache_key_changes_with_split_strategy() -> None:
    audit = {
        "train_tar_manifest": {"manifest_sha256": "train"},
        "validation_tar_manifest": {"manifest_sha256": "validation"},
    }
    app_disjoint = VJEPA2PilotConfig(dataset_split_strategy="app_disjoint")
    same_app = VJEPA2PilotConfig(dataset_split_strategy="same_app_holdout")
    assert encoded_feature_cache_key(app_disjoint, audit) != encoded_feature_cache_key(
        same_app, audit
    )


def test_frozen_feature_cache_key_changes_with_split_seed() -> None:
    audit = {
        "train_tar_manifest": {"manifest_sha256": "train"},
        "validation_tar_manifest": {"manifest_sha256": "validation"},
    }
    first = VJEPA2PilotConfig(dataset_split_seed=1)
    second = VJEPA2PilotConfig(dataset_split_seed=2)
    assert encoded_feature_cache_key(first, audit) != encoded_feature_cache_key(second, audit)


def test_qwen_feature_cache_key_changes_with_semantic_grid() -> None:
    audit = {
        "train_tar_manifest": {"manifest_sha256": "train"},
        "validation_tar_manifest": {"manifest_sha256": "validation"},
    }
    small = VJEPA2PilotConfig(qwen_semantic_grid_size=4)
    large = VJEPA2PilotConfig(qwen_semantic_grid_size=8)
    assert qwen_feature_cache_key(small, audit) != qwen_feature_cache_key(large, audit)


def test_qwen_tokens_resize_to_fixed_semantic_grid() -> None:
    tokens = torch.arange(24 * 32, dtype=torch.float32).reshape(24, 32)
    fixed = qwen_tokens_to_fixed_grid(tokens, torch.tensor([1, 8, 12]), size=4)
    assert fixed.shape == (16, 32)
    assert torch.isfinite(fixed).all()


def test_fusion_gate_metrics_report_effective_gate_size() -> None:
    predictor = QwenVJEPAFusionPredictor(
        latent_dim=8,
        semantic_dim=8,
        hidden_dim=8,
        action_dim=8,
        layers=2,
        heads=2,
    )
    assert fusion_gate_metrics(predictor)["fusion_gate_effective_l2"] == 0.0
    with torch.no_grad():
        predictor.cross_gates[0, 0] = 0.5
    metrics = fusion_gate_metrics(predictor)
    assert metrics["fusion_gate_effective_l2"] > 0.0
    assert len(metrics["fusion_gate_effective_l2_by_layer"]) == 2
    assert fusion_gate_metrics(torch.nn.Linear(2, 2)) == {}


def test_frozen_feature_cache_validator_rejects_wrong_count() -> None:
    config = VJEPA2PilotConfig(max_train_transitions=8, max_validation_transitions=8)
    try:
        validate_encoded_feature_cache(
            {"cache_key": "key", "train": [_bundle("one", "jira")], "validation": []},
            "key",
            config,
        )
    except ValueError as error:
        assert "count" in str(error).lower()
    else:
        raise AssertionError("An incomplete frozen feature cache was accepted")


def test_frozen_feature_cache_validator_rejects_wrong_bundle_ids() -> None:
    config = VJEPA2PilotConfig(max_train_transitions=4, max_validation_transitions=4)
    cached = {
        "cache_key": "key",
        "train": [_bundle("wrong-train", "jira")],
        "validation": [_bundle("right-validation", "slack")],
    }
    try:
        validate_encoded_feature_cache(
            cached,
            "key",
            config,
            expected_train_bundle_ids=["right-train"],
            expected_validation_bundle_ids=["right-validation"],
        )
    except ValueError as error:
        assert "bundle ids" in str(error).lower()
    else:
        raise AssertionError("A cache with the wrong bundles was accepted")


def test_pilot_success_requires_each_difficult_group() -> None:
    metrics = {
        "four_way_accuracy": 0.45,
        "action_accuracy_drop": 0.15,
        "current_screen_accuracy_drop": 0.08,
        "bundle_bootstrap_ci95": [0.35, 0.55],
        "by_app": {
            "jira": {"four_way_accuracy": 0.4},
            "slack": {"four_way_accuracy": 0.5},
        },
        "by_action_kind": {"click": {"four_way_accuracy": 0.4}},
        "by_changed_pixel_bucket": {"large_change": {"four_way_accuracy": 0.2}},
    }
    passed, checks = pilot_success(metrics)
    assert not passed
    assert not checks["large_change_at_least_30_percent"]


def test_artifact_validator_rejects_incomplete_run(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    try:
        validate_vjepa2_pilot_artifacts(tmp_path)
    except ValueError as error:
        assert "missing" in str(error).lower()
    else:
        raise AssertionError("Incomplete V-JEPA 2 artifacts were accepted")
