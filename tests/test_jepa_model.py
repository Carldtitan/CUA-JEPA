import torch
from PIL import Image, ImageDraw

from cua_jepa.jepa_model import (
    ActionConditionedPredictor,
    ActionEncoder,
    ActionTokenConditionedPredictor,
    TiledActionConditionedPredictor,
    IndependentTiledActionConditionedPredictor,
    VisualGatedActionConditionedPredictor,
    QwenVJEPAFusionPredictor,
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


def test_action_encoder_distinguishes_actions_and_backpropagates() -> None:
    encoder = ActionEncoder(output_dim=32, byte_dim=16, text_dim=16)
    actions = [
        {"kind": "click", "x_normalized": 0.2, "y_normalized": 0.4},
        {"kind": "type", "x_normalized": 0.2, "y_normalized": 0.4, "text": "Maya"},
    ]
    values = encoder(*actions_to_tensors(actions, torch.device("cpu"), max_text_bytes=16))
    assert values.shape == (2, 32)
    assert not torch.allclose(values[0], values[1])
    values.sum().backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_predictor_and_latent_loss_backpropagate() -> None:
    predictor = ActionConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=1, heads=4
    )
    current = torch.randn(6, 32, requires_grad=True)
    target = torch.randn(6, 32)
    action = torch.randn(1, 32)
    spatial_action = torch.randn(1, 6, 3)
    prediction = predictor(current, action, spatial_action)
    loss = latent_prediction_loss(prediction, target)
    loss.backward()
    assert prediction.shape == target.shape
    assert current.grad is not None
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_action_conditioning_changes_predictions() -> None:
    torch.manual_seed(3)
    predictor = ActionConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=2, heads=4
    )
    current = torch.randn(6, 32)
    actions = torch.stack((torch.zeros(32), torch.ones(32)))
    spatial = torch.zeros(2, 6, 3)
    predictions = predictor(current, actions, spatial)
    assert predictions.shape == (2, 6, 32)
    assert not torch.allclose(predictions[0], predictions[1])


def test_visual_gated_predictor_requires_visual_content() -> None:
    torch.manual_seed(13)
    predictor = VisualGatedActionConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=2, heads=4
    )
    actions = torch.stack((torch.zeros(32), torch.ones(32)))
    spatial = torch.randn(2, 6, 3)
    zero_predictions = predictor(torch.zeros(6, 32), actions, spatial)
    assert torch.count_nonzero(zero_predictions) == 0
    visual_predictions = predictor(torch.randn(6, 32), actions, spatial)
    assert visual_predictions.shape == (2, 6, 32)
    assert not torch.allclose(visual_predictions[0], visual_predictions[1])


def test_qwen_vjepa_fusion_starts_as_base_predictor_and_gets_gate_gradients() -> None:
    torch.manual_seed(17)
    predictor = QwenVJEPAFusionPredictor(
        latent_dim=32,
        semantic_dim=24,
        hidden_dim=32,
        action_dim=32,
        layers=2,
        heads=4,
    )
    current = torch.randn(6, 32)
    actions = torch.randn(2, 32)
    spatial = torch.zeros(2, 6, 3)
    semantic = torch.randn(8, 24)
    first = predictor(current, actions, spatial, semantic)
    changed_semantic = predictor(current, actions, spatial, semantic + 10.0)
    assert torch.allclose(first, changed_semantic)
    first.square().mean().backward()
    assert predictor.cross_gates.grad is not None
    assert torch.count_nonzero(predictor.cross_gates.grad) > 0


def test_action_token_predictor_keeps_visual_shape_and_uses_action() -> None:
    torch.manual_seed(4)
    predictor = ActionTokenConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=2, heads=4
    )
    current = torch.randn(6, 32)
    actions = torch.stack((torch.zeros(32), torch.ones(32)))
    spatial = torch.zeros(2, 6, 3)
    predictions = predictor(current, actions, spatial)
    assert predictions.shape == (2, 6, 32)
    assert not torch.allclose(predictions[0], predictions[1])


def test_click_coordinates_bind_to_visual_tokens() -> None:
    left, right = action_spatial_features(
        [
            {"kind": "click", "x_normalized": 0.1, "y_normalized": 0.5},
            {"kind": "click", "x_normalized": 0.9, "y_normalized": 0.5},
        ],
        grid_thw=torch.tensor([1, 4, 8]),
        device=torch.device("cpu"),
    )
    assert left.shape == (8, 3)
    assert left[:, 0].argmax() != right[:, 0].argmax()


def test_inactive_tile_has_no_pointer_heatmap() -> None:
    inactive = action_spatial_features(
        [{"kind": "click", "x_normalized": 0.5, "y_normalized": 0.5, "spatial_active": False}],
        grid_thw=torch.tensor([1, 4, 8]),
        device=torch.device("cpu"),
    )
    assert torch.count_nonzero(inactive) == 0


def test_tiled_predictor_uses_complete_screen_positions() -> None:
    torch.manual_seed(8)
    predictor = TiledActionConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=1, heads=4
    )
    current = torch.randn(8, 32)
    action = torch.randn(2, 32)
    spatial = torch.zeros(2, 8, 3)
    positions = torch.rand(8, 3)
    prediction = predictor(current, action, spatial, positions)
    prediction.sum().backward()
    assert prediction.shape == (2, 8, 32)
    assert predictor.screen_position_projection.weight.grad is not None


def test_independent_tiled_predictor_keeps_two_view_shape() -> None:
    torch.manual_seed(9)
    predictor = IndependentTiledActionConditionedPredictor(
        latent_dim=32, hidden_dim=32, action_dim=32, layers=1, heads=4
    )
    current = torch.randn(8, 32)
    actions = torch.randn(2, 32)
    spatial = torch.zeros(2, 8, 3)
    positions = torch.rand(8, 3)
    prediction = predictor(current, actions, spatial, positions)
    prediction.square().mean().backward()
    assert prediction.shape == (2, 8, 32)
    assert predictor.blocks[0].attention.in_proj_weight.grad is not None


def test_action_separation_prefers_matched_futures() -> None:
    targets = torch.eye(4).reshape(4, 1, 4)
    weights = torch.ones(4, 1)
    matched = action_separation_loss(targets, targets, weights)
    shuffled = action_separation_loss(targets.roll(1, dims=0), targets, weights)
    assert matched < shuffled


def test_action_separation_uses_one_shared_candidate_mask() -> None:
    torch.manual_seed(11)
    predictions = torch.randn(4, 2, 8)
    targets = torch.randn(4, 2, 8)
    candidate_specific = torch.tensor([[1.0, 0.05], [0.05, 1.0], [0.25, 0.75], [0.75, 0.25]])
    union = candidate_specific.max(dim=0).values.expand_as(candidate_specific)
    first = action_separation_loss(predictions, targets, candidate_specific)
    second = action_separation_loss(predictions, targets, union)
    assert torch.allclose(first, second)


def test_delta_prediction_uses_change_not_complete_future() -> None:
    current = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    future = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    target_delta = latent_delta(current, future)
    matched = latent_delta_loss(target_delta, target_delta)
    copied_screen = latent_delta_loss(torch.zeros_like(target_delta), target_delta)
    reconstructed = reconstruct_future_latent(current, target_delta)
    assert matched < copied_screen
    assert latent_prediction_loss(reconstructed, future) < 1e-6


def test_bundle_regularization_penalizes_identical_action_predictions() -> None:
    targets = torch.eye(4).reshape(4, 1, 4)
    weights = torch.ones(4, 1)
    matched = bundle_anti_collapse_losses(targets, targets, weights)
    collapsed_predictions = targets[:1].expand_as(targets)
    collapsed = bundle_anti_collapse_losses(collapsed_predictions, targets, weights)
    assert matched["variance"] < collapsed["variance"]
    assert matched["covariance"] < collapsed["covariance"]
    assert matched["relation"] < collapsed["relation"]


def test_change_weights_focus_on_modified_tokens() -> None:
    current = Image.new("RGB", (128, 64), "white")
    future = current.copy()
    ImageDraw.Draw(future).rectangle((0, 0, 31, 31), fill="black")
    weights = change_patch_weights(
        current,
        future,
        grid_thw=torch.tensor([1, 4, 8]),
        device=torch.device("cpu"),
        changed_weight=1.0,
        unchanged_weight=0.05,
    )
    assert weights.shape == (8,)
    assert weights.max() == 1.0
    assert weights.min() == 0.05
