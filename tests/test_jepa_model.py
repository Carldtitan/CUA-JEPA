import torch
from PIL import Image, ImageDraw

from cua_jepa.jepa_model import (
    ActionConditionedPredictor,
    ActionEncoder,
    action_spatial_features,
    actions_to_tensors,
    change_patch_weights,
    latent_prediction_loss,
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
