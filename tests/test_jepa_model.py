import torch
from PIL import Image, ImageDraw

from cua_jepa.jepa_model import (
    ActionConditionedPredictor,
    ActionEncoder,
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
    loss = latent_prediction_loss(predictor(current, action), target)
    loss.backward()
    assert current.grad is not None
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_change_weights_focus_on_modified_tokens() -> None:
    current = Image.new("RGB", (128, 64), "white")
    future = current.copy()
    ImageDraw.Draw(future).rectangle((0, 0, 31, 31), fill="black")
    weights = change_patch_weights(
        current,
        future,
        grid_thw=torch.tensor([1, 4, 8]),
        device=torch.device("cpu"),
        changed_weight=4.0,
    )
    assert weights.shape == (8,)
    assert weights.max() > 1.0
    assert weights.min() == 1.0
