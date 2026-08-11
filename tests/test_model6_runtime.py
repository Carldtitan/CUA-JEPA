from pathlib import Path

import torch

from cua_jepa.jepa_model import ActionConditionedPredictor, ActionEncoder
from cua_jepa.model6 import InverseDynamicsHead, ResidualLatentAdapter
from cua_jepa.model6_runtime import load_model6_dynamics, predict_candidate_futures


def _checkpoint(path: Path) -> None:
    action_encoder = ActionEncoder(8)
    predictor = ActionConditionedPredictor(1024, 8, 8, 1, 2)
    online = ResidualLatentAdapter(1024, 8)
    target = ResidualLatentAdapter(1024, 8)
    inverse = InverseDynamicsHead(1024, 8, 8)
    torch.save(
        {
            "config": {
                "architecture_name": "model6",
                "predictor_dim": 8,
                "predictor_layers": 1,
                "predictor_heads": 2,
                "model6_adapter_dim": 8,
                "model6_inverse_hidden_dim": 8,
            },
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "online_adapter": online.state_dict(),
            "target_adapter": target.state_dict(),
            "inverse_head": inverse.state_dict(),
        },
        path,
    )


def test_model6_runtime_loads_frozen_heads(tmp_path: Path) -> None:
    path = tmp_path / "model6.pt"
    _checkpoint(path)
    modules = load_model6_dynamics(path, torch.device("cpu"))
    for name in (
        "action_encoder",
        "predictor",
        "online_adapter",
        "target_adapter",
        "inverse_head",
    ):
        assert not any(parameter.requires_grad for parameter in modules[name].parameters())


def test_model6_runtime_batches_candidate_futures(tmp_path: Path) -> None:
    path = tmp_path / "model6.pt"
    _checkpoint(path)
    modules = load_model6_dynamics(path, torch.device("cpu"))
    futures, actions = predict_candidate_futures(
        torch.randn(256, 1024),
        [
            {"kind": "click", "x_normalized": 0.5, "y_normalized": 0.5},
            {"kind": "scroll", "delta_y": -5.0},
        ],
        1280,
        720,
        modules,
        torch.device("cpu"),
    )
    assert futures.shape == (2, 256, 1024)
    assert actions.shape == (2, 8)
    assert not futures.is_inference()
    assert not actions.is_inference()
