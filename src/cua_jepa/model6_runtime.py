"""Load Model 6 dynamics heads and predict candidate futures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cua_jepa.jepa_model import (
    ActionConditionedPredictor,
    ActionEncoder,
    action_spatial_features,
    actions_to_tensors,
)
from cua_jepa.model6 import InverseDynamicsHead, ResidualLatentAdapter
from cua_jepa.train_vjepa2 import letterbox_action_coordinates


def load_model6_dynamics(
    weights_path: str | Path,
    device: torch.device,
) -> dict[str, torch.nn.Module | dict[str, Any]]:
    saved = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config.get("architecture_name") != "model6":
        raise ValueError("The dynamics checkpoint is not Model 6")
    action_dim = int(config["predictor_dim"])
    action_encoder = ActionEncoder(action_dim)
    predictor = ActionConditionedPredictor(
        latent_dim=1024,
        hidden_dim=action_dim,
        action_dim=action_dim,
        layers=int(config["predictor_layers"]),
        heads=int(config["predictor_heads"]),
    )
    online_adapter = ResidualLatentAdapter(
        latent_dim=1024,
        adapter_dim=int(config["model6_adapter_dim"]),
    )
    target_adapter = ResidualLatentAdapter(
        latent_dim=1024,
        adapter_dim=int(config["model6_adapter_dim"]),
    )
    inverse_head = InverseDynamicsHead(
        latent_dim=1024,
        hidden_dim=int(config["model6_inverse_hidden_dim"]),
        action_dim=action_dim,
    )
    modules = {
        "action_encoder": action_encoder,
        "predictor": predictor,
        "online_adapter": online_adapter,
        "target_adapter": target_adapter,
        "inverse_head": inverse_head,
    }
    for name, module in modules.items():
        module.load_state_dict(saved[name])
        module.requires_grad_(False).eval().to(device)
    return {"config": config, **modules}


def encode_candidate_actions(
    actions: list[dict[str, Any]],
    action_encoder: ActionEncoder,
    device: torch.device,
) -> torch.Tensor:
    kinds, numeric, text_bytes, text_lengths = actions_to_tensors(actions, device)
    return action_encoder(kinds, numeric, text_bytes, text_lengths)


def predict_candidate_futures(
    current_tokens: torch.Tensor,
    actions: list[dict[str, Any]],
    source_width: int,
    source_height: int,
    modules: dict[str, torch.nn.Module | dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict all candidate futures after one frozen V-JEPA screen encoding."""

    if not actions:
        raise ValueError("Model 6 requires at least one candidate action")
    action_encoder = modules["action_encoder"]
    predictor = modules["predictor"]
    online_adapter = modules["online_adapter"]
    target_adapter = modules["target_adapter"]
    if not isinstance(action_encoder, ActionEncoder):
        raise TypeError("Model 6 action encoder has the wrong type")
    if not isinstance(predictor, ActionConditionedPredictor):
        raise TypeError("Model 6 predictor has the wrong type")
    if not isinstance(online_adapter, ResidualLatentAdapter) or not isinstance(
        target_adapter, ResidualLatentAdapter
    ):
        raise TypeError("Model 6 latent adapters have the wrong type")
    transformed_actions = [
        letterbox_action_coordinates(action, source_width, source_height)
        for action in actions
    ]
    grid = torch.tensor([1, 32, 32])
    with torch.inference_mode():
        embeddings = encode_candidate_actions(transformed_actions, action_encoder, device)
        spatial = action_spatial_features(transformed_actions, grid, device)
        current = current_tokens.to(device=device, dtype=torch.float32)
        online_current = online_adapter(current)
        target_current = target_adapter(current)
        predicted_delta = predictor(online_current, embeddings, spatial)
        predicted_futures = target_current.unsqueeze(0) + predicted_delta
    # Tensors made in inference mode cannot be inputs to trainable scorer
    # layers. Clone after the context so these are ordinary detached tensors.
    return predicted_futures.clone(), embeddings.clone()
