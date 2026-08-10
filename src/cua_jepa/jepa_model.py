from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageChops
from torch import nn
from torch.nn import functional as F


ACTION_KINDS = {"click": 0, "type": 1, "scroll": 2, "press": 3}


def actions_to_tensors(
    actions: Sequence[dict[str, Any]], device: torch.device, max_text_bytes: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kinds: list[int] = []
    numeric: list[list[float]] = []
    texts: list[list[int]] = []
    lengths: list[int] = []
    for action in actions:
        text_bytes = str(action.get("text", "")).encode("utf-8")[:max_text_bytes]
        encoded = [value + 1 for value in text_bytes]
        lengths.append(max(1, len(encoded)))
        texts.append(encoded + [0] * (max_text_bytes - len(encoded)))
        kinds.append(ACTION_KINDS.get(str(action.get("kind", "")), 4))
        numeric.append(
            [
                float(action.get("x_normalized", 0.0) or 0.0),
                float(action.get("y_normalized", 0.0) or 0.0),
                float(np.clip(float(action.get("delta_x", 0.0) or 0.0) / 1280.0, -1, 1)),
                float(np.clip(float(action.get("delta_y", 0.0) or 0.0) / 720.0, -1, 1)),
                min(len(text_bytes) / max_text_bytes, 1.0),
                1.0 if text_bytes else 0.0,
            ]
        )
    return (
        torch.tensor(kinds, dtype=torch.long, device=device),
        torch.tensor(numeric, dtype=torch.float32, device=device),
        torch.tensor(texts, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device="cpu"),
    )


class ActionEncoder(nn.Module):
    def __init__(self, output_dim: int = 384, byte_dim: int = 64, text_dim: int = 128) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(5, 32)
        self.byte_embedding = nn.Embedding(257, byte_dim, padding_idx=0)
        self.text_encoder = nn.GRU(byte_dim, text_dim, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(32 + text_dim + 6, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        kinds: torch.Tensor,
        numeric: torch.Tensor,
        text_bytes: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.byte_embedding(text_bytes)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.text_encoder(packed)
        features = torch.cat((self.kind_embedding(kinds), numeric, hidden[-1]), dim=-1)
        return self.output(features)


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2048,
        hidden_dim: int = 384,
        action_dim: int = 384,
        layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(latent_dim)
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, current_tokens: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        squeeze = current_tokens.ndim == 2
        if squeeze:
            current_tokens = current_tokens.unsqueeze(0)
        hidden = self.input_projection(self.input_norm(current_tokens.float()))
        hidden = hidden + self.action_projection(action_embedding.float()).unsqueeze(1)
        delta = self.output_projection(self.transformer(hidden))
        prediction = current_tokens.float() + self.delta_scale * delta
        return prediction.squeeze(0) if squeeze else prediction


def change_patch_weights(
    current: Image.Image,
    future: Image.Image,
    grid_thw: torch.Tensor,
    device: torch.device,
    changed_weight: float = 4.0,
    threshold: int = 12,
) -> torch.Tensor:
    """Create target-only loss weights aligned to Qwen's merged token grid."""

    t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    pooled_h = grid_h // 2
    pooled_w = grid_w // 2
    difference = np.asarray(ImageChops.difference(current, future), dtype=np.uint8)
    changed = (difference.max(axis=2) > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(changed).resize(
        (pooled_w, pooled_h), resample=Image.Resampling.BOX
    )
    values = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0).flatten()
    if t > 1:
        values = values.repeat(t)
    return (1.0 + changed_weight * values).to(device=device)


def latent_prediction_loss(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    per_token = 1.0 - (prediction * target).sum(dim=-1)
    if weights is None:
        return per_token.mean()
    if per_token.numel() != weights.numel():
        raise ValueError(f"Token/weight mismatch: {per_token.numel()} != {weights.numel()}")
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def mean_latent_distance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(latent_prediction_loss(prediction, target).item())
