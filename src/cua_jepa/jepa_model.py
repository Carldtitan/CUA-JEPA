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


def action_spatial_features(
    actions: Sequence[dict[str, Any]],
    grid_thw: torch.Tensor,
    device: torch.device,
    sigma: float = 0.08,
) -> torch.Tensor:
    """Bind pointer actions to Qwen's merged visual-token grid."""

    t, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    pooled_h = grid_h // 2
    pooled_w = grid_w // 2
    y_values = (torch.arange(pooled_h, device=device, dtype=torch.float32) + 0.5) / pooled_h
    x_values = (torch.arange(pooled_w, device=device, dtype=torch.float32) + 0.5) / pooled_w
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    features: list[torch.Tensor] = []
    for action in actions:
        kind = str(action.get("kind", ""))
        if kind not in {"click", "type"}:
            spatial = torch.zeros((pooled_h, pooled_w, 3), device=device)
        else:
            action_x = float(action.get("x_normalized", 0.0) or 0.0)
            action_y = float(action.get("y_normalized", 0.0) or 0.0)
            x_offset = x_grid - action_x
            y_offset = y_grid - action_y
            heat = torch.exp(-(x_offset.square() + y_offset.square()) / (2.0 * sigma**2))
            spatial = torch.stack((heat, heat * x_offset, heat * y_offset), dim=-1)
        flattened = spatial.reshape(-1, 3)
        if t > 1:
            flattened = flattened.repeat(t, 1)
        features.append(flattened)
    return torch.stack(features, dim=0)


class ActionConditionedBlock(nn.Module):
    """A transformer block with action conditioning in both normalization layers."""

    def __init__(self, hidden_dim: int, action_dim: int, heads: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.action_modulation = nn.Linear(action_dim, hidden_dim * 4)

    @staticmethod
    def _modulate(
        values: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        return values * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, hidden: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        attention_scale, attention_shift, mlp_scale, mlp_shift = self.action_modulation(
            action
        ).chunk(4, dim=-1)
        normalized = self._modulate(
            self.norm_attention(hidden), attention_scale, attention_shift
        )
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden = hidden + attended
        normalized = self._modulate(self.norm_mlp(hidden), mlp_scale, mlp_shift)
        return hidden + self.mlp(normalized)


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
        self.spatial_action_projection = nn.Linear(3, hidden_dim, bias=False)
        self.blocks = nn.ModuleList(
            ActionConditionedBlock(hidden_dim, action_dim, heads) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        current_tokens: torch.Tensor,
        action_embedding: torch.Tensor,
        spatial_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = current_tokens.ndim == 2
        if squeeze:
            current_tokens = current_tokens.unsqueeze(0)
        if current_tokens.shape[0] == 1 and action_embedding.shape[0] > 1:
            current_tokens = current_tokens.expand(action_embedding.shape[0], -1, -1)
        hidden = self.input_projection(self.input_norm(current_tokens.float()))
        if spatial_action is not None:
            if spatial_action.shape[:2] != hidden.shape[:2]:
                raise ValueError(
                    f"Spatial action shape mismatch: {spatial_action.shape} vs {hidden.shape}"
                )
            hidden = hidden + self.spatial_action_projection(spatial_action.float())
        for block in self.blocks:
            hidden = block(hidden, action_embedding.float())
        # There is no direct current-to-future copy path.
        prediction = self.output_projection(self.output_norm(hidden))
        return prediction.squeeze(0) if squeeze else prediction


def change_patch_weights(
    current: Image.Image,
    future: Image.Image,
    grid_thw: torch.Tensor,
    device: torch.device,
    changed_weight: float = 1.0,
    unchanged_weight: float = 0.05,
    changed_token_threshold: float = 0.01,
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
    values = (values >= changed_token_threshold).float()
    if t > 1:
        values = values.repeat(t)
    weights = unchanged_weight + (changed_weight - unchanged_weight) * values
    return weights.to(device=device)


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
