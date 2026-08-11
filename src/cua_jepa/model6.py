"""Model 6 heads for frozen Qwen and frozen V-JEPA 2."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Model6ArchitectureConfig:
    latent_dim: int = 1024
    adapter_dim: int = 256
    action_dim: int = 384
    inverse_hidden_dim: int = 384
    scorer_hidden_dim: int = 256
    goal_dim: int = 384
    ema_decay: float = 0.996
    inverse_loss_weight: float = 0.25
    inverse_temperature: float = 0.1


class ResidualLatentAdapter(nn.Module):
    """Adapt frozen visual tokens while preserving the initial representation."""

    def __init__(self, latent_dim: int = 1024, adapter_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.down = nn.Linear(latent_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, latent_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        update = self.up(F.gelu(self.down(self.norm(tokens.float()))))
        return tokens.float() + update


def make_ema_adapter(adapter: ResidualLatentAdapter) -> ResidualLatentAdapter:
    target = copy.deepcopy(adapter)
    target.requires_grad_(False).eval()
    return target


@torch.no_grad()
def update_ema_adapter(
    target: nn.Module,
    online: nn.Module,
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in the range [0, 1)")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("Online and target adapter parameters do not match")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(decay).add_(online_parameters[name], alpha=1.0 - decay)
    target_buffers = dict(target.named_buffers())
    for name, target_buffer in target_buffers.items():
        target_buffer.copy_(dict(online.named_buffers())[name])


class InverseDynamicsHead(nn.Module):
    """Recover which candidate action caused an observed future latent."""

    def __init__(
        self,
        latent_dim: int = 1024,
        hidden_dim: int = 384,
        action_dim: int = 384,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(latent_dim * 3),
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
        )
        self.token_attention = nn.Linear(hidden_dim, 1)
        self.query = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        current: torch.Tensor,
        futures: torch.Tensor,
        candidate_actions: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        if futures.ndim != 3 or candidate_actions.ndim != 2:
            raise ValueError("Inverse dynamics expects batched futures and actions")
        if current.ndim == 2:
            current = current.unsqueeze(0)
        if current.shape[0] == 1 and futures.shape[0] > 1:
            current = current.expand(futures.shape[0], -1, -1)
        if current.shape != futures.shape:
            raise ValueError("Current and future latent shapes do not match")
        if futures.shape[0] != candidate_actions.shape[0]:
            raise ValueError("Future and action candidate counts do not match")
        delta = futures.float() - current.float()
        tokens = self.token_projection(
            torch.cat((current.float(), futures.float(), delta), dim=-1)
        )
        attention = torch.softmax(self.token_attention(tokens).squeeze(-1), dim=-1)
        pooled = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
        query = F.normalize(self.query(pooled), dim=-1)
        actions = F.normalize(candidate_actions.float(), dim=-1)
        return query @ actions.transpose(0, 1) / max(temperature, 1e-6)


def inverse_dynamics_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Inverse dynamics logits must form a square candidate matrix")
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


class GoalConditionedFutureScorer(nn.Module):
    """Score a predicted future without reading the candidate action directly."""

    def __init__(
        self,
        latent_dim: int = 1024,
        goal_dim: int = 384,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.future_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.goal_projection = nn.Sequential(
            nn.LayerNorm(goal_dim),
            nn.Linear(goal_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, predicted_futures: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if predicted_futures.ndim != 3:
            raise ValueError("Predicted futures must have shape [candidates, tokens, latent]")
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[0] == 1 and predicted_futures.shape[0] > 1:
            goal = goal.expand(predicted_futures.shape[0], -1)
        if goal.shape[0] != predicted_futures.shape[0]:
            raise ValueError("Goal and future candidate counts do not match")
        future = self.future_projection(predicted_futures.float())
        goal_hidden = self.goal_projection(goal.float())
        attention_logits = torch.sum(
            future * goal_hidden.unsqueeze(1), dim=-1
        ) / math.sqrt(future.shape[-1])
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(future * attention.unsqueeze(-1), dim=1)
        features = torch.cat((pooled, goal_hidden, pooled * goal_hidden), dim=-1)
        return self.output(features).squeeze(-1)


class ActionOnlyScorer(nn.Module):
    """Control scorer that does not receive a predicted future."""

    def __init__(
        self,
        action_dim: int = 384,
        goal_dim: int = 384,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.goal_projection = nn.Linear(goal_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actions: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[0] == 1 and actions.shape[0] > 1:
            goal = goal.expand(actions.shape[0], -1)
        action_hidden = F.gelu(self.action_projection(actions.float()))
        goal_hidden = F.gelu(self.goal_projection(goal.float()))
        features = torch.cat(
            (action_hidden, goal_hidden, action_hidden * goal_hidden), dim=-1
        )
        return self.output(features).squeeze(-1)


def scorer_training_loss(
    logits: torch.Tensor,
    action_scores: torch.Tensor,
    rejection_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.ndim != 1 or action_scores.shape != logits.shape:
        raise ValueError("Scorer logits and action labels must be one-dimensional and equal")
    best_index = torch.argmax(action_scores)
    ranking = F.cross_entropy(logits.unsqueeze(0), best_index.unsqueeze(0))
    good = (action_scores > 0.0).to(dtype=logits.dtype)
    rejection = F.binary_cross_entropy_with_logits(logits, good)
    total = ranking + rejection_weight * rejection
    return total, {"ranking": ranking, "rejection": rejection}
