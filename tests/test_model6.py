import torch

from cua_jepa.model6 import (
    ActionOnlyScorer,
    GoalConditionedFutureScorer,
    InverseDynamicsHead,
    ResidualLatentAdapter,
    inverse_dynamics_loss,
    make_ema_adapter,
    scorer_training_loss,
    update_ema_adapter,
)


def test_model6_adapter_starts_as_identity() -> None:
    adapter = ResidualLatentAdapter(latent_dim=8, adapter_dim=4)
    values = torch.randn(2, 3, 8)
    assert torch.equal(adapter(values), values)


def test_model6_target_adapter_is_frozen_and_uses_ema() -> None:
    online = ResidualLatentAdapter(latent_dim=8, adapter_dim=4)
    target = make_ema_adapter(online)
    assert not any(parameter.requires_grad for parameter in target.parameters())
    with torch.no_grad():
        online.up.bias.fill_(2.0)
    update_ema_adapter(target, online, decay=0.5)
    assert torch.allclose(target.up.bias, torch.ones_like(target.up.bias))


def test_model6_inverse_head_returns_one_logit_per_action_pair() -> None:
    head = InverseDynamicsHead(latent_dim=8, hidden_dim=6, action_dim=5)
    current = torch.randn(4, 8)
    futures = torch.randn(4, 4, 8)
    actions = torch.randn(4, 5)
    logits = head(current, futures, actions)
    loss = inverse_dynamics_loss(logits)
    assert logits.shape == (4, 4)
    assert torch.isfinite(loss)


def test_model6_future_scorer_does_not_require_action_input() -> None:
    scorer = GoalConditionedFutureScorer(latent_dim=8, goal_dim=6, hidden_dim=4)
    scores = scorer(torch.randn(4, 5, 8), torch.randn(6))
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_model6_action_only_control_scores_the_same_candidates() -> None:
    scorer = ActionOnlyScorer(action_dim=5, goal_dim=6, hidden_dim=4)
    scores = scorer(torch.randn(4, 5), torch.randn(6))
    assert scores.shape == (4,)


def test_model6_scorer_loss_combines_ranking_and_rejection() -> None:
    logits = torch.tensor([0.0, 1.0, -1.0, 0.2], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 0.5])
    loss, parts = scorer_training_loss(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"ranking", "rejection"}
    assert logits.grad is not None
