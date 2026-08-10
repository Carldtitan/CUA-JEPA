from pathlib import Path

import torch

from cua_jepa.train_vjepa2 import (
    EncodedBundle,
    balanced_training_epoch,
    bundle_bootstrap_ci95,
    pilot_success,
    select_balanced_encoded_bundles,
    validate_vjepa2_pilot_artifacts,
)


def _bundle(bundle_id: str, app: str) -> EncodedBundle:
    return EncodedBundle(
        bundle_id=bundle_id,
        app=app,
        split="validation",
        actions=[{"kind": "click"}] * 4,
        changed_pixel_fractions=[0.3] * 4,
        current=torch.zeros(256, 1024),
        targets=torch.zeros(4, 256, 1024),
        target_weights=torch.ones(4, 256),
    )


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


def test_pilot_success_requires_each_difficult_group() -> None:
    metrics = {
        "four_way_accuracy": 0.45,
        "action_accuracy_drop": 0.15,
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
