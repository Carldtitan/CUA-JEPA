from pathlib import Path

from cua_jepa.jepa_data import (
    TransitionSample,
    balanced_bundle_groups,
    group_by_bundle,
    load_transition_tar,
)


ROOT = Path(__file__).parents[1]
TAR_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "clean-20260808-v7"
    / "full"
    / "train"
    / "github_mock"
    / "train-github_mock-shard-0000.tar"
)


def test_load_transition_tar_preserves_same_state_branches() -> None:
    samples = load_transition_tar(TAR_PATH, limit=8)
    assert len(samples) == 8
    groups = group_by_bundle(samples)
    assert len(groups) == 2
    for branches in groups.values():
        assert [sample.branch_index for sample in branches] == [0, 1, 2, 3]
        assert len({sample.current_webp for sample in branches}) == 1
        assert len({sample.future_webp for sample in branches}) == 4
        assert {sample.action["kind"] for sample in branches} <= {"click", "scroll", "type"}


def _sample(bundle: str, app: str, branch: int) -> TransitionSample:
    return TransitionSample(
        bundle_id=bundle,
        app=app,
        split="validation",
        branch_index=branch,
        action={"kind": "click"},
        current_webp=b"current",
        future_webp=b"future",
        changed_pixel_fraction=0.1,
    )


def test_balanced_bundle_groups_round_robins_applications() -> None:
    samples = [
        _sample(bundle, app, branch)
        for app in ("jira", "slack")
        for bundle in (f"{app}-0", f"{app}-1", f"{app}-2")
        for branch in range(4)
    ]
    selected = balanced_bundle_groups(samples, max_bundles=5)
    assert [branches[0].app for branches in selected] == [
        "jira",
        "slack",
        "jira",
        "slack",
        "jira",
    ]


def test_balanced_bundle_groups_returns_all_when_unlimited() -> None:
    samples = [_sample("jira-0", "jira", branch) for branch in range(4)] + [
        _sample("slack-0", "slack", branch) for branch in range(4)
    ]
    assert len(balanced_bundle_groups(samples, max_bundles=0)) == 2
