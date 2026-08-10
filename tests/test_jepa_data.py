from pathlib import Path

from cua_jepa.jepa_data import group_by_bundle, load_transition_tar


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
