from pathlib import Path

from cua_jepa.config import load_config


def test_dataset_target_is_36k_transitions() -> None:
    config = load_config(Path("configs/dataset_v1.json"))
    assert config.total_bundles == 9_000
    assert config.total_transitions == 36_000
    assert config.actions_per_bundle == 4
    assert config.quality.maximum_action_candidates_per_bundle >= config.actions_per_bundle
    assert config.quality.maximum_branch_reset_attempts >= 2
    assert (
        config.quality.maximum_reset_changed_pixel_fraction
        * config.viewport.width
        * config.viewport.height
        <= 24
    )


def test_app_splits_do_not_overlap() -> None:
    config = load_config(Path("configs/dataset_v1.json"))
    app_sets = [set(split.apps) for split in config.splits.values()]
    assert app_sets[0].isdisjoint(app_sets[1])
    assert app_sets[0].isdisjoint(app_sets[2])
    assert app_sets[1].isdisjoint(app_sets[2])
