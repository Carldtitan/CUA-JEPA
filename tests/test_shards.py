from cua_jepa.config import load_config
from cua_jepa.shards import build_full_specs, build_pilot_specs, deterministic_index


def test_full_shards_cover_exact_target() -> None:
    config = load_config("configs/dataset_v1.json")
    specs = build_full_specs(config)
    assert sum(spec.bundle_count for spec in specs) == 9_000
    assert len(specs) == 360


def test_pilot_covers_all_apps_and_100_bundles() -> None:
    config = load_config("configs/dataset_v1.json")
    specs = build_pilot_specs(config)
    assert len(specs) == 12
    assert sum(spec.bundle_count for spec in specs) == 100
    assert {spec.app for spec in specs} == {
        app for split in config.splits.values() for app in split.apps
    }


def test_catalog_selection_is_deterministic() -> None:
    assert deterministic_index(42, "gmail_mock", 7, 13) == deterministic_index(
        42, "gmail_mock", 7, 13
    )

