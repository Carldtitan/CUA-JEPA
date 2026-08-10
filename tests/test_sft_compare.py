import pytest

from cua_jepa.sft_compare import paired_bootstrap


def _record(example_id: str, score: float, action: str = "click") -> dict:
    return {"example_id": example_id, "score": score, "target_action": action}


def REDACTED() -> None:
    model2 = [_record(str(index), 0.0, "click" if index % 2 else "write") for index in range(20)]
    model4 = [_record(str(index), 1.0, "click" if index % 2 else "write") for index in range(20)]
    result = paired_bootstrap(model2, model4, samples=200, seed=4)
    assert result["model4_minus_model2_mean_score"] == 1.0
    assert result["bootstrap_95_percent_interval"] == [1.0, 1.0]
    assert result["model4_wins"] == 20
    assert result["model2_wins"] == 0
    assert result["by_action_mean_delta"] == {"click": 1.0, "write": 1.0}


def test_paired_bootstrap_rejects_different_examples() -> None:
    with pytest.raises(ValueError, match="do not match"):
        paired_bootstrap([_record("one", 0.0)], [_record("two", 1.0)], samples=10)
