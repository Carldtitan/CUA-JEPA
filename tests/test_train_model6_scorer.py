from cua_jepa.train_model6_scorer import (
    _latency_summary,
    binary_auc,
    summarize_qwen_greedy,
    summarize_selection,
)


def test_model6_binary_auc_handles_correct_order_and_ties() -> None:
    assert binary_auc([1, 0], [1.0, 0.0]) == 1.0
    assert binary_auc([1, 0], [0.0, 0.0]) == 0.5
    assert binary_auc([1, 1], [1.0, 0.0]) is None


def test_model6_selection_summary_keeps_supported_subset() -> None:
    rows = [
        {
            "selected_exact": True,
            "selected_action_score": 1.0,
            "oracle_action_score": 1.0,
            "candidate_labels": [1.0, 0.0],
            "candidate_logits": [2.0, -1.0],
            "target_dynamics_supported": True,
        },
        {
            "selected_exact": False,
            "selected_action_score": 0.0,
            "oracle_action_score": 1.0,
            "candidate_labels": [0.0, 1.0],
            "candidate_logits": [2.0, -1.0],
            "target_dynamics_supported": False,
        },
    ]
    summary = summarize_selection(rows)
    assert summary["exact_success_rate"] == 0.5
    assert summary["mean_regret"] == 0.5
    assert summary["supported_action_subset"]["exact_success_rate"] == 1.0


def test_model6_qwen_summary_uses_the_greedy_candidate() -> None:
    records = [
        {
            "target_action": {"action": "click", "x": 0.5, "y": 0.5},
            "candidates": [
                {"source": "greedy", "action_score": 0.0},
                {"source": "sampled", "action_score": 1.0},
            ],
        }
    ]
    result = summarize_qwen_greedy(records)
    assert result["exact_success_rate"] == 0.0
    assert result["oracle_exact_recall"] == 1.0
    assert result["mean_regret"] == 1.0


def test_model6_latency_summary_reports_milliseconds() -> None:
    result = _latency_summary([0.001, 0.003, 0.002])
    assert result == {
        "measurements": 3,
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 3.0,
    }
