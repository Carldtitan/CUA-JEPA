from cua_jepa.train_model6_scorer import binary_auc, summarize_selection


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
