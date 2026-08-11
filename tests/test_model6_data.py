from cua_jepa.model6_data import (
    build_candidate_record,
    candidate_generation_metrics,
    policy_action_to_dynamics,
    split_scorer_training_records,
)


def _record(index: int, split: str = "train", task: str | None = None) -> dict:
    return {
        "example_id": f"example-{index}",
        "split": split,
        "task_id": task or f"task-{index}",
        "instruction": "Click the center",
        "history": [],
        "system": "Ubuntu",
        "domain": "test",
        "stored_image": f"images/{index}.webp",
        "action": {"action": "click", "x": 0.5, "y": 0.5},
        "target": '{"action":"click","x":0.5,"y":0.5}',
    }


def test_model6_candidate_builder_deduplicates_and_labels_actions() -> None:
    correct = '{"action":"click","x":0.5,"y":0.5}'
    wrong = '{"action":"scroll","amount":-5}'
    result = build_candidate_record(
        _record(1),
        correct,
        [correct, wrong, "not json"],
        100,
        100,
        inject_training_oracle=False,
    )
    assert result["candidate_count"] == 2
    assert result["duplicate_candidates"] == 1
    assert result["parse_failures"] == 1
    assert result["candidates"][0]["action_score"] == 1.0


def test_model6_policy_action_conversion_marks_trained_action_kinds() -> None:
    click, click_supported = policy_action_to_dynamics(
        {"action": "click", "x": 0.25, "y": 0.75}
    )
    hotkey, hotkey_supported = policy_action_to_dynamics(
        {"action": "hotkey", "keys": ["ctrl", "s"]}
    )
    assert click == {"kind": "click", "x_normalized": 0.25, "y_normalized": 0.75}
    assert click_supported is True
    assert hotkey == {"kind": "press", "text": "ctrl+s"}
    assert hotkey_supported is False


def test_model6_training_oracle_replaces_a_bad_candidate() -> None:
    wrong = '{"action":"scroll","amount":-5}'
    result = build_candidate_record(
        _record(1),
        wrong,
        [wrong, wrong, wrong],
        100,
        100,
        inject_training_oracle=True,
    )
    assert result["oracle_injected"] is True
    assert any(item["source"] == "training_oracle" for item in result["candidates"])


def test_model6_validation_never_injects_an_oracle() -> None:
    wrong = '{"action":"scroll","amount":-5}'
    result = build_candidate_record(
        _record(1, split="validation"),
        wrong,
        [],
        100,
        100,
        inject_training_oracle=False,
    )
    assert result["oracle_injected"] is False
    assert result["candidates"][0]["action_score"] == 0.0


def test_model6_scorer_split_is_exact_and_task_disjoint() -> None:
    records = [_record(index) for index in range(20)]
    fit, development = split_scorer_training_records(
        records, development_examples=5, seed=7
    )
    assert len(fit) == 15
    assert len(development) == 5
    assert {item["task_id"] for item in fit}.isdisjoint(
        {item["task_id"] for item in development}
    )


def test_model6_candidate_metrics_report_the_reranking_ceiling() -> None:
    first = build_candidate_record(
        _record(1),
        '{"action":"scroll","amount":-5}',
        ['{"action":"click","x":0.5,"y":0.5}'],
        100,
        100,
        False,
    )
    second = build_candidate_record(
        _record(2),
        '{"action":"click","x":0.5,"y":0.5}',
        [],
        100,
        100,
        False,
    )
    metrics = candidate_generation_metrics([first, second])
    assert metrics["greedy_exact_success"] == 0.5
    assert metrics["candidate_recall_at_4"] == 1.0
