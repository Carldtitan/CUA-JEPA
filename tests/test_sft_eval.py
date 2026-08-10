from cua_jepa.sft_eval import parse_action_prediction, score_action, summarize_scores


def test_parses_json_from_plain_or_fenced_response() -> None:
    assert parse_action_prediction('{"action":"click","x":0.2,"y":0.3}') == {
        "action": "click",
        "x": 0.2,
        "y": 0.3,
    }
    assert parse_action_prediction('Result:\n```json\n{"action":"press","keys":["enter"]}\n```') == {
        "action": "press",
        "keys": ["enter"],
    }
    assert parse_action_prediction("not json") is None


def test_scores_coordinates_in_normalized_or_pixel_form() -> None:
    target = {"action": "click", "x": 0.5, "y": 0.5}
    close = score_action({"action": "click", "x": 520, "y": 490}, target, 1000, 1000)
    far = score_action({"action": "click", "x": 0.9, "y": 0.9}, target, 1000, 1000)
    assert close["score"] == 1.0
    assert far["score"] == 0.0


def test_scores_text_keys_and_scroll() -> None:
    write = score_action(
        {"action": "write", "text": "hello"},
        {"action": "write", "text": "hello"},
        100,
        100,
    )
    keys = score_action(
        {"action": "hotkey", "keys": ["CTRL", "s"]},
        {"action": "hotkey", "keys": ["ctrl", "s"]},
        100,
        100,
    )
    scroll = score_action(
        {"action": "scroll", "amount": -2},
        {"action": "scroll", "amount": -4},
        100,
        100,
    )
    assert write["score"] == 1.0
    assert keys["score"] == 1.0
    assert scroll["score"] == 0.5


def test_summary_keeps_action_and_system_breakdowns() -> None:
    summary = summarize_scores(
        [
            {
                "target_action": "click",
                "system": "Windows",
                "parsed": True,
                "type_correct": True,
                "score": 1.0,
            },
            {
                "target_action": "write",
                "system": "Ubuntu",
                "parsed": False,
                "type_correct": False,
                "score": 0.0,
            },
        ]
    )
    assert summary["examples"] == 2
    assert summary["mean_action_score"] == 0.5
    assert summary["macro_action_score"] == 0.5
    assert summary["coordinate"]["examples"] == 1
    assert summary["coordinate"]["hit_rate_at_0_02"] == 0.0
    assert summary["coordinate"]["hit_rate_at_0_05"] == 0.0
    assert summary["coordinate"]["hit_rate_at_0_1"] == 0.0
    assert summary["non_coordinate"]["examples"] == 1
    assert set(summary["by_action"]) == {"click", "write"}
    assert set(summary["by_system"]) == {"Ubuntu", "Windows"}


def test_summary_reports_strict_coordinate_thresholds() -> None:
    summary = summarize_scores(
        [
            {
                "target_action": "click",
                "system": "Ubuntu",
                "parsed": True,
                "type_correct": True,
                "score": 1.0,
                "coordinate_distance": 0.015,
            },
            {
                "target_action": "click",
                "system": "Ubuntu",
                "parsed": True,
                "type_correct": True,
                "score": 1.0,
                "coordinate_distance": 0.07,
            },
            {
                "target_action": "click",
                "system": "Ubuntu",
                "parsed": False,
                "type_correct": False,
                "score": 0.0,
                "coordinate_distance": None,
            },
        ]
    )
    assert summary["coordinate"]["hit_rate_at_0_02"] == 1 / 3
    assert summary["coordinate"]["hit_rate_at_0_05"] == 1 / 3
    assert summary["coordinate"]["hit_rate_at_0_1"] == 2 / 3
