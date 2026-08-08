from scripts.build_state_catalog import extract_literal_states


def test_extracts_literal_state_assignment() -> None:
    source = """
state = {"items": [{"id": "1", "done": False}]}
other = "ignored"
"""
    assert extract_literal_states(source) == [
        {"items": [{"id": "1", "done": False}]}
    ]


def test_skips_computed_state() -> None:
    source = "state = make_state()"
    assert extract_literal_states(source) == []
