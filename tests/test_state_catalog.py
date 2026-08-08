from scripts.build_state_catalog import extract_literal_states, filter_catalog


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


def test_filters_foreign_states_from_multi_app_setup() -> None:
    source = '''
gitlab_state = {"projects": [], "issues": [], "mergeRequests": []}
slack_state = {"channels": [], "messages": [], "workspace": {}}
'''
    assert extract_literal_states(source, app="gitlab_mock") == [
        {"projects": [], "issues": [], "mergeRequests": []}
    ]


def test_filters_an_existing_catalog() -> None:
    catalog = {
        "trello_mock": [
            {"source_task_id": "right", "state": {"boards": [], "lists": [], "cards": []}},
            {"source_task_id": "wrong", "state": {"emails": [], "labels": [], "drafts": []}},
        ]
    }
    filtered, stats = filter_catalog(catalog, {"trello_mock"})
    assert [entry["source_task_id"] for entry in filtered["trello_mock"]] == ["right"]
    assert stats["trello_mock:rejected_foreign_states"] == 1
