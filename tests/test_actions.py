from cua_jepa.actions import (
    Action,
    action_from_dict,
    choose_distinct_actions,
    varied_type_text,
)


def test_action_serializes_normalized_coordinates() -> None:
    action = Action(kind="click", x=640, y=360, element_hint="Compose", warmup_safe=True)
    value = action.as_dict(1280, 720)
    assert value["x_normalized"] == 0.5
    assert value["y_normalized"] == 0.5
    assert "element_hint" not in value
    assert "warmup_safe" not in value


def test_selection_prefers_action_type_diversity() -> None:
    candidates = [
        Action(kind="click", x=10, y=10),
        Action(kind="click", x=100, y=100),
        Action(kind="scroll", x=200, y=200, delta_y=400),
        Action(kind="type", text="hello"),
        Action(kind="press", key="Enter"),
    ]
    selected = choose_distinct_actions(candidates, count=4, seed=7)
    assert {action.kind for action in selected} == {"click", "scroll", "type", "press"}


def test_selection_removes_exact_duplicate_actions() -> None:
    duplicate = Action(kind="scroll", x=200, y=200, delta_y=400)
    candidates = [
        duplicate,
        duplicate,
        Action(kind="click", x=10, y=10),
        Action(kind="click", x=100, y=100),
        Action(kind="type", x=300, y=40, text="hello"),
    ]
    selected = choose_distinct_actions(candidates, count=4, seed=3)
    assert len(selected) == 4
    assert selected.count(duplicate) == 1


def test_action_from_dict_ignores_normalized_coordinates() -> None:
    action = action_from_dict(
        {
            "kind": "click",
            "x": 640,
            "y": 360,
            "x_normalized": 0.5,
            "y_normalized": 0.5,
        }
    )
    assert action == Action(kind="click", x=640, y=360)


def test_varied_type_text_is_deterministic_and_varied() -> None:
    first = varied_type_text("gmail_mock", "Search mail", 7, "bundle-7", 0)
    assert first == varied_type_text("gmail_mock", "Search mail", 7, "bundle-7", 0)
    values = {
        varied_type_text("gmail_mock", "Search mail", seed, f"bundle-{seed}", 0)
        for seed in range(100)
    }
    assert len(values) > 50
    assert len({len(value) for value in values}) >= 5


def test_varied_type_text_matches_special_fields() -> None:
    email = varied_type_text("gmail_mock", "Recipient email", 1, "bundle", 0)
    formula = varied_type_text("google_sheets_mock", "Formula bar", 1, "bundle", 0)
    assert "@" in email
    assert formula.startswith("=")
