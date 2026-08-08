from cua_jepa.actions import Action, choose_distinct_actions


def test_action_serializes_normalized_coordinates() -> None:
    action = Action(
        kind="click", x=640, y=360, element_hint="Compose", warmup_safe=True
    )
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
