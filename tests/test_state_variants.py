from cua_jepa.state_variants import make_state_variant


def test_variant_changes_visible_text_but_preserves_references() -> None:
    state = {
        "id": "record-1",
        "title": "Quarterly report",
        "ownerId": "user-7",
        "url": "https://example.com/file",
    }
    variant = make_state_variant(state, 42)
    assert variant["title"].startswith("Quarterly report V")
    assert variant["id"] == "record-1"
    assert variant["ownerId"] == "user-7"
    assert variant["url"] == "https://example.com/file"


def test_variant_is_deterministic() -> None:
    state = {"name": "Alice", "items": [{"text": "Hello"}]}
    assert make_state_variant(state, 9) == make_state_variant(state, 9)

