from cua_jepa.state_variants import make_state_variant, normalize_state_for_app


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


def test_outlook_state_is_adapted_to_rendered_mail_schema() -> None:
    state = {
        "user": {"displayName": "Katy Reid", "email": "katy@example.com"},
        "folders": [
            {
                "id": "folder-inbox",
                "displayName": "Inbox",
                "wellKnownName": "inbox",
                "icon": "inbox",
                "isSystem": True,
            }
        ],
        "messages": [
            {
                "id": "message-1",
                "parentFolderId": "folder-inbox",
                "subject": "Roadmap",
                "body": {"content": "Review this"},
                "from": {"name": "Marcus", "email": "m@example.com"},
                "toRecipients": [{"name": "Katy", "email": "katy@example.com"}],
                "receivedDateTime": "2026-03-30T09:15:00Z",
                "isRead": False,
            }
        ],
    }
    adapted = normalize_state_for_app("outlook_web_mock", state)
    assert adapted["folders"][0]["id"] == "inbox"
    assert adapted["messages"][0]["parentFolderId"] == "inbox"
    assert adapted["emails"][0]["folderId"] == "inbox"
    assert adapted["emails"][0]["body"] == "Review this"
    assert adapted["user"]["name"] == "Katy Reid"

