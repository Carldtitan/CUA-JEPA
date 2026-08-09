from __future__ import annotations

import copy
import hashlib
from typing import Any
from urllib.parse import quote


VISIBLE_TEXT_KEYS = {
    "body",
    "caption",
    "company",
    "content",
    "description",
    "displayname",
    "first_name",
    "firstname",
    "label",
    "last_name",
    "lastname",
    "message",
    "name",
    "notes",
    "subject",
    "summary",
    "text",
    "title",
}


def _variant_suffix(seed: int, path: str) -> str:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()
    return f" V{int(digest[:6], 16) % 10_000:04d}"


def make_state_variant(state: dict[str, Any], seed: int) -> dict[str, Any]:
    """Create deterministic visible-text variation without changing IDs or references."""
    value = copy.deepcopy(state)

    def visit(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            return {
                key: visit(child, f"{path}.{key}" if path else key)
                for key, child in node.items()
            }
        if isinstance(node, list):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(node)]
        key = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if (
            isinstance(node, str)
            and key in VISIBLE_TEXT_KEYS
            and 1 <= len(node.strip()) <= 240
            and not node.startswith(("http://", "https://"))
        ):
            return f"{node}{_variant_suffix(seed, path)}"
        return node

    return visit(value, "")


OUTLOOK_SYSTEM_FOLDER_IDS = {
    "inbox": "inbox",
    "sentitems": "sent",
    "drafts": "drafts",
    "archive": "archive",
    "deleteditems": "deleted",
    "junkemail": "junk",
}

OUTLOOK_FOLDER_ICONS = {
    "inbox": "Inbox",
    "send": "Send",
    "drafts": "File",
    "file": "File",
    "archive": "Archive",
    "trash": "Trash2",
    "warning": "Ban",
    "folder": "Folder",
}


def normalize_state_for_app(app: str, state: dict[str, Any]) -> dict[str, Any]:
    """Adapt task-state schemas to the exact schema rendered by each mock app."""
    if app != "outlook_web_mock":
        return state

    value = copy.deepcopy(state)
    folder_id_map: dict[str, str] = {}
    folders: list[dict[str, Any]] = []
    for folder in value.get("folders", []):
        old_id = str(folder.get("id", ""))
        well_known = str(folder.get("wellKnownName") or "").lower()
        new_id = OUTLOOK_SYSTEM_FOLDER_IDS.get(well_known, old_id)
        folder_id_map[old_id] = new_id
        icon = str(folder.get("icon") or "folder").lower()
        folders.append(
            {
                **folder,
                "id": new_id,
                "name": folder.get("displayName") or folder.get("name") or new_id,
                "icon": OUTLOOK_FOLDER_ICONS.get(icon, "Folder"),
                "type": "system" if folder.get("isSystem") else "custom",
            }
        )

    normalized_messages: list[dict[str, Any]] = []
    normalized_emails: list[dict[str, Any]] = []
    for index, message in enumerate(value.get("messages", [])):
        old_folder_id = str(message.get("parentFolderId", ""))
        folder_id = folder_id_map.get(old_folder_id, old_folder_id or "inbox")
        sender = message.get("from") or message.get("sender") or {
            "name": "Unknown",
            "email": "unknown@example.com",
        }
        recipients = message.get("toRecipients") or []
        body_value = message.get("body", "")
        if isinstance(body_value, dict):
            body_value = body_value.get("content", "")
        flag = message.get("flag") or {}
        attachments = [
            {
                **attachment,
                "name": attachment.get("name") or f"attachment-{attachment_index}",
                "url": attachment.get("url") or "#",
            }
            for attachment_index, attachment in enumerate(
                message.get("attachments") or []
            )
        ]
        normalized_messages.append({**message, "parentFolderId": folder_id})
        normalized_emails.append(
            {
                "id": message.get("id") or f"email_custom_{index}",
                "folderId": folder_id,
                "from": sender,
                "to": recipients,
                "subject": message.get("subject") or "(No Subject)",
                "body": body_value,
                "preview": message.get("bodyPreview") or str(body_value)[:50],
                "timestamp": (
                    message.get("receivedDateTime")
                    or message.get("sentDateTime")
                    or "2026-08-08T20:00:00Z"
                ),
                "read": bool(message.get("isRead")),
                "flagged": flag.get("flagStatus") == "flagged",
                "categories": list(message.get("categories") or []),
                "attachments": attachments,
                "isFocused": message.get("inferenceClassification") == "focused",
            }
        )

    user = value.get("user") or {}
    contacts = [
        {
            **contact,
            "name": contact.get("displayName") or contact.get("name") or "Unknown",
        }
        for contact in value.get("contacts", [])
    ]
    value.update(
        {
            "user": {
                **user,
                "name": user.get("displayName") or user.get("name") or "Admin User",
                "avatar": user.get("avatar") or "",
            },
            "folders": folders,
            "messages": normalized_messages,
            "emails": normalized_emails,
            "contacts": contacts,
            "selectedFolderId": folder_id_map.get(
                str(value.get("selectedFolderId", "")),
                value.get("selectedFolderId") or "inbox",
            ),
        }
    )
    return value


def initial_path_for_app(app: str, state: dict[str, Any], seed: int) -> str:
    """Choose a deterministic state-relevant initial view for a mock app."""
    if app == "google_docs_mock":
        documents = state.get("documents") or {}
        if isinstance(documents, dict) and documents and seed % 5:
            document_ids = sorted(str(document_id) for document_id in documents)
            requested = str((state.get("ui") or {}).get("currentDocId") or "")
            document_id = (
                requested
                if requested in documents
                else document_ids[seed % len(document_ids)]
            )
            return f"/document/{quote(document_id, safe='')}"
    if app == "outlook_web_mock":
        module = str(state.get("selectedModule") or "mail").lower()
        if module in {"calendar", "people", "tasks"}:
            return f"/{module}"
        folder_id = str(state.get("selectedFolderId") or "inbox")
        return f"/mail/{quote(folder_id, safe='')}"
    return "/"

