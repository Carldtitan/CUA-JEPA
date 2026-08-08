from __future__ import annotations

import copy
import hashlib
from typing import Any


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

