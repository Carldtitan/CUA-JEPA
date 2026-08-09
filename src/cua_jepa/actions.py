from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

from playwright.sync_api import Page


@dataclass(frozen=True)
class Action:
    kind: str
    x: float | None = None
    y: float | None = None
    text: str | None = None
    key: str | None = None
    delta_x: int | None = None
    delta_y: int | None = None
    element_hint: str | None = None
    warmup_safe: bool | None = None

    def as_dict(
        self, viewport_width: int, viewport_height: int, include_hint: bool = False
    ) -> dict[str, Any]:
        result = asdict(self)
        if not include_hint:
            result.pop("element_hint", None)
        result.pop("warmup_safe", None)
        if self.x is not None and self.y is not None:
            result["x_normalized"] = round(self.x / viewport_width, 6)
            result["y_normalized"] = round(self.y / viewport_height, 6)
        return {key: value for key, value in result.items() if value is not None}


def action_from_dict(value: dict[str, Any]) -> Action:
    """Recreate an action while ignoring derived serialized fields."""
    fields = {
        "kind",
        "x",
        "y",
        "text",
        "key",
        "delta_x",
        "delta_y",
        "element_hint",
        "warmup_safe",
    }
    return Action(**{key: item for key, item in value.items() if key in fields})


def varied_type_text(
    app: str,
    element_hint: str | None,
    seed: int,
    bundle_id: str,
    branch_index: int,
) -> str:
    """Produce deterministic, field-appropriate text with varied lengths."""
    digest = hashlib.sha256(
        f"{app}:{element_hint}:{seed}:{bundle_id}:{branch_index}".encode("utf-8")
    ).digest()
    number = int.from_bytes(digest[:4], "big") % 10_000
    hint = (element_hint or "").lower()

    if "formula" in hint:
        values = (
            f"=SUM(B2:B{2 + number % 18})",
            f"=AVERAGE(C2:C{3 + number % 15})",
            f"=B{2 + number % 20}*1.15",
            f'=COUNTIF(A:A,"open-{number % 90}")',
        )
    elif "email" in hint or "@" in hint or "recipient" in hint:
        values = (
            f"maya.chen{number % 100}@example.com",
            f"ops-{number % 1000}@example.org",
            f"founder{number % 500}@startup.test",
        )
    elif "url" in hint or "website" in hint:
        values = (
            f"https://example.com/brief-{number}",
            f"https://docs.example.org/q{1 + number % 4}",
            f"https://status.example.net/{number % 1000}",
        )
    elif "phone" in hint or "mobile" in hint or "tel" in hint:
        values = (
            f"415555{number % 10_000:04d}",
            f"+1202555{number % 10_000:04d}",
        )
    elif "search" in hint or "filter" in hint:
        values = (
            f"AI {number % 90}",
            f"invoice {number:04d}",
            f"Maya Chen {number % 100}",
            f"urgent customer issue {number % 500}",
            f"Q{1 + number % 4} roadmap review",
            f"open pull requests {number % 200}",
            f"SKU-{number:04d}",
        )
    elif any(
        word in hint for word in ("comment", "message", "description", "reply", "note", "body")
    ):
        values = (
            f"Please review item {number}.",
            f"The customer confirmed the update for ticket {number}.",
            f"Follow up with the design team before Friday ({number % 100}).",
            f"Blocked on approval {number}; the remaining work is ready.",
        )
    elif any(
        word in hint for word in ("name", "title", "subject", "company", "product", "project")
    ):
        values = (
            f"Atlas {number}",
            f"Northstar Planning {number % 1000}",
            f"Q{1 + number % 4} Launch Review",
            f"Customer Renewal {number}",
        )
    else:
        values = (
            f"Review {number}",
            f"Ready for approval {number % 1000}",
            f"Priority {1 + number % 5}",
            f"Planning note {number}: confirm the next step.",
            f"Maya Chen {number % 100}",
        )
    return values[digest[4] % len(values)]


ENUMERATE_SCRIPT = r"""
() => {
  const visible = (el) => {
    const style = getComputedStyle(el);
    const r = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' &&
      style.pointerEvents !== 'none' && Number(style.opacity || 1) > 0.05 &&
      r.width >= 8 && r.height >= 8 && r.bottom > 0 && r.right > 0 &&
      r.top < innerHeight && r.left < innerWidth && !el.disabled;
  };
  const hint = (el) => String(
    el.getAttribute('aria-label') || el.getAttribute('title') ||
    el.innerText || el.getAttribute('placeholder') || el.tagName
  ).replace(/\s+/g, ' ').trim().slice(0, 100);
  const selectors = [
    'button', 'a[href]', '[role="button"]', '[role="tab"]',
    '[role="menuitem"]', '[role="option"]', 'input[type="checkbox"]',
    'input[type="radio"]', 'input:not([type="hidden"])', 'textarea',
    '[contenteditable="true"]', 'select', '[tabindex]:not([tabindex="-1"])'
  ].join(',');
  const seen = new Set();
  const clicks = [];
  for (const el of document.querySelectorAll(selectors)) {
    if (!visible(el)) continue;
    const r = el.getBoundingClientRect();
    const x = Math.max(1, Math.min(innerWidth - 2, r.left + r.width / 2));
    const y = Math.max(1, Math.min(innerHeight - 2, r.top + r.height / 2));
    const key = `${Math.round(x / 4)}:${Math.round(y / 4)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    let warmupSafe = false;
    if (el.tagName === 'A' && el.href && el.target !== '_blank') {
      try {
        warmupSafe = new URL(el.href, location.href).origin === location.origin;
      } catch (_) {
        warmupSafe = false;
      }
    }
    clicks.push({
      kind: 'click', x, y, element_hint: hint(el), warmup_safe: warmupSafe
    });
  }

  const typeTargets = [];
  const typeSelector = [
    'input:not([type])', 'input[type="text"]', 'input[type="search"]',
    'input[type="email"]', 'input[type="url"]', 'input[type="tel"]',
    'textarea', '[contenteditable="true"]'
  ].join(',');
  for (const el of document.querySelectorAll(typeSelector)) {
    if (!visible(el) || el.readOnly) continue;
    const r = el.getBoundingClientRect();
    const centerX = Math.max(1, Math.min(innerWidth - 2, r.left + r.width / 2));
    const centerY = Math.max(1, Math.min(innerHeight - 2, r.top + r.height / 2));
    const hit = document.elementFromPoint(centerX, centerY);
    if (!hit || (hit !== el && !el.contains(hit))) continue;
    typeTargets.push({
      kind: 'type',
      x: centerX,
      y: centerY,
      element_hint: hint(el)
    });
    if (typeTargets.length >= 6) break;
  }

  const scrolls = [];
  const nodes = [document.scrollingElement, ...document.querySelectorAll('*')];
  for (const el of nodes) {
    if (!el || !visible(el)) continue;
    if (el.scrollHeight <= el.clientHeight + 80) continue;
    const r = el.getBoundingClientRect();
    scrolls.push({
      kind: 'scroll',
      x: Math.max(1, Math.min(innerWidth - 2, r.left + r.width / 2)),
      y: Math.max(1, Math.min(innerHeight - 2, r.top + r.height / 2)),
      delta_x: 0,
      delta_y: Math.min(540, Math.max(240, Math.round(el.clientHeight * 0.65))),
      element_hint: hint(el)
    });
    if (scrolls.length >= 4) break;
  }
  return {clicks, scrolls, type_targets: typeTargets};
}
"""


def enumerate_actions(page: Page, seed: int, text_suffix: str = "") -> list[Action]:
    raw = page.evaluate(ENUMERATE_SCRIPT)
    actions = [Action(**candidate) for candidate in raw["clicks"]]
    actions.extend(Action(**candidate) for candidate in raw["scrolls"])
    for index, candidate in enumerate(raw["type_targets"]):
        actions.append(
            Action(
                **candidate,
                text=f"Synthetic {text_suffix}-{index}".strip("-"),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(actions)
    return actions


def choose_distinct_actions(candidates: list[Action], count: int, seed: int) -> list[Action]:
    """Choose actions while avoiding four nearly identical coordinates."""
    rng = random.Random(seed)
    by_kind: dict[str, list[Action]] = {}
    seen: set[tuple] = set()
    for action in candidates:
        identity = (
            action.kind,
            action.x,
            action.y,
            action.text,
            action.key,
            action.delta_x,
            action.delta_y,
        )
        if identity in seen:
            continue
        seen.add(identity)
        by_kind.setdefault(action.kind, []).append(action)
    for group in by_kind.values():
        rng.shuffle(group)

    selected: list[Action] = []
    for kind in ("type", "press", "scroll", "click"):
        group = by_kind.get(kind)
        if group and len(selected) < count:
            selected.append(group.pop())

    remaining = [action for group in by_kind.values() for action in group]
    rng.shuffle(remaining)
    for action in remaining:
        if len(selected) >= count:
            break
        if action.kind == "click" and any(
            prior.kind == "click"
            and prior.x is not None
            and prior.y is not None
            and action.x is not None
            and action.y is not None
            and abs(prior.x - action.x) < 12
            and abs(prior.y - action.y) < 12
            for prior in selected
        ):
            continue
        selected.append(action)
    return selected


def execute_action(page: Page, action: Action) -> None:
    if action.kind == "click":
        assert action.x is not None and action.y is not None
        page.mouse.click(action.x, action.y)
    elif action.kind == "scroll":
        assert action.x is not None and action.y is not None
        page.mouse.move(action.x, action.y)
        page.mouse.wheel(action.delta_x or 0, action.delta_y or 0)
    elif action.kind == "type":
        if action.x is not None and action.y is not None:
            page.mouse.click(action.x, action.y)
            page.keyboard.press("ControlOrMeta+A")
        page.keyboard.type(action.text or "Synthetic note")
    elif action.kind == "press":
        page.keyboard.press(action.key or "Enter")
    else:
        raise ValueError(f"Unsupported action kind: {action.kind}")


def typed_text_is_present(page: Page, expected: str) -> bool:
    """Confirm that typing changed an editable field, not document selection."""
    return bool(
        page.evaluate(
            """
            expected => {
              const selector = [
                'input:not([type="hidden"])', 'textarea', '[contenteditable="true"]'
              ].join(',');
              for (const el of document.querySelectorAll(selector)) {
                const value = el.isContentEditable ? el.innerText : el.value;
                if (typeof value === 'string' && value.includes(expected)) return true;
              }
              return false;
            }
            """,
            expected,
        )
    )
