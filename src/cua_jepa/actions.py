from __future__ import annotations

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

  const active = document.activeElement;
  let canType = false;
  if (active && visible(active)) {
    const tag = active.tagName.toLowerCase();
    const type = String(active.getAttribute('type') || 'text').toLowerCase();
    canType = active.isContentEditable || tag === 'textarea' ||
      (tag === 'input' && !['button','checkbox','radio','submit','file','hidden'].includes(type));
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
  return {clicks, scrolls, can_type: canType};
}
"""


def enumerate_actions(page: Page, seed: int, text_suffix: str = "") -> list[Action]:
    raw = page.evaluate(ENUMERATE_SCRIPT)
    actions = [Action(**candidate) for candidate in raw["clicks"]]
    actions.extend(Action(**candidate) for candidate in raw["scrolls"])
    if raw["can_type"]:
        actions.append(Action(kind="type", text=f"Synthetic note {text_suffix}".strip()))
        actions.append(Action(kind="press", key="Enter"))
    rng = random.Random(seed)
    rng.shuffle(actions)
    return actions


def choose_distinct_actions(candidates: list[Action], count: int, seed: int) -> list[Action]:
    """Choose actions while avoiding four nearly identical coordinates."""
    rng = random.Random(seed)
    by_kind: dict[str, list[Action]] = {}
    for action in candidates:
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
        page.keyboard.type(action.text or "Synthetic note")
    elif action.kind == "press":
        page.keyboard.press(action.key or "Enter")
    else:
        raise ValueError(f"Unsupported action kind: {action.kind}")
