"""Prepare a small, controlled AgentNet next-action SFT dataset."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


ACTION_ALIASES = {
    "doubleClick": "double_click",
    "rightClick": "right_click",
    "typewrite": "write",
}
SUPPORTED_ACTIONS = {
    "click",
    "double_click",
    "right_click",
    "moveTo",
    "dragTo",
    "scroll",
    "write",
    "press",
    "hotkey",
    "terminate",
}


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    split: str
    task_id: str
    instruction: str
    image_file: str
    system: str
    domain: str
    applications: tuple[str, ...]
    action_kind: str
    action: dict[str, Any]
    target: str
    history: tuple[str, ...]
    source_file: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["applications"] = list(self.applications)
        value["history"] = list(self.history)
        return value


def _literal(node: ast.AST) -> Any:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ValueError("Action contains a non-literal value") from error
    if isinstance(value, (str, int, float, bool, type(None), list, tuple)):
        return value
    raise ValueError(f"Unsupported action value: {type(value).__name__}")


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        if function.value.id != "pyautogui":
            raise ValueError("Only pyautogui calls are supported")
        return function.attr
    if isinstance(function, ast.Name) and function.id == "terminate":
        return "terminate"
    raise ValueError("Unsupported action call")


def normalize_pyautogui_action(code: str) -> dict[str, Any]:
    """Parse one safe PyAutoGUI call into a stable JSON-compatible action."""
    try:
        tree = ast.parse(code.strip(), mode="exec")
    except SyntaxError as error:
        raise ValueError("Action is not valid Python syntax") from error
    calls = [node.value for node in tree.body if isinstance(node, ast.Expr)]
    if len(tree.body) != 1 or len(calls) != 1 or not isinstance(calls[0], ast.Call):
        raise ValueError("Action must contain exactly one function call")

    call = calls[0]
    raw_name = _call_name(call)
    name = ACTION_ALIASES.get(raw_name, raw_name)
    if name not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported action type: {raw_name}")

    positional = [_literal(node) for node in call.args]
    keyword = {item.arg: _literal(item.value) for item in call.keywords if item.arg}
    if len(keyword) != len(call.keywords):
        raise ValueError("Expanded keyword arguments are not supported")

    action: dict[str, Any] = {"action": name}
    if name in {"click", "double_click", "right_click", "moveTo", "dragTo"}:
        x = keyword.pop("x", positional.pop(0) if positional else None)
        y = keyword.pop("y", positional.pop(0) if positional else None)
        if x is None or y is None:
            raise ValueError(f"{name} requires x and y")
        action["x"] = round(float(x), 6)
        action["y"] = round(float(y), 6)
    elif name == "scroll":
        amount = keyword.pop("clicks", positional.pop(0) if positional else None)
        if amount is None:
            raise ValueError("scroll requires an amount")
        action["amount"] = float(amount)
    elif name == "write":
        text = keyword.pop("message", positional.pop(0) if positional else None)
        if text is None:
            raise ValueError("write requires text")
        action["text"] = str(text)
    elif name == "press":
        keys = keyword.pop("presses", positional.pop(0) if positional else None)
        if keys is None:
            raise ValueError("press requires a key")
        action["keys"] = list(keys) if isinstance(keys, (list, tuple)) else [str(keys)]
    elif name == "hotkey":
        if len(positional) == 1 and isinstance(positional[0], (list, tuple)):
            action["keys"] = [str(value) for value in positional[0]]
        else:
            action["keys"] = [str(value) for value in positional]
        positional.clear()

    for key in ("button", "clicks", "interval", "duration"):
        if key in keyword:
            action[key] = keyword.pop(key)
    if positional or keyword:
        raise ValueError("Action contains unsupported arguments")
    return action


def canonical_action(action: Mapping[str, Any]) -> str:
    return json.dumps(dict(action), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def candidate_examples(
    trajectories: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    source_file: str,
) -> list[SFTExample]:
    """Keep only completed, checked, non-redundant next-action examples."""
    result: list[SFTExample] = []
    seen_images: set[str] = set()
    for trajectory in trajectories:
        task_id = str(trajectory.get("task_id", ""))
        meta = metadata.get(task_id, {})
        if not task_id or not trajectory.get("task_completed"):
            continue
        if str(meta.get("domains", "")).lower() == "infeasible":
            continue
        if float(trajectory.get("alignment_score", 0) or 0) < 6:
            continue
        verify_feedback = meta.get("verify_feedback") or {}
        if verify_feedback.get("quality") not in {None, "ok", "good"}:
            continue
        if meta.get("task_description_alignment") not in {None, "High Alignment"}:
            continue

        history: deque[str] = deque(maxlen=2)
        for step in trajectory.get("traj", []):
            value = step.get("value", {})
            image_file = str(step.get("image", ""))
            code = str(value.get("code", ""))
            try:
                action = normalize_pyautogui_action(code)
            except (ValueError, TypeError, OverflowError):
                continue
            target = canonical_action(action)
            usable = (
                image_file
                and image_file not in seen_images
                and value.get("last_step_correct") is True
                and value.get("last_step_redundant") is False
            )
            if usable:
                index = int(step.get("index", len(result)))
                result.append(
                    SFTExample(
                        example_id=f"{task_id}:{index}",
                        split="candidate",
                        task_id=task_id,
                        instruction=str(trajectory.get("instruction", "")).strip(),
                        image_file=image_file,
                        system=str(meta.get("system", "unknown")),
                        domain=str(meta.get("domains", "unknown")),
                        applications=tuple(str(v) for v in meta.get("applications", [])),
                        action_kind=str(action["action"]),
                        action=action,
                        target=target,
                        history=tuple(history),
                        source_file=source_file,
                    )
                )
                seen_images.add(image_file)
            history.append(target)
    return result


def _balanced_take(
    examples: Iterable[SFTExample],
    count: int,
    split: str,
    seed: int,
    max_per_task: int = 3,
    max_action_share: Mapping[str, float] | None = None,
) -> list[SFTExample]:
    """Balance systems and domains without making rare actions unnaturally common."""
    by_system: dict[str, list[SFTExample]] = defaultdict(list)
    for value in examples:
        by_system[value.system].append(value)
    systems = sorted(by_system)
    if not systems:
        raise RuntimeError(f"No {split} examples are available")
    base_quota, remainder = divmod(count, len(systems))
    system_quotas = {
        system: base_quota + (index < remainder) for index, system in enumerate(systems)
    }
    selected: list[SFTExample] = []
    task_counts: dict[str, int] = defaultdict(int)
    selected_ids: set[str] = set()
    system_action_counts: dict[tuple[str, str], int] = defaultdict(int)

    def add(example: SFTExample) -> bool:
        if example.example_id in selected_ids or task_counts[example.task_id] >= max_per_task:
            return False
        selected_ids.add(example.example_id)
        task_counts[example.task_id] += 1
        system_action_counts[(example.system, example.action_kind)] += 1
        selected.append(
            SFTExample(
                **{
                    **example.to_dict(),
                    "split": split,
                    "applications": example.applications,
                    "history": example.history,
                }
            )
        )
        return True

    for system in systems:
        quota = system_quotas[system]
        system_values = sorted(
            by_system[system], key=lambda item: _stable_number(item.example_id, seed)
        )
        system_start = len(selected)

        # Give every available domain one place before the natural fill.
        domains: dict[str, list[SFTExample]] = defaultdict(list)
        for value in system_values:
            domains[value.domain].append(value)
        for domain in sorted(domains, key=lambda key: (_stable_number(key, seed), key)):
            for value in domains[domain]:
                if add(value):
                    break
            if len(selected) - system_start >= quota:
                break

        # Give each action type a small minimum. Do not make all types equal.
        minimum_per_action = max(2, quota // 100)
        action_counts: dict[str, int] = defaultdict(int)
        for value in selected[system_start:]:
            action_counts[value.action_kind] += 1
        by_action: dict[str, list[SFTExample]] = defaultdict(list)
        for value in system_values:
            by_action[value.action_kind].append(value)
        for action in sorted(by_action):
            for value in by_action[action]:
                if action_counts[action] >= minimum_per_action:
                    break
                if add(value):
                    action_counts[action] += 1

        deferred: list[SFTExample] = []
        for value in system_values:
            if len(selected) - system_start >= quota:
                break
            share = (max_action_share or {}).get(value.action_kind)
            action_limit = math.ceil(quota * share) if share is not None else None
            if (
                action_limit is not None
                and system_action_counts[(system, value.action_kind)] >= action_limit
            ):
                deferred.append(value)
                continue
            add(value)
        # Use capped actions only if the source does not contain enough alternatives.
        for value in deferred:
            if len(selected) - system_start >= quota:
                break
            add(value)

    if len(selected) != count:
        raise RuntimeError(f"Requested {count} {split} examples, but selected {len(selected)}")
    return selected


def split_examples(
    examples: Iterable[SFTExample],
    train_count: int = 2_000,
    validation_count: int = 250,
    seed: int = 20260810,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Make deterministic task-disjoint train and validation sets."""
    values = list(examples)
    validation_tasks = {
        example.task_id
        for example in values
        if _stable_number(example.task_id, seed) % 5 == 0
    }
    validation_pool = [value for value in values if value.task_id in validation_tasks]
    train_pool = [value for value in values if value.task_id not in validation_tasks]
    validation = _balanced_take(validation_pool, validation_count, "validation", seed + 1)
    train = _balanced_take(
        train_pool,
        train_count,
        "train",
        seed + 2,
        max_action_share={"click": 0.65},
    )
    if {value.task_id for value in train} & {value.task_id for value in validation}:
        raise RuntimeError("SFT train and validation task IDs overlap")
    if {value.image_file for value in train} & {value.image_file for value in validation}:
        raise RuntimeError("SFT train and validation screenshots overlap")
    return train, validation


def dataset_audit(train: Iterable[SFTExample], validation: Iterable[SFTExample]) -> dict[str, Any]:
    train_values = list(train)
    validation_values = list(validation)
    all_values = train_values + validation_values

    def counts(values: list[SFTExample], field: str) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for value in values:
            result[str(getattr(value, field))] += 1
        return dict(sorted(result.items()))

    return {
        "train_examples": len(train_values),
        "validation_examples": len(validation_values),
        "train_tasks": len({value.task_id for value in train_values}),
        "validation_tasks": len({value.task_id for value in validation_values}),
        "task_overlap": len(
            {value.task_id for value in train_values}
            & {value.task_id for value in validation_values}
        ),
        "screenshot_overlap": len(
            {value.image_file for value in train_values}
            & {value.image_file for value in validation_values}
        ),
        "exact_instruction_overlap": len(
            {value.instruction for value in train_values}
            & {value.instruction for value in validation_values}
        ),
        "duplicate_example_ids": len(all_values)
        - len({value.example_id for value in all_values}),
        "duplicate_screenshot_names": len(all_values)
        - len({value.image_file for value in all_values}),
        "empty_instructions": sum(not value.instruction.strip() for value in all_values),
        "target_action_mismatches": sum(
            value.target != canonical_action(value.action) for value in all_values
        ),
        "invalid_normalized_coordinates": sum(
            not (0.0 <= float(value.action["x"]) <= 1.0)
            or not (0.0 <= float(value.action["y"]) <= 1.0)
            for value in all_values
            if "x" in value.action or "y" in value.action
        ),
        "train_by_system": counts(train_values, "system"),
        "validation_by_system": counts(validation_values, "system"),
        "train_by_action": counts(train_values, "action_kind"),
        "validation_by_action": counts(validation_values, "action_kind"),
        "train_by_domain": counts(train_values, "domain"),
        "validation_by_domain": counts(validation_values, "domain"),
    }
