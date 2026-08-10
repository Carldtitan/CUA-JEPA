from dataclasses import replace

import pytest

from cua_jepa.sft_data import (
    _balanced_take,
    candidate_examples,
    canonical_action,
    dataset_audit,
    normalize_pyautogui_action,
    split_examples,
)


def test_normalizes_supported_actions_without_executing_code() -> None:
    assert normalize_pyautogui_action("pyautogui.click(x=0.25, y=0.75)") == {
        "action": "click",
        "x": 0.25,
        "y": 0.75,
    }
    assert normalize_pyautogui_action("pyautogui.hotkey('ctrl', 's')") == {
        "action": "hotkey",
        "keys": ["ctrl", "s"],
    }
    assert normalize_pyautogui_action("pyautogui.hotkey(['ctrl', 'u'])") == {
        "action": "hotkey",
        "keys": ["ctrl", "u"],
    }
    assert canonical_action(normalize_pyautogui_action("pyautogui.write('hello')")) == (
        '{"action":"write","text":"hello"}'
    )
    with pytest.raises(ValueError):
        normalize_pyautogui_action("__import__('os').system('whoami')")
    with pytest.raises(ValueError):
        normalize_pyautogui_action("pyautogui.click(0.1, 0.2); pyautogui.click(0.3, 0.4)")


def _trajectory(task_id: str, system: str, completed: bool = True) -> tuple[dict, dict]:
    trajectory = {
        "task_id": task_id,
        "instruction": f"Do task {task_id}",
        "task_completed": completed,
        "alignment_score": 7,
        "traj": [
            {
                "index": index,
                "image": f"{task_id}-{index}.png",
                "value": {
                    "code": f"pyautogui.click(x={index / 10}, y=0.5)",
                    "last_step_correct": True,
                    "last_step_redundant": False,
                },
            }
            for index in range(4)
        ],
    }
    metadata = {
        "task_id": task_id,
        "system": system,
        "domains": "Office",
        "applications": ["demo"],
        "verify_feedback": {"quality": "ok"},
        "task_description_alignment": "High Alignment",
    }
    return trajectory, metadata


def test_filters_bad_steps_and_builds_short_history() -> None:
    trajectory, metadata = _trajectory("task-1", "Windows")
    trajectory["traj"][1]["value"]["last_step_redundant"] = True
    candidates = candidate_examples([trajectory], {"task-1": metadata}, "source.jsonl")
    assert len(candidates) == 3
    assert candidates[0].history == ()
    assert len(candidates[-1].history) == 2


def test_accepts_official_quality_labels_and_null_feedback() -> None:
    first, first_metadata = _trajectory("ubuntu-good", "Ubuntu")
    first_metadata["verify_feedback"] = {"quality": "good"}
    second, second_metadata = _trajectory("win-null", "Windows")
    second_metadata["verify_feedback"] = None
    candidates = candidate_examples(
        [first, second],
        {"ubuntu-good": first_metadata, "win-null": second_metadata},
        "source.jsonl",
    )
    assert len(candidates) == 8


def test_excludes_tasks_marked_infeasible() -> None:
    trajectory, metadata = _trajectory("bad-task", "Windows")
    metadata["domains"] = "infeasible"
    assert candidate_examples([trajectory], {"bad-task": metadata}, "source.jsonl") == []


def test_training_click_cap_uses_available_other_actions() -> None:
    trajectories = []
    metadata = {}
    for index in range(40):
        trajectory, meta = _trajectory(f"mix-{index}", "Windows")
        trajectories.append(trajectory)
        metadata[meta["task_id"]] = meta
    candidates = candidate_examples(trajectories, metadata, "source.jsonl")
    mixed = [
        replace(
            value,
            action_kind="write",
            action={"action": "write", "text": "demo"},
            target='{"action":"write","text":"demo"}',
        )
        if index % 5 == 0
        else value
        for index, value in enumerate(candidates)
    ]
    selected = _balanced_take(
        mixed, 60, "train", seed=9, max_action_share={"click": 0.65}
    )
    assert sum(value.action_kind == "click" for value in selected) <= 39


def test_split_is_exact_deterministic_and_task_disjoint() -> None:
    trajectories = []
    metadata = {}
    systems = ["Windows", "Darwin", "Ubuntu"]
    for index in range(120):
        trajectory, meta = _trajectory(f"task-{index}", systems[index % 3])
        trajectories.append(trajectory)
        metadata[meta["task_id"]] = meta
    candidates = candidate_examples(trajectories, metadata, "source.jsonl")
    train, validation = split_examples(candidates, train_count=120, validation_count=24, seed=7)
    repeated = split_examples(candidates, train_count=120, validation_count=24, seed=7)
    assert [value.example_id for value in train] == [value.example_id for value in repeated[0]]
    audit = dataset_audit(train, validation)
    assert audit["train_examples"] == 120
    assert audit["validation_examples"] == 24
    assert audit["task_overlap"] == 0
    assert audit["screenshot_overlap"] == 0
    assert audit["exact_instruction_overlap"] == 0
    assert audit["duplicate_example_ids"] == 0
    assert audit["duplicate_screenshot_names"] == 0
    assert audit["empty_instructions"] == 0
    assert audit["target_action_mismatches"] == 0
    assert audit["invalid_normalized_coordinates"] == 0
    assert set(audit["train_by_system"]) == set(systems)
    assert max(audit["train_by_system"].values()) - min(audit["train_by_system"].values()) <= 1
