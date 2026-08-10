import os

import pytest
import torch
from torch import nn

os.environ.setdefault("USE_TF", "0")

from cua_jepa.jepa_data import TransitionSample
from cua_jepa.train_jepa import (
    JEPATrainConfig,
    _set_adapter,
    adapter_pair_max_difference,
    approved_vision_parameters,
    copy_online_adapter_to_target,
    training_actions_for_bundle,
    validate_dataset_assignments,
)


class FakePeftVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.adapters = nn.ModuleDict(
            {
                "online": nn.Linear(2, 2, bias=False),
                "target": nn.Linear(2, 2, bias=False),
            }
        )
        self.peft_config = {"online": object(), "target": object()}

    def set_adapter(self, name: str) -> None:
        del name
        for parameter in self.parameters():
            parameter.requires_grad_(True)


def _sample(bundle: str, split: str, current: bytes, future: bytes) -> TransitionSample:
    return TransitionSample(
        bundle_id=bundle,
        app=f"{split}_app",
        split=split,
        branch_index=0,
        action={"kind": "click"},
        current_webp=current,
        future_webp=future,
        changed_pixel_fraction=0.1,
    )


def test_qwen_revision_is_pinned() -> None:
    assert JEPATrainConfig().model_revision == (
        "89644892e4d85e24eaac8bacfd4f463576704203"
    )


def test_adapter_switch_never_unfreezes_base_or_target() -> None:
    vision = FakePeftVision()
    _set_adapter(vision, "online")
    online, target, base = approved_vision_parameters(vision, train_qwen_lora=True)
    assert online
    assert all(parameter.requires_grad for parameter in online)
    assert all(not parameter.requires_grad for parameter in target + base)

    _set_adapter(vision, "target")
    assert all(not parameter.requires_grad for parameter in vision.parameters())


def test_target_adapter_uses_ema_without_gradients() -> None:
    vision = FakePeftVision()
    _set_adapter(vision, "online")
    with torch.no_grad():
        vision.adapters["online"].weight.fill_(2.0)
        vision.adapters["target"].weight.zero_()
    copy_online_adapter_to_target(vision, decay=0.5)
    assert torch.allclose(vision.adapters["target"].weight, torch.ones(2, 2))
    assert adapter_pair_max_difference(vision) == 1.0
    assert not vision.adapters["target"].weight.requires_grad


def test_dataset_assignment_rejects_split_or_image_leakage() -> None:
    train = [_sample("train-1", "train", b"train-current", b"train-future")]
    validation = [
        _sample("validation-1", "validation", b"val-current", b"val-future")
    ]
    audit = validate_dataset_assignments(train, validation, 1, 1)
    assert audit["exact_screenshot_overlap"] == 0

    wrong_split = [_sample("validation-2", "train", b"other", b"future")]
    with pytest.raises(RuntimeError, match="wrong splits"):
        validate_dataset_assignments(train, wrong_split)

    leaked = [
        _sample("validation-3", "validation", b"train-current", b"different")
    ]
    with pytest.raises(RuntimeError, match="exact screenshots"):
        validate_dataset_assignments(train, leaked)


def test_model3_action_control_uses_one_fixed_no_action_input() -> None:
    branches = [
        TransitionSample(
            bundle_id="bundle-1",
            app="app",
            split="train",
            branch_index=index,
            action={"kind": f"action-{index}"},
            current_webp=b"current",
            future_webp=f"future-{index}".encode(),
            changed_pixel_fraction=0.1,
        )
        for index in range(4)
    ]
    correct = training_actions_for_bundle(branches, JEPATrainConfig())
    assert correct == [branch.action for branch in branches]

    config = JEPATrainConfig(training_action_assignment="no_action")
    first = training_actions_for_bundle(branches, config)
    second = training_actions_for_bundle(branches, config)
    assert first == second
    assert first == [
        {"kind": "NO_ACTION", "spatial_active": False},
        {"kind": "NO_ACTION", "spatial_active": False},
        {"kind": "NO_ACTION", "spatial_active": False},
        {"kind": "NO_ACTION", "spatial_active": False},
    ]
