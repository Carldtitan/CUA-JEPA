import pytest

import torch

from cua_jepa.train_sft import (
    SFTTrainConfig,
    action_prompt,
    find_last_subsequence,
    trainable_state_sha256,
)


def test_qwen_revision_and_small_lora_are_pinned() -> None:
    config = SFTTrainConfig()
    assert config.model_revision == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert config.lora_rank == 8
    assert config.gradient_accumulation_steps == 4
    assert config.initial_evaluation_examples == config.final_evaluation_examples == 250


def test_finds_last_target_token_sequence() -> None:
    assert find_last_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 3
    with pytest.raises(ValueError, match="not found"):
        find_last_subsequence([1, 2, 3], [4])


def test_prompt_contains_instruction_and_short_history() -> None:
    prompt = action_prompt(
        {
            "instruction": "Save the file",
            "history": ['{"action":"click","x":0.1,"y":0.2}'],
        }
    )
    assert "Save the file" in prompt
    assert '"action":"click"' in prompt
    assert "Return only the JSON object" in prompt


def test_trainable_state_hash_changes_only_with_trainable_parameters() -> None:
    module = torch.nn.Linear(2, 2)
    module.bias.requires_grad_(False)
    before = trainable_state_sha256(module)
    with torch.no_grad():
        module.bias.add_(1)
    assert trainable_state_sha256(module) == before
    with torch.no_grad():
        module.weight.add_(1)
    assert trainable_state_sha256(module) != before
