import pytest

import torch

from cua_jepa.train_sft import (
    SFTTrainConfig,
    action_prompt,
    find_last_subsequence,
    runtime_audit,
    select_evaluation_records,
    source_jepa_audit,
    trainable_state_sha256,
)


def test_qwen_revision_and_small_lora_are_pinned() -> None:
    config = SFTTrainConfig()
    assert config.model_revision == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert config.lora_rank == 8
    assert config.gradient_accumulation_steps == 4
    assert config.max_pixels == 1_048_576
    assert config.max_new_tokens == 128
    assert config.initial_evaluation_examples == config.final_evaluation_examples == 250
    assert config.cpu_cost_per_core_second == 3.942e-5
    assert config.memory_cost_per_gib_second == 6.67e-6


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


def test_small_evaluation_set_balances_operating_systems() -> None:
    records = [
        {"example_id": f"{system}-{index}", "system": system}
        for system in ("Darwin", "Ubuntu", "Windows")
        for index in range(20)
    ]
    selected = select_evaluation_records(records, limit=32, seed=7)
    counts = {
        system: sum(record["system"] == system for record in selected)
        for system in ("Darwin", "Ubuntu", "Windows")
    }
    assert len(selected) == 32
    assert max(counts.values()) - min(counts.values()) <= 1
    assert selected == select_evaluation_records(records, limit=32, seed=7)
    assert select_evaluation_records(records, limit=100, seed=7) == records


def test_runtime_audit_hashes_training_and_evaluation_code() -> None:
    audit = runtime_audit()
    assert audit["python"]
    assert audit["torch"]
    assert len(audit["training_code_sha256"]) == 64
    assert len(audit["evaluation_code_sha256"]) == 64


def test_source_jepa_audit_records_non_pure_objective(tmp_path) -> None:
    adapter = tmp_path / "run" / "qwen_vision_online_lora" / "online"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "run" / "final_metrics.json").write_text(
        '{"config":{"action_separation_weight":0.0,'
        '"variance_regularization_weight":0.05,'
        '"covariance_regularization_weight":0.05},'
        '"steps":10,"stop_reason":"done",'
        '"final_validation":{"four_way_accuracy":0.38}}',
        encoding="utf-8",
    )
    audit = source_jepa_audit(adapter)
    assert audit["uses_action_separation"] is True
    assert audit["uses_correct_action_information"] is True
    assert audit["objective"].startswith("latent regression plus")
    assert source_jepa_audit(None) == {"source": None}


def test_source_jepa_audit_identifies_model3_no_action_control(tmp_path) -> None:
    adapter = tmp_path / "run" / "qwen_vision_online_lora" / "online"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"model3-adapter")
    (tmp_path / "run" / "final_metrics.json").write_text(
        '{"config":{"action_separation_weight":0.25,'
        '"variance_regularization_weight":0.05,'
        '"covariance_regularization_weight":0.05,'
        '"training_action_assignment":"no_action"},'
        '"steps":7667,"stop_reason":"maximum_steps_completed",'
        '"final_validation":{"four_way_accuracy":0.25}}',
        encoding="utf-8",
    )
    audit = source_jepa_audit(adapter)
    assert audit["source"] == "no-action JEPA control"
    assert audit["uses_correct_action_information"] is False
    assert audit["training_action_assignment"] == "no_action"
    assert audit["uses_action_separation"] is False
    assert "fixed NO_ACTION" in audit["objective"]
