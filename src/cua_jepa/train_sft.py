"""Controlled next-action SFT for Model 2 and Model 4."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from cua_jepa.sft_eval import parse_action_prediction, score_action, summarize_scores


SYSTEM_PROMPT = (
    "You are a computer-use agent. Read the instruction, recent actions, and screenshot. "
    "Return only one JSON action. Use normalized x and y coordinates from 0 to 1. "
    "Valid actions are click, double_click, right_click, moveTo, dragTo, scroll, write, "
    "press, and hotkey. Coordinate actions use x and y. Scroll uses amount. Write uses "
    "text. Press and hotkey use a keys list."
)


@dataclass
class SFTTrainConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    model_revision: str = "89644892e4d85e24eaac8bacfd4f463576704203"
    seed: int = 20260810
    language_init_seed: int = 20260813
    max_pixels: int = 1_048_576
    min_pixels: int = 65_536
    lora_rank: int = 8
    lora_alpha: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_steps: int = 500
    warmup_steps: int = 15
    log_every: int = 10
    monitor_every: int = 100
    monitor_examples: int = 32
    initial_evaluation_examples: int = 250
    final_evaluation_examples: int = 250
    max_new_tokens: int = 128
    coordinate_threshold: float = 0.1
    max_runtime_seconds: int = 4 * 60 * 60
    approved_cost_limit_usd: float = 20.0
    gpu_cost_per_second: float = 0.000222
    cpu_cores: float = 4.0
    # Conservative Modal Starter rates as checked on 2026-08-10.
    cpu_cost_per_core_second: float = 3.942e-5
    memory_gib: float = 24.0
    memory_cost_per_gib_second: float = 6.67e-6


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_audit() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "pillow": version("pillow"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "training_code_sha256": sha256_file(Path(__file__)),
        "evaluation_code_sha256": sha256_file(
            Path(__file__).with_name("sft_eval.py")
        ),
    }


def tensor_state_sha256(parameters: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def trainable_state_sha256(module: torch.nn.Module) -> str:
    return tensor_state_sha256(
        (name, value) for name, value in module.named_parameters() if value.requires_grad
    )


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        raise ValueError("The target token sequence is empty")
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    raise ValueError("Target tokens were not found in the training sequence")


def action_prompt(record: dict[str, Any]) -> str:
    history = record.get("history") or []
    history_text = "\n".join(f"- {value}" for value in history) if history else "- none"
    return (
        f"Instruction: {record['instruction']}\n"
        f"Recent actions:\n{history_text}\n"
        "Choose the next action. Return only the JSON object."
    )


def select_evaluation_records(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic small set with balanced operating systems."""
    if not limit or limit >= len(records):
        return list(records)
    by_system: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_system.setdefault(str(record.get("system", "unknown")), []).append(record)
    for system, values in by_system.items():
        values.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}:{system}:{record['example_id']}".encode()
            ).digest()
        )
    selected = []
    position = 0
    systems = sorted(by_system)
    while len(selected) < limit:
        added = False
        for system in systems:
            values = by_system[system]
            if position < len(values):
                selected.append(values[position])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        position += 1
    return selected


def _messages(record: dict[str, Any], image: Image.Image, include_target: bool) -> list[dict]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": action_prompt(record)},
            ],
        },
    ]
    if include_target:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": record["target"]}]}
        )
    return messages


def make_training_inputs(processor, record: dict[str, Any], image_path: Path, device):
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        inputs = processor.apply_chat_template(
            _messages(record, image, include_target=True),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
    target_ids = processor.tokenizer(record["target"], add_special_tokens=False).input_ids
    input_ids = inputs["input_ids"][0].tolist()
    target_start = find_last_subsequence(input_ids, target_ids)
    labels = inputs["input_ids"].clone()
    labels[:, :target_start] = -100
    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def make_generation_inputs(processor, record: dict[str, Any], image_path: Path, device):
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        inputs = processor.apply_chat_template(
            _messages(record, image, include_target=False),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _load_dataset(dataset_root: Path) -> tuple[list[dict], list[dict], dict[str, dict], dict]:
    dataset_path = dataset_root / "dataset.jsonl"
    audit_path = dataset_root / "dataset_audit.json"
    manifest_path = dataset_root / "image_manifest.json"
    for path in (dataset_path, audit_path, manifest_path):
        if not path.is_file():
            raise RuntimeError(f"Required SFT data file is missing: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "ready" or audit.get("stored_images") != 2_250:
        raise RuntimeError("The AgentNet SFT audit is not ready")
    if sha256_file(dataset_path) != audit.get("dataset_sha256"):
        raise RuntimeError("The SFT dataset hash does not match its audit")
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    if len(train) != 2_000 or len(validation) != 250:
        raise RuntimeError(f"Wrong SFT split sizes: {len(train)} train, {len(validation)} validation")
    if {record["task_id"] for record in train} & {record["task_id"] for record in validation}:
        raise RuntimeError("SFT task IDs overlap")
    manifest_values = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {item["example_id"]: item for item in manifest_values}
    if len(manifest) != len(records):
        raise RuntimeError("The SFT image manifest has the wrong size")
    missing = [record["example_id"] for record in records if not (dataset_root / record["stored_image"]).is_file()]
    if missing:
        raise RuntimeError(f"SFT images are missing: {missing[:10]}")
    return train, validation, manifest, audit


def _configure_model(config: SFTTrainConfig, variant: str, jepa_adapter_path: Path | None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.manual_seed(config.seed)
    processor = AutoProcessor.from_pretrained(
        config.model_id, revision=config.model_revision, use_fast=False
    )
    processor.image_processor.max_pixels = config.max_pixels
    processor.image_processor.min_pixels = config.min_pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.config.use_cache = False

    vision_lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=r".*blocks\.\d+\.attn\.(qkv|proj)$",
    )
    if variant == "model2":
        torch.manual_seed(config.seed)
        model.model.visual = get_peft_model(
            model.model.visual, vision_lora, adapter_name="vision_sft"
        )
        source_adapter_sha256 = None
    elif variant == "model4":
        if jepa_adapter_path is None or not (jepa_adapter_path / "adapter_model.safetensors").is_file():
            raise RuntimeError("The Model 4 JEPA vision adapter is missing")
        source_adapter_sha256 = sha256_file(jepa_adapter_path / "adapter_model.safetensors")
        model.model.visual = PeftModel.from_pretrained(
            model.model.visual,
            str(jepa_adapter_path),
            adapter_name="vision_sft",
            is_trainable=True,
        )
    else:
        raise ValueError("variant must be 'model2' or 'model4'")

    torch.manual_seed(config.language_init_seed)
    language_lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.model.language_model = get_peft_model(
        model.model.language_model, language_lora, adapter_name="policy_sft"
    )
    return processor, model, source_adapter_sha256


def _parameter_groups(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    vision = [parameter for parameter in model.model.visual.parameters() if parameter.requires_grad]
    language = [
        parameter for parameter in model.model.language_model.parameters() if parameter.requires_grad
    ]
    if not vision or not language:
        raise RuntimeError("Both vision and language LoRA parameters must be trainable")
    other = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("model.visual")
        and not name.startswith("model.language_model")
    ]
    if other:
        raise RuntimeError(f"Unexpected trainable parameters: {other[:10]}")
    return vision, language


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    values = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm(2).item()) if values else 0.0


@torch.no_grad()
def evaluate_policy(
    model,
    processor,
    records: list[dict],
    manifest: dict[str, dict],
    dataset_root: Path,
    device,
    limit: int,
    config: SFTTrainConfig,
    output_path: Path,
    event: str,
    step: int,
) -> dict[str, Any]:
    model.eval()
    selected = select_evaluation_records(records, limit, config.seed)
    predictions = []
    for record in selected:
        inputs = make_generation_inputs(
            processor, record, dataset_root / record["stored_image"], device
        )
        input_length = inputs["input_ids"].shape[1]
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        raw = processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed = parse_action_prediction(raw)
        target = record["action"]
        image_info = manifest[record["example_id"]]
        score = score_action(
            parsed,
            target,
            int(image_info["width"]),
            int(image_info["height"]),
            coordinate_threshold=config.coordinate_threshold,
        )
        predictions.append(
            {
                "event": event,
                "step": step,
                "example_id": record["example_id"],
                "task_id": record["task_id"],
                "system": record["system"],
                "domain": record["domain"],
                "raw_response": raw,
                "parsed_action": parsed,
                "target": target,
                **score,
            }
        )
    summary = summarize_scores(predictions)
    summary.update({"event": event, "step": step})
    write_json(output_path / f"{event}_metrics.json", summary)
    prediction_path = output_path / f"{event}_predictions.jsonl"
    if prediction_path.exists():
        prediction_path.unlink()
    for prediction in predictions:
        append_jsonl(prediction_path, prediction)
    model.train()
    return summary


def _save_adapters(model, output_path: Path) -> None:
    model.model.visual.save_pretrained(
        output_path / "vision_lora", selected_adapters=["vision_sft"], safe_serialization=True
    )
    model.model.language_model.save_pretrained(
        output_path / "language_lora",
        selected_adapters=["policy_sft"],
        safe_serialization=True,
    )


def train_policy_sft(
    dataset_root: str | Path,
    output_dir: str | Path,
    variant: str,
    jepa_adapter_path: str | Path | None = None,
    config: SFTTrainConfig | None = None,
    persist_outputs=lambda: None,
) -> dict[str, Any]:
    config = config or SFTTrainConfig()
    dataset_root = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jepa_path = Path(jepa_adapter_path) if jepa_adapter_path else None
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, validation, manifest, dataset_audit = _load_dataset(dataset_root)
    processor, model, source_adapter_sha256 = _configure_model(config, variant, jepa_path)
    model.to(device)
    vision_parameters, language_parameters = _parameter_groups(model)
    trainable = vision_parameters + language_parameters
    initialization = {
        "variant": variant,
        "base_model": config.model_id,
        "base_revision": config.model_revision,
        "base_trainable_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ),
        "vision_lora_parameters": sum(parameter.numel() for parameter in vision_parameters),
        "language_lora_parameters": sum(parameter.numel() for parameter in language_parameters),
        "source_jepa_adapter_sha256": source_adapter_sha256,
        "initial_vision_lora_sha256": tensor_state_sha256(
            (name, value)
            for name, value in model.model.visual.named_parameters()
            if value.requires_grad
        ),
        "initial_language_lora_sha256": tensor_state_sha256(
            (name, value)
            for name, value in model.model.language_model.named_parameters()
            if value.requires_grad
        ),
    }
    if initialization["base_trainable_parameters"] != 0:
        raise RuntimeError("A non-LoRA base parameter is trainable")
    write_json(output_path / "config.json", asdict(config))
    write_json(output_path / "runtime_audit.json", runtime_audit())
    write_json(output_path / "dataset_audit.json", dataset_audit)
    write_json(output_path / "initialization_audit.json", initialization)
    write_json(
        output_path / "validation_example_ids.json",
        [record["example_id"] for record in validation],
    )

    initial_evaluation = evaluate_policy(
        model,
        processor,
        validation,
        manifest,
        dataset_root,
        device,
        min(config.initial_evaluation_examples, len(validation)),
        config,
        output_path,
        "initial_validation",
        0,
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def schedule(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            config.max_steps - config.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    order = list(range(len(train)))
    random.Random(config.seed).shuffle(order)
    write_json(
        output_path / "training_order.json",
        [
            {
                "position": position,
                "example_id": train[index]["example_id"],
                "task_id": train[index]["task_id"],
                "action_kind": train[index]["action_kind"],
            }
            for position, index in enumerate(order)
        ],
    )
    micro_step = 0
    optimizer_step = 0
    losses = []
    recent_losses = []
    recent_action_losses: dict[str, list[float]] = {}
    optimizer.zero_grad(set_to_none=True)
    model.train()
    stop_reason = "maximum_steps_completed"
    while optimizer_step < config.max_steps:
        if time.perf_counter() - started >= config.max_runtime_seconds:
            stop_reason = "runtime_limit"
            break
        record = train[order[micro_step % len(order)]]
        inputs = make_training_inputs(
            processor, record, dataset_root / record["stored_image"], device
        )
        output = model(**inputs)
        raw_loss = output.loss
        (raw_loss / config.gradient_accumulation_steps).backward()
        loss_value = float(raw_loss.detach().item())
        losses.append(loss_value)
        recent_losses.append(loss_value)
        recent_action_losses.setdefault(record["action_kind"], []).append(loss_value)
        micro_step += 1
        if micro_step % config.gradient_accumulation_steps:
            continue

        vision_gradient_norm = _gradient_norm(vision_parameters)
        language_gradient_norm = _gradient_norm(language_parameters)
        total_gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        if optimizer_step % config.log_every == 0 or optimizer_step == 1:
            event = {
                "step": optimizer_step,
                "micro_step": micro_step,
                "mean_loss": sum(recent_losses) / len(recent_losses),
                "mean_loss_by_action": {
                    action: sum(values) / len(values)
                    for action, values in sorted(recent_action_losses.items())
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
                "vision_lora_gradient_norm": vision_gradient_norm,
                "language_lora_gradient_norm": language_gradient_norm,
                "total_gradient_norm_before_clip": total_gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_memory_gib": (
                    torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                ),
            }
            append_jsonl(output_path / "train_metrics.jsonl", event)
            print(json.dumps(event, sort_keys=True), flush=True)
            recent_losses.clear()
            recent_action_losses.clear()
        if config.monitor_every and optimizer_step % config.monitor_every == 0:
            evaluate_policy(
                model,
                processor,
                validation,
                manifest,
                dataset_root,
                device,
                min(config.monitor_examples, len(validation)),
                config,
                output_path,
                f"monitor_step_{optimizer_step:06d}",
                optimizer_step,
            )
            _save_adapters(
                model, output_path / f"checkpoint_step_{optimizer_step:06d}"
            )
            persist_outputs()

    final_evaluation = evaluate_policy(
        model,
        processor,
        validation,
        manifest,
        dataset_root,
        device,
        min(config.final_evaluation_examples, len(validation)),
        config,
        output_path,
        "final_validation",
        optimizer_step,
    )
    _save_adapters(model, output_path)
    final_vision_lora_sha256 = trainable_state_sha256(model.model.visual)
    final_language_lora_sha256 = trainable_state_sha256(model.model.language_model)
    elapsed = time.perf_counter() - started
    estimated_cost = elapsed * (
        config.gpu_cost_per_second
        + config.cpu_cores * config.cpu_cost_per_core_second
        + config.memory_gib * config.memory_cost_per_gib_second
    )
    result = {
        "variant": variant,
        "steps": optimizer_step,
        "micro_steps": micro_step,
        "stop_reason": stop_reason,
        "mean_train_loss": sum(losses) / max(len(losses), 1),
        "first_20_loss": sum(losses[:20]) / max(min(len(losses), 20), 1),
        "last_20_loss": sum(losses[-20:]) / max(min(len(losses), 20), 1),
        "initial_validation": initial_evaluation,
        "final_validation": final_evaluation,
        "elapsed_seconds": elapsed,
        "estimated_modal_cost_usd": estimated_cost,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        ),
        "initialization_audit": initialization,
        "final_vision_lora_sha256": final_vision_lora_sha256,
        "final_language_lora_sha256": final_language_lora_sha256,
        "vision_lora_changed": (
            final_vision_lora_sha256 != initialization["initial_vision_lora_sha256"]
        ),
        "language_lora_changed": (
            final_language_lora_sha256 != initialization["initial_language_lora_sha256"]
        ),
    }
    if estimated_cost > config.approved_cost_limit_usd:
        raise RuntimeError(
            f"Estimated cost ${estimated_cost:.2f} exceeded ${config.approved_cost_limit_usd:.2f}"
        )
    write_json(output_path / "final_metrics.json", result)
    persist_outputs()
    return result
