from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    import aiohttp.connector
    import aiohttp.resolver

    aiohttp.connector.DefaultResolver = aiohttp.resolver.ThreadedResolver
    aiohttp.resolver.DefaultResolver = aiohttp.resolver.ThreadedResolver

import modal


APP_NAME = "cua-jepa-sft-transfer"
SFT_VOLUME_NAME = "cua-jepa-sft-v1"
TRAINING_VOLUME_NAME = "cua-jepa-training-v1"
HF_CACHE_VOLUME_NAME = "hf-cache"
DATASET_ROOT = "/sft/agentnet-v1"
JEPA_ADAPTER_PATH = (
    "/training/model4-model4_full-seed20260809-20260810T052900Z/"
    "qwen_vision_online_lora/online"
)

if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "accelerate>=1.13,<2",
            "numpy>=2,<3",
            "peft>=0.19,<1",
            "pillow>=10,<13",
            "safetensors>=0.5,<1",
            "torch>=2.8,<3",
            "torchvision>=0.23,<1",
            "transformers>=4.57,<5",
        )
        .add_local_python_source("cua_jepa", copy=True)
    )
else:
    image = modal.Image.debian_slim()

app = modal.App(APP_NAME)
sft_volume = modal.Volume.from_name(SFT_VOLUME_NAME)
training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=30 * 60,
    scaledown_window=60,
    volumes={
        "/sft": sft_volume,
        "/training": training_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def check_sft_model_setup(
    variant: str = "model4",
    max_pixels: int = 1_048_576,
    model3_adapter_path: str = "",
) -> dict:
    import torch
    from PIL import Image

    from cua_jepa.train_sft import (
        SFTTrainConfig,
        _configure_model,
        _gradient_norm,
        _parameter_groups,
        make_generation_inputs,
        make_training_inputs,
    )

    config = SFTTrainConfig(max_pixels=max_pixels, min_pixels=65_536)
    jepa_path = None
    if variant == "model4":
        jepa_path = Path(JEPA_ADAPTER_PATH)
    elif variant == "model3":
        if not model3_adapter_path:
            raise ValueError("Model 3 setup requires --model3-adapter-path")
        jepa_path = Path(model3_adapter_path)
    processor, model, source_sha256 = _configure_model(config, variant, jepa_path)
    model.to("cuda").train()
    vision_parameters, language_parameters = _parameter_groups(model)
    selection_path = Path(DATASET_ROOT) / "selection.jsonl"
    record = None
    image_path = None
    if selection_path.is_file():
        for line in selection_path.read_text(encoding="utf-8").splitlines():
            candidate = json.loads(line)
            stored_name = hashlib.sha256(candidate["example_id"].encode()).hexdigest()[:24]
            candidate_path = Path(DATASET_ROOT) / "images" / f"{stored_name}.webp"
            if candidate_path.is_file():
                record = candidate
                image_path = candidate_path
                break
    used_real_example = record is not None
    if record is None or image_path is None:
        image_path = Path("/tmp/sft-setup.webp")
        Image.new("RGB", (1920, 1080), "white").save(image_path)
        record = {
            "example_id": "synthetic-setup",
            "instruction": "Click the center of the screen.",
            "history": [],
            "target": '{"action":"click","x":0.5,"y":0.5}',
        }
    inputs = make_training_inputs(processor, record, image_path, torch.device("cuda"))
    output = model(**inputs)
    output.loss.backward()
    result = {
        "variant": variant,
        "max_pixels": max_pixels,
        "used_real_example": used_real_example,
        "example_id": record["example_id"],
        "loss": float(output.loss.detach().item()),
        "vision_lora_parameters": sum(value.numel() for value in vision_parameters),
        "language_lora_parameters": sum(value.numel() for value in language_parameters),
        "vision_gradient_norm": _gradient_norm(vision_parameters),
        "language_gradient_norm": _gradient_norm(language_parameters),
        "source_jepa_adapter_sha256": source_sha256,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    model.zero_grad(set_to_none=True)
    del output, inputs
    model.eval()
    generation_inputs = make_generation_inputs(
        processor, record, image_path, torch.device("cuda")
    )
    input_length = generation_inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
    result["generated_text"] = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result["peak_cuda_memory_gib_after_generation"] = (
        torch.cuda.max_memory_allocated() / 2**30
    )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24_576,
    timeout=4 * 60 * 60,
    scaledown_window=60,
    volumes={
        "/sft": sft_volume,
        "/training": training_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def run_sft_transfer(
    variant: str,
    mode: str,
    seed: int = 20260810,
    model3_adapter_path: str = "",
) -> dict:
    from cua_jepa.train_sft import SFTTrainConfig, train_policy_sft

    if variant not in {"model2", "model3", "model4"}:
        raise ValueError("variant must be 'model2', 'model3', or 'model4'")
    if variant == "model3" and not model3_adapter_path:
        raise ValueError("Model 3 SFT requires --model3-adapter-path")
    config = SFTTrainConfig(seed=seed)
    if mode == "smoke":
        config.max_steps = 2
        config.gradient_accumulation_steps = 1
        config.warmup_steps = 1
        config.log_every = 1
        config.monitor_every = 0
        config.monitor_examples = 4
        config.initial_evaluation_examples = 4
        config.final_evaluation_examples = 4
        config.max_runtime_seconds = 25 * 60
        config.approved_cost_limit_usd = 3.0
    elif mode == "pilot":
        config.max_steps = 25
        config.gradient_accumulation_steps = 2
        config.warmup_steps = 3
        config.log_every = 5
        config.monitor_every = 0
        config.monitor_examples = 16
        config.initial_evaluation_examples = 32
        config.final_evaluation_examples = 32
        config.max_runtime_seconds = 60 * 60
        config.approved_cost_limit_usd = 6.0
    elif mode == "full":
        pass
    else:
        raise ValueError("mode must be 'smoke', 'pilot', or 'full'")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"sft-{variant}-{mode}-seed{seed}-{timestamp}"
    output_path = Path("/training") / run_id
    result = train_policy_sft(
        dataset_root=DATASET_ROOT,
        output_dir=output_path,
        variant=variant,
        jepa_adapter_path=(
            JEPA_ADAPTER_PATH
            if variant == "model4"
            else model3_adapter_path if variant == "model3" else None
        ),
        config=config,
        persist_outputs=training_volume.commit,
    )
    result["run_id"] = run_id
    result["output_path"] = str(output_path)
    return result


@app.local_entrypoint()
def main(
    variant: str = "model4",
    mode: str = "smoke",
    seed: int = 20260810,
    max_pixels: int = 1_048_576,
    model3_adapter_path: str = "",
) -> None:
    if mode == "setup":
        if variant == "both":
            raise ValueError("Setup mode requires one variant")
        result = check_sft_model_setup.remote(
            variant=variant,
            max_pixels=max_pixels,
            model3_adapter_path=model3_adapter_path,
        )
    elif variant == "both":
        calls = {
            name: run_sft_transfer.spawn(variant=name, mode=mode, seed=seed)
            for name in ("model2", "model4")
        }
        result = {name: call.get() for name, call in calls.items()}
    else:
        result = run_sft_transfer.remote(
            variant=variant,
            mode=mode,
            seed=seed,
            model3_adapter_path=model3_adapter_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
