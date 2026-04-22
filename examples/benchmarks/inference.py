import gc
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from engram_peft import EngramLayer, EngramModel


def demo_base_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, inputs: dict[str, Any]
) -> None:
    print("\nGenerating with Base Model (Zero-shot)...")
    with torch.no_grad():
        output = cast(Any, model).generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (Base):   {tokenizer.decode(output[0], skip_special_tokens=True)}")


def demo_lora(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, Any],
    path: str = "outputs/benchmarks/lora_weights",
) -> None:
    print(f"Generating with LoRA ({path})...")
    model = PeftModel.from_pretrained(base_model, path)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (LoRA):   {tokenizer.decode(out[0], skip_special_tokens=True)}")
    model.unload()


def demo_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, Any],
    path: str = "outputs/benchmarks/engram_weights",
) -> None:
    print(f"Generating with Engram ({path})...")
    model = EngramModel.from_pretrained(base_model, path, tokenizer=tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (Engram): {tokenizer.decode(out[0], skip_special_tokens=True)}")

    # Optional gating visualization
    print("Gating Activation (Mean per branch):")
    for layer_id in model.config.target_layers:
        engram_layer = cast("EngramLayer", model.engram_layers[str(layer_id)])
        gate = engram_layer.gating.last_gate
        if gate is not None:
            mean_gates = gate.mean(dim=(0, 1, 3)).cpu().tolist()
            gate_str = " | ".join([f"B{i}: {g:.3f}" for i, g in enumerate(mean_gates)])
            print(f"  Layer {layer_id}: {gate_str}")
    model.unload_engram()


def demo_lora_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, Any],
    path: str = "outputs/benchmarks/lora_engram_weights",
) -> None:
    print(f"Generating with LoRA + Engram ({path})...")
    # Load LoRA first
    lora_model = PeftModel.from_pretrained(base_model, path)
    # Load Engram wrapper
    combined_model = EngramModel.from_pretrained(
        cast("PreTrainedModel", lora_model), path, tokenizer=tokenizer
    )
    with torch.no_grad():
        out = combined_model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (Combined): {tokenizer.decode(out[0], skip_special_tokens=True)}")
    combined_model.unload_engram()
    lora_model.unload()


def demo_full_finetune(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, Any],
    path: str = "outputs/benchmarks/full_ft_only_weights",
) -> None:
    print(f"Generating with Full FT ({path})...")
    ft_model = cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            path, dtype=base_model.dtype, device_map="auto"
        ),
    )
    with torch.no_grad():
        out = cast(Any, ft_model).generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (Full FT): {tokenizer.decode(out[0], skip_special_tokens=True)}")
    del ft_model
    gc.collect()


def demo_full_finetune_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, Any],
    path: str = "outputs/benchmarks/full_ft_engram_weights",
) -> None:
    print(f"Generating with Full FT + Engram ({path})...")
    # Load finetuned base model from subfolder
    sub_path = f"{path}/base_model"
    ft_base_model = cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            sub_path, dtype=base_model.dtype, device_map="auto"
        ),
    )
    # Load Engram wrapper
    model = EngramModel.from_pretrained(ft_base_model, path, tokenizer=tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=40, max_length=None, do_sample=False
        )
    print(f"Output (FT+Engram): {tokenizer.decode(out[0], skip_special_tokens=True)}")
    model.unload_engram()
    del ft_base_model
    gc.collect()


def run_inference_demo(
    engine: Any, prompt: str = "Once upon a time, there was a little robot named"
) -> None:
    """Runs a sequential inference demo for all methods trained in this session."""
    print("\n" + "=" * 20 + " Phase: Inference Demo " + "=" * 20)

    tokenizer = engine.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(
        engine.base_model.device if engine.base_model else "cuda"
    )

    print(f"\nPrompt: {prompt}")

    # 1. Base Model Inference
    base_model = engine.get_fresh_model()
    demo_base_model(base_model, tokenizer, inputs)

    for method_spec in engine.results.keys():
        if method_spec == "base":
            continue

        method_name = method_spec.split(":")[0]
        print(f"\n>>> Demo: {method_spec}")

        try:
            if method_name == "lora":
                demo_lora(base_model, tokenizer, inputs)
            elif method_name == "engram":
                demo_engram(base_model, tokenizer, inputs)
            elif method_name == "lora_engram":
                demo_lora_engram(base_model, tokenizer, inputs)
            elif method_name == "full_finetune":
                demo_full_finetune(base_model, tokenizer, inputs)
            elif method_name == "full_finetune_engram":
                demo_full_finetune_engram(base_model, tokenizer, inputs)
        except Exception as e:
            print(f"Error during inference demo for {method_spec}: {e}")

    print("\nInference demo completed.")
