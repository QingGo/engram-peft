import os
import sys

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Import set_submodule polyfill or logic if needed,
# but for init check we mainly care about config-model mapping.


def patch_transformers_mappings():
    """Apply the same defensive patches used in examples."""
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        if "mistral3" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mistral3"] = "Ministral3ForCausalLM"
    except ImportError:
        pass


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3.5-4B",
        "mistralai/Ministral-3-3B-Instruct-2512",
        "google/gemma-4-E2B-it",
    ],
)
def test_model_architecture_init(model_id):
    """
    Verify that the model architecture can be initialized from config
    without crashing due to missing attributes.
    """
    patch_transformers_mappings()

    print(f"\nVerifying architecture for: {model_id}")

    # 1. Load Tokenizer (needed for some attribute derivation)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load tokenizer for {model_id}: {e}")
        return

    # 2. Load Config
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        pytest.fail(f"Failed to load config for {model_id}: {e}")

    # 3. Apply defensive attribute patches (same as in examples)
    # Handle nested text_config which is common in new multimodal or complex architectures
    # We sync ALL attributes from text_config to the top-level config to ensure
    # compatibility with both model loading (path) and initialization (attributes).
    if hasattr(config, "text_config"):
        text_config_dict = config.text_config.to_dict()
        for attr, value in text_config_dict.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, value)

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = len(tokenizer)

    # 4. Attempt empty initialization
    # We use 'meta' device if available to avoid any memory usage,
    # or CPU with low_cpu_mem_usage.
    print(f"Attempting model __init__ for {model_id}...")
    try:
        # We use from_config to skip loading actual weights
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        assert model is not None
        print(f"Successfully initialized {model_id} architecture on meta device.")

    except Exception as e:
        pytest.fail(f"Architecture initialization FAILED for {model_id}: {e}")


def patch_peft_for_gemma4():
    """Apply monkey patch to PEFT for Gemma-4 support."""
    try:
        import peft.tuners.lora.model as lora_model
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

        original_method = lora_model.LoraModel._create_new_module
        # Avoid double patching
        if "patched_create_new_module" in str(original_method):
            return

        @staticmethod
        def patched_create_new_module(*args, **kwargs):
            target = None
            if len(args) >= 3:
                target = args[2]
                if isinstance(target, Gemma4ClippableLinear):
                    new_args = list(args)
                    new_args[2] = target.linear
                    return original_method(*new_args, **kwargs)
            elif "target" in kwargs:
                target = kwargs["target"]
                if isinstance(target, Gemma4ClippableLinear):
                    kwargs["target"] = target.linear
                    return original_method(*args, **kwargs)
            return original_method(*args, **kwargs)

        lora_model.LoraModel._create_new_module = patched_create_new_module
        print("Applied PEFT patch for Gemma-4")
    except ImportError:
        pass


@pytest.mark.parametrize(
    "model_id, target_modules",
    [
        ("Qwen/Qwen3.5-4B", ["q_proj", "v_proj"]),
        ("mistralai/Ministral-3-3B-Instruct-2512", ["q_proj", "v_proj"]),
        ("google/gemma-4-E2B-it", ["q_proj", "v_proj"]),
    ],
)
def test_lora_injection_on_architecture(model_id, target_modules):
    """
    Verify that LoRA can be injected into the architecture without crashing.
    """
    patch_transformers_mappings()
    if "gemma-4" in model_id.lower():
        patch_peft_for_gemma4()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Apply patches (sync attributes from text_config)
    if hasattr(config, "text_config"):
        text_config_dict = config.text_config.to_dict()
        for attr, value in text_config_dict.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, value)

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = len(tokenizer)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        lora_config = LoraConfig(
            r=8, target_modules=target_modules, task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        assert peft_model is not None
        print(f"Successfully injected LoRA into {model_id}")
