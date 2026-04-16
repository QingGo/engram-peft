import pytest
import torch
from transformers import AutoConfig, AutoModel, LlamaConfig, LlamaModel

from engram_peft import EngramConfig, get_engram_model


def test_architecture_discovery_on_meta_device() -> None:
    """
    Smoke test to verify that architecture discovery and injection
    works on the 'meta' device (CPU/local) without requiring real weights or GPU.
    """
    model_id = "gpt2"

    # 1. Load config only
    hf_config = AutoConfig.from_pretrained(model_id)

    # 2. Instantiate skeletal model on meta device (0 memory/CPU)
    with torch.device("meta"):
        model = AutoModel.from_config(hf_config)

    # 3. Attempt Engram injection
    engram_config = EngramConfig(target_layers=[0], tokenizer_name_or_path="gpt2")

    # This triggers ArchitectureResolver on the meta-model
    engram_model = get_engram_model(model, engram_config)

    # 4. Verify basic structural injection
    assert engram_model.config.hidden_size == hf_config.n_embd
    assert "0" in engram_model.engram_layers

    # Check if forward hooks are attached (conceptual check)
    # The meta device allows us to verify the graph without running it
    print("\n[Engram-PEFT] Meta-device smoke test passed for GPT2.")


def test_llama_discovery_on_meta_device() -> None:
    """Verify discovery logic for Llama-like architectures on meta device."""
    # Using tiny llama config for speed (Using dict for better Pyright compatibility)
    config = LlamaConfig.from_dict(
        {
            "hidden_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "vocab_size": 32000,
        }
    )

    with torch.device("meta"):
        model = LlamaModel(config)

    engram_config = EngramConfig(
        target_layers=[0, 1], enable_tokenizer_compression=False
    )
    engram_model = get_engram_model(model, engram_config)

    assert engram_model.config.hidden_size == 512
    assert len(engram_model.engram_layers) == 2
    print("[Engram-PEFT] Meta-device smoke test passed for Llama (skeleton).")
