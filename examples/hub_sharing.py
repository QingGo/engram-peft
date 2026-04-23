"""
Engram-PEFT Hugging Face Hub Sharing Example.

This script demonstrates how to:
1. Push a trained Engram adapter to the Hugging Face Hub.
2. Reload an Engram adapter directly from the Hub using its Repository ID.

Prerequisites:
    1. Login to your account: `huggingface-cli login`
    2. (Optional) Create a repository on the Hugging Face website,
       though `push_to_hub` will create it for you if it doesn't exist.

Usage:
    ```bash
    uv run python examples/hub_sharing.py --repo_id "your-username/tinyllama-engram-test"
    ```
"""

import argparse
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from engram_peft import EngramConfig, EngramModel, get_engram_model


def main():
    parser = argparse.ArgumentParser(description="Engram Hub Sharing Demo")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="Base model name",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/my-adapter')",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer & base model: {args.model_name}")
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(args.model_name)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We use CPU or half-precision to save memory in this demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ),
    )

    # 1. Initialize Engram Model
    print("Initializing Engram layers...")
    config = EngramConfig(
        target_layers=[2],  # Just one layer for demonstration
        engram_vocab_size_per_ngram=[128000],
        hidden_size=base_model.config.hidden_size,
        tokenizer_name_or_path=args.model_name,
    )

    model = get_engram_model(
        base_model,
        config,
        tokenizer,
        train_mode="engram_only",
    )

    # 2. Push to Hub
    # IMPORTANT: Ensure you are logged in via `huggingface-cli login`
    # The adapter weights and config.json will be uploaded.
    print(f"\n>>> Pushing Engram adapter to Hub: {args.repo_id}")
    print("Note: This requires an active Hugging Face token with write permissions.")

    try:
        model.push_to_hub(args.repo_id)
        print(f"Successfully pushed to https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        print("\nSkipping the rest of the demo as upload failed.")
        return

    # 3. Reload from Hub
    print(f"\n>>> Reloading Engram adapter from Hub ID: {args.repo_id}")

    # We can use EngramModel.from_pretrained with the Hub ID directly.
    # It will download the files to your cache and then load them.
    reloaded_model = EngramModel.from_pretrained(
        base_model, args.repo_id, tokenizer=tokenizer
    )

    print("Successfully reloaded model from Hub!")
    print(f"Active Adapter: {reloaded_model.active_adapter}")
    print(f"Engram Layers: {list(reloaded_model.engram_layers.keys())}")

    # 4. Cleanup
    reloaded_model.unload_engram()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
