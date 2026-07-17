# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none
"""
Gemma-4 (Effective 2B/4B) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA with Engram-PEFT for Google's Gemma
family, focusing on extreme on-device efficiency and knowledge storage.

For using unsloth,run "pip install unsloth" first!

Usage:
    uv run python examples/gemma4_engram_lora.py --model_id unsloth/gemma-4-E4B-it --max_steps 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Iterable
from typing import Any

from dotenv import load_dotenv

# Add the project root to sys.path to allow absolute imports from the 'examples' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import Dataset, load_dataset
from unsloth import FastModel
from peft import PeftModel
from transformers import (
    PreTrainedTokenizerBase,
    set_seed,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.types import SafeTrainingArguments, SizedEncoding
from engram_peft.utils import (
    apply_peft_patches,
    get_optimal_precision_config,
)
from engram_peft.utils.compat import (
    create_safe_training_args,
    wash_model,
    wash_tokenizer,
)
from examples.benchmarks.data_utils import get_dataset_template

# Try to import optional visualization components
try:
    from examples.benchmarks.persistence import BenchmarkResult
    from examples.benchmarks.plotting import plot_benchmark_comparison
except ImportError:
    BenchmarkResult = None
    plot_benchmark_comparison = None

load_dotenv()

# Defaults
DEFAULT_MODEL = "./gemma-4-E4B-it" # Changed to match Unsloth standard
OUTPUT_DIR = "outputs/gemma4_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, 
    data_size: int,
    max_length: int = 512, 
    eval_ratio: float = 0.05
) -> dict[str, Dataset]:
    """Load and format the Alpaca dataset using Gemma Instruct template."""
    dataset = load_dataset("parquet",data_files="test.parquet", split="train[600:]")
    # Aggressively cap dataset for fast example execution
    # dataset = dataset.select(range(min(600, len(dataset))))
    assert isinstance(dataset, Dataset)
    IGNORE_INDEX = -100
    MAX_LENGTH = args.max_seq_length  # 根据需要调整
    response_template = "<|turn>model\n"
    instruction_template = "<|turn>user\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    def find_all_subsequence_positions(sequence, subsequence):
        """找到 sequence 中所有 subsequence 出现的起始位置"""
        positions = []
        sub_len = len(subsequence)
        for i in range(len(sequence) - sub_len + 1):
            if sequence[i:i + sub_len] == subsequence:
                positions.append(i)
        return positions
    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        conversations = []
        conversations.append({"role": "user", "content": example["instruction"]})
        if example.get("input"):
            conversations[-1]["content"] += f"\nInput:{example['input']}"
        conversations.append({"role": "model", "content": example["output"]})
        full_text = tokenizer.apply_chat_template(
            conversations, 
            tokenize=False, 
            add_generation_prompt=True,
            padding="max_length",
            truncation=True,
            # enable_thinking=False
        )
        return {"text": full_text}


    def tokenize_and_mask(examples):
        texts = examples["text"]

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for text in texts:
            # Tokenize
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                add_special_tokens=True,  #add bos
                # padding=True,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 初始化 labels 为全部 IGNORE（不计算 loss）
            labels = [IGNORE_INDEX] * len(input_ids)

            # 找到所有 response_template 出现的位置
            response_positions = find_all_subsequence_positions(input_ids, response_template_ids)
            # 找到所有 instruction_template 出现的位置（用于确定 model 回复的结束边界）
            instruction_positions = find_all_subsequence_positions(input_ids, instruction_template_ids)

            for resp_start in response_positions:
                # model 回复内容从 response_template 之后开始
                content_start = resp_start + len(response_template_ids)

                # 找到该 response 之后最近的 instruction_template 位置作为结束边界
                # 如果没有后续的 instruction，则到序列末尾
                content_end = len(input_ids)
                for inst_pos in instruction_positions:
                    if inst_pos > resp_start:
                        content_end = inst_pos
                        break

                # 将 model 回复部分的 labels 设为对应的 input_ids（计算 loss）
                for i in range(content_start, content_end):
                    labels[i] = input_ids[i]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    tokenized_ds = dataset.map(format_alpaca, remove_columns=dataset.column_names)
    tokenized_ds = tokenized_ds.map(tokenize_and_mask, batched=True, remove_columns=tokenized_ds.column_names)
    # Split into train and eval
    if eval_ratio > 0:
        split_ds = tokenized_ds.train_test_split(test_size=eval_ratio, seed=SEED)
        return {"train": split_ds["train"], "eval": split_ds["test"]}
    return {"train": tokenized_ds}


def run_example(args: argparse.Namespace) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup file logging
    log_file = os.path.join(OUTPUT_DIR, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    logging.info(f"Starting Gemma Engram+LoRA example with model: {args.model_id}")

    if args.load_in_4bit and args.load_in_8bit:
        logging.warning("Both 4-bit and 8-bit loading enabled; defaulting to 4-bit.")
        args.load_in_8bit = False
    if not args.load_in_4bit and not args.load_in_8bit:
        logging.info("No quantization specified; loading in 16-bit (if supported).")

    # 0. Apply PEFT deep patches to support Gemma-4 custom layers
    apply_peft_patches()

    print(f"\n>>> Initializing Model: {args.model_id}")

    # 1. Load Processor & Model using Unsloth
    print(f"Loading tokenizer & model: {args.model_id}")

    model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_id if args.model_id else DEFAULT_MODEL,
    dtype = None, # None for auto detection
    max_seq_length = args.max_seq_length, # Choose any for long context!
    load_in_4bit = args.load_in_4bit,
    load_in_8bit = args.load_in_8bit,
    load_in_16bit = not args.load_in_4bit and not args.load_in_8bit,
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )   
    
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer) # Safe unwrap
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Apply LoRA
    print("Applying LoRA...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # Should leave on always!

        r = args.lora_r,           # Larger = higher accuracy, but might overfit
        lora_alpha = args.lora_alpha,  # Recommended alpha == r at least
        lora_dropout = args.lora_dropout,
        bias = "none",
        random_state = SEED,
    )

# ===============================================

    # 3. Apply Engram-PEFT
    print("Applying Engram-PEFT...")
    num_layers = getattr(model.config, "num_hidden_layers", 18)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        engram_dim=args.engram_dim,
        target_layers=target_layers,
    )

    model = get_engram_model(
        model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="preserve_trainable",
    )
    model.print_trainable_parameters()


    # 4. Prepare Dataset
    print(f"Preparing Alpaca subset (size: {args.data_size})...")
    datasets = prepare_alpaca_dataset(tokenizer, data_size=args.data_size)

    # 5. Training Arguments
    precision_config = get_optimal_precision_config()
    training_args_dict: SafeTrainingArguments = {
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": "steps" if "eval" in datasets else "no",
        "eval_steps": args.eval_steps,
        "remove_unused_columns": True,
        "report_to": "none",
        "bf16": precision_config.get("bf16", False),
        "fp16": precision_config.get("fp16", False),
    }

    training_args = create_safe_training_args(training_args_dict)

    # 6. Prepare Trainer
    print("Preparing Trainer...")
    data_collator = EngramDataCollator(
        tokenizer=wash_tokenizer(tokenizer), config=model.config
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        data_collator=data_collator,
    )

    print("\n>>> Initial (Zero-shot) Evaluation")
    initial_metrics = trainer.evaluate()
    print(f"Initial Eval Loss: {initial_metrics.get('eval_loss', 0.0):.4f}")
    print("\n>>> Starting combined LoRA + Engram training...")
    trainer.train()

    # 5.1 Plot Results
    if BenchmarkResult is not None:
        print("\n>>> Generating Training Plots")
        main_res = BenchmarkResult(
            method="gemma_lora_engram",
            metrics={
                "eval_loss": initial_metrics.get("eval_loss", 0.0),
                "log_history": trainer.state.log_history,
            },
            params=vars(args),
        )
        main_res.save(OUTPUT_DIR)

        base_res = BenchmarkResult(
            method="base",
            metrics={"eval_loss": initial_metrics.get("eval_loss", 0.0)},
            params=vars(args),
        )

        if plot_benchmark_comparison is not None:
            plot_benchmark_comparison(
                [main_res, base_res],
                output_path=os.path.join(OUTPUT_DIR, "training_curve.png"),
            )
    else:
        print("\n>>> Skipping plots (optional plotting tools not available)")

    # 6. Saving
    print(f"Saving combined adapters to {OUTPUT_DIR}")
    save_fn = getattr(model.base_model, "save_pretrained", None)
    if save_fn is not None:
        print("Saving LoRA adapters...")
        save_fn(OUTPUT_DIR)
    else:
        print("Warning: model.base_model does not have save_pretrained; LoRA adapter saving skipped.")

    print("Saving Engram adapters...")
    model.save_pretrained_engram(OUTPUT_DIR)

    # 7. Inference Demo (Original Model)
    model.gradient_checkpointing_disable()
    FastModel.for_inference(model) # FIX: Use Unsloth inference mode for speed
    
    print("\n>>> Inference Demo (Original Model)")
    messages = [{"role": "user", "content": "Tell me a short fact about the moon."}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
    )
    
    target_device = getattr(model.base_model, "device", "cuda")
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)
    print(f"Prompt: {prompt}")
    
    set_seed(SEED + 1)
    with torch.no_grad():
        gen_model = wash_model(model)
        output = gen_model.generate(
            **inputs,
            max_new_tokens=100,
            max_length=None,
            do_sample=False,
            stop_strings=["<end_of_turn>"], # FIX: Consistent stop string
            tokenizer=tokenizer,
        )
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    
    # 8. Reload and Verify
    print("\n>>> Reloading Model for Verification")
    model.unload_engram()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading a clean, fresh base model for verification...")
    fresh_base_model, _ = FastModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        load_in_16bit=not args.load_in_4bit and not args.load_in_8bit,
    )
    
    print("Loading saved LoRA adapters onto the clean base model...")
    reloaded_peft = PeftModel.from_pretrained(
        fresh_base_model, OUTPUT_DIR, trust_remote_code=True
    )
    
    try:
        print("Loading saved LoRA adapters onto the clean base model...")
        # FIX: Removed redundant PeftModel.from_pretrained block
        reloaded_base_model = PeftModel.from_pretrained(
            fresh_base_model, OUTPUT_DIR, trust_remote_code=True
        )
        print("Re-wrapping with Engram...")
        reloaded_model = EngramModel.from_pretrained(
            reloaded_peft, OUTPUT_DIR, tokenizer=wash_tokenizer(tokenizer)
        )
        FastModel.for_inference(reloaded_model)

        print("Inference with Fully Reloaded Model (LoRA + Engram):")
        set_seed(SEED + 2)
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs,
                max_new_tokens=100,
                max_length=None,
                do_sample=False,
                stop_strings=["<end_of_turn>"],
                tokenizer=tokenizer,
            )
        reloaded_resp = tokenizer.decode(reloaded_output[0], skip_special_tokens=True)
        print(f"Response: {reloaded_resp}")
    except Exception as e:
        print(f"Reloading failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma LoRA + Engram Example")
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL, help="Model ID on HuggingFace"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum training steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--data_size", type=int, default=600, help="Subset of Alpaca to use"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA dropout")
    parser.add_argument(
        "--engram_dim", type=int, default=1024, help="Engram embedding dimension"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load base model in 4-bit precision"
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load base model in 8-bit precision"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Logging frequency"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Maximum sequence length"
    )
    args = parser.parse_args()

    run_example(args)