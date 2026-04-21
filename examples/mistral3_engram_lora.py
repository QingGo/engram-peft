"""
Ministral-3 (3B Instruct) LoRA + Engram Fine-tuning Example.

This script demonstrates combining LoRA with Engram-PEFT for the Ministral-3
family of models, specifically optimized for edge devices and instruction following.

Usage:
    uv run python examples/mistral3_engram_lora.py --model_id mistralai/Ministral-3-3B-Instruct-2512 --max_steps 50
"""

import argparse
import logging
import os
import traceback
from typing import Any, cast

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from examples.benchmarks.data_utils import get_dataset_template

# Try to import safetensors
try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None  # type: ignore

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)

# Polyfill for set_submodule which is missing in some PyTorch versions
if not hasattr(nn.Module, "set_submodule"):

    def set_submodule(self: nn.Module, target: str, module: nn.Module) -> None:
        if not target:
            raise ValueError("Cannot set empty submodule")
        atoms = target.split(".")
        name = atoms.pop(-1)
        parent = self.get_submodule(".".join(atoms))
        setattr(parent, name, module)

    nn.Module.set_submodule = set_submodule  # type: ignore

# Try to import optional visualization components
try:
    from examples.benchmarks.persistence import BenchmarkResult
    from examples.benchmarks.plotting import plot_benchmark_comparison
except ImportError:
    BenchmarkResult = None  # type: ignore
    plot_benchmark_comparison = None  # type: ignore

# Defaults
DEFAULT_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"
OUTPUT_DIR = "outputs/mistral3_engram_lora"
SEED = 42

set_seed(SEED)


def prepare_alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase, max_length: int = 512, eval_ratio: float = 0.05
) -> dict[str, Dataset]:
    """Load and format the Alpaca dataset using Mistral Instruct template."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    # Aggressively cap dataset for fast example execution
    dataset = dataset.select(range(min(600, len(dataset))))
    assert isinstance(dataset, Dataset)

    template = get_dataset_template("mistral")

    def format_alpaca(example: dict[str, Any]) -> dict[str, Any]:
        prompt = template.format(
            instruction=example["instruction"],
            input=f"\n\nInput:\n{example['input']}" if example.get("input") else "",
        )
        response = f" {example['output']}</s>"
        full_text = prompt + response

        # Tokenize full text
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        labels = list(tokenized["input_ids"])

        # Mask the prompt part in labels (Padding masking handled by Collator)
        prompt_tokenized = cast(Any, tokenizer)(
            prompt, max_length=max_length, truncation=True
        )
        prompt_len = len(prompt_tokenized["input_ids"])
        for i in range(min(prompt_len, max_length)):
            labels[i] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    tokenized_ds = dataset.map(format_alpaca, remove_columns=dataset.column_names)

    # Split into train and eval
    if eval_ratio > 0:
        split_ds = tokenized_ds.train_test_split(test_size=eval_ratio, seed=SEED)
        return {"train": split_ds["train"], "eval": split_ds["test"]}
    return {"train": tokenized_ds}


def run_example(args: argparse.Namespace) -> None:
    # Defensive patch for Ministral-3 naming inconsistency in some transformers versions
    try:
        if "mistral3" not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mistral3"] = "Ministral3ForCausalLM"
    except ImportError:
        pass

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

    logging.info(f"Starting Mistral-3 Engram+LoRA example with model: {args.model_id}")

    print(f"\n>>> Initializing Ministral-3 Example with model: {args.model_id}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (load_in_4bit={args.load_in_4bit})...")

    # Pre-load config to check for existing quantization (e.g., FP8 models)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Defensive fix: ensure required attributes exist for Ministral3Model
    # Handle nested text_config which is common in new multimodal or complex architectures.
    # CRITICAL: We DO NOT unpack text_config here because that would lose the
    # 'quantization_config' which is essential for FP8 models.
    # Instead, we sync all attributes from text_config to the top-level config.
    if hasattr(config, "text_config"):
        print(
            "Detected nested text_config, synchronizing ALL attributes to top-level..."
        )
        text_config_dict = config.text_config.to_dict()
        for attr, value in text_config_dict.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, value)

        # CRITICAL MONKEY PATCH: PEFT's save_pretrained creates a fresh config instance
        # using model.config.__class__.from_pretrained(). We must ensure that NEW
        # instances also have vocab_size available on the top level.
        config_class = config.__class__
        if not hasattr(config_class, "vocab_size"):
            print(
                f"Monkey patching {config_class.__name__} to include vocab_size property..."
            )

            def get_vocab_size(self: Any) -> int | None:
                return (
                    self.text_config.vocab_size
                    if hasattr(self, "text_config")
                    else None
                )

            config_class.vocab_size = property(get_vocab_size)

    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = len(tokenizer)

    has_existing_quant = getattr(config, "quantization_config", None) is not None

    # Initialize bnb config for quantization if requested AND not already quantized
    quantization_config = None
    if has_existing_quant:
        print(
            f"Notice: Model {args.model_id} is already quantized ({config.quantization_config.get('quant_method')})."
        )
        print(
            "Ignoring bitsandbytes flags (--load_in_4bit/--load_in_8bit) to avoid conflicts."
        )
    else:
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if not quantization_config:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    print(f"Loading base model (load_in_4bit={args.load_in_4bit})...")

    # 1.1 Custom weight remapping for Mistral-3 prefix issue
    # The checkpoint has 'language_model.' prefix, but model class expects 'model.'
    state_dict = None
    if "Ministral-3-3B" in args.model_id and safe_load_file is not None:
        try:
            print("Downloading and remapping weights for Mistral-3 architecture...")
            st_path = hf_hub_download(
                repo_id=args.model_id, filename="model.safetensors"
            )
            raw_state_dict = safe_load_file(st_path)
            state_dict = {}
            for k, v in raw_state_dict.items():
                # Correct remapping: language_model. -> model.
                new_k = k
                if k.startswith("language_model."):
                    new_k = k.replace("language_model.", "model.")

                # Special case: 'model.model.' -> 'model.'
                if new_k.startswith("model.model."):
                    new_k = new_k.replace("model.model.", "model.")

                # Special case: embed_tokens and norm at root of language_model
                # language_model.embed_tokens -> model.embed_tokens
                # (Handled by the general replacement above)

                state_dict[new_k] = v

            # Handle lm_head: if missing, often tied to embed_tokens
            if (
                "lm_head.weight" not in state_dict
                and "model.embed_tokens.weight" in state_dict
            ):
                print("Note: lm_head.weight not found, will rely on weight tying.")

            print(f"Successfully remapped {len(state_dict)} tensors.")
        except Exception as e:
            print(f"Weight remapping failed: {e}")

    # 3. Load Base Model
    print("Loading base model structure on CPU to avoid initialization errors...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We load structure on CPU first because normal_kernel_cuda is missing for FP8.
    # We remove device_map="auto" to force CPU initialization.
    cpu_kwargs = model_kwargs.copy()
    cpu_kwargs["device_map"] = None
    cpu_kwargs["torch_dtype"] = torch.bfloat16  # Safer for structural creation

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, config=config, **cpu_kwargs
        )
        if state_dict:
            print("Injecting remapped state_dict on CPU...")
            base_model.load_state_dict(state_dict, strict=False)

        print(f"Moving model to {device}...")
        cast(Any, base_model).to(device)
        print("Model loaded successfully via CPU-to-GPU transition.")
    except Exception as e:
        print(
            f"Standard loading failed ({e}), falling back to manual weight injection..."
        )
        with torch.device("cpu"):
            base_model = AutoModelForCausalLM.from_config(config)
        base_model.load_state_dict(state_dict, strict=False)
        cast(Any, base_model).to(device)

        # Tie weights after loading to ensure lm_head is correct
    base_model.tie_weights()

    # 2. Apply LoRA
    print("Applying LoRA...")
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    # 3. Apply Engram-PEFT
    print("Applying Engram-PEFT...")
    # Mistral-3 layers
    num_layers = getattr(base_model.config, "num_hidden_layers", 32)
    target_layers = [num_layers // 2, num_layers - 2]

    engram_config = EngramConfig(
        engram_dim=args.engram_dim,
        target_layers=target_layers,
        k_dim=128,
        num_clusters=1024,
        backbone_freeze_steps=0,
    )
    # get_engram_model handles architecture-specific patching automatically!
    model = cast(Any, get_engram_model(model, engram_config, tokenizer=tokenizer))

    # 4. Prepare Dataset
    print("Preparing Alpaca subsets (train + eval)...")
    datasets = prepare_alpaca_dataset(tokenizer)

    # 5. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if "eval" in datasets else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        save_strategy="no",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=True,
    )

    data_collator = EngramDataCollator(tokenizer=tokenizer, config=model.config)

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
        # Prepare result objects for plotting
        main_res = BenchmarkResult(
            method="mistral3_lora_engram",
            metrics={
                "eval_loss": initial_metrics.get("eval_loss", 0.0),
                "log_history": trainer.state.log_history,
            },
            params=vars(args),
        )
        # Save structured logs
        main_res.save(OUTPUT_DIR)

        # Create a baseline result to trigger the horizontal line in plots
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
    # Explicitly save LoRA adapters first to ensure adapter_config.json exists
    if hasattr(model.base_model, "save_pretrained"):
        print("Saving LoRA adapters...")
        model.base_model.save_pretrained(OUTPUT_DIR)

    # Save Engram adapters separately for maximum robustness
    print("Saving Engram adapters...")
    model.save_pretrained_engram(OUTPUT_DIR)

    # 7. Inference Demo (Original Model)
    print("\n>>> Inference Demo (Original Model)")
    prompt = "<s>[INST] List three benefits of using parameter-efficient fine-tuning for edge models. [/INST]"
    inputs = cast(Any, tokenizer)(prompt, return_tensors="pt").to(
        model.base_model.device
    )

    print(f"Prompt: {prompt}")
    output = model.generate(
        **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
    )
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # 8. Reload and Verify
    print("\n>>> Reloading Model for Verification")
    # To fully verify, we should be able to load both LoRA and Engram back
    try:
        # 1. Load LoRA part onto a fresh base model (or reuse base_model for efficiency)
        reloaded_peft = PeftModel.from_pretrained(
            base_model, OUTPUT_DIR, trust_remote_code=True
        )

        # 2. Re-wrap with Engram using the class method which is cleaner
        reloaded_model = EngramModel.from_pretrained(
            reloaded_peft, OUTPUT_DIR, tokenizer=tokenizer
        )

        print("Inference with Fully Reloaded Model (LoRA + Engram):")
        with torch.no_grad():
            reloaded_output = reloaded_model.generate(
                **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
            )
        reloaded_resp = tokenizer.decode(reloaded_output[0], skip_special_tokens=True)
        print(f"Response: {reloaded_resp}")
    except Exception as e:
        print(f"Reloading failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ministral-3 LoRA + Engram Example")
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL, help="Model ID on HuggingFace"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum training steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--data_size", type=int, default=2000, help="Subset of Alpaca to use"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--engram_dim", type=int, default=1024, help="Engram embedding dimension"
    )
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
    args = parser.parse_args()

    run_example(args)
