# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportUnknownLambdaType=none, reportMissingTypeStubs=none, reportAssignmentType=none, reportArgumentType=none
"""
PopQA Knowledge Memorization Benchmark with Engram-PEFT.

Evaluates Engram's internal RAG capability on long-tail factual knowledge
(PopQA), comparing Exact Match (EM) accuracy across configurations:
  - Base model (no adapter)
  - + Engram adapter
  - + LoRA adapter
  - + Engram + LoRA (combined)

Protocol:
  1. PopQA 80/20 random split
  2. Train adapter(s) on held-in 80%
  3. Evaluate EM (generation, greedy) on held-out 20%
  4. Compare all available configurations in a table

Usage:
  # Train Engram only, then evaluate
  python examples/engram_knowledge_memory.py --mode train

  # Train Engram + LoRA, then evaluate
  python examples/engram_knowledge_memory.py --mode train --train_lora

  # Evaluate only (load saved adapters)
  python examples/engram_knowledge_memory.py --mode eval \\
    --engram_path outputs/popqa_benchmark/engram

  # Evaluate all three configs
  python examples/engram_knowledge_memory.py --mode eval \\
    --engram_path outputs/popqa_benchmark/engram \\
    --lora_path outputs/popqa_benchmark/lora

  # Distributed training (DDP, 8 GPUs)
  torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import textwrap
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramModel,
    EngramTrainer,
    get_engram_model,
)
from engram_peft.utils.compat import wash_tokenizer

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_OUTPUT_DIR = "outputs/popqa_benchmark"
DEFAULT_SEED = 42
TRAIN_RATIO = 0.8

# ──────────────────────────────────────────────────────────────────────
# Answer Normalization (Exact Match)
# ──────────────────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ──────────────────────────────────────────────────────────────────────
# Data Loading (PopQA)
# ──────────────────────────────────────────────────────────────────────


def format_qa_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def load_popqa(
    max_samples: int | None = None,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[Dataset, Dataset]:
    """Load PopQA, return (train_dataset, test_dataset).

    PopQA only has a ``test`` split on HF (14,267 rows), so we do an
    80/20 random split ourselves.  Each row is kept raw (with
    ``question`` and ``possible_answers`` columns) for later EM eval.
    A ``text`` column is added for training.
    """
    raw: Dataset = load_dataset("akariasai/PopQA", split="test")
    if max_samples is not None and max_samples < len(raw):
        raw = raw.select(range(max_samples))

    def add_text(example: dict[str, Any]) -> dict[str, Any]:
        answer: str = (
            example["possible_answers"][0] if example["possible_answers"] else ""
        )
        example["text"] = format_qa_text(example["question"], answer)
        return example

    raw = raw.map(add_text)

    split_result: DatasetDict = raw.train_test_split(
        test_size=1 - train_ratio, seed=DEFAULT_SEED
    )
    return split_result["train"], split_result["test"]


# ──────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize ``text`` column and produce causal-LM labels."""

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )


# ──────────────────────────────────────────────────────────────────────
# DeepSpeed Config
# ──────────────────────────────────────────────────────────────────────


def write_default_ds_config(output_dir: str) -> str:
    config = {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "gradient_clipping": 1.0,
        "bf16": {"enabled": "auto"},
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ──────────────────────────────────────────────────────────────────────
# 4-bit Backbone Loading
# ──────────────────────────────────────────────────────────────────────


def load_4bit_backbone(
    model_id: str,
    use_deepspeed: bool = False,
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if use_deepspeed:
        device_map: str | None = None
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    return cast(
        "AutoModelForCausalLM",
        AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Model Construction — Engram
# ──────────────────────────────────────────────────────────────────────


def build_engram_model(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    use_deepspeed: bool,
    engram_config: EngramConfig,
) -> EngramModel:
    base_model = load_4bit_backbone(model_id, use_deepspeed)

    if not use_deepspeed:
        base_model = prepare_model_for_kbit_training(base_model)

    model = get_engram_model(
        base_model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="engram_only",
    )
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────
# Model Construction — LoRA
# ──────────────────────────────────────────────────────────────────────


def build_lora_model(
    model_id: str,
    _tokenizer: PreTrainedTokenizerBase,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
) -> PeftModel:
    base_model = load_4bit_backbone(model_id)
    base_model = prepare_model_for_kbit_training(base_model)

    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return cast("PeftModel", model)


# ──────────────────────────────────────────────────────────────────────
# Exact Match Evaluation
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_em(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: Dataset,
    max_samples: int = 200,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    """Compute Exact Match accuracy on a held-out PopQA test set.

    For each sample: greedy-decode from ``"Question: {q}\\nAnswer:"``,
    extract the first line, normalize, compare with all possible answers.

    Returns:
        ``{"correct": int, "total": int, "accuracy": float}``
    """
    device = model.device
    total = min(max_samples, len(test_dataset))
    correct = 0

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(total):
        question: str = test_dataset[i]["question"]
        possible: list[str] = test_dataset[i]["possible_answers"]

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        model.eval()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        response = cast(
            "str", tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        )
        pred = response.split("\n")[0].strip()
        pred_norm = normalize_answer(pred)

        if any(normalize_answer(a) == pred_norm for a in possible):
            correct += 1

        if (i + 1) % 50 == 0:
            logging.info(
                "  EM progress: %d/%d  (acc so far: %.1f%%)",
                i + 1,
                total,
                correct / (i + 1) * 100,
            )

    tokenizer.padding_side = old_padding_side

    accuracy = correct / total * 100
    return {"correct": correct, "total": total, "accuracy": accuracy}


# ──────────────────────────────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────────────────────────────


def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    if not results:
        return

    base_acc = results.get("Base", {}).get("accuracy", 0.0)

    logging.info("")
    logging.info("=" * 60)
    logging.info("  PopQA Benchmark Results")
    logging.info("=" * 60)
    logging.info("  %-25s %10s %10s", "Config", "Accuracy", "Δ vs Base")
    logging.info("  " + "-" * 47)
    for name, res in results.items():
        acc = res.get("accuracy", 0.0)
        delta = acc - base_acc
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        if name == "Base":
            delta_str = "—"
        logging.info("  %-25s %8.1f%% %10s", name, acc, delta_str)
    logging.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Arg Parsing
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PopQA Knowledge Memorization Benchmark with Engram-PEFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Train Engram + evaluate
              python examples/engram_knowledge_memory.py --mode train

              # Train Engram + LoRA + evaluate
              python examples/engram_knowledge_memory.py --mode train --train_lora

              # Evaluate only (load saved adapters)
              python examples/engram_knowledge_memory.py --mode eval \\
                  --engram_path outputs/popqa_benchmark/engram

              # Distributed training
              torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py
        """),
    )

    # Mode
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    # Model
    p.add_argument("--model", default=DEFAULT_MODEL)
    # Data
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap PopQA samples (None = all 14,267)",
    )
    p.add_argument(
        "--eval_max_samples",
        type=int,
        default=200,
        help="Cap evaluation samples (0 = all)",
    )
    # Distributed
    p.add_argument("--use_deepspeed", action="store_true")
    # Training
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Adapters
    p.add_argument(
        "--train_engram", action="store_true", default=True, help="Train Engram adapter"
    )
    p.add_argument("--no_train_engram", action="store_false", dest="train_engram")
    p.add_argument(
        "--train_lora",
        action="store_true",
        default=False,
        help="Additionally train LoRA adapter",
    )
    p.add_argument(
        "--engram_path",
        type=str,
        default=None,
        help="Path to saved Engram adapter (eval mode or post-train load)",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to saved LoRA adapter (eval mode or post-train load)",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    # Engram
    p.add_argument("--embedding_dim", type=int, default=1280)
    p.add_argument("--target_layers", type=int, nargs="+", default=[2, 15])
    p.add_argument("--entropy_loss_weight", type=float, default=0.01)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Distributed detection ─────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", local_rank))
    is_main = global_rank <= 0
    use_deepspeed = args.use_deepspeed

    # ── Inject 'rank' into all log records so the format string can use %(rank)s ──
    _log_rank = global_rank

    def _log_record_factory(*args, **kwargs):
        record = logging.LogRecord(*args, **kwargs)
        record.rank = _log_rank
        return record

    if local_rank >= 0:
        logging.setLogRecordFactory(_log_record_factory)

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(levelname)s|rank=%(rank)s] %(message)s"
        if local_rank >= 0
        else "[%(levelname)s] %(message)s",
    )

    if is_main:
        logging.info("=" * 60)
        logging.info("PopQA Knowledge Memorization Benchmark")
        logging.info("  Mode:       %s", args.mode)
        logging.info("  Model:      %s", args.model)
        logging.info("  GPUs:       %d", world_size)
        logging.info(
            "  Backend:    %s", "DeepSpeed ZeRO-2" if use_deepspeed else "DDP + sparse"
        )
        logging.info("=" * 60)

    set_seed(args.seed)

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Data ──────────────────────────────────────────────────────
    train_raw, test_raw = load_popqa(max_samples=args.max_samples)

    if is_main:
        logging.info("PopQA: %d train / %d test", len(train_raw), len(test_raw))

    # ── Compute paths ─────────────────────────────────────────────
    engram_save_path = (
        os.path.join(args.output_dir, "engram")
        if args.mode == "train"
        else args.engram_path
    )
    lora_save_path = (
        os.path.join(args.output_dir, "lora")
        if args.mode == "train" and args.train_lora
        else args.lora_path
    )
    # Resolve eval-time paths from training defaults
    if args.mode == "eval":
        engram_save_path = args.engram_path
        lora_save_path = args.lora_path

    # ═══════════════════════════════════════════════════════════════
    #  TRAINING
    # ═══════════════════════════════════════════════════════════════
    if args.mode == "train":
        # ── Engram config ────────────────────────────────────────
        sparse_embeddings = not (
            use_deepspeed or int(os.environ.get("WORLD_SIZE", "1")) > 1
        )
        if use_deepspeed and args.entropy_loss_weight > 0:
            logging.warning(
                "DeepSpeed disables MixedOptimizer; entropy loss may not be applied."
            )

        engram_config = EngramConfig(
            embedding_dim=args.embedding_dim,
            target_layers=args.target_layers,
            use_sparse_embeddings=sparse_embeddings,
            entropy_loss_weight=args.entropy_loss_weight if not use_deepspeed else 0.0,
        )

        # ── Train Engram ─────────────────────────────────────────
        if args.train_engram:
            logging.info(">>> Training Engram adapter...")
            model = build_engram_model(
                args.model, tokenizer, use_deepspeed, engram_config
            )
            train_tokenized = tokenize_dataset(
                train_raw, tokenizer, max_length=args.max_length
            )

            deepspeed_config: str | None = None
            if use_deepspeed:
                deepspeed_config = write_default_ds_config(args.output_dir)

            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate,
                num_train_epochs=args.num_epochs,
                max_steps=args.max_steps if args.max_steps > 0 else -1,
                warmup_ratio=args.warmup_ratio,
                logging_steps=args.logging_steps,
                save_steps=0,
                save_total_limit=0,
                eval_strategy="no",
                bf16=True,
                deepspeed=deepspeed_config,
                ddp_find_unused_parameters=False,
                dataloader_num_workers=2,
                seed=args.seed,
                report_to="none",
                remove_unused_columns=False,
            )

            data_collator = EngramDataCollator(
                tokenizer=tokenizer, config=model.config, mlm=False
            )

            trainer = EngramTrainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                data_collator=data_collator,
            )

            trainer.train()

            if is_main:
                saved = trainer.accelerator.unwrap_model(trainer.model)
                if not isinstance(saved, EngramModel):
                    saved = model
                saved.save_pretrained(engram_save_path)
                logging.info("Engram adapter saved to %s", engram_save_path)

            del model, trainer, train_tokenized, data_collator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Train LoRA ────────────────────────────────────────────
        if args.train_lora:
            logging.info(">>> Training LoRA adapter...")
            lora_model = build_lora_model(
                args.model, tokenizer, r=args.lora_r, lora_alpha=args.lora_alpha
            )
            train_tokenized = tokenize_dataset(
                train_raw, tokenizer, max_length=args.max_length
            )

            lora_training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate * 0.5,
                num_train_epochs=args.num_epochs,
                max_steps=args.max_steps if args.max_steps > 0 else -1,
                warmup_ratio=args.warmup_ratio,
                logging_steps=args.logging_steps,
                save_steps=0,
                save_total_limit=0,
                eval_strategy="no",
                bf16=True,
                ddp_find_unused_parameters=False,
                dataloader_num_workers=2,
                seed=args.seed,
                report_to="none",
                remove_unused_columns=False,
            )

            lora_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )

            lora_trainer = Trainer(
                model=lora_model,
                args=lora_training_args,
                train_dataset=train_tokenized,
                data_collator=lora_collator,
            )

            lora_trainer.train()

            if is_main:
                os.makedirs(lora_save_path, exist_ok=True)
                lora_model.save_pretrained(lora_save_path)
                logging.info("LoRA adapter saved to %s", lora_save_path)

            del lora_model, lora_trainer, train_tokenized, lora_collator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    #  EVALUATION
    # ═══════════════════════════════════════════════════════════════
    eval_max = args.eval_max_samples if args.eval_max_samples > 0 else len(test_raw)
    results: dict[str, dict[str, Any]] = {}

    if not is_main:
        return  # evaluation runs on main process only

    logging.info("\n" + "=" * 60)
    logging.info("  Evaluation on %d held-out PopQA samples", eval_max)
    logging.info("=" * 60)

    # --- Base model ---
    logging.info("[1/4] Evaluating Base model...")
    base = load_4bit_backbone(args.model)
    results["Base"] = evaluate_em(base, tokenizer, test_raw, max_samples=eval_max)
    logging.info(
        "  Base accuracy: %.1f%% (%d/%d)",
        results["Base"]["accuracy"],
        results["Base"]["correct"],
        results["Base"]["total"],
    )
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- + Engram ---
    if engram_save_path and os.path.isdir(engram_save_path):
        logging.info("[2/4] Evaluating +Engram adapter...")
        base = load_4bit_backbone(args.model)
        engram_model: EngramModel = EngramModel.from_pretrained(
            base, engram_save_path, tokenizer=wash_tokenizer(tokenizer)
        )
        results["+Engram"] = evaluate_em(
            engram_model, tokenizer, test_raw, max_samples=eval_max
        )
        logging.info(
            "  +Engram accuracy: %.1f%% (%d/%d)",
            results["+Engram"]["accuracy"],
            results["+Engram"]["correct"],
            results["+Engram"]["total"],
        )
        del base, engram_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- + LoRA ---
    if lora_save_path and os.path.isdir(lora_save_path):
        logging.info("[3/4] Evaluating +LoRA adapter...")
        base = load_4bit_backbone(args.model)
        lora_model = PeftModel.from_pretrained(base, lora_save_path)
        results["+LoRA"] = evaluate_em(
            lora_model, tokenizer, test_raw, max_samples=eval_max
        )
        logging.info(
            "  +LoRA accuracy: %.1f%% (%d/%d)",
            results["+LoRA"]["accuracy"],
            results["+LoRA"]["correct"],
            results["+LoRA"]["total"],
        )
        del base, lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- + Engram + LoRA ---
    if (
        engram_save_path
        and os.path.isdir(engram_save_path)
        and lora_save_path
        and os.path.isdir(lora_save_path)
    ):
        logging.info("[4/4] Evaluating +Engram+LoRA combined...")
        base = load_4bit_backbone(args.model)
        combined = PeftModel.from_pretrained(base, lora_save_path)
        combined = EngramModel.from_pretrained(
            combined, engram_save_path, tokenizer=wash_tokenizer(tokenizer)
        )
        results["+Engram+LoRA"] = evaluate_em(
            combined, tokenizer, test_raw, max_samples=eval_max
        )
        logging.info(
            "  +Engram+LoRA accuracy: %.1f%% (%d/%d)",
            results["+Engram+LoRA"]["accuracy"],
            results["+Engram+LoRA"]["correct"],
            results["+Engram+LoRA"]["total"],
        )
        del base, combined
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Print table ---
    print_comparison(results)

    if args.mode == "train" and is_main:
        logging.info("\nAdapters saved to: %s", args.output_dir)
        logging.info(
            "Re-run with: --mode eval --engram_path %s [--lora_path %s]",
            engram_save_path,
            lora_save_path,
        )


if __name__ == "__main__":
    main()
