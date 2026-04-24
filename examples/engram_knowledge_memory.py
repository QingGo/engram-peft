# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none, reportUnknownParameterType=none, reportUnknownLambdaType=none, reportMissingTypeStubs=none
"""
Internal RAG: Knowledge Memorization with Engram-PEFT.

Trains an Engram adapter to memorize factual QA knowledge, leveraging Engram's
sparse n-gram hashing as an internal retrieval mechanism that replaces external
RAG systems. During training, n-grams from question-answer pairs are hashed into
Engram embedding tables. At inference, n-gram overlap triggers retrieval of
previously memorized information—no external vector database needed.

Target: Qwen2.5-32B backbone (4-bit) on 8 x RTX 4090 (24 GB).

Supports two distributed backends:
  DDP + sparse   (recommended) — native SparseCUDA all_reduce, MixedOptimizer
  ZeRO-2 + dense (experimental) — partitions engram grads+opt across GPUs

Usage:
  # DDP with 8 GPUs (recommended)
  torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py

  # ZeRO-2 with 8 GPUs (experimental)
  deepspeed --num_gpus=8 examples/engram_knowledge_memory.py --use_deepspeed

  # Quick smoke test on a single GPU with a small model
  python examples/engram_knowledge_memory.py \
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --max_samples 200 --batch_size 2 --max_steps 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from typing import Any, cast

import torch
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset, DatasetDict, load_dataset
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)

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

DEFAULT_MODEL = "Qwen/Qwen2.5-32B"
DEFAULT_OUTPUT_DIR = "outputs/engram_knowledge"
DEFAULT_SEED = 42

# QA datasets with documented structure. Add more entries as needed.
QA_DATASETS: dict[str, dict[str, Any]] = {
    "nq_open": {
        "path": "nq_open",
        "question_key": "question",
        "answer_key": "answer",
        "description": "Natural Questions Open (61K train, 3.6K validation)",
    },
    "trivia_qa": {
        "path": "trivia_qa",
        "name": "rc.nocontext",
        "question_key": "question",
        "answer_key": "answer",
        "description": "TriviaQA without context (138K train, 11K validation)",
    },
}

# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────


def format_qa_text(question: str, answer: str | list[str] | dict[str, Any]) -> str:
    """Format a single QA pair into training text.

    The simple 'Question: ... Answer: ...' format maximizes n-gram overlap
    between training and inference, letting Engram's sparse hashing act as
    an internal retrieval key.
    """
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    elif isinstance(answer, dict):
        # TriviaQA answer dict: {"value": "...", "aliases": [...], ...}
        answer = str(answer.get("value", answer.get("aliases", [""])[0] if answer.get("aliases") else ""))
    return f"Question: {question}\nAnswer: {answer}"


def load_qa_dataset(
    dataset_name: str,
    max_samples: int | None = None,
    eval_ratio: float = 0.05,
) -> tuple[Dataset, Dataset | None]:
    """Load and format a QA dataset for knowledge memorization training.

    Args:
        dataset_name: Key in QA_DATASETS dict.
        max_samples: Cap total samples (useful for smoke tests).
        eval_ratio: Fraction of data held out for evaluation.

    Returns:
        (train_dataset, eval_dataset) where each has a single ``text`` column.
    """
    ds_info = QA_DATASETS[dataset_name]
    load_kwargs: dict[str, Any] = {"path": ds_info["path"]}
    if "name" in ds_info:
        load_kwargs["name"] = ds_info["name"]

    logging.info("Loading %s dataset (%s)...", dataset_name, ds_info.get("description", ""))

    raw = load_dataset(**load_kwargs, split="train", trust_remote_code=True)
    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    question_key: str = ds_info["question_key"]
    answer_key: str = ds_info["answer_key"]

    texts: list[dict[str, str]] = []
    for example in raw:
        texts.append(
            {
                "text": format_qa_text(
                    example[question_key],
                    example[answer_key],
                )
            }
        )

    formatted: Dataset = Dataset.from_list(texts)

    if eval_ratio > 0 and len(formatted) > 1:
        split: DatasetDict = formatted.train_test_split(test_size=eval_ratio, seed=DEFAULT_SEED)
        return split["train"], split["test"]

    return formatted, None


# ──────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────


def tokenize_qa_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize QA text and produce causal-LM labels."""

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # collator handles padding
        )

        # labels = input_ids (standard causal LM)
        raw_ids: Any = tokenized["input_ids"]
        labels: list[list[int]] = [list(ids) for ids in raw_ids]

        # Replace pad tokens in labels with -100
        if tokenizer.pad_token_id is not None:
            for label in labels:
                for i in range(len(label)):
                    if label[i] == tokenizer.pad_token_id:
                        label[i] = -100

        tokenized["labels"] = labels
        return cast("dict[str, Any]", tokenized)

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc,
    )


# ──────────────────────────────────────────────────────────────────────
# DeepSpeed Config
# ──────────────────────────────────────────────────────────────────────


def write_default_ds_config(output_dir: str) -> str:
    """Generate a minimal DeepSpeed ZeRO-2 JSON config for dense engram training."""
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
# Model Construction
# ──────────────────────────────────────────────────────────────────────


def build_engram_model(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    use_deepspeed: bool,
    engram_config: EngramConfig,
) -> EngramModel:
    """Load 4-bit backbone, inject Engram layers, return EngramModel.

    For DDP, the model is loaded with explicit device_map pointing to the
    local GPU rank. For DeepSpeed, device placement is left to accelerate's
    deepspeed plugin.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if use_deepspeed:
        # DeepSpeed + 4-bit is experimental. Let accelerate handle device_map.
        logging.warning(
            "DeepSpeed with 4-bit quantization is experimental. "
            + "If you encounter device-related errors, try DDP mode instead."
        )
        device_map = None
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    logging.info("Loading base model: %s", model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if not use_deepspeed:
        logging.info("Preparing model for k-bit training...")
        base_model = prepare_model_for_kbit_training(base_model)

    logging.info("Injecting Engram layers...")
    model = get_engram_model(
        base_model,
        engram_config,
        tokenizer=wash_tokenizer(tokenizer),
        train_mode="engram_only",
    )

    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────
# Inference Demo
# ──────────────────────────────────────────────────────────────────────


def run_inference_demo(
    model: EngramModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate a response and return the decoded text."""
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = cast("str", tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True))
    return response


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Engram-PEFT Knowledge Memorization (Internal RAG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Smoke test (small model, CPU-friendly)
              python examples/engram_knowledge_memory.py \\
                  --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \\
                  --max_samples 200 --max_steps 50

              # Full DDP training (8 GPUs)
              torchrun --nproc_per_node=8 examples/engram_knowledge_memory.py

              # Full ZeRO-2 training (8 GPUs, experimental)
              deepspeed --num_gpus=8 examples/engram_knowledge_memory.py --use_deepspeed
        """),
    )
    # Model
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base model ID")
    # Dataset
    p.add_argument("--dataset", default="nq_open", choices=list(QA_DATASETS),
                   help="QA dataset to use")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap training samples (None = full dataset)")
    # Distributed
    p.add_argument("--use_deepspeed", action="store_true",
                   help="Enable DeepSpeed ZeRO-2 (disables sparse embeddings)")
    # Training
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-device micro batch size")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--max_length", type=int, default=256,
                   help="Max tokenized sequence length")
    p.add_argument("--learning_rate", type=float, default=2e-4,
                   help="Peak learning rate (Engram uses LR multiplier)")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Max training steps (-1 = use epochs)")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    # Engram
    p.add_argument("--embedding_dim", type=int, default=1280,
                   help="Engram internal embedding dimension")
    p.add_argument("--target_layers", type=int, nargs="+", default=[2, 15],
                   help="Layer indices for Engram injection")
    p.add_argument("--entropy_loss_weight", type=float, default=0.01,
                   help="Gating entropy penalty (0 = disabled)")
    # Logging
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Distributed environment detection ─────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", local_rank))
    is_main = global_rank <= 0
    use_deepspeed = args.use_deepspeed

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(levelname)s|rank=%(rank)s] %(message)s" if local_rank >= 0 else "[%(levelname)s] %(message)s",
    )

    if is_main:
        logging.info("=" * 60)
        logging.info("Engram Knowledge Memorization Training")
        logging.info("  Backend:    %s", "DeepSpeed ZeRO-2" if use_deepspeed else "DDP + sparse")
        logging.info("  Model:      %s", args.model)
        logging.info("  Dataset:    %s", args.dataset)
        logging.info("  GPUs:       %d", world_size)
        logging.info("=" * 60)

    set_seed(args.seed)

    # ── Build engram config ───────────────────────────────────────
    sparse_embeddings = not use_deepspeed
    if use_deepspeed and args.entropy_loss_weight > 0:
        logging.warning("DeepSpeed disables MixedOptimizer; entropy loss may not be applied.")

    engram_config = EngramConfig(
        embedding_dim=args.embedding_dim,
        target_layers=args.target_layers,
        use_sparse_embeddings=sparse_embeddings,
        entropy_loss_weight=args.entropy_loss_weight if not use_deepspeed else 0.0,
    )

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────
    model = build_engram_model(
        args.model,
        tokenizer,
        use_deepspeed=use_deepspeed,
        engram_config=engram_config,
    )

    # ── Dataset ───────────────────────────────────────────────────
    train_dataset, eval_dataset = load_qa_dataset(
        args.dataset,
        max_samples=args.max_samples,
    )

    if is_main:
        logging.info("Train: %d examples", len(train_dataset))
        if eval_dataset:
            logging.info("Eval:  %d examples", len(eval_dataset))

    train_dataset = tokenize_qa_dataset(
        train_dataset, tokenizer, max_length=args.max_length
    )
    if eval_dataset is not None:
        eval_dataset = tokenize_qa_dataset(
            eval_dataset, tokenizer, max_length=args.max_length
        )

    # ── Training arguments ────────────────────────────────────────
    deepspeed_config: str | None = None
    if use_deepspeed:
        deepspeed_config = write_default_ds_config(args.output_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=False,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── Collator ──────────────────────────────────────────────────
    data_collator = EngramDataCollator(
        tokenizer=tokenizer,
        config=model.config,
        mlm=False,
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────
    logging.info("Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────
    if is_main:
        # Unwrap model (DDP / DeepSpeed) to ensure trained weights are captured
        saved_model = trainer.accelerator.unwrap_model(trainer.model)
        if not isinstance(saved_model, EngramModel):
            saved_model = model  # fallback to original if unwrap failed

        logging.info("Saving Engram adapter to %s", args.output_dir)
        saved_model.save_pretrained(args.output_dir)

        # ── Inference demo ────────────────────────────────────────
        demo_questions = [
            "What is the largest planet in our solar system?",
            "Who wrote the novel '1984'?",
        ]
        # Try to grab a real question from the dataset
        if len(train_dataset) > 0:
            raw_raw = load_dataset(
                QA_DATASETS[args.dataset]["path"],
                **({"name": QA_DATASETS[args.dataset]["name"]} if "name" in QA_DATASETS[args.dataset] else {}),
                split="train",
                trust_remote_code=True,
            )
            demo_questions = [str(raw_raw[i][QA_DATASETS[args.dataset]["question_key"]]) for i in range(min(3, len(raw_raw)))]

        logging.info("\n" + "=" * 60)
        logging.info("Inference Demo (main process only)")
        logging.info("=" * 60)

        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        for i, question in enumerate(demo_questions):
            prompt = f"Question: {question}\nAnswer:"
            response = run_inference_demo(saved_model, tokenizer, prompt)
            logging.info("[%d] Q: %s", i + 1, question)
            logging.info("[%d] A: %s", i + 1, response)
        tokenizer.padding_side = old_padding_side

        logging.info("\nAdapter saved to: %s", args.output_dir)
        logging.info(
            "Reload with: EngramModel.from_pretrained(base_model, '%s', tokenizer=tokenizer)",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
