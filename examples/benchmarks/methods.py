from typing import Any, cast

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.adamw import AdamW
from transformers import (
    DefaultDataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from engram_peft import (
    EngramConfig,
    EngramDataCollator,
    EngramTrainer,
    get_engram_model,
)


def extract_trainer_metrics(trainer: Trainer, train_result: Any) -> dict[str, Any]:
    """Helper to extract common metrics from a Trainer object."""
    eval_results = trainer.evaluate()
    eval_loss = cast("float", eval_results.get("eval_loss", 0.0))

    avg_time_per_step = train_result.metrics.get("train_runtime", 0) / max(
        1, train_result.global_step
    )

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    log_history = []
    for log in trainer.state.log_history:
        if "step" in log:
            entry = {"step": log["step"]}
            if "loss" in log:
                entry["loss"] = log["loss"]
            if "eval_loss" in log:
                entry["eval_loss"] = log["eval_loss"]
            if len(entry) > 1:
                log_history.append(entry)

    return {
        "log_history": log_history,
        "peak_memory_gb": peak_memory,
        "avg_time_per_step": avg_time_per_step,
        "eval_loss": eval_loss,
    }


def train_lora(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: LoRA")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/lora",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(peft_config, k):
            setattr(peft_config, k, v)
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/lora_weights")
    # Clean up
    model = model.unload()
    return metrics


def train_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Engram Only")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = EngramConfig(
        target_layers=[2, 11],
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
        embedding_dim=1024,
    )
    # Apply overrides to config before model creation
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = get_engram_model(base_model, config, tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/engram",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/engram_weights")
    model.unload_engram()
    return metrics


def train_full_finetune(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Full Finetune")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_model.requires_grad_(True)
    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/full_ft",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    base_model.save_pretrained("outputs/benchmarks/full_ft_only_weights")
    return metrics


def train_lora_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: LoRA + Engram")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    # Apply overrides to peft_config
    for k, v in overrides.items():
        if hasattr(peft_config, k):
            setattr(peft_config, k, v)

    lora_model = get_peft_model(base_model, peft_config)

    # Load Engram wrapper
    config = EngramConfig(
        target_layers=[2, 11],
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
    )
    # Apply overrides to engram config
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = get_engram_model(lora_model, config, tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/lora_engram",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/lora_engram_weights")
    model.unload_engram()
    lora_model.unload()
    return metrics


def train_full_finetune_engram(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Any,
    eval_dataset: Any,
    args: Any,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    print("\n>>> Method: Full Finetune + Engram")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = EngramConfig(
        target_layers=[2, 11],
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
        learning_rate_multiplier=3.0,
    )
    # Apply overrides to config
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)

    model = get_engram_model(
        base_model,
        config,
        tokenizer,
        train_mode="full_finetune",
    )

    training_args = TrainingArguments(
        output_dir="outputs/benchmarks/tmp/full_ft_engram",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=3e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        report_to="wandb" if args.wandb else "none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    # Apply overrides to training_args
    for k, v in overrides.items():
        if hasattr(training_args, k):
            setattr(training_args, k, v)

    collator = EngramDataCollator(tokenizer=tokenizer, config=config)
    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        optimizer_kwargs={
            "backbone_learning_rate": 5e-5,
            "engram_dense_learning_rate": 3e-4,
            "engram_sparse_learning_rate": 9e-4,
            "backbone_optimizer": AdamW,
        },
    )

    train_result = trainer.train()
    metrics = extract_trainer_metrics(trainer, train_result)

    model.save_pretrained("outputs/benchmarks/full_ft_engram_weights")
    model.unload_engram()
    return metrics
