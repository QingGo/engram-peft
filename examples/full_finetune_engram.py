from torch.optim import AdamW  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram_peft import EngramConfig, get_engram_model, get_optimizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = EngramConfig(
        target_layers=[2, 11],
        tokenizer_name_or_path=MODEL_NAME,
        engram_vocab_size_per_ngram=[256000, 256000],
        hidden_size=base_model.config.hidden_size,
        learning_rate_multiplier=5.0,
        enable_tokenizer_compression=True,
        pad_id=tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else 0,
    )

    model = get_engram_model(
        base_model,
        config,
        tokenizer,
        train_mode="full_finetune",
    )
    model.print_trainable_parameters()

    optimizer = get_optimizer(
        model,
        backbone_learning_rate=5e-5,
        engram_dense_learning_rate=4e-4,
        engram_sparse_learning_rate=2e-3,
        backbone_optimizer=AdamW,
        engram_dense_optimizer="adamw",
        engram_sparse_optimizer="sparse_adam",
    )

    print("Optimizer layout:")
    for idx, opt in enumerate(optimizer.optimizers):
        numel = sum(p.numel() for group in opt.param_groups for p in group["params"])
        print(
            f"  optimizer {idx}: {type(opt).__name__} "
            f"params={numel:,} lr={opt.param_groups[0].get('lr')}"
        )

    inputs = tokenizer("Engram full finetuning example", return_tensors="pt")
    model.train()
    output = model(**inputs, labels=inputs["input_ids"])
    loss = output.loss
    loss.backward()
    print(f"Sanity-check loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
