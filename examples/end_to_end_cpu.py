"""
Engram-PEFT End-to-End CPU Example.

This script demonstrates a full workflow using Engram-PEFT on CPU:
1. Training: Setup a tiny Transformer (< 10M params), inject Engram, and train on TinyStories.
2. Inference: Generate text with trained Engram weights.
3. Visualization: Observing the Context-Aware Gating activation levels.
4. Dynamic Management: Loading and unloading Engram packs at runtime.

Usage:
    uv run python examples/end_to_end_cpu.py
"""

import os
import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, cast, Optional, Tuple

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset  # type: ignore[import-untyped]

from engram_peft import (
    EngramConfig,
    get_engram_model,
    EngramDataCollator,
    get_optimizer,
    get_scheduler,
    EngramModel,
    EngramLayer,
)

# 1. Constants & Device Detection
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "outputs/engram_cpu_demo"
ENGRAM_WEIGHT_DIR = os.path.join(OUTPUT_DIR, "engram_weights")
SEED = 42

set_seed(SEED)


def get_device() -> torch.device:
    """Detects available device and provides hints."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print(
            "\n[INFO] No GPU (CUDA/MPS) detected. Running on CPU (Expected time < 5 mins)."
        )
        return torch.device("cpu")


DEVICE = get_device()


# 2. Minimal Nano-style Transformer Implementation
class SimpleConfig(PretrainedConfig):
    model_type = "simple_transformer"

    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        n_layer: int = 4,
        n_head: int = 2,
        max_position_embeddings: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.num_hidden_layers = n_layer  # Compatibility with transformers Cache
        self.n_head = n_head
        self.max_position_embeddings = max_position_embeddings


class SimpleAttention(nn.Module):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.hidden_size // config.n_head
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.max_position_embeddings, config.max_position_embeddings
                )
            ).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        bias = cast(torch.Tensor, self.bias)
        att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return cast(torch.Tensor, self.c_proj(y))


class SimpleMLP(nn.Module):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class SimpleBlock(nn.Module):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = SimpleAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = SimpleMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


from transformers import GenerationMixin


class SimpleTransformer(PreTrainedModel, GenerationMixin):
    config_class = SimpleConfig

    def __init__(self, config: SimpleConfig) -> None:
        super().__init__(config)
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer.wpe = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.transformer.h = nn.ModuleList(
            [SimpleBlock(config) for _ in range(config.n_layer)]
        )
        self.transformer.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.wte  # type: ignore

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.transformer.wte = value  # type: ignore

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = tok_emb + pos_emb

        for block in self.transformer.h:  # type: ignore
            x = block(x)

        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        next_sequence_length: Optional[int] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_first_iteration: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        return {
            "input_ids": input_ids,
            "next_sequence_length": next_sequence_length,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "is_first_iteration": is_first_iteration,
        }


# 3. Main Logic Functions
class EngramTrainer(Trainer):
    """Custom Trainer to handle SparseCPU gradients which don't support linalg_vector_norm."""

    def _get_grad_norm(
        self, model: nn.Module, grad_norm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.tensor(0.0).to(self.args.device)


def train_engram() -> Tuple[EngramModel, PreTrainedTokenizer, EngramConfig]:
    print("\n>>> Stage 1: Initializing Tiny Model & Engram")

    # Load Tokenizer (using GPT2 as base)
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(MODEL_NAME))
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    # Create Tiny Base Model
    # 32,000 vocab * 128 hidden * 2 (emb+head) + 1M layers = ~9.2M params
    base_config = SimpleConfig(
        vocab_size=32000,
        hidden_size=128,
        n_layer=4,
        n_head=2,
        max_position_embeddings=128,
    )
    base_model = SimpleTransformer(base_config)
    base_model.to(DEVICE)  # type: ignore[arg-type]
    print(
        f"Base model created with {sum(p.numel() for p in base_model.parameters())/1e6:.2f}M parameters."
    )

    # Engram Configuration (CPU Friendly)
    engram_config = EngramConfig(
        target_layers=[1, 2],
        hidden_size=128,
        embedding_dim=64,
        n_head_per_ngram=2,
        engram_vocab_size_per_ngram=[10000, 10000],
        enable_tokenizer_compression=True,
        tokenizer_name_or_path=MODEL_NAME,
        pad_id=tokenizer.pad_token_id,
        seed=SEED,
    )

    # Inject Engram
    print("Injecting Engram layers...")
    model = get_engram_model(base_model, engram_config, tokenizer)

    # Load Tiny Dataset
    print("Loading TinyStories subset (1000 samples)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            tokenizer(
                examples["text"], truncation=True, max_length=128, padding="max_length"
            ),
        )

    data_list = []
    for i, item in enumerate(dataset):
        if i >= 1000:
            break
        data_list.append(item)
    train_dataset = Dataset.from_list(data_list).map(
        tokenize_fn, batched=True, remove_columns=["text"]
    )

    # Components
    collator = EngramDataCollator(tokenizer=tokenizer, config=engram_config)
    optimizer = get_optimizer(model, base_learning_rate=4e-4)
    scheduler = get_scheduler(optimizer, num_training_steps=100, warmup_steps=10)

    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        max_steps=100,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.0,  # Disable clipping to avoid SparseCPU error
    )

    trainer = EngramTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
    )

    print("Starting training (100 steps)...")
    start_time = time.time()
    trainer.train()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Save
    print(f"Saving Engram weights to {ENGRAM_WEIGHT_DIR}")
    model.save_pretrained(ENGRAM_WEIGHT_DIR)

    return model, tokenizer, engram_config


def inference_demo(tokenizer: PreTrainedTokenizer, config: EngramConfig) -> None:
    print("\n>>> Stage 2: Inference & Visualization")

    # Clean Base Model
    base_config = SimpleConfig(
        vocab_size=32000,
        hidden_size=128,
        n_layer=4,
        n_head=2,
        max_position_embeddings=128,
    )
    base_model = SimpleTransformer(base_config)
    base_model.to(DEVICE)  # type: ignore[arg-type]
    model = EngramModel.from_pretrained(base_model, ENGRAM_WEIGHT_DIR)
    model.to(DEVICE)  # type: ignore[arg-type]

    prompt = "Once upon a time, there was a tiny"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    print(f"\nPrompt: {prompt}")

    # Generate with Engram (Manual loop to ensure hooks trigger correctly)
    print("Generating with Engram ENABLED...")
    curr_ids = inputs["input_ids"]
    for _ in range(15):
        with torch.no_grad():
            outputs = model(input_ids=curr_ids)
            logits = outputs.logits[:, -1, :]
            # Simple greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    print(f"Output: {tokenizer.decode(curr_ids[0], skip_special_tokens=True)}")

    # Gating Visualization
    print("\n[Visualization] Context-Aware Gating activation:")
    for layer_id in config.target_layers:
        engram_layer = cast(EngramLayer, model.engram_layers[str(layer_id)])
        gate = engram_layer.gating.last_gate
        if gate is not None:
            # Mean gate value per branch across batch and sequence
            mean_vals = gate.mean(dim=(0, 1, 3)).cpu().tolist()
            gate_str = " | ".join(
                [f"Branch {i}: {val:.4f}" for i, val in enumerate(mean_vals)]
            )
            print(f"Layer {layer_id}: {gate_str}")

    # Dynamic Switching Demo
    print("\n>>> Stage 3: Dynamic Switching Demo")
    print("Unloading Engram (Running base model)...")
    model.unload_engram()
    curr_ids_base = inputs["input_ids"]
    for _ in range(15):
        with torch.no_grad():
            outputs = model(input_ids=curr_ids_base)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            curr_ids_base = torch.cat([curr_ids_base, next_token], dim=-1)
    print(
        f"Base Output: {tokenizer.decode(curr_ids_base[0], skip_special_tokens=True)}"
    )

    print("\nReloading Engram...")
    model.load_engram(ENGRAM_WEIGHT_DIR)
    print("Engram reloaded successfully.")


if __name__ == "__main__":
    model, tokenizer, config = train_engram()
    inference_demo(tokenizer, config)
    print("\nAll done!")
