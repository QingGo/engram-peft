# Distributed Training

This document describes how to use `engram-peft` with distributed training frameworks: **PyTorch DDP** (Distributed Data Parallel) and **DeepSpeed ZeRO-1/2**.

---

## DDP (Distributed Data Parallel)

`EngramTrainer` inherits from `transformers.Trainer`, which integrates with Hugging Face `accelerate` to support DDP out of the box.

### How to Enable

Use `torchrun` to launch your training script on multiple GPUs:

```bash
torchrun --nproc_per_node=N \
    -m engram_peft.cli train \
    --model meta-llama/Llama-2-7b \
    --dataset your_dataset \
    --output_dir ./output
```

Or equivalently, using `accelerate`:

```bash
accelerate launch --num_processes=N \
    -m engram_peft.cli train \
    --model meta-llama/Llama-2-7b \
    --dataset your_dataset \
    --output_dir ./output
```

No code changes are needed. `transformers.Trainer` (and therefore `EngramTrainer`) automatically handles:

- **Gradient synchronization** across GPUs via `accelerate`
- **Learning rate scheduling** on the main process only
- **Checkpoint saving** (only on rank 0)

### Sparse Gradients in DDP

**DDP does not support sparse gradients** regardless of the backend (NCCL or Gloo). PyTorch's `DistributedDataParallel` flattens all gradients into **dense buckets** before performing all_reduce — this is a fundamental aspect of DDP's design, not a limitation of the communication backend.

When `use_sparse_embeddings=True` is set and DDP is detected:

1. `MultiHeadEmbedding` automatically forces `sparse=False` on the `nn.Embedding` layer.
2. `EngramTrainer` skips `MixedOptimizer` (SparseAdam + Adam) and uses a standard `AdamW` for all parameters.
3. A warning is printed at trainer initialization.

**What this means for training:**

- **Memory**: Same as single-GPU mode — no additional penalty from sparse-to-dense conversion at sync time. The Engram memory formula (`16E`) covers weights, gradients, and Adam states in dense format.
- **Performance**: Dense all_reduce is the fastest NCCL operation. No sparse sync overhead.
- **Gloo is not a workaround**: Gloo `all_reduce` supports sparse tensors, but DDP's gradient bucket mechanism converts all gradients to dense before calling it. Changing the backend alone does not enable sparse gradients.

**Single GPU workaround:** If you need the memory/compute benefits of SparseAdam, train on a single GPU with `use_sparse_embeddings=True`. DDP's throughput scaling offsets the loss of sparse optimizer efficiency on multi-GPU setups.

---

## DeepSpeed (ZeRO-1/2)

DeepSpeed ZeRO stages 1 and 2 are supported, but with important limitations.

### Installation

Install `engram-peft` with DeepSpeed support:

```bash
uv sync --group dev --group deepspeed
```

Or manually:

```bash
pip install deepspeed>=0.15.0
```

### Limitations

1. **`MixedOptimizer` is not supported**: DeepSpeed takes over optimizer creation internally. When DeepSpeed is detected, `EngramTrainer` automatically skips `MixedOptimizer` and falls back to a standard optimizer (AdamW).
2. **`SparseAdam` is not compatible**: DeepSpeed does not support sparse gradient handling (same as DDP — all gradients are flattened to dense internally). You **must** set `use_sparse_embeddings=False` in `EngramConfig` when using DeepSpeed. The trainer automatically forces dense mode and skips MixedOptimizer.
3. **Gradient clipping is handled by DeepSpeed**: The `_clip_grad_norm` override in `EngramTrainer` is bypassed when DeepSpeed is active.
4. **ZeRO-3 is not tested**: ZeRO stage 3 partitions model parameters, which may conflict with the Engram hook injection mechanism. Use at your own risk.

### DeepSpeed Configuration Example

Create a `ds_config.json` file:

```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_clipping": 1.0,
    "bf16": {
        "enabled": "auto"
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto"
}
```

### Launching with DeepSpeed

```bash
deepspeed --num_gpus=N \
    -m engram_peft.cli train \
    --model meta-llama/Llama-2-7b \
    --dataset your_dataset \
    --output_dir ./output \
    --deepspeed ds_config.json
```

Or equivalently, via `TrainingArguments`:

```python
from engram_peft import EngramConfig, get_engram_model
from engram_peft.trainer import EngramTrainer
from transformers import TrainingArguments

# IMPORTANT: disable sparse embeddings for DeepSpeed
config = EngramConfig(
    use_sparse_embeddings=False,
    ...
)

model = get_engram_model(base_model, config)

args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",
    per_device_train_batch_size=4,
    ...
)

trainer = EngramTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
)
trainer.train()
```

### Expected Warning

When DeepSpeed is enabled with `use_sparse_embeddings=True`, you will see:

```
[Engram-PEFT] Warning: DeepSpeed is enabled but use_sparse_embeddings=True. ...
[Engram-PEFT] DeepSpeed detected: skipping MixedOptimizer. ...
```

These are informational. To eliminate them, set `use_sparse_embeddings=False` in your `EngramConfig`.

### Troubleshooting

#### "Expected all tensors to be on the same device" with DeepSpeed

Ensure all model parameters are moved to the correct device before creating the trainer. Engram layers are automatically placed on the backbone's device during `load_engram()`, but if you manually move the model, also call:

```python
model.adapters.to(device)
model.load_engram()
```

#### Sparse gradient error with DeepSpeed

```
RuntimeError: sparse tensors not supported in DeepSpeed
```

**Fix:** Set `use_sparse_embeddings=False` in `EngramConfig` and reinitialize the model.

#### NCCL timeout with many GPUs

Increase the NCCL timeout:

```bash
export NCCL_TIMEOUT=600
```

---

## Memory Estimation & Hardware Strategy

This section helps you choose the right hardware and training mode for your model size.

### Memory Model

**Notation:**

| Symbol | Meaning |
|--------|---------|
| B | Backbone parameter count (e.g., 7 × 10⁹) |
| E | Engram parameter count (≈ **362M** with `embedding_dim=1280`, 2 injection layers, default hash table) |
| L | LoRA parameter count (≈ **0.005B** with rank=16 on all linear layers) |
| N | Number of GPUs |
| A | Activation memory (varies by mode and hardware) |

**Formula by training mode:**

| Mode | Per-GPU Memory |
|------|----------------|
| `engram_only` (single GPU) | `2B + 16E + 0.1` GB (uses SparseAdam for memory-efficient embedding lookup) |
| `engram_only` (DDP) | `2B + 16E + 0.1` GB (dense gradients, AdamW) |
| `engram_only` (ZeRO-2, N GPUs) | `2B + 2E + 10E/N + 0.1` GB |
| `engram+LoRA` (single GPU) | `2B + 12L + 16E + 0.1` GB |
| `engram+LoRA` (DDP) | `2B + 12L + 16E + 0.1` GB (dense gradients, AdamW) |
| `engram+LoRA` (ZeRO-2, N GPUs) | `2B + 2E + 10E/N + 12L/N + 0.1` GB |
| `engram+full_finetune` (single GPU / DDP) | `12B + 16E + 8` GB (needs grad ckpt) |
| `engram+full_finetune` (ZeRO-2, N GPUs) | `2B + 10B/N + 2E + 10E/N + 8` GB (needs grad ckpt) |

**Why the formulas differ:**

- `engram_only` freezes the backbone → no gradients or optimizer states for backbone params (2B → **12× savings** vs `full_finetune`). Backbone activations are not retained, giving **A ≈ 0.1 GB**.
- `full_finetune` needs to store gradients (2B) + Adam optimizer states (8B) for every parameter. Without ZeRO, this is `12B + 16E` per GPU. Must use **gradient checkpointing** to control activations (A ≈ 8 GB vs 68 GB).
- **ZeRO-2 partitions gradients and optimizer states but NOT weights.** Weights stay replicated on every GPU. For `engram_only`, the backbone weights (2B) are frozen and stay full-sized on every GPU — ZeRO-2 doesn't help them since they have no grads/opt states. For `full_finetune`, the replicated backbone weights (2B) are the dominant cost — ZeRO-2 only helps with `10B/N` (grads + opt).

### Activation Memory Detail

| Mode | Gradient Checkpointing | Activation per GPU (bs=1, seq=2048) |
|------|:----------------------:|:-----------------------------------:|
| `engram_only` | Not needed | **0.1 GB** (backbone frozen, no activation retention) |
| `engram+full_finetune` | **Required** | **8 GB** (4 × bs × seq_len × hidden × layers × 2 bytes) |
| `engram+full_finetune` | Disabled (OOM) | ~68 GB for 32B model |

### Hardware Decision Tables

All numbers in GB, assuming bs=1, seq=2048, fp16/bf16 mixed precision.

<details open>
<summary><b>Single GPU — RTX 4090 (24 GB)</b></summary>

| Backbone | `engram_only` (sparse) | `engram+LoRA` (dense) | `engram+full_finetune` |
|----------|:----------------------:|:---------------------:|:----------------------:|
| 1.5B | **9** ✅ | **9** ✅ | ❌ (32) |
| 7B | **20** ✅ | **20** ✅ | ❌ (98) |
| 13B | ❌ (32) | ❌ (33) | ❌ (170) |
| 32B | ❌ (70) | ❌ (72) | ❌ (398) |

```text
Best: engram_only + sparse. 2B backbone weights (14 GB) + ~6 GB Engram ≈ 20 GB.
full_finetune needs 12B weight+grad+opt (84 GB for 7B) — impossible on consumer hardware.
```
</details>

<details open>
<summary><b>Single GPU — A100 (80 GB)</b></summary>

| Backbone | `engram_only` (sparse) | `engram+LoRA` (dense) | `engram+full_finetune` |
|----------|:----------------------:|:---------------------:|:----------------------:|
| 1.5B | **9** ✅ | **9** ✅ | **32** ✅ |
| 7B | **20** ✅ | **20** ✅ | **98** ❌ |
| 13B | **32** ✅ | **33** ✅ | ❌ (170) |
| **32B** | **70** ✅ | **72** ✅ | ❌ (398) |
| 70B | ❌ (146) | ❌ (150) | ❌ (854) |

```text
Best: engram_only + sparse — runs 32B on a single A100.
full_finetune tops out at 1.5B on a single A100 (32 GB).
```
</details>

<details open>
<summary><b>Multi-GPU DDP (8 × RTX 4090, 24 GB)</b></summary>

DDP replicates everything on each GPU — same per-GPU memory as single 4090. Gains come from throughput (8× tokens/sec). DDP forces dense gradients, so the memory is identical to single-GPU dense mode.

| Backbone | `engram_only` (dense, DDP) | `engram+LoRA` (dense, DDP) | `engram+full_finetune` |
|----------|:--------------------------:|:--------------------------:|:----------------------:|
| 1.5B | **9** ✅ | **9** ✅ | ❌ (32) |
| **7B** | **20** ✅ | **20** ✅ | ❌ (98) |
| 13B | ❌ (32) | ❌ (33) | ❌ (170) |

**7B is the max** for `engram_only` on 4090. The 2B backbone weights (14 GB) + Engram (~6 GB) leaves ~4 GB headroom.

</details>

<details open>
<summary><b>DeepSpeed ZeRO-2 (8 × RTX 4090, 24 GB)</b></summary>

ZeRO-2 partitions gradients + optimizer states across 8 cards. **Weights stay replicated.**

| Backbone | `engram_only` (dense, ZeRO) | `engram+full_finetune` (dense, ZeRO, grad ckpt) |
|----------|:---------------------------:|:-----------------------------------------------:|
| 1.5B | **4** ✅ | **14** ✅ |
| 7B | **15** ✅ | **32** ❌ |
| 13B | **27** ❌ | **51** ❌ |

**Key insight:** `engram_only` is bottlenecked by the replicated 2B backbone weights (26 GB for 13B > 24 GB). `full_finetune` on 7B hits 32 GB — ZeRO-2 helps with `10B/N` (grads+opt) but the replicated 2B weights (14 GB) remain the dominant cost. 13B `full_finetune` at 51 GB is far beyond 24 GB — the only viable mode on 8×4090 is `engram_only` up to 7B.

> ⚠️ With 4-bit backbone quantization, `engram_only` on 8×4090 can reach 32B (22 GB/card). See Quantization section.
</details>

<details open>
<summary><b>Multi-GPU DDP (8 × A100-80 GB)</b></summary>

With DDP, each GPU holds a complete copy — same per-GPU memory as single GPU. DDP forces dense gradients, so the Engram layers use standard AdamW (same memory as single-GPU dense mode).

| Backbone | `engram_only` (dense, DDP) | `engram+LoRA` (dense, DDP) | `engram+full_finetune` |
|----------|:--------------------------:|:--------------------------:|:----------------------:|
| 7B | **20** ✅ | **20** ✅ | **98** ❌ |
| 13B | **32** ✅ | **33** ✅ | ❌ (170) |
| **32B** | **70** ✅ | **72** ✅ | ❌ (398) |
| 70B | ❌ (146) | ❌ (150) | ❌ (854) |

**Why no `full_finetune`?** DDP doesn't partition anything. `12B` (84 GB for 7B) must fit on every card — it doesn't.

**Recommendation:** Use DDP with `engram_only` for throughput scaling. For `full_finetune`, use ZeRO-2.

> DDP does not support sparse gradients — all gradients are flattened to dense buckets during all_reduce. The optimizer is automatically switched to standard AdamW. Train on a single GPU with `use_sparse_embeddings=True` if you need SparseAdam.
</details>

<details open>
<summary><b>DeepSpeed ZeRO-2 (8 × A100-80 GB)</b></summary>

| Backbone | `engram_only` (dense, ZeRO) | `engram+LoRA` (dense, ZeRO) | `engram+full_finetune` (dense, ZeRO, grad ckpt) |
|----------|:---------------------------:|:---------------------------:|:----------------------------------------------:|
| 1.5B | **4** ✅ | **4** ✅ | **14** ✅ |
| 7B | **15** ✅ | **15** ✅ | **32** ✅ |
| 13B | **27** ✅ | **27** ✅ | **51** ✅ |
| **32B** | **65** ✅ | **65** ✅ | **113** ❌ |
| 70B | ❌ (141) | ❌ (141) | ❌ (237) |

```text
ZeRO-2 partitions optimizer states + gradients across GPUs, but NOT weights.
So engram_only still pays 2B (64 GB for 32B) per card for backbone weights.
full_finetune pays 2B (weights replicated) + 10B/N (grads+opt partitioned).
On 8 GPUs, 32B full_finetune: 64 + 40 + engram + activation ≈ 113 GB — exceeds 80 GB.
```
</details>

<details open>
<summary><b>DeepSpeed ZeRO-2 (16 × A100-80 GB)</b></summary>

| Backbone | `engram_only` (dense, ZeRO) | `engram+full_finetune` (dense, ZeRO, grad ckpt) |
|----------|:---------------------------:|:----------------------------------------------:|
| 7B | **15** ✅ | **27** ✅ |
| 13B | **27** ✅ | **43** ✅ |
| **32B** | **65** ✅ | **93** ❌ |
| 70B | ❌ (141) | ❌ (193) |

With 16 GPUs, `full_finetune` up to 13B fits comfortably. 32B `full_finetune` requires ~93 GB — still beyond 80 GB even with 16 GPUs, because the replicated 2B weights (64 GB) don't shrink with more GPUs.

`engram_only` on 70B is blocked by unreplicated 2B weights (140 GB).
</details>

### Decision Flowchart

```
Q: What is your GPU budget?
├─ Single GPU (24 GB)
│  ├─ engram_only + sparse SparseAdam → up to 7B
│  └─ engram_only + 4-bit → up to 32B (!)
├─ Single GPU (80 GB)
│  ├─ engram_only + sparse SparseAdam → up to 32B
│  ├─ engram_only + 4-bit → up to 130B (!)
│  └─ engram_only + 8-bit → up to 70B
├─ Multi-GPU DDP (24 GB cards, e.g. 8×4090)
│  ├─ engram_only (dense AdamW) → up to 7B (throughput)
│  └─ engram_only + 4-bit → up to 32B (throughput, comfortable headroom)
├─ Multi-GPU ZeRO-2 (24 GB cards, e.g. 8×4090)
│  ├─ engram+full_finetune → up to 1.5B (ZeRO-2 helps grads+opt, but replicated weights block larger models)
│  └─ engram_only → 4-bit recommended (ZeRO-2 adds no value for frozen backbone)
├─ Multi-GPU DDP (80 GB cards)
│  ├─ engram_only (dense AdamW) → up to 32B (throughput scaling)
│  └─ engram+full_finetune → use ZeRO-2 instead (DDP replicates all 12B per card)
└─ Multi-GPU ZeRO-2 (80 GB cards)
   ├─ engram_only → up to 32B (8 or 16 GPUs), bottleneck at replicated 2B weights
   ├─ engram+full_finetune → up to 13B (8 GPUs), up to 13B (16 GPUs)
   └─ engram+LoRA → similar to engram_only (LoRA is tiny)
```

### Key Takeaways

1. **Single GPU with `engram_only` + sparse SparseAdam** is the most memory-efficient without quantization because the backbone (7B, 13B, 32B) requires zero gradients or optimizer states. It's the only way to train a 7B model on a single 24 GB card, or a 32B model on a single 80 GB card.

2. **Add 4-bit quantization and `engram_only` goes even further**: 32B on a single 4090 (22 GB), 70B on a single A100 (41 GB), 130B on a single A100 (71 GB). This is the single biggest memory lever.

3. **DDP does NOT support sparse gradients**: PyTorch's `DistributedDataParallel` flattens all gradients into dense buckets before all_reduce, regardless of the backend (NCCL or Gloo). When DDP is active, `nn.Embedding(sparse=True)` is automatically overridden to `sparse=False`, and `EngramTrainer` switches from `MixedOptimizer` to standard `AdamW`. The memory formula is identical to single-GPU dense mode — DDP provides throughput scaling, not memory savings. For sparse SparseAdam benefits (optimizer step efficiency), use single-GPU training.

4. **ZeRO-2's value is limited for `engram_only` and `full_finetune`** because **weights are replicated**. For `engram_only`, the backbone weights (2B) are the dominant cost and ZeRO-2 doesn't touch them. For `full_finetune`, the replicated 2B backbone weights are the floor — ZeRO-2 only partitions `10B/N` (grads+opt). On 8×A100, `full_finetune` maxes out at 13B (51 GB).

5. **`full_finetune` with ZeRO-2 on 16 × A100-80GB** can handle up to 13B comfortably (43 GB). 32B `full_finetune` requires ~93 GB — still beyond 80 GB because the 64 GB of replicated backbone weights can't be partitioned by ZeRO-2. To go beyond 13B with `full_finetune`, ZeRO-3 (which partitions weights) would be needed, but ZeRO-3 compatibility is not yet tested.

6. **8 × RTX 4090 (24 GB) with ZeRO-2** can `full_finetune` only up to 1.5B (14 GB). Even 7B `full_finetune` (32 GB) exceeds 24 GB. For these cards, `engram_only` (up to 7B) or `engram_only` + 4-bit (up to 32B) is the only viable path.

7. **With 4-bit + DDP on 8 × 4090**, you can train 32B `engram_only` comfortably (22 GB/card) with high throughput — the best price-performance ratio in this document.

---

## Quantization & Memory

Quantization (4-bit NF4 or 8-bit via `bitsandbytes`) shrinks the **backbone weights only** — Engram layers stay in full precision for training stability. The effect on memory depends heavily on whether the backbone is frozen or trainable.

### Updated Memory Formulas with Quantization

| Mode | Quant | Formula | Notes |
|------|-------|---------|-------|
| `engram_only` | None | `2B + 16E + 0.1` GB | backbone frozen, fp16 weights |
| `engram_only` | 4-bit NF4 | `0.5B + 16E + 0.1` GB | backbone frozen, nf4 weights |
| `engram_only` | 8-bit | `1B + 16E + 0.1` GB | backbone frozen, 8-bit weights |
| `engram_only` ZeRO-2 (N GPUs) | 4-bit NF4 | `0.5B + 2E + 10E/N + 0.1` GB | ZeRO-2 partitions only engram grads+opt |
| `engram+LoRA` | 4-bit NF4 | `0.5B + 16E + 0.1` GB | backbone frozen (QLoRA-style), LoRA adds <0.1 GB |
| `engram+full_finetune` | None | `12B + 16E + 8` GB | needs grad ckpt |
| `engram+full_finetune` (ZeRO-2, N GPUs) | None | `2B + 10B/N + 2E + 10E/N + 8` GB | weights replicated, grads+opt partitioned |
| `engram+full_finetune` (ZeRO-2, N GPUs) | 8-bit | `1B + 4B/N + 2E + 10E/N + 8` GB† | Adam8bit on backbone, weights replicated in 8-bit |

> † 8-bit `full_finetune` ZeRO-2: weights in 8-bit (1B), gradients in fp16 (2B/N), Adam states in 8-bit (2B/N) ≈ `1B + 4B/N` per backbone param. This is a rough estimate — actual memory depends on `bitsandbytes` internals.
>
> †† All ZeRO-2 formulas reflect that **weights are replicated** (ZeRO-2 partitions only gradients and optimizer states). For frozen-backbone modes with 4-bit quantization, ZeRO-2 adds negligible benefit (backbone is 0.5B/card, engram grads+opt are already small).

### How Much Does Quantization Save?

The saving is purely on **backbone weights** (2B → 0.5B or 1B). For frozen-backbone modes (`engram_only`, `engram+LoRA`), there are no backbone grads/opt to save on — everything goes to shrinking the static weight footprint.

| Backbone | 2B (fp16) | 0.5B (4-bit) | Saving |
|----------|:---------:|:------------:|:------:|
| 7B | 14 GB | 3.5 GB | **10.5 GB** |
| 13B | 26 GB | 6.5 GB | **19.5 GB** |
| 32B | 64 GB | 16 GB | **48 GB** |
| 70B | 140 GB | 35 GB | **105 GB** |

### Practical Impact — Single GPU

**4-bit + `engram_only` changes everything on consumer GPUs:**

<details open>
<summary><b>Single GPU — RTX 4090 (24 GB) + 4-bit backbone</b></summary>

| Backbone | `engram_only` (4-bit) | `engram+LoRA` (4-bit) |
|----------|:---------------------:|:---------------------:|
| 7B | **9** ✅ | **9** ✅ |
| 13B | **12** ✅ | **12** ✅ |
| **32B** | **22** ✅ | **22** ✅ |
| 70B | **41** ❌ | **41** ❌ |

```text
Without quantization: 32B hits 70 GB — need A100.
With 4-bit: 32B drops to 22 GB — fits on a single RTX 4090.
```
</details>

<details>
<summary><b>Single GPU — A100 (80 GB) + 4-bit backbone</b></summary>

| Backbone | `engram_only` (4-bit) | `engram+full_finetune` (none) |
|----------|:---------------------:|:----------------------------:|
| 32B | **22** ✅ | **98** ❌ |
| **70B** | **41** ✅ | ❌ |
| 130B | **71** ✅ | ❌ |

```text
4-bit + engram_only pushes the reach from 32B to 130B on a single A100.
```
</details>

<details>
<summary><b>Single GPU — A100 (80 GB) + 8-bit backbone</b></summary>

| Backbone | `engram_only` (8-bit) | `engram+full_finetune` (8-bit)† |
|----------|:---------------------:|:------------------------------:|
| 32B | **38** ✅ | **142** ❌ |
| 70B | **76** ✅ | **294** ❌ |
| 130B | ❌ (136) | ❌ |

> † 8-bit `full_finetune` single GPU: 4B (8-bit weights + fp16 grads + 8-bit Adam) + 16E + 8. ~142 GB for 32B — infeasible on a single GPU. Use ZeRO-2 to partition grads+opt.

```text
8-bit engram_only fits 70B on single A100 with padding (76 GB).
8-bit full_finetune on a single GPU is only feasible for tiny models (<1B).
```
</details>

### Practical Impact — Multi-GPU

<details open>
<summary><b>Multi-GPU DDP (8 × RTX 4090, 24 GB, 4-bit backbone)</b></summary>

DDP replicates everything per GPU — but with 4-bit, each card's footprint is tiny. DDP forces dense gradients, but the memory formula is the same (Engram optimizer state dominates at 6 GB, not the gradient format).

| Backbone | `engram_only` (4-bit, DDP) |
|----------|:--------------------------:|
| 7B | **9** ✅ |
| 13B | **12** ✅ |
| **32B** | **22** ✅ |
| 70B | **41** ❌ |

```text
32B engram_only trains on 8×4090 DDP with huge headroom (22 GB / 24 GB).
Perfect scenario: cheap hardware, large model, high throughput, dense AdamW optimizer.
```
</details>

<details open>
<summary><b>Multi-GPU ZeRO-2 (8 × RTX 4090, 24 GB, 4-bit backbone)</b></summary>

With 4-bit backbone, ZeRO-2's partitioning of optimizer states only helps the engram parameters (the only trainable params). The backbone weights are 0.5B per card — too small to benefit from further partitioning.

| Backbone | `engram_only` (4-bit, ZeRO) | `engram+LoRA` (4-bit, ZeRO) |
|----------|:---------------------------:|:---------------------------:|
| 7B | **5** ✅ | **5** ✅ |
| 13B | **8** ✅ | **8** ✅ |
| **32B** | **17** ✅ | **17** ✅ |
| 70B | ❌ (36) | ❌ (36) |

```text
4-bit backbone shrinks the weight floor dramatically: 32B backbone = 16 GB (4-bit) + ~1.3 GB engram = 17 GB.
70B backbone = 35 GB + 1.3 GB = 36 GB — exceeds 24 GB even with 4-bit.
engram_only and engram+LoRA converge to nearly identical memory (Engram dominates over LoRA).
```
</details>

### Quantization Decision Rules

```
Q: Can I use quantization?
├─ Do I have enough VRAM without it?
│  └─ Yes → skip quantization (faster, simpler)
│  └─ No → which mode?
│     ├─ engram_only → 4-bit NF4 backbone → huge reach gain (32B → 130B on A100)
│     ├─ engram+LoRA → 4-bit NF4 backbone → same reach as engram_only
│     └─ engram+full_finetune → consider 8-bit + ZeRO-2 or skip quantization
```

### Key Quantization Takeaways

1. **4-bit + `engram_only` is the killer combination**: A 32B model goes from requiring A100 (70 GB) to fitting on RTX 4090 (22 GB). A 70B model fits on a single A100 (41 GB).

2. **ZeRO-2 adds minimal value with a 4-bit backbone**: The backbone is frozen and quantized (0.5B/card). Enram trainable overhead with ZeRO-2 is only ~1.3 GB (`2E + 10E/N + 0.1`). The limiting factor is the 4-bit backbone weight floor (0.5B), which ZeRO-2 can't partition.

3. **DDP + 4-bit + `engram_only`** is the cheapest path to 32B training: 8×RTX 4090, each card at 22 GB — comfortable headroom for throughput scaling with standard AdamW.

4. **`full_finetune` benefits least from quantization**: The backbone needs to be trainable, so 4-bit doesn't help (must dequantize). 8-bit helps through Adam8bit, but `full_finetune` still needs fp16 gradients + replicated weights — infeasible on single GPU beyond tiny models.

5. **QLoRA (`engram+LoRA` with 4-bit) has similar memory to `engram_only` + 4-bit**: LoRA overhead is negligible (<0.1 GB). The engram memory (16E ≈ 5.8 GB single GPU, or ~1.3 GB ZeRO-2) is the dominant trainable cost.


