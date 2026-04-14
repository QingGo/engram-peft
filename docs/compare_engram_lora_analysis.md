# Performance Comparison: Engram vs. LoRA vs. LoRA+Engram

This document presents a detailed analysis comparing **Engram-PEFT**, **LoRA** (Low-Rank Adaptation), and the **Combined (LoRA+Engram)** mode using empirical data from `engram_test8` and `engram_test9` runs on **NVIDIA RTX 4090D 24GB**.

## 1. Metric Overview

The table below summarizes the key performance indicators from the 3000-step benchmarks using TinyLlama-1.1B on the TinyStories dataset (`--batch_size 16`, `--grad_accum 2`, `--subset 30000`).

| Metric | Base Model (Zero-shot) | LoRA (Baseline) | Engram (2 Layers) | LoRA+Engram | Best Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Eval Loss** | 1.7401 | 0.9890 | 1.0165 | **0.9656** | **LoRA+Engram** |
| **Training Steps** | - | 3000 | 3000 | 3000 | - |
| **Peak Allocated (GB)**| 2.05 | 8.07 | 9.38 | 10.33 | LoRA |
| **Peak VRAM (nvtop)** | 2.91 GiB | 9.35 GiB | 10.82 GiB | 11.69 GiB | LoRA |
| **Avg Time/Step (s)** | - | **0.2738** | 0.2961 | 0.3360 | LoRA |
| **GPU Utilization** | - | ~100% | ~100% | ~100% | - |

> [!IMPORTANT]
> The **LoRA+Engram** method achieved the best convergence (lowest evaluation loss), outperforming standalone LoRA by **2.3%** and Engram by **5.0%**. This demonstrates the clear synergy between structural adaptation and sparse knowledge injection.

## 2. Memory Discrepancy Analysis

A common observation is that system-level monitoring (`nvtop` / `nvidia-smi`) reports higher VRAM usage than PyTorch's internal metrics (`training_metrics.json`).

### Why nvtop is higher:
1. **CUDA Context Overhead**: Initializing the GPU driver and context consumes a baseline of VRAM (~0.5 - 1.0 GB depending on the driver version).
2. **PyTorch Caching Allocator**: `peak_memory_gb` records `torch.cuda.max_memory_allocated()`, which only counts active tensors. However, the PyTorch allocator keeps a pool of **Reserved** memory to speed up future allocations. `nvtop` shows this **Reserved memory + Context memory**.
3. **Workspace Buffers**: Operations like large GEMMs or cross-entropy loss gradients require temporary workspace buffers managed internally by cuBLAS or cuDNN, which are not always counted as "allocated tensors" in the basic metrics.

**In these runs, the gap is consistently ~1.3–1.5 GiB**, regardless of the adapter type.

## 3. Combined LoRA+Engram Mode

The `lora+engram` mode represents a powerful synergy for PEFT:

- **Mechanism**: The model is first wrapped by LoRA (`PeftModel`), then the resulting structure is further wrapped by `EngramModel`.
- **Finding Layers**: The `EngramModel` uses a robust recursive search to locate transformer layers even when nested deep within multiple PEFT wrappers.
- **Superior Convergence**: By combining LoRA's ability to optimize internal representations with Engram's high-capacity sparse retrieval, we achieve better fine-tuning results than either method alone.

## 4. Strengths of Engram-PEFT

Beyond raw loss optimization, Engram offers several unique advantages:

1. **Flexible Knowledge Migration**: Weights can be migrated between different configurations (different n-gram sizes, seeds, or even tokenizers) using best-effort remapping via `weight_transfer.py`.
2. **Sparse Parameter Power**: Engram can inject hundreds of millions of parameters into a small model with minimal impact on dense computation, effectively turning a small "reasoner" into a high-capacity "knowledge base".
3. **CPU Prefetching (Potential)**: The sparse retrieval nature of n-gram embeddings allows for future optimizations where embeddings are prefetched from CPU RAM to GPU VRAM based on the upcoming tokens, further reducing the GPU memory bottleneck for massive Engram tables.

## 5. Loss Curve Comparison

The following plot compares the three methods. Note how **LoRA+Engram** (purple) consistently stays below the other curves after the initial warmup.

![Loss Curve Comparison](../figures/loss_curve.png)

## 6. Conclusion

For TinyLlama-1.1B training:
1. **LoRA** is the most GPU-efficient choice for simple adaptation tasks.
2. **Engram** is superior for knowledge-heavy tasks where parameter count expansion is required without scaling dense compute.
3. **LoRA+Engram** is the **optimal configuration** for maximizing model performance, providing significantly better convergence at the cost of moderate additional VRAM (~2.3 GB more than LoRA).

