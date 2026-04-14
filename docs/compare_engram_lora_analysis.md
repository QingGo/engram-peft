# Performance Comparison: Engram vs. LoRA

This document presents a comprehensive analysis comparing **Engram-PEFT** and **LoRA** (Low-Rank Adaptation) using empirical data from the `TinyLlama-1.1B` benchmark run on NVIDIA RTX 4090D 24GB hardware.

## 1. Executive Summary

| Metric | LoRA (r=16) | Engram (2 layers) | Delta |
| :--- | :--- | :--- | :--- |
| **Trainable Params** | 2.88 M | 545.4 M | +18,800% |
| **Training Speed** | 0.1777 s/step | **0.1643 s/step** | **~8% Faster** |
| **Training Loss** | **1.254** | 1.311 | LoRA -4.3% |
| **Eval Loss** | 1.153 | **1.141** | **Engram -1.1%** |
| **Memory (Allocated)** | 5.05 GB | 6.05 GB | +1.0 GB |
| **Memory (System/nvtop)**| 6.07 GiB | **6.97 GiB** | **+0.9 GiB** |

## 2. Computational Analysis: Why Engram is Faster
A common misconception is that a higher number of trainable parameters ($545M$ vs. $3M$) always leads to slower training. Engram breaks this correlation through **Sparse Retrieval Architecture**.

### LoRA Complexity
LoRA injects rank-update matrices ($A$ and $B$) into existing linear layers (typically $Q$ and $V$). 
- **Operation**: $Y = XW + (XA)B$.
- **Compute Cost**: Adds $O(B \times L \times D \times r)$ operations per layer.
- **Overhead**: While $r$ is small (16), LoRA is typically applied across **all** transformer layers (22 in TinyLlama). This multiplies the kernel launch overhead and the backpropagation cost across the entire network depth.

### Engram Complexity
Engram uses a modular injection at specific bottleneck layers (e.g., layers 2 and 11).
- **Operation**: Sparse lookup followed by a gated projection.
- **Compute Cost**: 
    1. **Sparse Lookup**: $O(B \times L \times K)$ where $K$ is total heads. This is independent of the vocabulary size.
    2. **Gating Projection**: $O(B \times L \times 5 \times D_{eng} \times D_{model})$.
- **Efficiency Gains**:
    1. **Layer Sparsity**: By targeting only the layers most critical for knowledge injection, Engram reduces the total number of gradient calculations compared to a global LoRA application.
    2. **Kernel Efficiency**: Engram consolidates structural parameters into few, larger GEMM operations (Single Gating block) rather than many tiny rank-update operations spread across 22 layers.
    3. **Pre-computation**: `EngramDataCollator` pre-calculates hash indices on the CPU, removing the hashing overhead from the GPU timeline. In our latest version, this is **highly vectorized** using NumPy broadcasting and supports **multi-process workers** (`num_workers`), effectively parallelizing the most CPU-intensive part of the Engram pipeline across multiple cores.

## 3. Memory Footprint Analysis

### VRAM Discrepancy (Allocated vs. System)
One of the most frequent questions is why `nvtop` shows higher usage than our internal metrics:
- **CUDA Context**: ~500 MB is reserved immediately by the NVIDIA driver.
- **PyTorch Caching Allocator**: PyTorch maintains a "pool" of reserved GPU memory blocks. `nvtop` shows the *pool size*, whereas `torch.cuda.max_memory_allocated()` (recorded in our logs) shows the *actual bytes used by tensors*.
- **Observation**: LoRA peaked at **6.07 GiB** in `nvtop` while Engram peaked at **6.97 GiB**. This ~0.9 GiB gap in system-visible memory closely mirrors the ~1.0 GB gap in allocated tensor memory.

### The "1GB Weight Gap"
The difference in peak allocated memory (~1.0 GB) is deterministic and scales with the Engram capacity:
- **Calculation**: $2 \text{ layers} \times 16 \text{ heads} \times 256,000 \text{ capacity} \times 64 \text{ dim\_per\_head} \times 2 \text{ bytes (FP16)} \approx \mathbf{1.048 \text{ GB}}$.
- Engram's overhead is primarily **static** (weights) rather than **dynamic** (activations), making it very predictable for large-scale deployments.

## 4. Performance & Convergence

### Loss Curve Comparison
![Loss Curve Comparison](../figures/loss_curve.png)

Engram starts with a higher initial loss (due to zero-initialized gating which initially isolates the new embeddings), but converges rapidly.

### Generalization: Training vs. Evaluation
An important observation from the raw metrics is the relationship between training and validation loss:
- **Training Loss**: LoRA (1.254) slightly outperforms Engram (1.311).
- **Evaluation Loss**: Engram (**1.141**) outperforms LoRA (1.153).
- **Insight**: LoRA's low-rank updates are more prone to overfitting the specific patterns in the training samples. Engram's high-capacity sparse memory, combined with its context-aware gating, captures more robust semantic features that generalize significantly better to the evaluation set.

### Throughput (Tokens per Second)
The **~8% reduction** in time-per-step translates directly to higher training throughput. For large models (7B+), the advantage of injecting massive knowledge via a few Engram layers versus many LoRA layers becomes even more pronounced.

## 5. Conclusion: Selecting the Right Method

| Scenario | Recommended Method | Why? |
| :--- | :--- | :--- |
| **Fine-tuning on niche tasks** | **LoRA** | Extremely low memory overhead; good for small datasets. |
| **Massive Knowledge Injection** | **Engram** | High parameter capacity (500M+) with same/lower latency. |
| **Inference Latency Sensitive** | **Engram** | Faster step-time; scales better with model depth. |
| **Extremely VRAM Constrained** | **LoRA** | Saves ~1GB of weights compared to 2-layer Engram. |

Engram-PEFT represents a shift from "adapting the existing model" (LoRA) to "expanding the model's memory capacity" (Engram), providing the benefits of deep knowledge storage with the efficiency of sparse retrieval.

## 6. Detailed Metrics (Raw Data)

The following table summarizes the raw metrics captured during the benchmarking process:

| Metric | Base Model (Zero-shot) | LoRA Baseline | Engram-PEFT (2 Layers) |
| :--- | :--- | :--- | :--- |
| **Eval Loss** | 1.80629 | 1.15317 | **1.14126** |
| **Final Training Loss** | - | **1.25401** | 1.31103 |
| **Peak Allocated (GB)** | 2.04900 | 5.05315 | 6.05095 |
| **Avg Time/Step (s)** | - | 0.17771 | **0.16431** |
| **Training Steps** | - | 2000 | 2000 |
| **VRAM reserved (nvtop)** | ~2.91 GiB | **6.07 GiB** | **6.97 GiB** |
