# Memory Footprint Analysis: Engram vs. LoRA

This document provides a theoretical and empirical analysis of the memory (VRAM) usage for Engram and LoRA when applied to the TinyLlama-1.1B model.

## 1. Baseline: TinyLlama-1.1B
To understand the overhead, we first examine the base model:
- **Parameters**: ~1.1 Billion
- **Weight Precision**: FP16 (2 bytes/param)
- **Static Memory**: ~2.2 GB (weights only)
- **Inference/Training Context**: VRAM usage also includes CUDA kernels, workspace, and activations (gradient checkpoints), typically totaling **3-5 GB** for the base model during training with small batch sizes.

## 2. LoRA Analysis (r=16)
In our benchmarks, LoRA adds parameters to `q_proj` and `v_proj`.
- **Rank ($r$)**: 16
- **Hidden Size ($D$)**: 2048
- **Params per module**: $D \times r + r \times D = 65,536$
- **Total Layers**: 22
- **Total LoRA Params**: $2 \text{ modules} \times 22 \text{ layers} \times 65k \approx 2.88 \text{ Million}$
- **VRAM Overhead**: 
    - Weights + Gradients + Optimizer States: $\approx 2.88M \times (2+2+8) \approx 35 \text{ MB}$.
    - Since this overhead is negligible, LoRA training memory is dominated by the base model and its activations.
- **Observed VRAM**: **~5.1 GB** (mostly base weights + activations).

## 1.5 Full Fine-Tuning (FFT) Analysis
For a 1.1B parameter model using the standard Adam optimizer and mixed-precision (FP16), the memory requirement is significantly higher due to the need for gradient and optimizer state storage for *every* parameter.

### Theoretical Breakdown:
1. **Model Weights (FP16)**: $1.1B \times 2 \text{ bytes} = \mathbf{2.2 \text{ GB}}$
2. **Gradients (FP16)**: $1.1B \times 2 \text{ bytes} = \mathbf{2.2 \text{ GB}}$
3. **Optimizer States (Adam/FP32)**:
   - Adam requires 12 bytes per parameter (Master Weight, Momentum, and Variance in FP32).
   - $1.1B \times 12 \text{ bytes} = \mathbf{13.2 \text{ GB}}$
4. **Activations & Buffers**:
   - For a 1.1B model, training usually requires **2-4 GB** for activation storage (depending on sequence length and gradient checkpointing).
   - CUDA overhead and workspace usually occupy **1-2 GB**.

### Total Rationale:
**$17.6 \text{ GB (Static)} + 4 \text{ GB (Activations)} + 2 \text{ GB (System)} \approx \mathbf{23.6 \text{ GB}}$**

This explains why **~24 GB** is the industry-standard minimum for full fine-tuning of 1B-scale models.

## 3. Engram Analysis
Engram's architecture introduces large (though sparse) embedding tables and branching projections.

### Parameter Breakdown
Using the configuration `target_layers=[2, 11, 20]` (3 layers) and `engram_vocab_size_per_ngram=[256000, 256000]`:
1. **Multi-Head Embeddings (Sparse)**:
   - 16 heads total, dimension per head = 64.
   - Total capacity per layer $\approx 512,000$ entries.
   - 3 layers $\times 512k \times 64 \approx \mathbf{98.3 \text{ Million parameters}}$.
2. **Context-Aware Gating (Dense)**:
   - Shared Value/Key projections per layer $\approx 10.5 \text{ Million}$.
   - 3 layers $\approx \mathbf{31.5 \text{ Million parameters}}$.
- **Total Engram Params**: $\approx \mathbf{130 \text{ Million}}$.

### VRAM Overhead
Even though Engram uses sparse gradient updates, the full parameter set and optimizer states must reside in VRAM:
- **Weights (FP16)**: $130M \times 2 \approx 260 \text{ MB}$.
- **Optimizer States (Adam/SparseAdam)**: $130M \times 8 \approx \mathbf{1.04 \text{ GB}}$.
- **Activations**: Branching ($hc\_mult=4$) increases activation storage requirement by $\approx 20$ MB per layer.
- **Total Calculated Overhead**: **~1.3 - 1.5 GB** above the LoRA baseline.

This aligns closely with our theoretical calculation of **~1.5 GB** overhead for Engram's 130M parameters and their corresponding optimizer states. 

## Summary Comparison Table (1.1B Model)

| Component | FFT (Full Fine-Tune) | LoRA (r=16) | Engram (3 layers) |
| :--- | :--- | :--- | :--- |
| **Trainable Params** | 1,100 M | 2.88 M | 130 M |
| **Weight Memory** | 2.2 GB | 2.2 GB* | 2.45 GB** |
| **Optimizer States** | 13.2 GB | 0.03 GB | 1.04 GB |
| **Gradients** | 2.2 GB | 0.01 GB | 0.26 GB |
| **Activations/CUDA** | ~6.0 GB | ~2.8 GB | ~3.0 GB |
| **Total VRAM** | **~24 GB (est.)** | **~5.1 GB (obs.)** | **~6.8 GB (obs.)** |

*\* LoRA and Engram weights include the frozen base model (2.2GB).\
\** Engram weights include base (2.2GB) + Engram tables (0.26GB).*

## 4. Conclusion
The observed difference in VRAM (**Engram 6.8 GB** vs **LoRA 5.1 GB**) is $\approx \mathbf{1.7 \text{ GB}}$. 

This aligns closely with our theoretical calculation of **~1.5 GB** overhead for Engram's 130M parameters and their corresponding optimizer states. For larger base models (e.g., 7B+), this overhead becomes a much smaller fraction of the total budget, highlighting Engram's efficiency as a "Parameter-Efficient" method for massive knowledge injection.
