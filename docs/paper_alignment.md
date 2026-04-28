# Paper Alignment

This document details the alignment between the `engram-peft` implementation and the official DeepSeek Engram paper: **"Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"** (arXiv:2601.07372).

## Core Architecture Mapping

| Paper Component | Code Implementation | Implementation Detail |
| :--- | :--- | :--- |
| **Tokenizer Compression** (Sec 2.2) | `compression.py` | Implements `P: V→V'` surjective mapping using `NFKC` + `Lowercase` + `Strip`. |
| **Multi-Head Hashing** (Sec 2.2) | `hashing.py` | Polynomial hashing followed by bitwise XOR. Uses prime-sized tables $M_{n,k}$. |
| **Context-Aware Gating** (Sec 2.3) | `layer.py:ContextAwareGating` | Formula: $\alpha_t = \sigma( \text{RMSNorm}(h_t)^\top \text{RMSNorm}(k_t) / \sqrt{d} )$. |
| **Short-term Memory** (Sec 2.3) | `layer.py:ShortConv` | Depth-wise 1D Conv with kernel size 4 and dilation equal to max N-gram size. |
| **Mixed Optimization** (Sec 4.1) | `utils.py:get_optimizer` | `SparseAdam` for retrieval embeddings with $5\times$ learning rate multiplier. |

## Hyperparameters (Appendix A Table 5)

Our `EngramConfig` defaults are 100% aligned with the configurations for the 27B and 40B models specified in the paper.

| Parameter | Paper Value (27B) | `EngramConfig` Default |
| :--- | :--- | :--- |
| Engram Dim $d_{mem}$ | 1280 | `embedding_dim: 1280` |
| Engram Vocab Size | 2,262,400 | `engram_vocab_size_per_ngram: [1131200, 1131200]` |
| Hash Heads $K$ | 8 | `n_head_per_ngram: 8` |
| Target Layers | [1, 14] | `target_layers: [2, 15] (0-indexed)` |
| N-gram Orders $n$ | [2, 3] | `ngram_sizes: [2, 3]` |
| LR Multiplier | $5\times$ | `learning_rate_multiplier: 5.0` |
| Conv Zero Init | True | `conv_zero_init: True` |

## Official Implementation Details (via Demo)

The following specific implementation details from the DeepSeek official demo are incorporated:

1.  **Gating Activation**:
    `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    This ensures numerical stability and matches the exact gating behavior of the original research.
2.  **Polynomial Hash Coefficients**:
    Unique random multipliers are generated for each head and each layer to minimize hash collisions across the architecture.
3.  **Efficiency**:
    Embedding tables for all heads in a layer are concatenated into a single larger `nn.Embedding` and indexed using offsets for maximum GPU throughput.
4.  **mHC (multi-Head Hyper-connection)**:
    We support `hc_mult=4` which expands the hidden states before gating, as described in the paper's efficient hyper-connection section.

## Weight Reuse & Knowledge Transfer

One of the key practical advantages of the Engram design—implied by its deterministic hashing and modular nature—is the ability to reuse learned memory across different environments. We implement several enhancements beyond the paper's base training logic:

1.  **Structural Invariance**:
    Because each N-gram head is independent, weights can be migrated between models with different `target_layers` or different `engram_vocab_size_per_ngram` (via slicing/padding).
2.  **Logic Alignment (Seeds & Tokenizers)**:
    The paper emphasizes "normalized textual equivalence." By leveraging this, we can align weights between different tokenizers (e.g., Llama-2 vs Qwen) by using character-level offset mappings on a reference corpus to synchronize the logical hashes.
3.  **Cross-Model Knowledge Distillation**:
    A trained Engram module can be treated as a portable "knowledge pack." Our implementation supports loading weights even when hashing seeds differ, by using a best-effort remapping strategy that recovers the semantic mapping from a sample of text data.
