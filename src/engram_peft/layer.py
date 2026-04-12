import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, cast, Dict, Tuple, List
from engram_peft.config import EngramConfig
from engram_peft.compression import TokenizerCompressor
from engram_peft.hashing import MultiHeadHash


class ContextAwareGating(nn.Module):
    """
    Context-Aware Gating module as described in Section 2.3 and 2.4 of the Engram paper.

    This module computes a gating signal based on the context (h_t) and the retrieved
    Engram embeddings (e_t), and applies it to the value projection of the embeddings.
    It also includes a depthwise causal convolution and a residual connection.

    Attributes:
        embedding_dim (int): Total dimension of the concatenated Engram embeddings (e_t).
                             e_t is formed by concatenating embeddings from all N-gram orders
                             and all hash heads: e_t = ||_{n} ||_{k} e_{t,n,k}, e_t ∈ R^{d_mem}.
        hidden_dim (int): Dimension of the Transformer hidden states (h_t).
        num_branches (int): Number of branches in the multi-branch architecture.
        kernel_size (int): Size of the convolution kernel (fixed to 4 in the paper).
        dilation (int): Dilation rate for the convolution.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,  # Total concatenated dimension (e.g. 1024 dim)
        hidden_dim: int = 2560,
        num_branches: int = 1,
        kernel_size: int = 4,
        dilation: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        self.kernel_size = kernel_size
        self.dilation = dilation if dilation is not None else 1

        # Step 1: Shared Value projection: W_V
        self.w_v = nn.Linear(embedding_dim, hidden_dim, bias=False)

        # Step 2: Branch-specific Key projection: W_K
        # We use a single linear layer to project to all branches at once
        self.w_k = nn.Linear(embedding_dim, num_branches * hidden_dim, bias=False)

        # Step 3: Branch-specific RMSNorm for h_t and k_t
        self.norm_h = nn.ModuleList(
            [nn.RMSNorm(hidden_dim) for _ in range(num_branches)]
        )
        self.norm_k = nn.ModuleList(
            [nn.RMSNorm(hidden_dim) for _ in range(num_branches)]
        )

        # Step 6: Depthwise causal convolution
        # parameters are shared across branches if num_branches > 1
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=hidden_dim,  # Depthwise
            bias=True,
        )

        # Step 8: Branch-specific RMSNorm before convolution
        self.norm_v_tilde = nn.ModuleList(
            [nn.RMSNorm(hidden_dim) for _ in range(num_branches)]
        )

        # Initialize convolution to zero to preserve identity mapping (Y = v_tilde initially)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters as per paper requirements."""
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        # Linear layers use default initialization (Kaiming Uniform)

    def forward(self, e_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ContextAwareGating module.

        Args:
            e_t: [batch_size, seq_len, embedding_dim] - Concatenated Engram embeddings.
            h_t: [batch_size, seq_len, num_branches, hidden_dim] if num_branches > 1,
                 else [batch_size, seq_len, hidden_dim].

        Returns:
            torch.Tensor: Output tensor Y with the same shape as h_t.
        """
        batch_size, seq_len, _ = e_t.shape

        # Step 1: Shared Value projection: v_t = W_V e_t
        v_t = self.w_v(e_t)  # [B, L, D]

        # Step 2: Branch-specific Key projection: k_t = W_K e_t
        k_t_raw = self.w_k(e_t)  # [B, L, M * D]

        if self.num_branches > 1:
            k_t = k_t_raw.view(batch_size, seq_len, self.num_branches, self.hidden_dim)
            # h_t is already [B, L, M, D]
        else:
            k_t = k_t_raw  # [B, L, D]
            # h_t is [B, L, D]

        # Step 3: Branch-specific RMSNorm h_t and k_t
        if self.num_branches > 1:
            h_t_norm_list = []
            k_t_norm_list = []
            for i in range(self.num_branches):
                h_t_norm_list.append(self.norm_h[i](h_t[:, :, i, :]))
                k_t_norm_list.append(self.norm_k[i](k_t[:, :, i, :]))
            h_t_norm = torch.stack(h_t_norm_list, dim=2)
            k_t_norm = torch.stack(k_t_norm_list, dim=2)
        else:
            h_t_norm = self.norm_h[0](h_t)
            k_t_norm = self.norm_k[0](k_t)

        # Step 4: Calculate gating α_t = σ( (h_t_norm · k_t_norm) / sqrt(D) )
        # dot product over the last dimension D
        dot_product = (h_t_norm * k_t_norm).sum(dim=-1) / (self.hidden_dim**0.5)

        # Stability trick from official demo:
        # gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        stable_dot_product = (
            dot_product.abs().clamp_min(1e-6).sqrt() * dot_product.sign()
        )
        alpha_t = torch.sigmoid(stable_dot_product)

        # Step 5: Gating modulation: ṽ_t = α_t · v_t
        if self.num_branches > 1:
            # alpha_t: [B, L, M] -> [B, L, M, 1]
            # v_t: [B, L, D] -> [B, L, 1, D]
            v_tilde = alpha_t.unsqueeze(-1) * v_t.unsqueeze(2)  # [B, L, M, D]
        else:
            # alpha_t: [B, L] -> [B, L, 1]
            # v_t: [B, L, D]
            v_tilde = alpha_t.unsqueeze(-1) * v_t  # [B, L, D]

        # Step 6 & 7: Depthwise causal convolution
        # Step 8: Branch-specific RMSNorm(ṽ) before convolution
        if self.num_branches > 1:
            # Apply branch-specific RMSNorm
            v_tilde_norm_list = []
            for i in range(self.num_branches):
                v_tilde_norm_list.append(self.norm_v_tilde[i](v_tilde[:, :, i, :]))
            v_tilde_norm = torch.stack(v_tilde_norm_list, dim=2)

            # [B, L, M, D] -> [B, M, D, L] -> [B*M, D, L]
            x_norm = (
                v_tilde_norm.permute(0, 2, 3, 1)
                .reshape(batch_size * self.num_branches, self.hidden_dim, seq_len)
                .contiguous()
            )
        else:
            v_tilde_norm = self.norm_v_tilde[0](v_tilde)
            # [B, L, D] -> [B, D, L]
            x_norm = v_tilde_norm.permute(0, 2, 1).contiguous()

        # Manual padding for causal convolution
        padding_size = (self.kernel_size - 1) * self.dilation
        if padding_size > 0:
            x_padded = F.pad(x_norm, (padding_size, 0))
        else:
            x_padded = x_norm

        # Apply convolution
        conv_out = self.conv(x_padded)

        # Step 8: SiLU activation + residual connection: Y = SiLU(Conv(RMSNorm(ṽ))) + ṽ
        y = F.silu(conv_out)

        # Reshape back to match input shape
        if self.num_branches > 1:
            # [B*M, D, L] -> [B, M, D, L] -> [B, L, M, D]
            y = (
                y.view(batch_size, self.num_branches, self.hidden_dim, seq_len)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        else:
            # [B, D, L] -> [B, L, D]
            y = y.permute(0, 2, 1).contiguous()

        return cast(torch.Tensor, y + v_tilde)


class EngramLayer(nn.Module):
    """
    Complete Engram Layer as described in Section 2.1-2.3 of the Engram paper.

    1. Extracts suffix N-grams (via TokenizerCompressor and MultiHeadHash)
    2. Computes indices via Multi-Head Hashing
    3. Retrieves vectors from K independent embedding tables
    4. Applies Context-Aware Gating modulation
    5. Residual connection to Transformer Block hidden states
    """

    def __init__(
        self,
        config: EngramConfig,
        layer_id: int,
        primes: List[int],
        compressor: Optional[TokenizerCompressor] = None,
    ):
        """
        Initialize the EngramLayer.

        Args:
            config: EngramConfig containing hyperparameters.
            layer_id: The ID of this layer.
            primes: List of pre-calculated primes for this layer's heads.
            compressor: Optional TokenizerCompressor for token mapping.
        """
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.compressor = compressor

        # 2. Multi-Head Hashing
        self.multi_head_hash = MultiHeadHash(
            layer_id=layer_id,
            primes=primes,
            ngram_sizes=config.ngram_sizes,
            hash_heads=config.hash_heads,
            seed=config.seed,
            tokenizer_vocab_size=(
                compressor.compressed_vocab_size if compressor else 129280
            ),
        )

        # 3. Single shared embedding table with offsets
        # Matches Demo's MultiHeadEmbedding logic
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_capacity = sum(primes)
        self.embedding = nn.Embedding(total_capacity, config.embedding_dim_per_head)
        # Weight initialization: mean 0, std 0.02
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # 4. Context-Aware Gating
        self.gating = ContextAwareGating(
            embedding_dim=config.total_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_branches=config.num_branches,
            kernel_size=config.kernel_size,
            dilation=config.dilation,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        compressed_ids: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor = cast(torch.Tensor, None),
        engram_hash_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the EngramLayer.

        Args:
            input_ids: [batch_size, seq_len] Original token IDs.
            compressed_ids: [batch_size, seq_len] Compressed token IDs.
            hidden_states: [batch_size, seq_len, hidden_dim] or [B, L, M, D].
            engram_hash_indices: Optional precomputed hash indices [B, L, total_heads].

        Returns:
            torch.Tensor: Modified hidden states with Engram contributions.
        """
        if hidden_states is None:
            raise ValueError("hidden_states must be provided.")

        # Step 1: Get hash indices (Priority: engram_hash_indices > compressed_ids > input_ids)
        if engram_hash_indices is None:
            if compressed_ids is None:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids, compressed_ids, or engram_hash_indices must be provided."
                    )
                if self.compressor is None:
                    raise ValueError(
                        "TokenizerCompressor is required if only input_ids are provided."
                    )
                # Use the pre-computed lookup table for fast mapping
                compressed_ids = self.compressor.lookup.to(input_ids.device)[input_ids]

            # Compute hashes using MultiHeadHash
            # At this point, compressed_ids is guaranteed not to be None
            engram_hash_indices = self.multi_head_hash.compute_hashes(compressed_ids)

        # Step 2: Retrieve vectors from single shared embedding table
        # engram_hash_indices: [B, L, total_heads]
        indices_on_device = engram_hash_indices.to(hidden_states.device)
        shifted_indices = indices_on_device + cast(torch.Tensor, self.offsets)

        # [B, L, total_heads, d_per_head]
        all_embeddings_tensor = self.embedding(shifted_indices)

        # Step 3: Concatenate all N-gram and hash head vectors: e_t = ||_{n} ||_{k} e_{t,n,k}
        # [batch_size, seq_len, total_embedding_dim]
        # Flatten the last two dimensions: total_heads * d_per_head = total_embedding_dim
        e_t = all_embeddings_tensor.flatten(start_dim=-2)

        # Step 4: Context-Aware Gating modulation
        # y has the same shape as hidden_states
        y = self.gating(e_t, hidden_states)

        # Step 5: Residual connection: hidden_states + Y
        return cast(torch.Tensor, hidden_states + y)
