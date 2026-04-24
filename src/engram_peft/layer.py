# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
from typing import Any, cast, final, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int64

from engram_peft.compression import CompressedTokenizer
from engram_peft.config import EngramConfig
from engram_peft.hashing import NgramHashMapping
from engram_peft.types import jaxtyped
from engram_peft.utils import safe_from_numpy


@final
class ShortConv(nn.Module):
    """
    ShortConv module as described in the Engram paper.

    Y = SiLU( Conv1D( RMSNorm(Ṽ) ) ) + Ṽ
    Applies independent RMSNorm per branch, depthwise causal convolution,
    and optional SiLU activation, followed by a residual connection.
    """

    hidden_size: int
    kernel_size: int
    dilation: int
    hc_mult: int
    activation: bool
    norms: nn.ModuleList
    conv: nn.Conv1d

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
        zero_init: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hc_mult = hc_mult
        self.activation = activation

        # Step 1: Independent RMSNorm for each branch
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )

        # Depthwise convolution (groups=total_channels)
        total_channels = hc_mult * hidden_size
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=total_channels,
            bias=False,
        )

        # Weight/bias initialization
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            # Use normal initialization if zero_init is False
            nn.init.normal_(self.conv.weight, std=0.02)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    @jaxtyped
    @override
    def forward(
        self, x: Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
    ) -> Float[torch.Tensor, "batch seq hc_mult hidden_dim"]:
        """
        Forward pass for ShortConv.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hc_mult, hidden_size]

        Returns:
            torch.Tensor: Output tensor of same shape, calculated as SiLU(Conv(Norm(x))) + x
        """
        batch_size, seq_len, hc_mult, hidden_size = x.shape

        # Step 1: Independent RMSNorm per branch
        normed_branches: list[torch.Tensor] = []
        for i in range(hc_mult):
            normed_branches.append(self.norms[i](x[:, :, i, :]))
        x_norm = torch.stack(normed_branches, dim=2)

        # Step 2: Reshape for Conv1D -> [batch_size, total_channels, seq_len]
        # x_norm: [B, L, M, D] -> [B, M*D, L]
        x_conv_in = x_norm.permute(0, 2, 3, 1).reshape(
            batch_size, hc_mult * hidden_size, seq_len
        )

        # Step 4: Causal padding (shift sequence right so conv output corresponds to current pos)
        pad_len = (self.kernel_size - 1) * self.dilation
        if pad_len > 0:
            x_padded = F.pad(x_conv_in, (pad_len, 0))
        else:
            x_padded = x_conv_in

        # Step 3: Depthwise convolution
        conv_out = self.conv(x_padded)

        # Step 5: SiLU activation (in-place saves one intermediate tensor)
        if self.activation:
            conv_out = F.silu(conv_out, inplace=True)

        # Step 6: Convert back to [batch_size, seq_len, hc_mult, hidden_size]
        out = (
            conv_out.view(batch_size, hc_mult, hidden_size, seq_len)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Matches the formula: Y = SiLU(Conv(Norm(V))) + V
        return (out + x).to(x.dtype)


@final
class ContextAwareGating(nn.Module):
    """
    Context-Aware Gating module as described in Section 2.3 and 2.4 of the Engram paper.

    This module computes a gating signal based on the context (h_t) and the retrieved
    Engram embeddings (e_t), and applies it to the value projection of the embeddings.
    """

    config: EngramConfig
    engram_hidden_size: int
    hidden_size: int
    hc_mult: int
    w_v: nn.Linear
    w_k: nn.ModuleList
    norm_h: nn.ModuleList
    norm_k: nn.ModuleList
    last_gate: torch.Tensor | None
    last_entropy: float
    gating_entropy: torch.Tensor

    def __init__(
        self,
        config: EngramConfig,
        engram_hidden_size: int,
        hidden_size: int,
        hc_mult: int = 4,
        zero_init: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.engram_hidden_size = engram_hidden_size
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        # Step 1: Shared Value projection
        self.w_v = nn.Linear(engram_hidden_size, hidden_size, bias=False)
        if zero_init:
            nn.init.zeros_(self.w_v.weight)

        # Step 2: Branch-specific Key projection W_K^(m)
        self.w_k = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, hidden_size, bias=False)
                for _ in range(hc_mult)
            ]
        )

        # Step 3: Independent RMSNorm
        self.norm_h = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.last_gate = None
        self.last_entropy = 0.0  # Default to zero

    @jaxtyped
    @override
    def forward(
        self,
        embeddings: Float[torch.Tensor, "batch seq engram_hidden"],
        hidden_states: Float[torch.Tensor, "batch seq hc_mult hidden_dim"],
    ) -> Float[torch.Tensor, "batch seq hc_mult hidden_dim"]:
        """
        Forward pass of the ContextAwareGating module.

        Args:
            embeddings: [batch_size, seq_len, engram_hidden_size]
            hidden_states: [batch_size, seq_len, hc_mult, hidden_size]

        Returns:
            torch.Tensor: gated_value of shape [batch_size, seq_len, hc_mult, hidden_size]
        """
        # Step 1: Shared Value projection
        value = self.w_v(embeddings)  # [B, L, D]

        # Compute gate scores per branch and stack small intermediates
        # (only stores M * [B, L, 1] in the list, avoids stacking full [B, L, D] tensors)
        gate_scores: list[torch.Tensor] = []
        for m in range(self.hc_mult):
            key_m = self.w_k[m](embeddings)
            normed_key = self.norm_k[m](key_m)
            normed_query = self.norm_h[m](hidden_states[:, :, m, :])
            gate_scores.append((normed_key * normed_query).sum(dim=-1, keepdim=True))

        gate = torch.stack(gate_scores, dim=2)  # [B, L, M, 1]
        gate = gate / (self.hidden_size**0.5)

        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()  # [B, L, M, 1]

        # Store for visualization
        self.last_gate = gate.detach()

        # Calculate Entropy
        # p=gate, 1-p=(1-gate)
        p = gate.clamp(1e-6, 1 - 1e-6)
        # We always compute the tensor version for loss regularization support
        self.gating_entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        if self.config.enable_telemetry:
            with torch.no_grad():
                self.last_entropy = self.gating_entropy.item()

        # Step 5: Gating modulation
        gated_value = gate * value.unsqueeze(
            2
        )  # [B, L, M, 1] * [B, L, 1, D] -> [B, L, M, D]

        return gated_value


@final
class MultiHeadEmbedding(nn.Module):
    """
    Concatenated embedding table for all hash heads across all N-gram sizes.
    Retrieves vectors from K independent virtual embedding tables using offset indices.
    """

    offsets: torch.Tensor
    embedding: nn.Embedding

    def __init__(
        self,
        primes: list[int],
        embedding_dim_per_head: int,
        sparse: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        offsets = [0]
        for p in primes[:-1]:
            offsets.append(offsets[-1] + p)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_capacity = sum(primes)
        self.embedding = nn.Embedding(
            total_capacity, embedding_dim_per_head, sparse=sparse
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @jaxtyped
    @override
    def forward(
        self, hash_indices: Int64[torch.Tensor, "batch seq total_heads"]
    ) -> Float[torch.Tensor, "batch seq total_heads dim_per_head"]:
        """
        Retrieves embedding vectors for pre-computed hash indices.
        Args:
            hash_indices: [batch_size, seq_len, total_heads]
        Returns:
            torch.Tensor: [batch_size, seq_len, total_heads, embedding_dim_per_head]
        """
        assert isinstance(self.offsets, torch.Tensor)
        shifted_indices = hash_indices.to(self.offsets.device) + self.offsets
        return self.embedding(shifted_indices)


@final
class EngramLayer(nn.Module):
    """
    Complete Engram Layer as described in Section 2.1-2.3 of the Engram paper.

    1. Extracts suffix N-grams (via CompressedTokenizer and MultiHeadHash)
    2. Computes indices via Multi-Head Hashing
    3. Retrieves vectors from K independent embedding tables
    4. Applies Context-Aware Gating modulation
    5. Residual connection to Transformer Block hidden states
    """

    config: EngramConfig
    layer_id: int
    compressor: CompressedTokenizer | None
    ngram_sizes: list[int]
    hash_heads: int
    num_branches: int
    kernel_size: int
    dilation: int
    hidden_dim: int
    total_embedding_dim: int
    embedding_dim_per_head: int
    hash_mapping: NgramHashMapping
    multi_head_embedding: MultiHeadEmbedding
    gating: ContextAwareGating
    short_conv: ShortConv
    last_norm_ratio: float

    def __init__(
        self,
        config: EngramConfig,
        layer_id: int,
        primes: list[int],
        compressor: CompressedTokenizer | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the EngramLayer.

        Args:
            config: EngramConfig containing hyperparameters.
            layer_id: The ID of this layer.
            primes: List of pre-calculated primes for this layer's heads.
            compressor: Optional CompressedTokenizer for token mapping.
        """
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.compressor = compressor

        self.ngram_sizes = config.ngram_sizes
        self.hash_heads = config.n_head_per_ngram
        self.num_branches = config.hc_mult
        self.kernel_size = config.conv_kernel_size
        self.dilation = (
            config.conv_dilation
            if config.conv_dilation is not None
            else config.max_ngram_size
        )
        assert config.hidden_size is not None
        assert config.embedding_dim is not None
        self.hidden_dim = config.hidden_size

        self.total_embedding_dim = config.embedding_dim
        self.embedding_dim_per_head = self.total_embedding_dim // (
            len(self.ngram_sizes) * self.hash_heads
        )

        # 0. Hash Mapping
        # Map pad_id to compressed space for hashing consistency
        mapped_pad_id = config.pad_id
        if self.compressor is not None:
            assert config.pad_id is not None
            mapped_pad_id = self.compressor.map_id(config.pad_id)

        assert config.compressed_vocab_size is not None
        assert mapped_pad_id is not None

        self.hash_mapping = NgramHashMapping(
            compressed_vocab_size=config.compressed_vocab_size,
            engram_vocab_size_per_ngram=config.engram_vocab_size_per_ngram,
            ngram_sizes=config.ngram_sizes,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=[layer_id],
            pad_id=mapped_pad_id,
            seed=config.seed,
        )

        # 1. MultiHeadEmbedding
        self.multi_head_embedding = MultiHeadEmbedding(
            primes=primes,
            embedding_dim_per_head=self.embedding_dim_per_head,
            sparse=config.use_sparse_embeddings,
        )

        # 2. Context-Aware Gating
        self.gating = ContextAwareGating(
            config=config,
            engram_hidden_size=self.total_embedding_dim,
            hidden_size=self.hidden_dim,
            hc_mult=self.num_branches,
            zero_init=config.gating_zero_init,
        )

        # 3. ShortConv
        self.short_conv = ShortConv(
            hidden_size=self.hidden_dim,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            hc_mult=self.num_branches,
            activation=True,
            zero_init=config.conv_zero_init,
        )
        self.last_norm_ratio = 0.0  # Default to zero

    @property
    def value_proj(self) -> nn.Linear:
        return self.gating.w_v

    @property
    def key_projs(self) -> nn.ModuleList:
        return self.gating.w_k

    @property
    def norm1(self) -> nn.ModuleList:
        return self.gating.norm_k

    @property
    def norm2(self) -> nn.ModuleList:
        return self.gating.norm_h

    @jaxtyped
    @override
    def forward(
        self,
        input_ids: Int64[torch.Tensor, "batch seq"] | None = None,
        compressed_ids: Int64[torch.Tensor, "batch seq"] | None = None,
        hidden_states: Float[torch.Tensor, "batch seq hidden_dim"]
        | Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
        | None = None,
        engram_hash_indices: Int64[torch.Tensor, "batch seq total_heads"] | None = None,
    ) -> (
        Float[torch.Tensor, "batch seq hidden_dim"]
        | Float[torch.Tensor, "batch seq hc_mult hidden_dim"]
    ):
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
            raise ValueError("hidden_states must be provided to EngramLayer.forward()")

        if engram_hash_indices is None:
            if input_ids is None:
                raise ValueError(
                    "Either engram_hash_indices or input_ids must be provided."
                )
            if self.compressor is None:
                raise ValueError(
                    "Compressor must be provided to compute hashes from input_ids."
                )
            # Step 1: Compress and hash
            c_ids = self.compressor.compress(input_ids)
            hashes_np = self.hash_mapping.hash(c_ids)[self.layer_id]
            engram_hash_indices = safe_from_numpy(hashes_np).to(hidden_states.device)

        # Step 1: Retrieve vectors from MultiHeadEmbedding and flatten
        all_embeddings = self.multi_head_embedding(engram_hash_indices)
        e_t = all_embeddings.flatten(start_dim=-2).to(hidden_states.device)

        # Ensure hidden_states is [B, L, M, D] if it's not already
        is_3d = hidden_states.dim() == 3
        if is_3d:
            hidden_states_m = hidden_states.unsqueeze(2).expand(
                -1, -1, self.num_branches, -1
            )
        else:
            hidden_states_m = hidden_states

        # Step 4: Context-Aware Gating modulation
        # gated_value has shape [B, L, M, D]
        gated_value = self.gating(e_t, hidden_states_m)

        # Step 5: ShortConv module
        # y has shape [B, L, M, D]
        y = self.short_conv(gated_value)

        # Step 6: Residual connection to Transformer Block hidden states
        if is_3d:
            if self.num_branches == 1:
                y = y.squeeze(2)
            else:
                # Sum branches if no out_proj is provided
                y = y.sum(dim=2)

        if self.config.enable_telemetry:
            with torch.no_grad():
                y_norm = cast(
                    "torch.Tensor", torch.linalg.vector_norm(y.float(), ord=2)
                )
                h_norm = cast(
                    "torch.Tensor",
                    torch.linalg.vector_norm(hidden_states.float(), ord=2)
                    if is_3d
                    else torch.linalg.vector_norm(hidden_states_m.float(), ord=2),
                )
                self.last_norm_ratio = (y_norm / (h_norm + 1e-8)).item()

        # Final result matches the input hidden_states shape
        return (hidden_states + y).to(hidden_states.dtype)
