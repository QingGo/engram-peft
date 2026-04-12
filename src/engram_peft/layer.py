import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, cast


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

        # Step 3: RMSNorm for h_t and k_t
        self.norm_h = nn.RMSNorm(hidden_dim)
        self.norm_k = nn.RMSNorm(hidden_dim)

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

        # Step 8: RMSNorm before convolution
        self.norm_v_tilde = nn.RMSNorm(hidden_dim)

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

        # Step 3: RMSNorm h_t and k_t
        h_t_norm = self.norm_h(h_t)
        k_t_norm = self.norm_k(k_t)

        # Step 4: Calculate gating α_t = σ( (h_t_norm · k_t_norm) / sqrt(D) )
        # dot product over the last dimension D
        dot_product = (h_t_norm * k_t_norm).sum(dim=-1)
        alpha_t = torch.sigmoid(dot_product / (self.hidden_dim**0.5))

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
        # Prepare input for Conv1d: [N, C, L]
        if self.num_branches > 1:
            # [B, L, M, D] -> [B, M, D, L] -> [B*M, D, L]
            x = (
                v_tilde.permute(0, 2, 3, 1)
                .reshape(batch_size * self.num_branches, self.hidden_dim, seq_len)
                .contiguous()
            )
        else:
            # [B, L, D] -> [B, D, L]
            x = v_tilde.permute(0, 2, 1).contiguous()

        # Step 8: RMSNorm(ṽ) before convolution
        # RMSNorm expects [..., D], x is [N, D, L]
        x_norm = self.norm_v_tilde(x.transpose(1, 2)).transpose(1, 2)

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
