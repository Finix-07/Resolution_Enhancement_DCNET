# selective_fusion.py

import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(x)


class TransformerBranch(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 bias: bool = False):
        super().__init__()
        # Positional embedding (broadcast over tokens)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, dim))

        # Project to QKV via a 1×1 conv
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            bias=bias,
            batch_first=True
        )

        # Pre-norm for MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1) QKV projection + reshape to (B, N, C)
        qkv = self.qkv_conv(x)               # (B, 3C, H, W)
        qkv = qkv.view(B, 3, C, H * W)       # (B, 3, C, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # (B, N, C)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # 2) Add positional embedding
        q = q + self.pos_emb

        # 3) MHSA with pre-norm + residual
        y = self.norm1(q)
        attn_out, _ = self.attn(y, y, y)
        x2 = q + attn_out

        # 4) MLP with pre-norm + residual
        y2 = self.norm2(x2)
        y2 = self.mlp(y2)
        out = x2 + y2

        # 5) Reshape back to (B, C, H, W)
        return out.transpose(1, 2).view(B, C, H, W)


class SelectiveFusionBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 fusion_type: str = "conv",
                 bias: bool = False):
        super().__init__()
        self.cnn_branch = CNNBranch(dim, bias=bias)
        self.transformer_branch = TransformerBranch(
            dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            bias=bias
        )
        self.fusion_type = fusion_type

        if fusion_type == "conv":
            self.fusion = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias),
                nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local  = self.cnn_branch(x)
        x_global = self.transformer_branch(x)

        if self.fusion_type == "add":
            return x_local + x_global
        elif self.fusion_type == "concat":
            return torch.cat([x_local, x_global], dim=1)
        elif self.fusion_type == "conv":
            return self.fusion(torch.cat([x_local, x_global], dim=1))
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
