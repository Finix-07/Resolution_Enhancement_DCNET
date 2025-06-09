import torch
import torch.nn as nn
import torch.nn.functional as F

# Dual Attention FeedForward Network (DA-FFN)
# This module implements a feedforward network with dual attention mechanisms:
# 1. Channel Attention: Enhances channel-wise features.
# 2. Spatial Attention: Enhances spatial features.
# The network consists of a gated feedforward structure with depthwise convolution and adaptive pooling.
# according to the paper section 3.3

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.mlp(y)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)

        # Dual attention components (DA-FFN)
        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention()

        # FeedForward with gating
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dual Attention applied first
        x = self.channel_attn(x)
        x = self.spatial_attn(x)

        # Gated FeedForward
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


if __name__ == "__main__":
    # Example usage
    model = FeedForward(dim=64)
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    # Output: torch.Size([1, 64, 32, 32])
    print("FeedForward model initialized and executed successfully.")
    print("Output shape:", output_tensor.shape)
