import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A simple 2-layer residual block:
      x → Conv(3×3) → ReLU → Conv(3×3) → +x
    """
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual


class ShallowFeatureExtractor(nn.Module):
    """
    Maps raw input image to a 'dim'-channel feature map using:
      1) A 3×3 convolution head
      2) A stack of ResidualBlocks for low-level feature learning
    """
    def __init__(self,
                 in_channels: int = 1,
                 dim: int = 64,
                 num_res_blocks: int = 2,
                 bias: bool = False):
        super().__init__()
        # 3×3 conv to project input → dim channels
        self.head = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Body: sequence of ResidualBlocks
        body_layers = []
        for _ in range(num_res_blocks):
            body_layers.append(ResidualBlock(dim, bias=bias))
        self.body = nn.Sequential(*body_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        return x


if __name__ == "__main__":
    # Sanity check
    # For a single‐channel 64×64 input and dim=64, output should be (1, 64, 64, 64)
    input_tensor = torch.randn(1, 1, 64, 64)
    sfe = ShallowFeatureExtractor(in_channels=1, dim=64, num_res_blocks=2)
    output = sfe(input_tensor)
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
