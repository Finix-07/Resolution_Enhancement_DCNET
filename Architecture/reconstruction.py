import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    """
    Upsamples by an integer scale factor using sub-pixel convolution.
    """
    def __init__(self, dim: int, scale: int = 2, bias: bool = False):
        super().__init__()
        # Expand channels to dim * (scale^2), then PixelShuffle
        self.conv = nn.Conv2d(dim, dim * (scale ** 2), kernel_size=1, bias=bias)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ps(x)
        return self.act(x)


class ProgressiveReconstruction(nn.Module):
    """
    Chains multiple UpsampleBlocks to reach a desired total scale factor,
    then applies a final convolution to map to image channels.
    
    Example:
      - 4×: scale_factors=[2,2]
      - 6×: scale_factors=[2,3]
      - 8×: scale_factors=[2,2,2]
    """
    def __init__(self,
                 dim: int,
                 scale_factors: list[int],
                 out_channels: int = 1,
                 bias: bool = False):
        super().__init__()
        # Create one UpsampleBlock per factor
        self.blocks = nn.ModuleList([
            UpsampleBlock(dim, scale=s, bias=bias)
            for s in scale_factors
        ])
        # Final conv to get the desired output channels
        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


if __name__ == "__main__":
    # Minimal sanity check
    x = torch.randn(2, 64, 32, 32)          # Batch of 2, C=64, H=W=32
    recon4x = ProgressiveReconstruction(dim=64, scale_factors=[2, 2], out_channels=1)
    y4 = recon4x(x)
    print("4× upsampled shape:", y4.shape)  # → (2, 1, 128, 128)

    recon6x = ProgressiveReconstruction(dim=64, scale_factors=[2, 3], out_channels=1)
    y6 = recon6x(x)
    print("6× upsampled shape:", y6.shape)  # → (2, 1, 192, 192)
