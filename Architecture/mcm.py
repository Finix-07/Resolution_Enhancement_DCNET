# mcm.py

import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    """
    Multi-scale Convolution Module (MCM):
      - Convd_i with dilation rates i ∈ {1,3,5}
      - Concatenate [X_in, Conv_d1(X), Conv_d3(X), Conv_d5(X)]
      - 1×1 conv to reduce back to dim channels
      - Residual add: X_out = Conv1x1(concat) + X_in
    (Eq.14 in Sect. 3.3) :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.conv_d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, bias=bias)
        self.conv_d3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, bias=bias)
        self.conv_d5 = nn.Conv2d(dim, dim, kernel_size=3, padding=5, dilation=5, bias=bias)
        # after concatenation we have 4× dim channels
        self.conv1x1 = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.conv_d1(x)
        d3 = self.conv_d3(x)
        d5 = self.conv_d5(x)
        # concat along channel axis
        combined = torch.cat([x, d1, d3, d5], dim=1)
        out = self.conv1x1(combined)
        return x + out


if __name__ == "__main__":
    # Quick sanity check
    x = torch.randn(1, 180, 64, 64)             # B=1, C=180, H=W=64
    mcm = MultiScaleConv(dim=180, bias=False)
    y = mcm(x)
    print("MCM output shape:", y.shape)         # → (1, 180, 64, 64)
