# residual_group.py

import torch
import torch.nn as nn
from dcb import DCB

class ResidualGroup(nn.Module):
    """
    A Residual Group (RG) consists of `num_blocks` DCBs followed by
    a 3×3 conv, with a residual add from the group’s input:
    
      F_out = Conv3x3( DCB_{n}(…(DCB1(F_in))… ) ) + F_in
    
    (Eq. 3, Sect. 3.1) 
    """
    def __init__(self,
                 channels: int,
                 num_blocks: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 window_size: tuple[int,int] = (8,8),
                 fusion_bias: bool = False,
                 conv_bias: bool = False):
        super().__init__()
        # Stack of DCBs
        self.blocks = nn.ModuleList([
            DCB(
                channels=channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                bias=fusion_bias
            )
            for _ in range(num_blocks)
        ])
        # 3×3 conv to mash together and keep same channel count
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.conv(out)
        return residual + out
