# residual_group.py

import torch.nn as nn
from dcb import DCB

class ResidualGroup(nn.Module):
    """
    RG = num_blocks × DCB + Conv3×3 + group residual
    """
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: tuple[int,int],
        fusion_bias: bool,
        conv_bias: bool,
        drop_path_rates: list[float],      # <— new!
    ):
        super().__init__()
        assert len(drop_path_rates) == num_blocks, "Need one drop_rate per DCB"
        self.blocks = nn.ModuleList([
            DCB(
                channels=channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                bias=fusion_bias,
                drop_path=drop_path_rates[i],
            )
            for i in range(num_blocks)
        ])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)

    def forward(self, x):
        res = x
        out = x
        for blk in self.blocks:
            out = blk(out)
        return res + self.conv(out)
