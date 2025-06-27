# dcb.py

import torch
import torch.nn as nn
from datm import DualAttentionTransformerModule
from mcm import MultiScaleConv
from selective_fusion import SelectiveFusionModule

class DCB(nn.Module):
    """
    Dual‐path Collaborative Block:
      - DATM (with drop_path)
      - MultiScaleConv
      - SelectiveFusion
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: tuple[int,int] = (8,8),
        bias: bool = False,
        drop_path: float = 0.0,          # <— new!
    ):
        super().__init__()
        # 1) Global branch: DATM with stochastic depth
        self.datm = DualAttentionTransformerModule(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            bias=bias,
            drop_path=drop_path,
        )
        # 2) Local branch
        self.mcm = MultiScaleConv(dim=channels, bias=bias)
        # 3) Fusion
        self.sfm = SelectiveFusionModule(channels=channels, bias=bias)

    def forward(self, x):
        x_t = self.datm(x)
        x_c = self.mcm(x)
        return self.sfm(x_t, x_c)



if __name__ == "__main__":
    # Quick sanity check
    B, C, H, W = 1, 180, 64, 64
    x = torch.randn(B, C, H, W)
    dcb = DCB(
        channels=C,
        num_heads=6,
        mlp_ratio=2.0,
        window_size=(4, 4),
        bias=False
    )
    y = dcb(x)
    print("DCB output shape:", y.shape)  # → (1, 180, 64, 64)
