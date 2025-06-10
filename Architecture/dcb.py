# dcb.py

import torch
import torch.nn as nn
from mcm import MultiScaleConv
from datm import DualAttentionTransformerModule
from selective_fusion import SelectiveFusionModule


class DCB(nn.Module):
    """
    Dual-path Collaborative Block (DCB):
      - Dual Attention Transformer Module (DATM)
      - Multi-scale Convolution Module (MCM)
      - Selective Fusion Module (SFM)

    Each DCB takes input features X, processes them in parallel
    through DATM and MCM, then fuses via the SFM:
        X_trans = DATM(X)
        X_cnn   = MCM(X)
        X_out   = SFM(X_trans, X_cnn)

    (Sect. 3.1, Fig. 1b) :contentReference[oaicite:3]{index=3}
    """
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 window_size: tuple[int, int] = (8, 8),
                 bias: bool = False):
        super().__init__()
        # Global branch: Transformer with dual attention
        self.datm = DualAttentionTransformerModule(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            bias=bias
        )
        # Local branch: multi-scale dilated convolutions
        self.mcm = MultiScaleConv(
            dim=channels,
            bias=bias
        )
        # Fusion: similarity-guided selective fusion
        self.sfm = SelectiveFusionModule(
            channels=channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Global features via DATM
        x_trans = self.datm(x)
        # 2) Local features via MCM
        x_cnn   = self.mcm(x)
        # 3) Selective fusion of global & local
        out     = self.sfm(x_trans, x_cnn)
        return out


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
