# dcnet.py

import torch
import torch.nn as nn
from feature_extraction import ShallowFeatureExtractor
from residual_group import ResidualGroup
from reconstruction import ProgressiveReconstruction

class DCNet(nn.Module):
    """
    Full DCNet:
      1) ShallowFeatureExtractor → F_0
      2) N ResidualGroups (RG1…RGN) → F_N
      3) Global 3×3 conv + add: F_out = Conv3x3(F_N) + F_0
      4) ProgressiveReconstruction → HR image

    Matches Eq. (3) and Fig. 2 of the paper.
    """
    def __init__(self,
                 in_channels: int = 1,
                 channels: int = 180,
                 rg_depths: list[int] = [2,2,2,2,2],
                 num_heads: list[int] = [6,6,6,6,6],
                 mlp_ratio: float = 2.0,
                 window_size: tuple[int,int] = (8,8),
                 fusion_bias: bool = False,
                 conv_bias: bool = False,
                 scale_factors: list[int] = [2,2],
                 out_channels: int = 1):
        super().__init__()

        # 1) Shallow feature extraction
        self.shallow = ShallowFeatureExtractor(
            in_channels=in_channels,
            dim=channels,
            num_res_blocks=2,
            bias=conv_bias
        )

        # 2) Stack of Residual Groups
        assert len(rg_depths) == len(num_heads), "rg_depths and num_heads must align"
        self.rgs = nn.ModuleList()
        for n_blocks, heads in zip(rg_depths, num_heads):
            self.rgs.append(
                ResidualGroup(
                    channels=channels,
                    num_blocks=n_blocks,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size,
                    fusion_bias=fusion_bias,
                    conv_bias=conv_bias
                )
            )

        # 3) Global residual conv
        self.conv_after_rg = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)

        # 4) Reconstruction head
        self.reconstruction = ProgressiveReconstruction(
            dim=channels,
            scale_factors=scale_factors,
            out_channels=out_channels,
            bias=conv_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow features
        f0 = self.shallow(x)

        # Deep features via RGs
        out = f0
        for rg in self.rgs:
            out = rg(out)

        # Global residual
        out = self.conv_after_rg(out) + f0

        # Upsample to HR
        return self.reconstruction(out)


if __name__ == "__main__":
    # Sanity-check forward shapes
    model = DCNet(
        in_channels=1,
        channels=180,
        rg_depths=[2,2,2,2,2],   # 5 RGs × 2 DCBs = 10 blocks
        num_heads=[6,6,6,6,6],
        mlp_ratio=2.0,
        window_size=(4,4),
        scale_factors=[4],
        out_channels=1
    )
    # dummy = torch.randn(1, 1, 64, 64)
    # out = model(dummy)
    # print("Input:", dummy.shape, " → Output:", out.shape)  # → (1,1,256,256)
    from torchinfo import summary
    summary(model, input_size=(1, 1, 64, 64), col_names=["input_size", "output_size", "num_params", "trainable"], depth=3)
