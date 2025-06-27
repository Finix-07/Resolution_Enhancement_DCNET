
# dcnet.py

import torch
import torch.nn as nn
from feature_extraction import ShallowFeatureExtractor
from residual_group import ResidualGroup
from reconstruction import ProgressiveReconstruction

class DCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 180,
        rg_depths: list[int] = [2,2,2,2,2],
        num_heads: list[int] = [6,6,6,6,6],
        mlp_ratio: float = 2.0,
        window_size: tuple[int,int] = (8,8),
        fusion_bias: bool = False,
        conv_bias: bool = False,
        scale_factors: list[int] = [2,2],
        out_channels: int = 1,
        max_drop_path: float = 0.1,     # <— new!
    ):
        super().__init__()
        # 1) Shallow feature extractor
        self.shallow = ShallowFeatureExtractor(
            in_channels=in_channels,
            dim=channels,
            num_res_blocks=2,
            bias=conv_bias
        )

        # 2) Stochastic‐depth schedule
        total_blocks = sum(rg_depths)
        dp_rates = torch.linspace(0.0, max_drop_path, total_blocks).tolist()

        # 3) ResidualGroups with per-block drop rates
        self.rgs = nn.ModuleList()
        idx = 0
        for n_blocks, heads in zip(rg_depths, num_heads):
            rates = dp_rates[idx: idx + n_blocks]
            idx += n_blocks
            self.rgs.append(
                ResidualGroup(
                    channels=channels,
                    num_blocks=n_blocks,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size,
                    fusion_bias=fusion_bias,
                    conv_bias=conv_bias,
                    drop_path_rates=rates,
                )
            )

        # 4) Global residual conv + reconstruction
        self.conv_after_rg    = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=conv_bias)
        self.reconstruction   = ProgressiveReconstruction(
            dim=channels,
            scale_factors=scale_factors,
            out_channels=out_channels,
            bias=conv_bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.shallow(x)
        out = f0
        for rg in self.rgs:
            out = rg(out)
        out = self.conv_after_rg(out) + f0
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
