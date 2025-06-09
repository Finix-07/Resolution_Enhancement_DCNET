import torch
import torch.nn as nn
from torchinfo import summary

from feature_extraction import ShallowFeatureExtractor
from selective_fusion import SelectiveFusionBlock
from ffn import FeedForward
from reconstruction import ProgressiveReconstruction

class DCNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dim: int,
                 depth: list[int],
                 num_heads: list[int],
                 mlp_ratio: float,
                 fusion_type: str,
                 scale_factors: list[int],
                 out_channels: int,
                 bias: bool = False):
        super().__init__()

        # 1) Shallow feature extractor
        self.shallow = ShallowFeatureExtractor(
            in_channels=in_channels,
            dim=dim,
            num_res_blocks=2,
            bias=bias
        )

        # 2) Build the exact sequence of DATM blocks
        self.datm_blocks = nn.ModuleList()
        for d_i, heads_i in zip(depth, num_heads):
            for _ in range(d_i):
                self.datm_blocks.append(nn.Sequential(
                    SelectiveFusionBlock(dim, num_heads=heads_i, mlp_ratio=mlp_ratio, fusion_type=fusion_type, bias=bias),
                    FeedForward(dim, expansion_factor=mlp_ratio, bias=bias)
                ))

        # 3) Upsampling head
        self.reconstruction = ProgressiveReconstruction(
            dim=dim,
            scale_factors=scale_factors,
            out_channels=out_channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallow(x)
        for block in self.datm_blocks:
            x = block(x)
        return self.reconstruction(x)


if __name__ == "__main__":
    # Instantiate with your YAML settings
    model = DCNet(
        in_channels=1,
        dim=180,
        depth=[2,2,2,2,2],
        num_heads=[6,6,6,6,6],
        mlp_ratio=2.0,
        fusion_type="conv",
        scale_factors=[2,2],   # 4× total
        out_channels=1,
        bias=False
    )
    # dummy = torch.randn(1, 1, 64, 64)
    # out = model(dummy)
    # print("Output shape:", out.shape)  # -> (1,3,256,256)

    
    summary(model, input_size=(1, 1, 128, 128), col_names=["input_size","output_size","num_params"])


