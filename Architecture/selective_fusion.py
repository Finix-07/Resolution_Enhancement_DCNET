# selective_fusion_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFT(nn.Module):
    """
    Spatial Feature Transform (SFT) module that, given dissimilar features X,
    produces modulation parameters γ and β via a small conv‐ReLU‐conv.
    (Eq. 17) :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.relu  = nn.ReLU(inplace=True)
        # produce 2×channels: γ and β
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        gamma, beta = y.chunk(2, dim=1)
        return gamma, beta


class SelectiveFusionModule(nn.Module):
    """
    SFM: given two feature maps from Transformer (X_trans) and CNN (X_cnn),
    1) compute cosine similarities at spatial (Ms) and channel (Mc) levels,
       then build M = sigmoid(Mc)^T * sigmoid(Ms) reshaped to (C,H,W) :contentReference[oaicite:5]{index=5}
    2) split into similar (X * M) vs dissimilar (X * (1−M))
    3) Fsim = concat(Xsim_trans, Xsim_cnn)             (Eq. 16)
    4) Fdis  = concat( Xdis_trans⋅γ_cnn+β_cnn,
                       Xdis_cnn⋅γ_trans+β_trans )      (Eqs. 17–18)
    5) Output = Conv1×1(Fsim + Fdis)                   (Eq. 19)
    """
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.sft  = SFT(channels, bias=bias)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=bias)

    def forward(self, x_trans: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_trans.size()
        N = H * W
        eps = 1e-8

        # flatten for similarity computations
        t_flat = x_trans.view(B, C, N)   # (B, C, L)
        c_flat = x_cnn.view(B, C, N)     # (B, C, L)

        # 1) Spatial similarity Ms ∈ R^{B×L}: cosine across channel dimension
        t_norm  = F.normalize(t_flat, dim=1, eps=eps)
        c_norm  = F.normalize(c_flat, dim=1, eps=eps)
        Ms      = torch.sum(t_norm * c_norm, dim=1)          # (B, L)

        # 2) Channel similarity Mc ∈ R^{B×C}: cosine across spatial dimension
        t_spat  = F.normalize(t_flat, dim=2, eps=eps)
        c_spat  = F.normalize(c_flat, dim=2, eps=eps)
        Mc      = torch.sum(t_spat * c_spat, dim=2)          # (B, C)

        # 3) Feature selection matrix M: outer product, reshape to (B, C, H, W)
        Mc_s    = torch.sigmoid(Mc).unsqueeze(2)             # (B, C, 1)
        Ms_s    = torch.sigmoid(Ms).unsqueeze(1)             # (B, 1, L)
        M_flat  = Mc_s * Ms_s                                # (B, C, L)
        M_map   = M_flat.view(B, C, H, W)                    # (B, C, H, W)

        # 4) Split into similar vs dissimilar
        Xsim_t  = x_trans * M_map
        Xsim_c  = x_cnn   * M_map
        Xdis_t  = x_trans * (1 - M_map)
        Xdis_c  = x_cnn   * (1 - M_map)

        # 5) Similar content fusion
        Fsim    = torch.cat([Xsim_t, Xsim_c], dim=1)         # (B, 2C, H, W)

        # 6) Dissimilar content: SFT modulation
        γ_c, β_c = self.sft(Xdis_c)  # params from CNN branch
        Xp_t     = Xdis_t * γ_c + β_c
        γ_t, β_t = self.sft(Xdis_t)  # params from Transformer branch
        Xp_c     = Xdis_c * γ_t + β_t
        Fdis     = torch.cat([Xp_t, Xp_c], dim=1)             # (B, 2C, H, W)

        # 7) Merge and reduce channels
        out = Fsim + Fdis                                    # (B, 2C, H, W)
        out = self.conv(out)                                 # (B, C, H, W)
        return out
