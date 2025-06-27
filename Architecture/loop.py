import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((1.0 ** 2) / mse)

def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """Compute SSIM between super-resolution and high-resolution images."""
    # Convert tensors to numpy arrays
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    
    # Fix 1: Print shapes for debugging
    # print(f"SR shape: {sr_np.shape}, HR shape: {hr_np.shape}")
    
    # Fix 2: Find minimum spatial dimension to determine window size
    if len(sr_np.shape) == 4:  # [B,C,H,W]
        spatial_dims = sr_np.shape[2:]  # [H,W]
    elif len(sr_np.shape) == 3:  # [C,H,W]
        spatial_dims = sr_np.shape[1:]  # [H,W]
    else:  # [H,W]
        spatial_dims = sr_np.shape
    
    min_dim = min(spatial_dims)
    
    # Fix 3: Set appropriate window size (must be odd)
    if min_dim < 7:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        win_size = max(win_size, 3)  # minimum valid window size is 3
    else:
        win_size = 7  # default
    
    # Fix 4: Handle grayscale images correctly
    if len(sr_np.shape) == 4:  # Batched input [B,C,H,W]
        batch_size = sr_np.shape[0]
        ssim_vals = []
        
        for b in range(batch_size):
            if sr_np.shape[1] == 1:  # Grayscale
                s_val = sk_ssim(hr_np[b, 0], sr_np[b, 0], 
                               data_range=1.0, win_size=win_size)
            else:  # Multi-channel
                # Transpose to [H,W,C] format for scikit-image
                sr_img = np.transpose(sr_np[b], (1, 2, 0))
                hr_img = np.transpose(hr_np[b], (1, 2, 0))
                s_val = sk_ssim(hr_img, sr_img, data_range=1.0, 
                               win_size=win_size, channel_axis=2)
            ssim_vals.append(s_val)
        
        return float(np.mean(ssim_vals))
        
    elif len(sr_np.shape) == 3:  # Single image [C,H,W]
        if sr_np.shape[0] == 1:  # Grayscale
            return float(sk_ssim(hr_np[0], sr_np[0], 
                              data_range=1.0, win_size=win_size))
        else:  # Multi-channel
            sr_img = np.transpose(sr_np, (1, 2, 0))
            hr_img = np.transpose(hr_np, (1, 2, 0))
            return float(sk_ssim(hr_img, sr_img, data_range=1.0, 
                              win_size=win_size, channel_axis=2))
    
    else:  # Single image, single channel [H,W]
        return float(sk_ssim(hr_np, sr_np, data_range=1.0, win_size=win_size))

def compute_rmse(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr, reduction='mean').item()
    return float(np.sqrt(mse))

def compute_pcc(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_flat = sr.cpu().view(-1).numpy()
    hr_flat = hr.cpu().view(-1).numpy()
    return float(np.corrcoef(sr_flat, hr_flat)[0, 1])
from tqdm import tqdm  # Import tqdm for progress bars
from torch.cuda.amp import autocast, GradScaler  # Import autocast for mixed precision training
from kornia.losses import SSIMLoss  # Import SSIM loss from Kornia

class DynamicCompositeLoss(nn.Module):
    def __init__(self, window_size: int = 11, ssim_weight: float = 5.0, mse_weight: float = 1.0):
        super().__init__()
        # Initialize with bias toward SSIM (-1.6 gives exp(-(-1.6)) ≈ 5)
        self.log_sigma_mse = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_ssim = nn.Parameter(torch.tensor(-1.6))  # Initial higher weight
        
        # Static weights to further control the balance
        self.ssim_weight = ssim_weight  # Higher weight for SSIM
        self.mse_weight = mse_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=window_size, reduction='mean')
        
    def forward(self, sr, hr, epoch=None):
        L_mse = self.mse_loss(sr, hr)
        L_ssim = self.ssim_loss(sr, hr)  # returns (1−SSIM)
        
        # Dynamic weights with regularization
        mse_weight = torch.exp(-self.log_sigma_mse) * self.mse_weight
        ssim_weight = torch.exp(-self.log_sigma_ssim) * self.ssim_weight
        
        # Apply progressive weighting if epoch is provided
        if epoch is not None:
            # Gradually increase SSIM importance over training
            epoch_factor = min(1.0 + (epoch / 10), 3.0)  # Cap at 3x boost
            ssim_weight = ssim_weight * epoch_factor
        
        # Composite loss with enhanced SSIM weight
        loss = (
            mse_weight * L_mse + self.log_sigma_mse +
            ssim_weight * L_ssim + self.log_sigma_ssim
        )
        
        # For monitoring weight evolution
        with torch.no_grad():
            self.current_mse_weight = mse_weight.item()
            self.current_ssim_weight = ssim_weight.item()
            
        return loss

def train_loop(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               num_epochs: int = 100):
    # Setup optimizer, scheduler, loss

    criterion = DynamicCompositeLoss(window_size=11).to(device)

    optimizer = Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=2e-4, betas=(0.9, 0.999)
    )
    scheduler = MultiStepLR(optimizer, milestones=[25, 45, 65, 85], gamma=0.5)

    # Prepare metrics storage

    psnr_list, ssim_list, rmse_list, pcc_list = [], [], [], []

    # Create models directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 0. Before everything:
    scaler = GradScaler()
    best_ssim = 0.0


    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}", leave=False)
        for batch in train_bar:
            lr_imgs = batch['image'].to(device)
            hr_imgs = batch['label'].to(device)

            optimizer.zero_grad()

            # 1. Mixed-precision forward + loss
            with autocast():
                sr = model(lr_imgs)
                loss = criterion(sr, hr_imgs)

            # 2. Scale, backprop, unscale, clip, step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            train_bar.set_postfix(loss=f"{batch_loss:.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        epoch_psnr, epoch_ssim, epoch_rmse, epoch_pcc = [], [], [], []
        val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch}/{num_epochs}", leave=False)

        with torch.no_grad():
            for batch in val_bar:
                lr_imgs = batch['image'].to(device)
                hr_imgs = batch['label'].to(device)

                # 3. You can also wrap inference in autocast for speed
                with autocast():
                    sr = model(lr_imgs)

                psnr = compute_psnr(sr, hr_imgs)
                ssim = (compute_ssim(sr, hr_imgs) + 1) / 2
                rmse = compute_rmse(sr, hr_imgs)
                pcc = compute_pcc(sr, hr_imgs)

                epoch_psnr.append(psnr)
                epoch_ssim.append(ssim)
                epoch_rmse.append(rmse)
                epoch_pcc.append(pcc)

                val_bar.set_postfix(PSNR=f"{psnr:.2f}", SSIM=f"{ssim:.4f}")

        # Average metrics
        avg_psnr = np.mean(epoch_psnr)
        avg_ssim = np.mean(epoch_ssim)
        avg_rmse = np.mean(epoch_rmse)
        avg_pcc = np.mean(epoch_pcc)

        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)
        rmse_list.append(avg_rmse)
        pcc_list.append(avg_pcc)

        print(f"Epoch {epoch}/{num_epochs}  "
            f"Loss: {avg_train_loss:.4f}  "
            f"PSNR: {avg_psnr:.4f}  "
            f"SSIM: {avg_ssim:.4f}  "
            f"RMSE: {avg_rmse:.4f}  "
            f"PCC: {avg_pcc:.4f}")

        # 4. Checkpoint on best SSIM (optional) & periodic saves
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(model.state_dict(), os.path.join(save_dir, "best_ssim.pth"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pth"))

    # After training, save metrics arrays
    metrics = np.vstack([psnr_list, ssim_list, rmse_list, pcc_list]).T
    metrics_path = os.path.join(save_dir, "metrics.txt")
    np.savetxt(metrics_path, metrics, header="PSNR SSIM RMSE PCC", fmt="%.6f")

    # Plot and save each metric
    epochs = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, psnr_list)
    plt.title("Validation PSNR over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.savefig(os.path.join(save_dir, "psnr_plot.png"))

    plt.figure()
    plt.plot(epochs, ssim_list)
    plt.title("Validation SSIM over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.savefig(os.path.join(save_dir, "ssim_plot.png"))

    plt.figure()
    plt.plot(epochs, rmse_list)
    plt.title("Validation RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(save_dir, "rmse_plot.png"))

    plt.figure()
    plt.plot(epochs, pcc_list)
    plt.title("Validation PCC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("PCC")
    plt.savefig(os.path.join(save_dir, "pcc_plot.png"))