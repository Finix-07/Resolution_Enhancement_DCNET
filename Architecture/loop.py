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

# def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
#     sr_ = sr.squeeze().cpu().numpy()
#     print(sr_.shape)
#     hr_ = hr.squeeze().cpu().numpy()
#     print(hr_.shape)
#     # sr_np = sr_.transpose(1, 2, 0)
#     # hr_np = hr_.transpose(1, 2, 0)

#     # sr_np = sr_.transpose(1,2,0)
#     # hr_np = hr_.transpose(1,2,0)
#     sr_np = np.expand_dims(sr.cpu().numpy(), axis=-1) 
#     hr_np = np.expand_dims(hr.cpu().numpy(), axis=-1)
    
#     # Compute SSIM on each channel and average
#     ssim_vals = [sk_ssim(hr_np[..., c], sr_np[..., c], data_range=1.0) for c in range(hr_np.shape[2])]
#     return float(np.mean(ssim_vals))

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

# def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
#     sr_ = sr.squeeze().cpu().numpy()
#     hr_ = hr.squeeze().cpu().numpy()

#     print("SR shape:", sr_.shape)
#     print("HR shape:", hr_.shape)

#     # If image is grayscale (2D), SSIM can be computed directly
#     if sr_.ndim == 2 and hr_.ndim == 2:
#         return float(sk_ssim(hr_, sr_, data_range=1.0))

#     # If image is color (3D, shape: C x H x W), transpose to H x W x C
#     elif sr_.ndim == 3 and hr_.ndim == 3:
#         sr_np = sr_.transpose(1, 2, 0)
#         hr_np = hr_.transpose(1, 2, 0)
#         # Compute SSIM per channel and average
#         ssim_vals = [sk_ssim(hr_np[..., c], sr_np[..., c], data_range=1.0) for c in range(hr_np.shape[2])]
#         return float(np.mean(ssim_vals))

#     else:
#         raise ValueError(f"Unexpected tensor shapes: SR {sr_.shape}, HR {hr_.shape}")

def compute_rmse(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr, reduction='mean').item()
    return float(np.sqrt(mse))

def compute_pcc(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_flat = sr.cpu().view(-1).numpy()
    hr_flat = hr.cpu().view(-1).numpy()
    return float(np.corrcoef(sr_flat, hr_flat)[0, 1])

def train_loop(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device,
               num_epochs: int = 100):
    # Setup optimizer, scheduler, loss
    optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=[25, 45, 65, 85], gamma=0.5)
    criterion = nn.L1Loss()

    # Prepare metrics storage
    psnr_list, ssim_list, rmse_list, pcc_list = [], [], [], []

    # Create models directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("models", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            # Extract 'image' and 'label' from the batch dictionary
            lr_imgs = batch['image'].to(device)
            hr_imgs = batch['label'].to(device)

            optimizer.zero_grad()
            sr = model(lr_imgs)
            loss = criterion(sr, hr_imgs)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            epoch_psnr = []
            epoch_ssim = []
            epoch_rmse = []
            epoch_pcc = []
            for batch in val_loader:
                lr_imgs = batch['image'].to(device)
                hr_imgs = batch['label'].to(device)
                sr = model(lr_imgs)
                epoch_psnr.append(compute_psnr(sr, hr_imgs))
                epoch_ssim.append(compute_ssim(sr, hr_imgs))
                epoch_rmse.append(compute_rmse(sr, hr_imgs))
                epoch_pcc.append(compute_pcc(sr, hr_imgs))
            # average metrics for epoch
            psnr_list.append(np.mean(epoch_psnr))
            ssim_list.append(np.mean(epoch_ssim))
            rmse_list.append(np.mean(epoch_rmse))
            pcc_list.append(np.mean(epoch_pcc))

        # Save model every 10 epochs
        if epoch % 10 == 0:
            model_path = os.path.join(save_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), model_path)

        print(f"Epoch {epoch}/{num_epochs}  PSNR: {psnr_list[-1]:.4f}  SSIM: {ssim_list[-1]:.4f}  RMSE: {rmse_list[-1]:.4f}  PCC: {pcc_list[-1]:.4f}")

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