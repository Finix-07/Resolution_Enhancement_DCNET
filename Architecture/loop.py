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

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    mse = F.mse_loss(sr, hr, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10((1.0 ** 2) / mse)

def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
    hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
    # Compute SSIM on each channel and average
    ssim_vals = [sk_ssim(hr_np[..., c], sr_np[..., c], data_range=1.0) for c in range(hr_np.shape[2])]
    return float(np.mean(ssim_vals))

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
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
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
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
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

