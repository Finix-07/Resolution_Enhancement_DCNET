#!/usr/bin/env python3
"""
Super-Resolution Training Script for DCNet model
"""

import os
import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Import local modules
from loop import train_loop
from dcnet import DCNet


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, lr_transform=None, hr_transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = os.listdir(lr_dir)
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.lr_images[idx])

        lr_image = Image.open(lr_image_path).convert("L").resize((64, 64))  # Resize to 64x64
        hr_image = Image.open(hr_image_path).convert("L").resize((256, 256))  # Resize to 256x256

        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)
        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)

        return {'image': lr_image, 'label': hr_image}


def dataloaders(train_dataset, val_dataset, batch_size=8, num_train_samples=None, num_val_samples=None):
    """
    Create dataloaders for training and validation datasets with an optional limit 
    on the number of samples to use.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        batch_size: The batch size for the dataloaders.
        num_train_samples: The number of samples to use from the training dataset. If None, use all samples.
        num_val_samples: The number of samples to use from the validation dataset. If None, use all samples.

    Returns:
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
    """
    # Limit the number of samples if specified
    if num_train_samples is not None:
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, 
            [num_train_samples, len(train_dataset) - num_train_samples]
        )
    if num_val_samples is not None:
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset, 
            [num_val_samples, len(val_dataset) - num_val_samples]
        )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def setup_training(model, device, train_loader, val_loader, epochs=1000, patience=10):
    """
    Set up and start the training process.
    
    Args:
        model: The model to train.
        device: The device to use (CPU or GPU).
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        epochs: Number of epochs to train for.
        patience: Early stopping patience.
    """
    # Ensure the model is moved to the correct device
    model = model.to(device)

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("models", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Start training loop
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=epochs
    )


def main():
    """Main training function."""
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    # Initialize the DCNet model
    model = DCNet(
        in_channels=1,
        channels=180,
        rg_depths=[1,1,1],  # Residual Groups × 2 DCBs
        num_heads=[6,6,6],
        mlp_ratio=2.0,
        window_size=(4,4),
        scale_factors=[4],
        out_channels=1
    )

    # Setup transformations
    transform_lr = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_hr = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the dataset
    lr_dir = "/media/user/9c7eaef1-35fa-4210-889c-9e2b99342586/user/abul/RESM/sdo patches/dataset sdo patches low res"
    hr_dir = "/media/user/9c7eaef1-35fa-4210-889c-9e2b99342586/user/abul/RESM/sdo patches/dataset sdo patches"

    # Create dataset
    print("Loading dataset...")
    full_dataset = SuperResolutionDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        lr_transform=transform_lr,
        hr_transform=transform_hr
    )
    print(f"Dataset loaded with {len(full_dataset)} samples")

    # Splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"Splitting dataset: {train_size} training samples, {val_size} validation samples")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    batch_size = 16  # Adjust based on your GPU memory
    train_loader, val_loader = dataloaders(
        train_dataset, 
        val_dataset, 
        batch_size=batch_size,
        num_train_samples=None,  # Use all samples
        num_val_samples=None     # Use all samples
    )

    # Optional: Print model summary
    # print(summary(
    #     model, 
    #     input_size=(1, 1, 64, 64),  # Batch size = 1, Channels = 1, Height = 64, Width = 64
    #     device=device, 
    #     col_names=["input_size", "output_size", "num_params", "trainable"], 
    #     depth=4
    # ))

    # Start training
    print("Starting training...")
    setup_training(model, device, train_loader, val_loader)
    print("Training complete!")


if __name__ == "__main__":
    main()