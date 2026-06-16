import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.generator import RedesignGenerator
from src.models.discriminator import Discriminator
from src.models.losses import PerceptualLoss
from src.data.dataset_redesign import RedesignDataset
import os
import time

# Training Settings
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100
BATCH_SIZE = 4
EPOCHS = 50

def train():
    # 1. Models
    generator = RedesignGenerator(n_channels=5).to(DEVICE)
    discriminator = Discriminator(in_channels=6).to(DEVICE)
    
    # 2. Dataset
    root_dir = "/Users/snehapatel/visionary/data/raw/huggingface_rooms"
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found. Create a paired dataset first.")
        return
    
    dataset = RedesignDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Loss & Optimizers
    criterion_gan = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    # criterion_perceptual = PerceptualLoss(device=DEVICE)
    
    LAMBDA_PERCEPTUAL = 0.0
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    print(f"Training initialized on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            fake_targets = generator(inputs)
            
            # Real loss
            pred_real = discriminator(inputs[:, :3, :, :], targets)
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake loss
            pred_fake = discriminator(inputs[:, :3, :, :], fake_targets.detach())
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            pred_fake = discriminator(inputs[:, :3, :, :], fake_targets)
            
            # GAN loss
            loss_GAN = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_targets, targets)
            # Perceptual loss
            # loss_perceptual = criterion_perceptual(fake_targets, targets)
            
            loss_G = loss_GAN + (LAMBDA_L1 * loss_pixel) # + (LAMBDA_PERCEPTUAL * loss_perceptual)
            loss_G.backward()
            optimizer_G.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}] [Loss D: {loss_D.item():.4f}] [Loss G: {loss_G.item():.4f}]")

    # Save trained model
    os.makedirs("/Users/snehapatel/visionary/models", exist_ok=True)
    torch.save(generator.state_dict(), "/Users/snehapatel/visionary/models/redesign_generator.pth")
    print("Core Structural Model saved to models/redesign_generator.pth")

if __name__ == "__main__":
    train()
