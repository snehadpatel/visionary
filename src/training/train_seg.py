import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tqdm
from src.models.unet import UNet
from src.models.losses import dice_loss
from src.data.loaders import VisionarySegDataset

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    MODEL_DIR = "/Users/snehapatel/visionary/models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Dataset
    seg_train_dir = "/Users/snehapatel/visionary/data/seg_labels/train"
    train_dataset = VisionarySegDataset(
        jsonl_path="/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        seg_dir=seg_train_dir,
        size=(384, 384)
    )
    # Filter for items with labels
    train_dataset.items = [item for item in train_dataset.items 
                          if os.path.exists(os.path.join(seg_train_dir, os.path.basename(item['path']).split('.')[0] + "_mask.png"))]
    
    print(f"Training on {len(train_dataset)} items with segmentation masks.")
    
    seg_val_dir = "/Users/snehapatel/visionary/data/seg_labels/val"
    val_dataset = VisionarySegDataset(
        jsonl_path="/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        seg_dir=seg_val_dir,
        size=(384, 384)
    )
    val_dataset.items = [item for item in val_dataset.items 
                        if os.path.exists(os.path.join(seg_val_dir, os.path.basename(item['path']).split('.')[0] + "_mask.png"))]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(n_classes=21).to(device) # DeepLabV3 has 21 classes (COCO VOC)
    # Note: Our UNet was built for 10 classes in the definition, 
    # but the distillation model (DeepLabV3) has 21. 
    # I'll use 21 for now to match exactly.
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs) # (B, C, H, W)
            
            loss_ce = criterion(preds, masks)
            loss_dice = dice_loss(preds, masks)
            loss = loss_ce + 0.5 * loss_dice
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = criterion(preds, masks) + 0.5 * dice_loss(preds, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(val_loader) + 1e-8)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "seg_model.pth"))
            print("Model saved!")

if __name__ == "__main__":
    train()
