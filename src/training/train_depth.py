import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tqdm
from src.models.fast_depth_net import FastDepthNet
from src.models.losses import berhu_loss
from src.data.loaders import VisionaryDepthDataset

def train():
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    MODEL_DIR = "/Users/snehapatel/visionary/models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Dataset Loaders
    train_dataset = VisionaryDepthDataset(
        jsonl_path="/Users/snehapatel/visionary/data/annotations/rooms.jsonl", # Or a specific split later
        depth_dir="/Users/snehapatel/visionary/data/depth_labels/train",
        size=(256, 256)
    )
    # Since rooms.jsonl has all data, we only take items that have a depth map in the depth_dir
    # Our VisionaryDepthDataset handles missing depth, but we should probably filter for those that have labels.
    
    # Filtering dataset for items with existing labels
    depth_dir = "/Users/snehapatel/visionary/data/depth_labels/train"
    train_dataset.items = [item for item in train_dataset.items 
                          if os.path.exists(os.path.join(depth_dir, os.path.basename(item['path']).replace('.jpg', '_depth.npy').replace('.webp', '_depth.npy')))]
    
    print(f"Training on {len(train_dataset)} items with depth labels.")
    
    val_dataset = VisionaryDepthDataset(
        jsonl_path="/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        depth_dir="/Users/snehapatel/visionary/data/depth_labels/val",
        size=(256, 256)
    )
    val_dataset.items = [item for item in val_dataset.items 
                        if os.path.exists(os.path.join("/Users/snehapatel/visionary/data/depth_labels/val", os.path.basename(item['path']).replace('.jpg', '_depth.npy').replace('.webp', '_depth.npy')))]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = FastDepthNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = berhu_loss(preds, gts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, gts in val_loader:
                imgs, gts = imgs.to(device), gts.to(device)
                preds = model(imgs)
                loss = berhu_loss(preds, gts)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(val_loader) + 1e-8)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "depth_model.pth"))
            print("Model saved!")

if __name__ == "__main__":
    train()
