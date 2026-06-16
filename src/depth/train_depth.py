import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import cv2
import tqdm
from src.depth.fast_depth_net import FastDepthNet, berhu_loss
from src.depth.depth_evaluator import compute_depth_metrics

class RoomDepthDataset(Dataset):
    def __init__(self, split_jsonl, depth_dir, transform=None, limit=None):
        with open(split_jsonl, 'r') as f:
            self.items = [json.loads(line) for line in f]
        if limit:
            self.items = self.items[:limit]
        self.depth_dir = depth_dir
        self.transform = transform
        self.project_root = "/Users/snehapatel/visionary"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.project_root, item['local_path'])
        depth_path = os.path.join(self.depth_dir, f"{item['id']}_depth.npy")
        
        # Load RGB
        img = cv2.imread(img_path)
        if img is None:
            # Fallback if image missing
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        # Load Depth
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            depth = cv2.resize(depth, (256, 256))
            # Normalize to [0, 1] for training stability since FastDepthNet has Sigmoid output
            # MiDaS values can be large or negative.
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            depth = torch.from_numpy(depth).unsqueeze(0)
        else:
            depth = torch.zeros((1, 256, 256))
            
        return img, depth

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Config
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    
    # Dataset
    train_dataset = RoomDepthDataset(
        "/Users/snehapatel/visionary/data/splits/train.jsonl",
        "/Users/snehapatel/visionary/data/depth_labels/train",
        limit=500
    )
    val_dataset = RoomDepthDataset(
        "/Users/snehapatel/visionary/data/splits/val.jsonl",
        "/Users/snehapatel/visionary/data/depth_labels/val",
        limit=100
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FastDepthNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_loss = float('inf')
    
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
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_state_dict(), "/Users/snehapatel/visionary/models/depth_model.pth")
            print("Model saved!")

if __name__ == "__main__":
    train()
