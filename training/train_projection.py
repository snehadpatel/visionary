"""
Train the projection MLP on interior design image-text pairs.
This maps CLIP visual features (768-dim) to TinyLlama's language space (2048-dim).
Run this once before starting the server for best VLM quality.

Usage:
    cd training
    python train_projection.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip
import json
from PIL import Image
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from vlm.projection import ProjectionMLP

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load CLIP encoder
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()


class InteriorDataset(Dataset):
    """
    Expects data/interior_qa.json:
    [{"image_path": "...", "description": "modern living room with grey sofa..."}, ...]
    """
    def __init__(self, json_path: str):
        with open(json_path) as f:
            self.data = json.load(f)
        # Filter out entries with missing images
        self.data = [d for d in self.data if Path(d["image_path"]).exists()]
        print(f"Dataset loaded: {len(self.data)} valid samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        img_tensor = clip_preprocess(img)
        return img_tensor, item["description"]


def train():
    data_path = Path(__file__).parent / "data" / "interior_qa.json"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Create training/data/interior_qa.json with image-description pairs first.")
        return

    dataset = InteriorDataset(str(data_path))
    if len(dataset) == 0:
        print("No valid training samples found. Check image paths in interior_qa.json.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    projection = ProjectionMLP(clip_dim=768, llm_dim=2048).to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * 10)
    criterion = nn.MSELoss()

    print("Starting projection MLP training...")
    for epoch in range(2):
        total_loss = 0
        for i, (imgs, texts) in enumerate(loader):
            imgs = imgs.to(device)

            with torch.no_grad():
                img_feats = clip_model.encode_image(imgs).float()  # (B, 768)
                text_tokens = clip.tokenize(texts, truncate=True).to(device)
                text_feats = clip_model.encode_text(text_tokens).float()  # (B, 768)
                # Expand text features to target llm_dim
                targets = text_feats.repeat(1, 3)[:, :2048]

            projected = projection(img_feats)
            loss = criterion(projected, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/2 - Batch {i}/{len(loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}/2 - Avg Loss: {avg_loss:.4f}")

    save_path = Path(__file__).parent.parent / "models" / "projection_layer.pt"
    torch.save(projection.state_dict(), save_path)
    print(f"Saved projection_layer.pt to {save_path}")


if __name__ == "__main__":
    train()
