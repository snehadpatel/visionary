import torch
from torch.utils.data import Dataset
import json
import os
import cv2
import numpy as np
from pathlib import Path

class VisionaryDepthDataset(Dataset):
    def __init__(self, jsonl_path, depth_dir, transform=None, size=(256, 256)):
        with open(jsonl_path, 'r') as f:
            self.items = [json.loads(line) for line in f]
        self.depth_dir = Path(depth_dir)
        self.transform = transform
        self.size = size
        self.project_root = Path("/Users/snehapatel/visionary")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.project_root / item['path']
        img_id = Path(item['path']).stem
        
        # In our distillation, we used item['id'] which was just the filename stem for some or random ID
        # Let's check how distill_depth.py saves it.
        # It used item['id'] from metadata.jsonl.
        # But buildings rooms.jsonl uses 'path'. 
        # I'll use Path(item['path']).stem as ID.
        depth_path = self.depth_dir / f"{img_id}_depth.npy"
        
        # RGB
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((*self.size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size)
            
        img = img.astype(np.float32) / 255.0
        # Normalize (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Depth
        if depth_path.exists():
            depth = np.load(depth_path).astype(np.float32)
            depth = cv2.resize(depth, self.size)
            # Normalize to [0, 1] - FastDepthNet has Sigmoid
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                depth = (depth - d_min) / (d_max - d_min)
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        else:
            depth = torch.zeros((1, *self.size)).float()
            
        return img, depth

class VisionarySegDataset(Dataset):
    def __init__(self, jsonl_path, seg_dir, transform=None, size=(384, 384)):
        with open(jsonl_path, 'r') as f:
            self.items = [json.loads(line) for line in f]
        self.seg_dir = Path(seg_dir)
        self.transform = transform
        self.size = size
        self.project_root = Path("/Users/snehapatel/visionary")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.project_root / item['path']
        img_id = Path(item['path']).stem
        mask_path = self.seg_dir / f"{img_id}_mask.png"
        
        # RGB
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((*self.size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size)
            
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Mask
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long()
        else:
            mask = torch.zeros(self.size).long()
            
        return img, mask
