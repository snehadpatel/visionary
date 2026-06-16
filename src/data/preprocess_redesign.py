import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from src.models.fast_depth_net import FastDepthNet
from src.models.unet import UNet

def preprocess_dataset(root_dir, device="mps"):
    """
    Pre-processes the huggingface_rooms dataset by caching depth and segmentation maps.
    """
    depth_dir = os.path.join(root_dir, "depth")
    seg_dir = os.path.join(root_dir, "seg")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    # Load models once
    print(f"Loading models on {device}...")
    depth_net = FastDepthNet().to(device).eval()
    seg_net = UNet(n_classes=21).to(device).eval()
    
    # Weights are assumed to be in the project root /models/ as per src/pipeline.py
    # But for a dry run/preprocessing, we can use the default initialization if weights are missing
    
    all_files = os.listdir(root_dir)
    before_files = [f for f in all_files if f.endswith("_before.jpg")]
    
    print(f"Pre-processing {len(before_files)} images...")
    
    for f in tqdm(before_files):
        base = f.replace("_before.jpg", "")
        img_path = os.path.join(root_dir, f)
        image_bgr = cv2.imread(img_path)
        
        if image_bgr is None: continue
        
        # 1. Inference
        with torch.no_grad():
            depth_map = depth_net.infer(image_bgr, device=device)
            seg_mask = seg_net.infer(image_bgr, device=device)
        
        # 2. Save Depth (normalized to 0-255)
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255
        cv2.imwrite(os.path.join(depth_dir, f"{base}.png"), depth_norm.astype(np.uint8))
        
        # 3. Save Segmentation
        cv2.imwrite(os.path.join(seg_dir, f"{base}.png"), seg_mask.astype(np.uint8))

if __name__ == "__main__":
    dataset_root = "/Users/snehapatel/visionary/data/raw/huggingface_rooms"
    if os.path.exists(dataset_root):
        preprocess_dataset(dataset_root)
    else:
        print(f"Dataset root {dataset_root} not found.")
