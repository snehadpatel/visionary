import os
import json
import torch
import cv2
import numpy as np
import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def load_midas():
    print("Loading MiDaS DPT_Large...")
    model_type = "DPT_Large"  # or "MiDaS_small" for faster processing
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
        
    return midas, transform, device

def distill_dataset(split_path: str, output_dir: str, limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    
    midas, transform, device = load_midas()
    
    project_root = "/Users/snehapatel/visionary"
    
    with open(split_path, 'r') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]
        
    print(f"Distilling {len(lines)} images from {split_path}...")
    
    processed_count = 0
    
    for line in tqdm.tqdm(lines, desc="MiDaS Inference"):
        item = json.loads(line)
        img_id = item['id']
        rel_path = item['local_path']
        img_path = os.path.join(project_root, rel_path)
        
        if not os.path.exists(img_path):
            continue
            
        # 1. Load image
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Transform and inference
        input_batch = transform(img).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth = prediction.cpu().numpy()
        
        # 3. Save depth map (normalized float32)
        # MiDaS outputs relative inverse depth.
        # We'll save it as a .npy for training.
        depth_out_path = os.path.join(output_dir, f"{img_id}_depth.npy")
        np.save(depth_out_path, depth)
        
        processed_count += 1
        
    print(f"Finished distilling {processed_count} images.")

if __name__ == "__main__":
    # We'll distill the train split first.
    # We can also distill val split for evaluation.
    distill_dataset(
        "/Users/snehapatel/visionary/data/splits/train.jsonl",
        "/Users/snehapatel/visionary/data/depth_labels/train",
        limit=500  # Start with a subset to verify
    )
    distill_dataset(
        "/Users/snehapatel/visionary/data/splits/val.jsonl",
        "/Users/snehapatel/visionary/data/depth_labels/val",
        limit=100
    )
