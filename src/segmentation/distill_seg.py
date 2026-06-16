import os
import json
import torch
import cv2
import numpy as np
import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

def load_segformer():
    print("Loading SegFormer B5 for ADE20K...")
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, processor, device

def distill_segmentation(split_path: str, output_dir: str, limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    
    model, processor, device = load_segformer()
    
    project_root = "/Users/snehapatel/visionary"
    
    with open(split_path, 'r') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]
        
    print(f"Generating segmentation labels for {len(lines)} images...")
    
    processed_count = 0
    
    for line in tqdm.tqdm(lines, desc="SegFormer Inference"):
        item = json.loads(line)
        img_id = item['id']
        rel_path = item['local_path']
        img_path = os.path.join(project_root, rel_path)
        
        if not os.path.exists(img_path):
            continue
            
        # 1. Load image
        img = Image.open(img_path).convert("RGB")
        
        # 2. Preprocess and inference
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
            
            # Rescale logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=img.size[::-1], # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            prediction = upsampled_logits.argmax(dim=1).squeeze()
            
        mask = prediction.cpu().numpy().astype(np.uint8)
        
        # 3. Save mask as .png (efficient for categorical data)
        mask_out_path = os.path.join(output_dir, f"{img_id}_mask.png")
        cv2.imwrite(mask_out_path, mask)
        
        processed_count += 1
        
    print(f"Finished generating {processed_count} masks.")

if __name__ == "__main__":
    distill_segmentation(
        "/Users/snehapatel/visionary/data/splits/train.jsonl",
        "/Users/snehapatel/visionary/data/seg_labels/train",
        limit=500
    )
    distill_segmentation(
        "/Users/snehapatel/visionary/data/splits/val.jsonl",
        "/Users/snehapatel/visionary/data/seg_labels/val",
        limit=100
    )
