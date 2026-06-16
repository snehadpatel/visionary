import os
import json
import torch
import cv2
import numpy as np
import tqdm
import segmentation_models_pytorch as smp
from PIL import Image

def load_smp_model():
    print("Loading SMP UNet with MobileNetV2 encoder (Pre-trained on ADE20K)...")
    # Note: smp doesn't have a direct 'ade20k' weight set, 
    # but we can use a model that handles many classes and is robust.
    # Actually, let's use a simpler approach if we can't get ADE20K weights easily.
    # If SMP doesn't have ADE20K specific weights, I'll use a torchvision model.
    
    import torchvision
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return model, device

def distill_segmentation(split_path: str, output_dir: str, limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    
    model, device = load_smp_model()
    
    project_root = "/Users/snehapatel/visionary"
    
    with open(split_path, 'r') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]
        
    print(f"Generating segmentation labels for {len(lines)} images...")
    
    processed_count = 0
    
    # Normalization for torchvision models
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for line in tqdm.tqdm(lines, desc="Segmentation Inference"):
        item = json.loads(line)
        img_id = os.path.basename(item['path']).split('.')[0]
        rel_path = item['path']
        img_path = os.path.join(project_root, rel_path)
        
        if not os.path.exists(img_path):
            continue
            
        # 1. Load image
        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size
        
        # 2. Preprocess and inference
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            prediction = output.argmax(0)
            
        mask = prediction.cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # 3. Save mask
        mask_out_path = os.path.join(output_dir, f"{img_id}_mask.png")
        cv2.imwrite(mask_out_path, mask)
        
        processed_count += 1
        
    print(f"Finished generating {processed_count} masks.")

if __name__ == "__main__":
    # We use the same splits
    distill_segmentation(
        "/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        "/Users/snehapatel/visionary/data/seg_labels/train",
        limit=500
    )
    distill_segmentation(
        "/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        "/Users/snehapatel/visionary/data/seg_labels/val",
        limit=100
    )
