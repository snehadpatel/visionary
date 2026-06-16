import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src.models.fast_depth_net import FastDepthNet
from src.models.unet import UNet

class RedesignDataset(Dataset):
    """
    Dataset for paired room redesign training.
    Handles 'room_XXXX_before.jpg' and 'room_XXXX_after.jpg' structure.
    """
    def __init__(self, root_dir, transform=None, device="mps"):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        
        # We'll use the pre-trained models to generate features for training pairs
        self.depth_net = FastDepthNet().to(device).eval()
        self.seg_net = UNet(n_classes=21).to(device).eval()
        
        # Find all 'before' images
        all_files = os.listdir(root_dir)
        self.base_names = [f.replace("_before.jpg", "") for f in all_files if f.endswith("_before.jpg")]

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base = self.base_names[idx]
        
        # Load Original (Before)
        orig_path = os.path.join(self.root_dir, f"{base}_before.jpg")
        image_bgr = cv2.imread(orig_path)
        orig_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        # Load Target (After)
        target_path = os.path.join(self.root_dir, f"{base}_after.jpg")
        target_img = Image.open(target_path).convert("RGB")
        
        # --- Feature Loading (Cached) ---
        depth_path = os.path.join(self.root_dir, "depth", f"{base}.png")
        seg_path = os.path.join(self.root_dir, "seg", f"{base}.png")
        
        if os.path.exists(depth_path) and os.path.exists(seg_path):
            depth_img = Image.open(depth_path).convert("L")
            seg_mask = np.array(Image.open(seg_path))
        else:
            # Fallback to on-the-fly inference if cache missing
            with torch.no_grad():
                depth_map = self.depth_net.infer(image_bgr, device=self.device)
                seg_mask = self.seg_net.infer(image_bgr, device=self.device)
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255
            depth_img = Image.fromarray(depth_norm.astype(np.uint8))
        
        # Mask (all non-background objects)
        mask_img = Image.fromarray((seg_mask > 0).astype(np.uint8) * 255)
        
        # Standard Resize
        resize = transforms.Resize((256, 256), Image.BICUBIC)
        orig_img = resize(orig_img)
        target_img = resize(target_img)
        depth_img = resize(depth_img)
        mask_img = resize(mask_img)
        
        # Transformation
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        orig_tensor = t(orig_img)
        target_tensor = t(target_img)
        depth_tensor = t(depth_img)
        mask_tensor = t(mask_img)
        
        input_tensor = torch.cat((orig_tensor, depth_tensor, mask_tensor), 0)
        return input_tensor, target_tensor

if __name__ == "__main__":
    # Dry run
    print("Dataset module ready.")
