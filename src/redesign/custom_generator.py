import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from src.models.generator import RedesignGenerator
import os

class CustomNeuralGenerator:
    """
    VLM-Free Neural Redesign Generator.
    Uses a custom Pix2Pix UNet trained on room transformation pairs.
    """
    def __init__(self, weights_path="models/redesign_generator.pth", device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"CustomNeuralGenerator initializing on {self.device}...")
        
        # Initialize model with 5 input channels (RGB + Depth + Mask)
        self.model = RedesignGenerator(n_channels=5, n_classes=3).to(self.device)
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded custom weights from {weights_path}")
        else:
            print(f"WARNING: Weights not found at {weights_path}. Running with random initialization.")
            
        self.model.eval()
        
        # Normalization matching the training script
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _correct_color(self, source_img: np.ndarray, target_img: np.ndarray, mask: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        Luminance-preserving color transfer.
        Keep Target L (lighting), shift AB (color) towards Source.
        """
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        if not np.any(mask):
            return target_img
            
        mask_bool = mask > 0
        
        # Calculate stats for masked region
        s_mean = source_lab[mask_bool].mean(axis=0)
        t_mean = target_lab[mask_bool].mean(axis=0)
        
        # Perform shift only on AB channels (1 and 2), preserving L (0)
        corrected_lab = target_lab.copy()
        for i in [1, 2]:
            # Blend original color with target style color
            corrected_lab[:,:,i] = (target_lab[:,:,i] - t_mean[i]) + s_mean[i]
            # Smooth blend based on alpha
            corrected_lab[:,:,i] = cv2.addWeighted(target_lab[:,:,i], 1-alpha, corrected_lab[:,:,i], alpha, 0)
            
        corrected_rgb = cv2.cvtColor(corrected_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return corrected_rgb

    def generate_redesign(
        self,
        original_img: Image.Image,
        depth_map: np.ndarray,
        masks: list[np.ndarray],
        target_style: str = None,
        floor_mask: np.ndarray = None,
    ) -> Image.Image:
        """
        Neural Transformation with Style-Based Color Correction
        """
        orig_w, orig_h = original_img.size
        orig_np = np.array(original_img.convert("RGB"))
        
        # 0. Get Style Palette if requested
        target_palette = None
        if target_style:
            try:
                from src.redesign.style_engine import StyleEngine
                # Assuming StyleEngine can be initialized without args or with default path
                se = StyleEngine("/Users/snehapatel/visionary/data/annotations/furniture_index.json", 
                                "/Users/snehapatel/visionary/data/annotations/style_signatures.json")
                inspiration = se.get_style_inspiration(target_style)
                target_palette = inspiration.get("palette")
            except:
                pass
        
        # 1. Prepare Mask
        if masks:
            # Handle both list of dicts (from segment_objects) and list of arrays
            m0 = masks[0]
            m0_arr = np.array(m0["mask"]) if isinstance(m0, dict) else m0
            
            combined_h, combined_w = m0_arr.shape
            combined_mask = np.zeros((combined_h, combined_w), dtype=np.uint8)
            for m in masks:
                m_arr = np.array(m["mask"]) if isinstance(m, dict) else m
                combined_mask = np.maximum(combined_mask, (m_arr > 0).astype(np.uint8))
            mask_pil = Image.fromarray(combined_mask * 255)
        else:
            mask_pil = Image.new("L", (256, 256), 0)
            
        # 2. Prepare RGB with Architectural Inpainting
        orig_pil_resized = original_img.resize((256, 256), Image.BICUBIC).convert("RGB")
        mask_pil_resized = mask_pil.resize((256, 256), Image.NEAREST)
        
        rgb_np = np.array(orig_pil_resized).astype(np.float32)
        m_np = (np.array(mask_pil_resized) > 0).astype(np.float32)
        rgb_np[m_np > 0] = [128, 128, 128] # Neutral gray void
        
        inpainted_pil = Image.fromarray(rgb_np.astype(np.uint8))
        rgb_tensor = self.transform(inpainted_pil).to(self.device)
        
        # 3. Depth Fix (Near=1.0)
        depth_norm = 1.0 - (depth_map / 255.0).clip(0, 1)
        depth_pil = Image.fromarray((depth_norm * 255).astype(np.uint8))
        depth_tensor = self.transform(depth_pil).to(self.device)[0:1, :, :]
        
        mask_tensor = self.transform(mask_pil).to(self.device)[0:1, :, :]
        
        # 2. Inference
        input_tensor = torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0).unsqueeze(0)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            
        # 3. Denormalize & Correction
        output_np = output_tensor.squeeze(0).cpu().numpy()
        output_np = (output_np * 0.5 + 0.5).clip(0, 1)
        output_np = (output_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Resize GAN output back to original for color correction
        gan_rgb = cv2.resize(output_np, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 4. Neural Color Correction
        # If we have a target palette from the style engine, we use it. 
        # Otherwise, we fallback to original image stats to maintain lighting.
        full_mask = cv2.resize((np.array(mask_pil) > 0).astype(np.uint8) * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        if target_palette and len(target_palette) > 0:
            # Create a synthetic 'ideal' image from the palette for color transfer
            palette_np = np.array(target_palette, dtype=np.uint8).reshape(1, -1, 3)
            palette_res = cv2.resize(palette_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            corrected_rgb = self._correct_color(palette_res, gan_rgb, full_mask)
        else:
            corrected_rgb = self._correct_color(orig_np, gan_rgb, full_mask)
        
        # 5. Seamless Blending with edge feathering
        mask_f = full_mask.astype(float) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (21, 21), 0)
        mask_f = np.expand_dims(mask_f, axis=-1)
        
        final_rgb = (orig_np * (1.0 - mask_f) + corrected_rgb * mask_f).astype(np.uint8)
        return Image.fromarray(final_rgb)

if __name__ == "__main__":
    # Quick Test
    try:
        gen = CustomNeuralGenerator()
        print("Success: CustomNeuralGenerator ready.")
    except Exception as e:
        print(f"Error initializing: {e}")
