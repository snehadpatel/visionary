import torch
import numpy as np
import cv2
from PIL import Image
import os
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# Use mps for Mac M4
_device = "mps" if torch.backends.mps.is_available() else "cpu"
_dtype = torch.float32  # MPS often prefers float32 for certain ops, though float16 can work.

print(f"Image Generator initializing on {_device}...")

class ImageGenerator:
    def __init__(self):
        # ControlNet for Canny (strictly preserves hard geometry/edges)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=_dtype,
        )

        # Inpainting Pipeline with ControlNet
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=_dtype,
            safety_checker=None,
        ).to(_device)

        # Higher quality scheduler for SD v1.5
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, 
            use_karras_sigmas=True
        )
        
        self.pipe.enable_attention_slicing()

    def generate_redesign(
        self,
        original_img: Image.Image,
        depth_map: np.ndarray,
        masks: list[np.ndarray],
        redesign_spec: dict,
        floor_mask: np.ndarray = None,
    ) -> Image.Image:
        """
        High-Speed Total Room Transformation: Reimagines the entire room at 512px with heavy Canny lock.
        """
        # 1. Standard resolution for high-speed WOW demo
        gen_w, gen_h = 512, 512 
        orig_w, orig_h = original_img.size
        img_resized = original_img.resize((gen_w, gen_h), Image.LANCZOS)
        
        # 2. Extract detailed Canny Edges
        img_np = np.array(img_resized)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(img_gray, 50, 150)
        canny_edges = canny_edges[:, :, None]
        canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)
        control_image = Image.fromarray(canny_edges)

        # 3. Create a 'Full Selection' Mask
        full_mask = Image.new("RGB", (gen_w, gen_h), (255, 255, 255))

        # 4. Run the pipeline (FULL IMAGE Transformation)
        print(">>> COMMENCING HIGH-FIDELITY TOTAL ROOM REDESIGN...")
        result = self.pipe(
            prompt=redesign_spec["sd_prompt"] + ", architectural photography, high-end interior design, cinematic lighting, sharp details, luxury textures, masterpiece, 8k",
            negative_prompt=redesign_spec["negative_sd_prompt"] + ", blurry, distorted, painting, lowres, foggy, out of focus, hazy, messy",
            image=img_resized,
            mask_image=full_mask, 
            control_image=control_image,
            num_inference_steps=50, 
            guidance_scale=9.0, 
            strength=0.85, # Very high strength for "WOW" re-texturing
            controlnet_conditioning_scale=1.1, # Strong edge lock
        ).images[0]

        # 5. Resize back to original
        return result.resize((orig_w, orig_h), Image.LANCZOS)

if __name__ == "__main__":
    # Quick sanity check/dry run if weights are present
    try:
        gen = ImageGenerator()
        print("Success: Pipeline ready.")
    except Exception as e:
        print(f"Error initializing: {e}")
