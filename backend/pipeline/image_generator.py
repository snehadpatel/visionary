"""
Visionary Pipeline — Stable Diffusion v1.5 + ControlNet Canny Image Generator.
Generates photorealistic room redesigns preserving structural elements.
Optimized for Apple MPS (no xformers, float32).
"""
import asyncio
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# Device selection
if torch.backends.mps.is_available():
    _device = "mps"
    # Ensure fallback is enabled for any unsupported ops
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
elif torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"

# Use float32 on MPS for stability (prevents black images), float16 on CUDA
_dtype = torch.float32 if _device == "mps" else torch.float16

# Lazy loading
_pipe = None
_custom_gen = None

def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return
    
    print(f"[Image Generator] Loading SD v1.5 + ControlNet Canny on {_device} ({_dtype})...")
    
    # Depth ControlNet (more professional for interior design)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=_dtype,
    )
    
    _pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=_dtype,
        safety_checker=None,
        feature_extractor=None,
    ).to(_device)
    
    _pipe.scheduler = UniPCMultistepScheduler.from_config(_pipe.scheduler.config)
    
    # Optimization for 8GB Mac
    _pipe.enable_attention_slicing()
    
    print("[Image Generator] SD v1.5 + ControlNet ready (MPS Stability Mode).")


def _make_canny_map(img: Image.Image) -> Image.Image:
    """
    Extract structural edges only.
    Heavy blur first removes furniture texture noise,
    leaving only hard structural lines: walls, windows, doors.
    """
    arr = np.array(img.convert("L"))
    blurred = cv2.GaussianBlur(arr, (15, 15), 0)
    canny = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return Image.fromarray(canny).convert("RGB")


def generate_redesigned_image(
    original_img: Image.Image,
    depth_map: np.ndarray,
    masks: list[dict],
    sd_prompt: str,
    negative_prompt: str,
) -> Image.Image:
    """
    Generate a photorealistic redesigned room image using SD v1.5 + ControlNet Canny.
    
    Uses high strength (0.97) for full redesign rather than just retexturing.
    Canny ControlNet preserves structural elements (walls, windows, doors)
    while allowing creative freedom for furniture and decor.
    
    Args:
        original_img: Original room image (PIL)
        depth_map: MiDaS depth estimation
        masks: SAM segmentation masks
        sd_prompt: Positive prompt from sd_prompt_builder
        negative_prompt: Negative prompt from sd_prompt_builder
    
    Returns:
        Redesigned room as PIL Image
    """
    _load_pipeline()
    
    # Resize for SD (must be multiple of 8)
    gen_size = (768, 768)  # Good balance of quality and speed on MPS
    original_resized = original_img.resize(gen_size, Image.LANCZOS)
    
    canny_map = _make_canny_map(original_resized)
    
    result = _pipe(
        prompt=sd_prompt,
        negative_prompt=negative_prompt,
        image=original_resized,
        control_image=Image.fromarray(depth_map).convert("RGB"),
        strength=0.97,                        # high = full redesign, not just retexture
        num_inference_steps=30,               # Re-calibrated for quality/speed balance
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.35,   # lower = more creative transformation
        width=gen_size[0],
        height=gen_size[1],
    ).images[0]
    
    # Resize back to original dimensions
    result = result.resize(original_img.size, Image.LANCZOS)
    
    return result
async def generate_redesigned_image_streaming(
    original_img: Image.Image,
    depth_map: np.ndarray,
    sd_prompt: str,
    negative_prompt: str,
    client_id: str,
    manager: any,
) -> Image.Image:
    """
    Generate image with real-time previews streamed to frontend via WebSocket.
    """
    _load_pipeline()
    
    gen_size = (768, 768)
    original_resized = original_img.resize(gen_size, Image.LANCZOS)
    
    # Use Depth instead of Canny for professional volume preservation
    depth_resized = cv2.resize(depth_map, gen_size, interpolation=cv2.INTER_LINEAR)
    control_image = Image.fromarray(depth_resized).convert("RGB")
    
    total_steps = 30
    
    # Run in executor to not block the event loop.
    # We capture the main loop here so callback can safely schedule preview sends.
    loop = asyncio.get_event_loop()

    def callback_wrapper(pipe, step_index, timestep, callback_kwargs):
        # Stream preview every 10 steps
        if step_index % 10 == 0:
            latents = callback_kwargs["latents"]
            with torch.no_grad():
                # Decode latents to approximate image
                # This is a bit complex for v1.5, usually we use a fast VAE or just skip
                # For simplicity and speed, we will use the current pipeline's VAE
                # NOTE: In a real production app, you might want a tiny decoder
                try:
                    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                    # Since this is a sync callback in a thread, we use a future or just run_coroutine_threadsafe
                    asyncio.run_coroutine_threadsafe(
                        manager.send_image_preview(client_id, image, step_index, total_steps),
                        loop
                    )
                except Exception as e:
                    print(f"Preview error: {e}")
        return callback_kwargs

    def run_sd():
        return _pipe(
            prompt=sd_prompt,
            negative_prompt=negative_prompt,
            image=original_resized,
            control_image=control_image,
            strength=0.97,
            num_inference_steps=30,
            guidance_scale=9.0,
            controlnet_conditioning_scale=0.55,
            width=gen_size[0],
            height=gen_size[1],
            callback_on_step_end=callback_wrapper,
            callback_on_step_end_tensor_inputs=["latents"]
        ).images[0]

    result = await loop.run_in_executor(None, run_sd)
    return result.resize(original_img.size, Image.LANCZOS)

def _load_custom_gen():
    global _custom_gen
    if _custom_gen is not None:
        return
    
    print(f"[Image Generator] Loading Custom Neural Engine (UNet) on {_device}...")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.redesign.custom_generator import CustomNeuralGenerator
    
    weights = "/Users/snehapatel/visionary/models/redesign_generator.pth"
    _custom_gen = CustomNeuralGenerator(weights, device=_device)
    print("[Image Generator] Custom Neural Engine ready.")


async def generate_custom_redesign_streaming(
    original_img: Image.Image,
    depth_map: np.ndarray,
    masks: list[np.ndarray],
    client_id: str,
    manager: any,
    target_style: str = "modern",
) -> Image.Image:
    """
    High-speed redesign generator for 'Live Mode' feeling.
    Completes in < 1 second.
    """
    _load_custom_gen()
    
    # 1. Start progress
    await manager.send(client_id, "pipeline_step", {
        "step": "Neural Image Synthesis",
        "progress": 50,
        "elapsed_seconds": 0.1
    })
    
    loop = asyncio.get_event_loop()
    
    # Run in executor to prevent blocking
    def run_custom():
        return _custom_gen.generate_redesign(
            original_img=original_img,
            depth_map=depth_map,
            masks=masks,
            target_style=target_style
        )
        
    result = await loop.run_in_executor(None, run_custom)
    
    await manager.send(client_id, "pipeline_step", {
        "step": "Synthesis Complete",
        "progress": 100,
        "elapsed_seconds": 0.5
    })
    
    return result
