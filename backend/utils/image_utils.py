"""
Visionary — Image utility functions.
"""
from PIL import Image
import io
import numpy as np


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (H, W, 3) uint8."""
    return np.array(img.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array (H, W, 3) to PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def resize_maintain_aspect(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image maintaining aspect ratio, fitting within max_size."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)
