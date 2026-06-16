"""
Visionary Pipeline — MiDaS DPT_Large Depth Estimation.
Produces dense depth maps from single room images.
Auto-downloads MiDaS weights (~400MB) on first use.
"""
import torch
import numpy as np
from PIL import Image

# Device selection
# Using CPU for Depth to leverage 10 cores and save GPU VRAM
_device = "cpu"

# Lazy loading
_midas = None
_transform = None


def _load_midas():
    global _midas, _transform
    if _midas is not None:
        return
    
    print(f"[Depth Estimator] Loading MiDaS DPT_Large on {_device}...")
    _midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    _transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _transform = _transforms.dpt_transform
    _midas.to(_device).eval()
    print("[Depth Estimator] MiDaS DPT_Large ready.")


def estimate_depth(img: Image.Image) -> np.ndarray:
    """
    Estimate depth from a single room image using MiDaS DPT_Large.
    
    Args:
        img: PIL Image of the room
    
    Returns:
        Normalized depth map as uint8 numpy array (0=near, 255=far)
    """
    _load_midas()
    
    arr = np.array(img)
    batch = _transform(arr).to(_device)
    
    with torch.no_grad():
        pred = _midas(batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=arr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = pred.cpu().numpy()
    # Normalize to 0-255 range
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
    return depth.astype(np.uint8)
