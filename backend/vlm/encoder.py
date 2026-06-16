"""
Visionary VLM — CLIP ViT-L/14 Vision Encoder.
Extracts rich visual features (768-dim) from room images.
Runs on Apple MPS for GPU acceleration.
"""
import torch
import clip
from PIL import Image

# Device selection — prefer MPS on Apple Silicon
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

_model = None
_preprocess = None


def _load_clip():
    global _model, _preprocess
    if _model is not None:
        return
    print(f"[VLM Encoder] Loading CLIP ViT-L/14 on {device}...")
    _model, _preprocess = clip.load("ViT-L/14", device=device)
    _model.eval()
    print("[VLM Encoder] CLIP ViT-L/14 ready.")


def encode_image(img: Image.Image) -> torch.Tensor:
    """
    Encode a PIL Image into a normalized CLIP embedding.
    
    Returns:
        torch.Tensor: Normalized image embedding of shape (1, 768)
    """
    _load_clip()
    with torch.no_grad():
        inp = _preprocess(img).unsqueeze(0).to(device)
        features = _model.encode_image(inp)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.float()  # (1, 768) — ensure float32 for projection


def encode_text(text: str) -> torch.Tensor:
    """
    Encode text into a normalized CLIP embedding (for style classification).
    
    Returns:
        torch.Tensor: Normalized text embedding of shape (1, 768)
    """
    _load_clip()
    with torch.no_grad():
        tokens = clip.tokenize([text], truncate=True).to(device)
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.float()
