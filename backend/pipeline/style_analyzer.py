"""
Visionary Pipeline — CLIP Zero-Shot Style Classifier.
Classifies room style using CLIP text-image similarity.
Reuses the CLIP encoder already loaded for the VLM.
"""
import torch
import clip
from PIL import Image

# Device selection
if torch.backends.mps.is_available():
    _device = "mps"
elif torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"

STYLES = [
    "scandinavian minimalist interior",
    "industrial loft interior",
    "bohemian eclectic interior",
    "mid-century modern interior",
    "japandi interior",
    "minimalist interior",
    "coastal beach house interior",
    "luxury contemporary interior",
    "rustic farmhouse interior",
    "traditional classic interior",
    "modern interior",
    "contemporary interior",
    "art deco interior",
]

STYLE_NAMES = [
    "scandinavian",
    "industrial",
    "bohemian",
    "mid-century modern",
    "japandi",
    "minimalist",
    "coastal",
    "luxury",
    "rustic",
    "traditional",
    "modern",
    "contemporary",
    "art deco",
]

# Lazy loading
_model = None
_preprocess = None
_text_features = None


def _load():
    global _model, _preprocess, _text_features
    if _model is not None:
        return
    
    print(f"[Style Analyzer] Loading CLIP for style classification on {_device}...")
    _model, _preprocess = clip.load("ViT-L/14", device=_device)
    _model.eval()
    
    # Pre-compute text embeddings for all styles
    with torch.no_grad():
        tokens = clip.tokenize(STYLES, truncate=True).to(_device)
        _text_features = _model.encode_text(tokens)
        _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)
    
    print("[Style Analyzer] Ready.")


def classify_style(img: Image.Image, top_k: int = 3) -> list[dict]:
    """
    Classify the interior style of a room image using CLIP zero-shot.
    
    Args:
        img: PIL Image of the room
        top_k: Number of top style matches to return
    
    Returns:
        List of dicts with style name and confidence score
    """
    _load()
    
    with torch.no_grad():
        inp = _preprocess(img).unsqueeze(0).to(_device)
        img_features = _model.encode_image(inp)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = (img_features @ _text_features.T).squeeze(0)
        # Convert to probabilities
        probs = (similarity * 100).softmax(dim=-1)
    
    probs_list = probs.cpu().tolist()
    results = [
        {"style": STYLE_NAMES[i], "confidence": round(probs_list[i], 4)}
        for i in range(len(STYLE_NAMES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_k]


def get_dominant_style(img: Image.Image) -> str:
    """Get the single most likely style for a room image."""
    results = classify_style(img, top_k=1)
    return results[0]["style"] if results else "modern"
