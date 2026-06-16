"""
Visionary Pipeline — SAM (Segment Anything Model) Segmenter.
Uses box-prompted segmentation from YOLO detections to get precise object masks.
SAM vit_h for best quality (2.5GB weights).
"""
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import torch
from pathlib import Path

# Device selection — prefer MPS on Apple Silicon
if torch.backends.mps.is_available():
    _device = "mps"
elif torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"

STRUCTURAL_ELEMENTS = {"window", "door"}

# SAM vit_b for balanced quality and speed (375MB weights)
_predictor = None
_sam_checkpoint = str(Path(__file__).resolve().parent.parent.parent / "models" / "sam_vit_b_01ec17.pth")

def _get_predictor():
    global _predictor
    if _predictor is None:
        print(f"[Segmenter] Loading SAM vit_b on {_device}...")
        try:
            _sam = sam_model_registry["vit_b"](checkpoint=_sam_checkpoint)
            _sam.to(_device)
            _predictor = SamPredictor(_sam)
            print("[Segmenter] SAM vit_b ready.")
        except FileNotFoundError:
            print(f"[Segmenter] SAM vit_b checkpoint not found at {_sam_checkpoint}")
            print("[Segmenter] Falling back to fast bbox-masking.")
            _predictor = None
        except Exception as e:
            print(f"[Segmenter] SAM load error: {e}. Falling back to bbox masks.")
            _predictor = None
    return _predictor


def segment_objects(img: Image.Image, detections: list[dict]) -> list[dict]:
    """
    Generate precise segmentation masks for detected objects using SAM.
    Falls back to rectangular bbox masks if SAM is unavailable.
    
    Args:
        img: PIL Image of the room
        detections: List of YOLO detections with bbox info
    
    Returns:
        List of segmented objects with masks and metadata
    """
    predictor = _get_predictor()
    img_array = np.array(img)
    
    if predictor is not None:
        predictor.set_image(img_array)
    
    segments = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        
        if predictor is not None:
            try:
                masks, scores, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]),
                    multimask_output=False,
                )
                mask = masks[0]
                score = float(scores[0])
            except Exception as e:
                print(f"[Segmenter] SAM prediction failed for {det['label']}: {e}")
                mask = _bbox_mask(img_array.shape[:2], det["bbox"])
                score = det["confidence"]
        else:
            # Fallback: use bounding box as mask
            mask = _bbox_mask(img_array.shape[:2], det["bbox"])
            score = det["confidence"]
        
        segments.append({
            "label": det["label"],
            "mask": mask.tolist(),
            "score": round(score, 3),
            "is_structural": det["label"] in STRUCTURAL_ELEMENTS,
            "bbox": det["bbox"],
        })
    
    return segments


def _bbox_mask(img_shape: tuple, bbox: list) -> np.ndarray:
    """Create a simple rectangular mask from bounding box (fallback)."""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = True
    return mask
