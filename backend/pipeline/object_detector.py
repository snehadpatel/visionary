"""
Visionary Pipeline — YOLO-World Open-Vocabulary Detection.
Detects a vast range of furniture and decor items globally.
Supports any custom class label without retraining.
"""
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

# Lazy-load the model
_model = None

# Comprehensive list of furniture and interior objects (Open-Vocabulary)
FURNITURE_PROMPT = [
    "bed", "couch", "sofa", "chair", "armchair", "dining table", "coffee table",
    "bookshelf", "wardrobe", "sideboard", "nightstand", "desk", "lamp", "chandelier",
    "pendant light", "rug", "curtain", "mirror", "potted plant", "vase", "television",
    "fireplace", "window", "door", "ceiling fan", "ottoman", "pouffe", "vanity",
    "chest of drawers", "bench", "bar stool", "bookshelf", "cabinet", "shelf"
]

def _get_model():
    global _model
    if _model is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = YOLOWorld("yolov8s-worldv2.pt")
        _model.to(device)
        _model.set_classes(FURNITURE_PROMPT)
        print(f"[Object Detector] YOLO-World ready on {device} with {len(FURNITURE_PROMPT)} classes.")
    return _model


def detect_objects(img: Image.Image) -> list[dict]:
    """
    Run YOLO-World detection on a room image.
    
    Args:
        img: PIL Image of the room
    
    Returns:
        List of detected furniture items with bounding boxes, confidence,
        center points, and size information.
    """
    model = _get_model()
    # YOLO-World works better with a slightly lower confidence threshold for niche items
    results = model.predict(np.array(img), conf=0.15, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        detections.append({
            "label": label,
            "confidence": round(float(box.conf), 3),
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "size": [x2 - x1, y2 - y1],
            "is_structural": label in {"window", "door", "fireplace", "staircase"}
        })
    
    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
