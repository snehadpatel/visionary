"""
Visionary Pipeline — Scene Graph Builder.
Combines VLM analysis, YOLO detections, SAM masks, and MiDaS depth
into a unified structured scene representation.
"""
import numpy as np
from PIL import Image


def build_scene_graph(
    scene_analysis: dict,
    detections: list[dict],
    masks: list[dict],
    depth_map: np.ndarray,
    image_size: tuple,
) -> dict:
    """
    Build a complete scene graph combining all pipeline outputs.
    
    Args:
        scene_analysis: Structured JSON from SceneNet room analysis
        detections: YOLO detection results
        masks: SAM segmentation masks
        depth_map: MiDaS depth estimation (H, W) uint8
        image_size: (width, height) of the image
    
    Returns:
        Complete scene graph with spatial reasoning
    """
    w, h = image_size
    objects = []
    
    for det, m in zip(detections, masks):
        x1, y1, x2, y2 = det["bbox"]
        
        # Calculate average depth in object region
        depth_region = depth_map[y1:y2, x1:x2]
        avg_depth = float(np.mean(depth_region)) if depth_region.size > 0 else 128.0
        
        cx, cy = det["center"]
        
        objects.append({
            "label": det["label"],
            "confidence": det["confidence"],
            "zone": _zone(cx / w, cy / h),
            "position": {"x": cx, "y": cy},
            "size_pct": {
                "w": round((det["size"][0] / w) * 100, 1),
                "h": round((det["size"][1] / h) * 100, 1),
            },
            "depth": round(avg_depth, 1),
            "is_structural": m.get("is_structural", False),
            "bbox": det["bbox"],
        })
    
    # Sort by depth (nearest first)
    objects.sort(key=lambda o: o["depth"])
    
    # Compute spatial relationships
    relationships = _compute_relationships(objects)
    
    return {
        "image_size": {"w": w, "h": h},
        "scene_analysis": scene_analysis,
        "objects": objects,
        "relationships": relationships,
        "room_type": scene_analysis.get("room_type", "living room"),
        "current_style": scene_analysis.get("style", "unknown"),
        "color_palette": scene_analysis.get("palette", []),
        "room_size": "medium", # Fallback
        "natural_light": scene_analysis.get("lighting", "medium"),
    }


def _zone(xr: float, yr: float) -> str:
    """Determine spatial zone from normalized coordinates."""
    col = "left" if xr < 0.33 else ("center" if xr < 0.66 else "right")
    row = "top" if yr < 0.33 else ("middle" if yr < 0.66 else "bottom")
    return f"{row}-{col}"


def _compute_relationships(objects: list[dict]) -> list[dict]:
    """Compute spatial relationships between detected objects."""
    relationships = []
    
    for i, obj_a in enumerate(objects):
        for j, obj_b in enumerate(objects):
            if i >= j:
                continue
            
            ax, ay = obj_a["position"]["x"], obj_a["position"]["y"]
            bx, by = obj_b["position"]["x"], obj_b["position"]["y"]
            
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            
            # Determine relationship
            rel_type = "near" if dist < 200 else "far"
            if abs(obj_a["depth"] - obj_b["depth"]) < 20:
                depth_rel = "same_plane"
            elif obj_a["depth"] < obj_b["depth"]:
                depth_rel = "in_front"
            else:
                depth_rel = "behind"
            
            relationships.append({
                "object_a": obj_a["label"],
                "object_b": obj_b["label"],
                "spatial": rel_type,
                "depth_relation": depth_rel,
                "distance_px": round(dist),
            })
    
    return relationships
