"""
Real-Time Analyzer — Lightweight analysis pipeline for Visionary Live mode.

Three-tier inference strategy:
  Tier 1 (every frame, ~15ms):  SceneNet  — room type, style, lighting, palette
  Tier 2 (every 2-3s, ~200ms):  YOLO      — object detection + bounding boxes
  Tier 3 (on-demand, ~2000ms):  VLM       — deep conversational understanding

Includes:
  - Frame deduplication (skip unchanged frames)
  - Rolling SceneState with exponential moving average smoothing
  - Fast and full analysis modes
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from PIL import Image
from pathlib import Path

from pipeline.frame_dedup import FrameDeduplicator


# ─── Scene State (Smoothed across frames) ───

# Global model cache to avoid re-loading on every frame/upload
_scene_net_global = None
_yolo_global = None

def _get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

@dataclass
class SceneState:
    """
    Accumulated scene understanding smoothed across frames.
    Uses exponential moving average to avoid jittery predictions.
    """
    room_type: str = "unknown"
    room_type_confidence: float = 0.0
    style: str = "unknown"
    style_confidence: float = 0.0
    lighting: str = "medium"
    palette: List[List[int]] = field(default_factory=lambda: [[128, 128, 128]] * 3)
    condition: str = "fair"
    
    # Smoothed probability distributions
    room_type_probs: Dict[str, float] = field(default_factory=dict)
    style_probs: Dict[str, float] = field(default_factory=dict)
    
    # Detection state
    detections: list = field(default_factory=list)
    object_count: int = 0
    
    # Timing
    last_scenenet_ms: float = 0
    last_yolo_ms: float = 0
    frame_count: int = 0

    def to_dict(self) -> dict:
        return {
            "room_type": self.room_type,
            "room_type_confidence": round(self.room_type_confidence, 3),
            "style": self.style,
            "style_confidence": round(self.style_confidence, 3),
            "lighting": self.lighting,
            "palette": self.palette,
            "condition": self.condition,
            "room_type_probs": {k: round(v, 3) for k, v in self.room_type_probs.items()},
            "style_probs": {k: round(v, 3) for k, v in self.style_probs.items()},
            "detections": self.detections,
            "object_count": self.object_count,
            "last_scenenet_ms": round(self.last_scenenet_ms, 1),
            "last_yolo_ms": round(self.last_yolo_ms, 1),
            "frame_count": self.frame_count,
        }


class RealtimeAnalyzer:
    """
    Manages the real-time frame analysis pipeline.
    
    Lazily loads models on first use. Maintains a rolling SceneState
    that smooths predictions across frames using EMA.
    """

    def __init__(self, ema_alpha: float = 0.3):
        """
        Args:
            ema_alpha: Smoothing factor for EMA (0 = no update, 1 = no smoothing)
        """
        self.ema_alpha = ema_alpha
        self.deduplicator = FrameDeduplicator(similarity_threshold=5)
        self.state = SceneState()
        
    def _get_scene_net(self):
        """Lazy load SceneNet model with global caching."""
        global _scene_net_global
        if _scene_net_global is None:
            import torch
            import sys
            
            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from src.models.scene_net import SceneNet
            
            device = _get_device()
            _scene_net_global = SceneNet(pretrained=False).to(device)
            
            weights_path = project_root / "models" / "scene_net.pth"
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=device)
                if "model_state_dict" in checkpoint:
                    _scene_net_global.load_state_dict(checkpoint["model_state_dict"])
                else:
                    _scene_net_global.load_state_dict(checkpoint)
                print(f"[RealtimeAnalyzer] SceneNet weights loaded into global cache from {weights_path}")
            else:
                print(f"[RealtimeAnalyzer] ⚠ No SceneNet weights found at {weights_path}")
            
            _scene_net_global.eval()
        return _scene_net_global

    def _get_yolo(self):
        """Lazy load YOLO model with global caching."""
        global _yolo_global
        if _yolo_global is None:
            from pipeline.object_detector import _get_model
            _yolo_global = _get_model()
            print("[RealtimeAnalyzer] YOLO loaded into global cache")
        return _yolo_global

    def analyze_frame_fast(self, img: Image.Image) -> dict:
        """
        Tier 1: SceneNet only (~15ms).
        Run on every incoming frame for real-time scene cards.
        
        Args:
            img: PIL Image
            
        Returns:
            SceneNet results as dict
        """
        import cv2
        import numpy as np
        
        # Convert PIL to numpy for dedup check
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Dedup check
        if not self.deduplicator.is_new_frame(img_bgr):
            # Return cached state without re-processing
            res = self.state.to_dict()
            res["is_new_frame"] = False
            return res
        
        self.state.frame_count += 1
        
        # Run SceneNet
        t0 = time.perf_counter()
        scene_net = self._get_scene_net()
        result = scene_net.infer(img_bgr, device=_get_device())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        # Update state with EMA smoothing
        self._update_state_ema(result)
        self.state.last_scenenet_ms = elapsed_ms
        
        res = self.state.to_dict()
        res["is_new_frame"] = True
        res["features"] = result.features # Pass raw features for VLM
        return res

    def analyze_frame_full(self, img: Image.Image) -> dict:
        """
        Tier 2: SceneNet + YOLO (~200ms).
        Run every 2-3 seconds for object detection overlay.
        
        Args:
            img: PIL Image
            
        Returns:
            Full analysis with detections
        """
        # First run SceneNet
        result = self.analyze_frame_fast(img)
        
        # Then run YOLO
        t0 = time.perf_counter()
        try:
            from pipeline.object_detector import detect_objects
            detections = detect_objects(img)
            
            self.state.detections = [
                {
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                    "is_structural": d.get("is_structural", False),
                }
                for d in detections
            ]
            self.state.object_count = len(detections)
        except Exception as e:
            print(f"[RealtimeAnalyzer] YOLO error: {e}")
        
        self.state.last_yolo_ms = (time.perf_counter() - t0) * 1000
        
        return self.state.to_dict()

    def _update_state_ema(self, scene_result):
        """Apply exponential moving average to smooth SceneNet predictions."""
        alpha = self.ema_alpha
        
        # For classification, we smooth the probabilities then take argmax
        if scene_result.room_type_probs:
            if not self.state.room_type_probs:
                self.state.room_type_probs = dict(scene_result.room_type_probs)
            else:
                for k in scene_result.room_type_probs:
                    old = self.state.room_type_probs.get(k, 0)
                    self.state.room_type_probs[k] = alpha * scene_result.room_type_probs[k] + (1 - alpha) * old
            
            best_room = max(self.state.room_type_probs, key=self.state.room_type_probs.get)
            self.state.room_type = best_room
            self.state.room_type_confidence = self.state.room_type_probs[best_room]

        if scene_result.style_probs:
            if not self.state.style_probs:
                self.state.style_probs = dict(scene_result.style_probs)
            else:
                for k in scene_result.style_probs:
                    old = self.state.style_probs.get(k, 0)
                    self.state.style_probs[k] = alpha * scene_result.style_probs[k] + (1 - alpha) * old
            
            best_style = max(self.state.style_probs, key=self.state.style_probs.get)
            self.state.style = best_style
            self.state.style_confidence = self.state.style_probs[best_style]

        # Direct updates for simpler fields
        self.state.lighting = scene_result.lighting
        self.state.condition = scene_result.condition
        
        # Palette: smooth RGB values
        new_palette = scene_result.palette
        old_palette = self.state.palette
        smoothed = []
        for i in range(3):
            r = int(alpha * new_palette[i][0] + (1 - alpha) * old_palette[i][0])
            g = int(alpha * new_palette[i][1] + (1 - alpha) * old_palette[i][1])
            b = int(alpha * new_palette[i][2] + (1 - alpha) * old_palette[i][2])
            smoothed.append([r, g, b])
        self.state.palette = smoothed

    def reset(self):
        """Reset for a new live session."""
        self.deduplicator.reset()
        self.state = SceneState()
        # Don't unload models — keep them warm

    def get_dedup_stats(self) -> dict:
        """Get frame deduplication statistics."""
        return self.deduplicator.get_stats()
