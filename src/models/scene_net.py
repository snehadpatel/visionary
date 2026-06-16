"""
SceneNet — Custom Multi-Task Scene Understanding Model for Visionary Live.

A lightweight EfficientNet-B0 backbone with 5 task-specific heads that
classifies room scenes in real-time (~15ms on Apple MPS):

  1. Room Type   (bedroom, living_room, kitchen, bathroom, dining_room, office)
  2. Style       (13 interior design styles)
  3. Lighting    (low, medium, high)
  4. Palette     (3 dominant RGB colors — regression)
  5. Condition   (good, fair, needs_work)

Designed for real-time camera feed analysis in Visionary Live mode.
130x faster than VLM inference (15ms vs ~2000ms).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─── Label Definitions ───

ROOM_TYPES = ["bedroom", "living_room", "kitchen", "bathroom", "dining_room", "office"]
STYLES = [
    "scandinavian", "modern", "industrial", "bohemian", "minimalist",
    "rustic", "luxury", "mid_century", "contemporary", "transitional",
    "eclectic", "cozy", "farmhouse",
]
LIGHTING_LEVELS = ["low", "medium", "high"]
CONDITIONS = ["good", "fair", "needs_work"]


@dataclass
class SceneNetResult:
    """Structured output from SceneNet inference."""
    room_type: str
    room_type_confidence: float
    style: str
    style_confidence: float
    lighting: str
    lighting_confidence: float
    palette: List[Tuple[int, int, int]]   # 3 dominant RGB colors
    condition: str
    condition_confidence: float
    features: Optional[torch.Tensor] = None  # Added for VLM visual prefix

    # Full probability distributions for UI confidence bars
    room_type_probs: dict = field(default_factory=dict)
    style_probs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "room_type": self.room_type,
            "room_type_confidence": round(self.room_type_confidence, 3),
            "style": self.style,
            "style_confidence": round(self.style_confidence, 3),
            "lighting": self.lighting,
            "lighting_confidence": round(self.lighting_confidence, 3),
            "palette": [list(c) for c in self.palette],
            "condition": self.condition,
            "condition_confidence": round(self.condition_confidence, 3),
            "room_type_probs": {k: round(v, 3) for k, v in self.room_type_probs.items()},
            "style_probs": {k: round(v, 3) for k, v in self.style_probs.items()},
        }


# ─── Task Head ───

class TaskHead(nn.Module):
    """Small MLP head for a single classification/regression task."""
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ─── SceneNet ───

class SceneNet(nn.Module):
    """
    Multi-task scene understanding CNN.
    
    Architecture:
        EfficientNet-B0 (pretrained) → Global Average Pool → 1280-dim feature
        → 5 parallel task heads
    
    Inference: ~15ms on Apple MPS at 224×224 input.
    Model size: ~16MB.
    """

    def __init__(self, pretrained: bool = True):
        super(SceneNet, self).__init__()

        # ── Shared Backbone ──
        # Use torchvision's EfficientNet-B0 as feature extractor
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(weights=None)

        # Remove the classifier head — we'll add our own
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # EfficientNet-B0 outputs 1280-dim features
        feat_dim = 1280

        # ── Task Heads ──
        self.room_type_head = TaskHead(feat_dim, len(ROOM_TYPES))
        self.style_head = TaskHead(feat_dim, len(STYLES))
        self.lighting_head = TaskHead(feat_dim, len(LIGHTING_LEVELS))
        self.palette_head = TaskHead(feat_dim, 9)  # 3 colors × 3 channels (RGB)
        self.condition_head = TaskHead(feat_dim, len(CONDITIONS))

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: (B, 3, 224, 224) normalized image tensor
            
        Returns:
            dict with raw logits/predictions for each task
        """
        # Shared backbone
        features = self.features(x)                 # (B, 1280, 7, 7)
        features = self.pool(features)              # (B, 1280, 1, 1)
        features = features.flatten(1)              # (B, 1280)

        return {
            "features": features,                           # (B, 1280)
            "room_type": self.room_type_head(features),     # (B, 6)
            "style": self.style_head(features),             # (B, 13)
            "lighting": self.lighting_head(features),       # (B, 3)
            "palette": torch.sigmoid(self.palette_head(features)),  # (B, 9)
            "condition": self.condition_head(features),     # (B, 3)
        }

    def infer(self, image: np.ndarray, device: str = "mps") -> SceneNetResult:
        """
        Single-image inference with full preprocessing.
        
        Args:
            image: BGR uint8 numpy array (any size) or RGB PIL Image
            device: torch device string
            
        Returns:
            SceneNetResult with all predictions and confidence scores
        """
        # Handle PIL Image input
        if hasattr(image, 'convert'):
            image = np.array(image.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = self.forward(tensor)

        # ── Decode outputs ──

        # Room type
        room_probs = F.softmax(outputs["room_type"][0], dim=0).cpu().numpy()
        room_idx = int(np.argmax(room_probs))
        room_type_probs = {ROOM_TYPES[i]: float(room_probs[i]) for i in range(len(ROOM_TYPES))}

        # Style
        style_probs = F.softmax(outputs["style"][0], dim=0).cpu().numpy()
        style_idx = int(np.argmax(style_probs))
        style_probs_dict = {STYLES[i]: float(style_probs[i]) for i in range(len(STYLES))}

        # Lighting
        light_probs = F.softmax(outputs["lighting"][0], dim=0).cpu().numpy()
        light_idx = int(np.argmax(light_probs))

        # Palette — 9 values → 3 RGB tuples
        palette_raw = outputs["palette"][0].cpu().numpy()  # (9,) in [0, 1]
        palette = [
            (int(palette_raw[i*3] * 255), int(palette_raw[i*3+1] * 255), int(palette_raw[i*3+2] * 255))
            for i in range(3)
        ]

        # Condition
        cond_probs = F.softmax(outputs["condition"][0], dim=0).cpu().numpy()
        cond_idx = int(np.argmax(cond_probs))

        return SceneNetResult(
            room_type=ROOM_TYPES[room_idx],
            room_type_confidence=float(room_probs[room_idx]),
            style=STYLES[style_idx],
            style_confidence=float(style_probs[style_idx]),
            lighting=LIGHTING_LEVELS[light_idx],
            lighting_confidence=float(light_probs[light_idx]),
            palette=palette,
            condition=CONDITIONS[cond_idx],
            condition_confidence=float(cond_probs[cond_idx]),
            room_type_probs=room_type_probs,
            style_probs=style_probs_dict,
            features=outputs["features"], # Pass raw backbone features for VLM
        )


# ─── Multi-Task Loss ───

class SceneNetLoss(nn.Module):
    """
    Weighted multi-task loss for SceneNet training.
    
    - Classification heads: CrossEntropy with optional class weights
    - Palette head: MSE (regression)
    - Supports masked labels (label = -1 means ignore)
    - Inverse-frequency class weighting for imbalanced datasets
    """

    def __init__(
        self,
        weight_room: float = 1.0,
        weight_style: float = 1.5,     # Style is hardest, give it more weight
        weight_light: float = 0.8,
        weight_palette: float = 0.5,
        weight_condition: float = 0.8,
        style_class_weights: Optional[torch.Tensor] = None,
        room_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weights = {
            "room_type": weight_room,
            "style": weight_style,
            "lighting": weight_light,
            "palette": weight_palette,
            "condition": weight_condition,
        }
        # Per-head CE losses with optional class weighting
        self.ce_losses = nn.ModuleDict({
            "room_type": nn.CrossEntropyLoss(weight=room_class_weights, ignore_index=-1),
            "style": nn.CrossEntropyLoss(weight=style_class_weights, ignore_index=-1),
            "lighting": nn.CrossEntropyLoss(ignore_index=-1),
            "condition": nn.CrossEntropyLoss(ignore_index=-1),
        })
        self.mse = nn.MSELoss()

    def forward(self, predictions: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute weighted multi-task loss.
        
        Returns:
            total_loss: weighted sum
            losses_dict: individual losses for logging
        """
        losses = {}

        # Classification losses (per-head, with class weights)
        for key in ["room_type", "style", "lighting", "condition"]:
            if key in targets and targets[key] is not None:
                mask = targets[key] != -1
                if mask.any():
                    losses[key] = self.ce_losses[key](predictions[key][mask], targets[key][mask])
                else:
                    losses[key] = torch.tensor(0.0, device=predictions[key].device)
            else:
                losses[key] = torch.tensor(0.0, device=predictions[key].device)

        # Palette regression loss
        if "palette" in targets and targets["palette"] is not None:
            # Only compute on valid palette entries (non-zero)
            valid_mask = targets["palette"].sum(dim=1) > 0
            if valid_mask.any():
                losses["palette"] = self.mse(
                    predictions["palette"][valid_mask],
                    targets["palette"][valid_mask]
                )
            else:
                losses["palette"] = torch.tensor(0.0, device=predictions["palette"].device)
        else:
            losses["palette"] = torch.tensor(0.0, device=predictions["palette"].device)

        # Weighted sum
        total = sum(self.weights[k] * losses[k] for k in losses)

        return total, {k: v.item() for k, v in losses.items()}


def compute_class_weights(dataset, key: str, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a dataset.
    
    Classes with fewer samples get higher weight, forcing the model
    to pay equal attention to minority classes like 'bohemian' (39 samples)
    vs 'modern' (3617 samples).
    
    Uses sqrt-smoothed inverse frequency to avoid extreme weights.
    """
    from collections import Counter
    counts = Counter()
    for entry in dataset.entries:
        label = entry[key]
        if label >= 0:
            counts[label] += 1
    
    total = sum(counts.values())
    weights = torch.ones(num_classes)
    for cls_idx, count in counts.items():
        if count > 0:
            # sqrt-smoothed inverse frequency: prevents extreme weights
            weights[cls_idx] = np.sqrt(total / (num_classes * count))
    
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return weights


if __name__ == "__main__":
    # Smoke test
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SceneNet(pretrained=False).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Forward pass
    dummy = torch.randn(2, 3, 224, 224).to(device)
    outputs = model(dummy)
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")

    # Inference test
    import time
    fake_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    _ = model.infer(fake_img, device=str(device))
    
    # Benchmark
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        result = model.infer(fake_img, device=str(device))
        times.append((time.perf_counter() - t0) * 1000)
    
    print(f"\nInference: {np.mean(times):.1f}ms avg (±{np.std(times):.1f}ms)")
    print(f"Result: {result.room_type} ({result.room_type_confidence:.1%}), "
          f"style={result.style}, light={result.lighting}")
    print(f"Device: {device}")
