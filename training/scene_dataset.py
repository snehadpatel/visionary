"""
SceneNet Dataset — Loads room images with multi-task labels for training SceneNet.

Merges data from two sources:
  1. data/datasets/labeled_metadata.jsonl — has style, room_type, dominant_colors, has_natural_light
  2. data/annotations/rooms.jsonl — has room_type, style, dominant_colors

Handles missing labels gracefully with masked loss (label = -1).
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Import label definitions from model
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.scene_net import ROOM_TYPES, STYLES, LIGHTING_LEVELS, CONDITIONS


# ─── Room type normalization map ───
ROOM_TYPE_MAP = {
    "bedroom": "bedroom",
    "living_room": "living_room",
    "living room": "living_room",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "dining_room": "dining_room",
    "dining room": "dining_room",
    "office": "office",
    "home_office": "office",
    "home office": "office",
    # Common variants
    "other": None,        # Will be skipped
    "unknown": None,
}

# ─── Style normalization map ───
STYLE_MAP = {
    "scandinavian": "scandinavian",
    "modern": "modern",
    "industrial": "industrial",
    "bohemian": "bohemian",
    "minimalist": "minimalist",
    "rustic": "rustic",
    "luxury": "luxury",
    "mid_century": "mid_century",
    "mid-century": "mid_century",
    "mid-century modern": "mid_century",
    "mid century modern": "mid_century",
    "contemporary": "contemporary",
    "transitional": "transitional",
    "eclectic": "eclectic",
    "cozy": "cozy",
    "farmhouse": "farmhouse",
    # Common variants that map to closest
    "traditional": "transitional",
    "classic": "transitional",
    "art_deco": "luxury",
    "art deco": "luxury",
    "japandi": "scandinavian",
    "unknown": None,
}


def _hex_to_rgb(hex_str: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def _infer_lighting(entry: dict) -> int:
    """Infer lighting level from available data."""
    # Direct field
    if "has_natural_light" in entry:
        if entry["has_natural_light"] is True:
            return LIGHTING_LEVELS.index("high")
        elif entry["has_natural_light"] is False:
            return LIGHTING_LEVELS.index("low")

    if "natural_light" in entry:
        val = str(entry["natural_light"]).lower()
        if val in LIGHTING_LEVELS:
            return LIGHTING_LEVELS.index(val)

    # Heuristic from dominant colors brightness
    if "dominant_colors" in entry and entry["dominant_colors"]:
        try:
            colors = entry["dominant_colors"]
            avg_brightness = 0
            for c in colors:
                if isinstance(c, str):
                    r, g, b = _hex_to_rgb(c)
                else:
                    r, g, b = c
                avg_brightness += (0.299 * r + 0.587 * g + 0.114 * b)
            avg_brightness /= len(colors)
            
            if avg_brightness > 170:
                return LIGHTING_LEVELS.index("high")
            elif avg_brightness > 100:
                return LIGHTING_LEVELS.index("medium")
            else:
                return LIGHTING_LEVELS.index("low")
        except Exception:
            pass

    return -1  # Unknown — will be masked in loss


def _infer_condition(entry: dict) -> int:
    """Infer room condition heuristically."""
    # If CLIP score is available, use as proxy for image quality
    clip_score = entry.get("clip_score", None)
    if clip_score is not None:
        if clip_score > 0.27:
            return CONDITIONS.index("good")
        elif clip_score > 0.25:
            return CONDITIONS.index("fair")
        else:
            return CONDITIONS.index("needs_work")
    
    return -1  # Unknown


def _parse_palette(entry: dict) -> np.ndarray:
    """Extract palette as normalized float array (9,) from entry."""
    colors = entry.get("dominant_colors", [])
    palette = np.zeros(9, dtype=np.float32)

    for i, c in enumerate(colors[:3]):
        try:
            if isinstance(c, str):
                r, g, b = _hex_to_rgb(c)
            else:
                r, g, b = c
            palette[i*3] = r / 255.0
            palette[i*3 + 1] = g / 255.0
            palette[i*3 + 2] = b / 255.0
        except Exception:
            pass

    return palette


class SceneNetDataset(Dataset):
    """
    PyTorch dataset for SceneNet multi-task training.
    
    Loads images and multi-task labels from JSONL metadata files.
    Missing labels are set to -1 (masked in loss computation).
    
    Args:
        project_root: Path to the visionary project root
        split: "train" or "val"
        image_size: Target image size (square)
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        project_root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = True,
    ):
        self.project_root = Path(project_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        
        self.entries = []
        self._load_metadata()
        
        print(f"[SceneNetDataset] {split}: {len(self.entries)} samples loaded")

    def _load_metadata(self):
        """Load and merge entries from both metadata sources."""
        seen_paths = set()

        # Source 1: labeled_metadata.jsonl (richer labels)
        labeled_path = self.project_root / "data" / "datasets" / "labeled_metadata.jsonl"
        if labeled_path.exists():
            with open(labeled_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    img_path = self.project_root / entry.get("local_path", entry.get("path", ""))
                    if not img_path.exists():
                        continue

                    # Check split
                    entry_split = entry.get("split", "train")
                    if entry_split != self.split:
                        continue

                    parsed = self._parse_entry(entry, str(img_path))
                    if parsed:
                        self.entries.append(parsed)
                        seen_paths.add(str(img_path))

        # Source 2: rooms.jsonl (broader coverage)
        rooms_path = self.project_root / "data" / "annotations" / "rooms.jsonl"
        if rooms_path.exists():
            with open(rooms_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    img_path = self.project_root / entry.get("path", "")
                    if str(img_path) in seen_paths:
                        continue
                    if not img_path.exists():
                        continue

                    entry_split = entry.get("split", "train")
                    if entry_split != self.split:
                        continue

                    parsed = self._parse_entry(entry, str(img_path))
                    if parsed:
                        self.entries.append(parsed)

    def _parse_entry(self, entry: dict, img_path: str) -> Optional[dict]:
        """Parse a metadata entry into training-ready format."""
        # Room type
        raw_room = entry.get("room_type", "unknown").lower().strip()
        room_type_str = ROOM_TYPE_MAP.get(raw_room)
        room_type_idx = ROOM_TYPES.index(room_type_str) if room_type_str and room_type_str in ROOM_TYPES else -1

        # Style
        raw_style = entry.get("style", "unknown").lower().strip()
        style_str = STYLE_MAP.get(raw_style)
        style_idx = STYLES.index(style_str) if style_str and style_str in STYLES else -1

        # Skip entries with no useful labels
        if room_type_idx == -1 and style_idx == -1:
            return None

        # Lighting
        lighting_idx = _infer_lighting(entry)

        # Condition
        condition_idx = _infer_condition(entry)

        # Palette
        palette = _parse_palette(entry)

        return {
            "image_path": img_path,
            "room_type": room_type_idx,
            "style": style_idx,
            "lighting": lighting_idx,
            "palette": palette,
            "condition": condition_idx,
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load image
        img = cv2.imread(entry["image_path"])
        if img is None:
            # Fallback: return a random other sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        # Augmentation
        if self.augment:
            img = self._augment(img)

        # Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, 224, 224)

        return {
            "image": tensor,
            "room_type": torch.tensor(entry["room_type"], dtype=torch.long),
            "style": torch.tensor(entry["style"], dtype=torch.long),
            "lighting": torch.tensor(entry["lighting"], dtype=torch.long),
            "palette": torch.from_numpy(entry["palette"]),
            "condition": torch.tensor(entry["condition"], dtype=torch.long),
        }

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        h, w = img.shape[:2]

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()

        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            alpha = 0.8 + random.random() * 0.4  # contrast [0.8, 1.2]
            beta = random.randint(-20, 20)         # brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Random color jitter
        if random.random() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + random.random() * 0.4), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random small rotation (-5 to 5 degrees)
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        return img


if __name__ == "__main__":
    # Test dataset loading
    project_root = "/Users/snehapatel/visionary"

    train_ds = SceneNetDataset(project_root, split="train", augment=True)
    val_ds = SceneNetDataset(project_root, split="val", augment=False)

    print(f"\nTrain: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples")

    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"\nSample:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")

        # Label distribution
        from collections import Counter
        room_counts = Counter()
        style_counts = Counter()
        for e in train_ds.entries:
            if e["room_type"] >= 0:
                room_counts[ROOM_TYPES[e["room_type"]]] += 1
            if e["style"] >= 0:
                style_counts[STYLES[e["style"]]] += 1

        print(f"\nRoom type distribution:")
        for k, v in room_counts.most_common():
            print(f"  {k}: {v}")

        print(f"\nStyle distribution:")
        for k, v in style_counts.most_common():
            print(f"  {k}: {v}")
