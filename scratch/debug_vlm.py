import sys
import os
from PIL import Image
import torch

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from vlm.visionary_vlm import analyse_room

IMG_PATH = "outputs/hotel_redesign_luxury.jpg"
if not os.path.exists(IMG_PATH):
    print("Image not found")
    sys.exit(1)

img = Image.open(IMG_PATH).convert("RGB")
print(f"Analyzing {IMG_PATH}...")
analysis = analyse_room(img)
import json
print(json.dumps(analysis, indent=2))
