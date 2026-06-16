import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def aggregate_style_inspiration(rooms_jsonl: str, output_path: str):
    """
    Extracts dominant color palettes and common furniture archetypes for each style.
    """
    style_data = defaultdict(lambda: {"colors": [], "rooms": 0})
    
    with open(rooms_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            style = data.get('style', 'unknown')
            if style == 'unknown': continue
            
            style_data[style]["colors"].extend(data.get('dominant_colors', []))
            style_data[style]["rooms"] += 1
            
    # Finalize signatures
    signatures = {}
    for style, data in style_data.items():
        # Get top 5 most frequent hex colors
        all_colors = data["colors"]
        if not all_colors: continue
        
        counts = defaultdict(int)
        for c in all_colors: counts[c] += 1
        top_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        palette = [c for c, _ in top_colors]
        
        signatures[style] = {
            "palette": palette,
            "room_count": data["rooms"],
            "starter_pack": get_starter_pack(style)
        }
        
    with open(output_path, 'w') as f:
        json.dump(signatures, f, indent=4)
    print(f"Style signatures saved to {output_path}")

def get_starter_pack(style: str):
    """
    Returns a set of common furniture types for a style.
    """
    packs = {
        "bohemian": ["pillow", "rug", "pottedplant", "sofa"],
        "scandinavian": ["chair", "table", "lamp", "storage"],
        "minimalist": ["sofa", "table", "lamp"],
        "industrial": ["table", "chair", "storage", "lamp"],
        "modern": ["sofa", "chair", "table", "tvmonitor"]
    }
    return packs.get(style, ["sofa", "table", "chair"])

if __name__ == "__main__":
    aggregate_style_inspiration(
        "/Users/snehapatel/visionary/data/annotations/rooms.jsonl",
        "/Users/snehapatel/visionary/data/annotations/style_signatures.json"
    )
