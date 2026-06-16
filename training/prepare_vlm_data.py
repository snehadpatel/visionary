import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "datasets" / "labeled_metadata.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "training" / "data" / "interior_qa.json"

def prepare():
    if not METADATA_PATH.exists():
        print(f"Metadata not found at {METADATA_PATH}")
        return

    qa_data = []
    with open(METADATA_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except:
                continue
            
            # Map paths correctly
            img_path = str(PROJECT_ROOT / entry.get("local_path", entry.get("path", "")))
            if not os.path.exists(img_path):
                continue
            
            room = entry.get("room_type", "room")
            style = entry.get("style", "contemporary")
            light = "natural light" if entry.get("has_natural_light") else "ambient light"
            
            # Construct a descriptive sentence
            description = f"A {style} {room} with {light}. The room features a clean layout and professional interior design."
            
            qa_data.append({
                "image_path": img_path,
                "description": description
            })

    print(f"Prepared {len(qa_data)} samples for VLM training.")
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(qa_data, f, indent=2)

if __name__ == "__main__":
    prepare()
