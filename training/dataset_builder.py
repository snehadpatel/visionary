"""
Dataset Builder — Creates training data for the Visionary VLM.
Generates image-description pairs from room images for fine-tuning the projection MLP.
"""
import json
import os
from pathlib import Path
from PIL import Image


def build_dataset_from_directory(
    image_dir: str,
    output_json: str,
    default_descriptions: dict = None,
):
    """
    Build a training dataset from a directory of room images.
    Creates a JSON file with image_path and description pairs.
    
    For best results, manually annotate descriptions in the output file.
    """
    if default_descriptions is None:
        default_descriptions = {
            "bedroom": "A bedroom with a bed, nightstand, and warm lighting",
            "living": "A living room with sofa, coffee table, and decorative elements",
            "kitchen": "A kitchen with cabinets, countertop, and appliances",
            "bathroom": "A bathroom with modern fixtures and clean tiles",
            "dining": "A dining area with table and chairs",
        }
    
    entries = []
    image_dir = Path(image_dir)
    
    for img_path in sorted(image_dir.rglob("*.jpg")):
        # Try to infer room type from filename/directory
        name_lower = str(img_path).lower()
        desc = "An interior room with furniture and decor"
        
        for room_type, room_desc in default_descriptions.items():
            if room_type in name_lower:
                desc = room_desc
                break
        
        entries.append({
            "image_path": str(img_path),
            "description": desc,
        })
    
    # Also check PNG files
    for img_path in sorted(image_dir.rglob("*.png")):
        entries.append({
            "image_path": str(img_path),
            "description": "An interior room with furniture and decor",
        })
    
    with open(output_json, "w") as f:
        json.dump(entries, f, indent=2)
    
    print(f"Built dataset with {len(entries)} entries -> {output_json}")
    print("NOTE: Edit the descriptions manually for best training results!")
    return entries


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "data" / "datasets" / "raw"
    output = Path(__file__).parent / "data" / "interior_qa.json"
    
    if image_dir.exists():
        build_dataset_from_directory(str(image_dir), str(output))
    else:
        print(f"Image directory not found: {image_dir}")
        print("Place room images in data/datasets/raw/ first.")
