import torch
import cv2
import numpy as np
import os
from PIL import Image
from src.pipeline import VisionaryPipeline
from src.redesign.redesign_spec_engine import build_redesign_spec

def generate_synthetic_dataset(input_dir, output_dir, styles=["scandinavian", "industrial", "bohemian"]):
    """
    Uses the existing high-fidelity pipeline to generate paired images for training.
    """
    pipeline = VisionaryPipeline(visualize=False)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    
    img_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Generating synthetic pairs for {len(img_files)} images...")

    for fname in img_files:
        img_path = os.path.join(input_dir, fname)
        image_bgr = cv2.imread(img_path)
        h, w = image_bgr.shape[:2]
        
        # 1. Run Pipeline to get Redesign + Depth + Masks
        # We'll run it once and save the components
        for style in styles:
            spec = build_redesign_spec(style, room_type="bedroom")
            
            # Run the full pipeline
            # Note: We need a way to get the intermediate depth and mask from the pipeline
            # Let's assume the pipeline already saves or returns these
            result_pil = pipeline.process_redesign(image_bgr, style)
            
            if result_pil:
                base_name = os.path.splitext(fname)[0]
                save_name = f"{base_name}_{style}.png"
                
                # Save Original (Input)
                orig_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                orig_pil.save(os.path.join(output_dir, "original", save_name))
                
                # Save Target (Redesign)
                result_pil.save(os.path.join(output_dir, "target", save_name))
                
                # We need to extract depth and mask from the pipeline run
                # For now, let's assume we rerun the inference to save them for the dataset
                depth_map, seg_mask, _ = pipeline.inference_engine.run_inference(image_bgr)
                
                # Save Depth
                depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                depth_pil = Image.fromarray(depth_norm.astype(np.uint8))
                depth_pil.save(os.path.join(output_dir, "depth", save_name))
                
                # Save Mask (Furniture + Floor)
                # In this demo, we'll just save a combined mask of everything redesigned
                mask_pil = Image.fromarray((seg_mask > 0).astype(np.uint8) * 255)
                mask_pil.save(os.path.join(output_dir, "mask", save_name))
                
                print(f"Saved synthetic pair: {save_name}")

if __name__ == "__main__":
    # Example usage
    input_images = "/Users/snehapatel/visionary/data/uploads"
    output_dataset = "/Users/snehapatel/visionary/data/paired_redesign"
    
    if os.path.exists(input_images):
        generate_synthetic_dataset(input_images, output_dataset)
    else:
        print(f"Input directory {input_images} not found.")
