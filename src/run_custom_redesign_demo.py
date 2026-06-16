import os
import cv2
import torch
from src.pipeline import VisionaryPipeline

def run_neural_demo():
    """
    VLM-Free Demo: Validates the custom Pix2Pix UNet redesign pipeline.
    Ensures structural changes without generic Stable Diffusion 'coloring'.
    """
    print("🚀 INITIALIZING VISIONARY CUSTOM NEURAL PIPELINE...")
    
    # 1. Initialize Pipeline (No VLM/SD)
    # This loads FastDepthNet, SegmentationUNet, and RedesignGenerator
    pipeline = VisionaryPipeline(visualize=True)
    
    # 2. Test Images from the dataset
    test_rooms = [
        "/Users/snehapatel/visionary/data/raw/huggingface_rooms/room_0001.jpg",
        "/Users/snehapatel/visionary/data/raw/huggingface_rooms/room_0002.jpg",
        "/Users/snehapatel/visionary/data/raw/huggingface_rooms/room_0003.jpg"
    ]
    
    output_dir = "/Users/snehapatel/visionary/outputs/neural_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    for room_path in test_rooms:
        if not os.path.exists(room_path):
            print(f"Skipping {room_path}, not found.")
            continue
            
        print(f"\n>>> STARTING NEURAL REDESIGN FOR: {os.path.basename(room_path)}")
        try:
            # The pipeline now uses the CustomNeuralGenerator (Pix2Pix)
            # which takes RGB + Depth + Mask as input.
            result = pipeline.process_room(
                room_path, 
                target_style="custom_neural", 
                generate_image=True
            )
            
            out_path = result['redesign'].get('visualized_image')
            print(f"✅ SUCCESS: High-fidelity redesign saved to {out_path}")
            
        except Exception as e:
            print(f"❌ ERROR processing {room_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_neural_demo()
