import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path("/Users/snehapatel/visionary")))

from src.pipeline import VisionaryPipeline

def main():
    test_img = "data/datasets/raw/small_bedroom_design/yW7Yx2CHNSI.jpg"
    
    if not os.path.exists(test_img):
        print(f"Error: Test image not found at {test_img}")
        return

    print("Initializing Visionary Pipeline...")
    pipeline = VisionaryPipeline(visualize=True)
    
    print(f"Processing image: {test_img} with AUTO LAYOUT...")
    # Using 'layout' in user_prompt to trigger FurniturePlacer.suggest_layout_heuristic
    result = pipeline.process_room(
        test_img, 
        target_style="scandinavian", 
        user_prompt="auto layout", 
        generate_image=True
    )
    
    print("\nResult Summary:")
    print(f"- Reconfigured Image: {result['redesign'].get('visualized_image')}")
    print(f"- Number of replacements: {len(result['redesign'].get('replacements', []))}")
    
    if result['redesign'].get('visualized_image') and os.path.exists(result['redesign']['visualized_image']):
        print("\nSUCCESS: Reconfigured redesign image generated successfully.")
    else:
        print("\nFAILURE: Reconfigured redesign image was not generated.")

if __name__ == "__main__":
    main()
