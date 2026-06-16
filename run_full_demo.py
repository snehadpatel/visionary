import os
import cv2
from src.pipeline import VisionaryPipeline
from pathlib import Path

def run_demo():
    # 1. Setup
    project_root = Path("/Users/snehapatel/visionary")
    output_base = project_root / "outputs" / "demo_gallery"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 2. Pick a test image
    # We'll use one from the data/uploads which is likely a "real" room
    uploads_dir = project_root / "data" / "uploads"
    test_images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.webp"))
    if not test_images:
        print("No test images found in data/uploads.")
        return
    
    test_img = str(test_images[0])
    print(f"Selected test image: {test_img}")
    
    # 3. Initialize Pipeline
    pipeline = VisionaryPipeline(visualize=True)
    
    # 4. Styles to test
    styles = ["scandinavian", "industrial", "mid-century modern", "bohemian", "luxury", "rustic", "japandi"]
    
    # 5. Run!
    for style in styles:
        print(f"\n>>> Running Redesign for Style: {style.upper()}...")
        try:
            result = pipeline.process_room(
                test_img, 
                target_style=style, 
                user_prompt=f"a beautiful {style} room", 
                generate_image=True
            )
            
            # Move the generated image to our demo gallery for easy viewing
            viz_path = result['redesign'].get('visualized_image')
            if viz_path and os.path.exists(viz_path):
                dest_path = output_base / f"redesign_{style.replace(' ', '_')}.jpg"
                import shutil
                shutil.copy(viz_path, dest_path)
                print(f"Result saved to: {dest_path}")
            else:
                print(f"Failed to generate image for style: {style}")
        except Exception as e:
            print(f"Error processing style {style}: {e}")

    print(f"\nDemo complete! Results are in {output_base}")

if __name__ == "__main__":
    run_demo()
