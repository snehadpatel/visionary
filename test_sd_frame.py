import cv2
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# Setup paths
PROJECT_ROOT = Path("/Users/snehapatel/visionary")
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.object_detector import detect_objects
from pipeline.depth_estimator import estimate_depth
from pipeline.image_generator import generate_redesigned_image
from pipeline.sd_prompt_builder import build_sd_prompt
from pipeline.budget_engine import allocate_budget

def test_single_frame():
    video_path = "/Users/snehapatel/visionary/mixkit-hotel-room-with-breakfast-served-4019-hd-ready.mp4"
    output_path = "/Users/snehapatel/visionary/outputs/hotel_redesign_hq_test.jpg"
    
    # 1. Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    print("🧠 Analyzing scene...")
    # 2. Analyze
    detections = detect_objects(pil_img)
    depth_map = estimate_depth(pil_img)
    
    # 3. Build Redesign Plan
    target_style = "scandinavian"
    user_prompt = "a luxurious scandinavian hotel suite with light wood and minimal decor"
    
    # Mock budget for prompt building
    budget_plan = allocate_budget(50000, {}, target_style, user_prompt)
    sd_prompt, neg_prompt = build_sd_prompt(target_style, budget_plan, {}, user_prompt)
    
    print(f"🎨 Generating redesign with SD...\nPrompt: {sd_prompt}")
    
    # 4. Generate
    # For the mask, we'll redesign the whole room area where furniture is or just the whole frame for a test
    # Let's create a mask for all detected furniture
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        mask[y1:y2, x1:x2] = 255
    
    # Dilate mask for better inpainting
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    
    # Run SD
    result_pil = generate_redesigned_image(
        pil_img,
        depth_map,
        [{"mask": mask}], # Simple mask list
        sd_prompt,
        neg_prompt
    )
    
    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_pil.save(output_path)
    print(f"✅ Redesign saved to: {output_path}")

if __name__ == "__main__":
    test_single_frame()
