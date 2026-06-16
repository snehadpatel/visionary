import cv2
import os
import sys
from pathlib import Path
from PIL import Image
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

def redesign_key_frames():
    video_path = "/Users/snehapatel/visionary/mixkit-hotel-room-with-breakfast-served-4019-hd-ready.mp4"
    output_dir = "/Users/snehapatel/visionary/outputs/hq_redesign_preview"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Frames to process (0s, 1s, 2s, 3s, 4s)
    key_frame_indices = [0, int(fps), int(fps*2), int(fps*3), int(fps*4)]
    
    target_style = "scandinavian"
    user_prompt = "modern luxury scandinavian hotel room, light wood, white linens, minimalist furniture"
    
    # 1. Pre-build prompts
    budget_plan = allocate_budget(50000, {}, target_style, user_prompt)
    sd_prompt, neg_prompt = build_sd_prompt(target_style, budget_plan, {}, user_prompt)
    
    print(f"🎨 Style: {target_style.upper()}")
    print(f"📝 Prompt: {sd_prompt}")

    for idx in key_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if not ret: break
        
        print(f"\n📸 Processing Frame {idx} (approx {idx/fps:.1f}s)...")
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Fast analysis for each frame
        depth_map = estimate_depth(pil_img)
        detections = detect_objects(pil_img)
        
        # Build furniture mask
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            mask[y1:y2, x1:x2] = 255
        
        # Redesign!
        result_pil = generate_redesigned_image(
            pil_img,
            depth_map,
            [{"mask": mask}],
            sd_prompt,
            neg_prompt
        )
        
        out_path = os.path.join(output_dir, f"frame_{idx:04d}_redesigned.jpg")
        result_pil.save(out_path)
        print(f"✅ Saved: {out_path}")

    cap.release()
    print("\n✨ Keyframe redesign complete!")

if __name__ == "__main__":
    redesign_key_frames()
