"""
Visionary Live — Robust HTTP-based Live Pipeline.
Alternative to WebSockets for flaky networks.
"""
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import base64
import asyncio
import time
import numpy as np
import cv2

from pipeline.realtime_analyzer import RealtimeAnalyzer
from pipeline import image_generator
from pipeline.depth_estimator import estimate_depth
from pipeline.object_detector import detect_objects

router = APIRouter(prefix="/api/live", tags=["live"])

# Lazy-load analyzer
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = RealtimeAnalyzer()
    return _analyzer

@router.post("/frame")
async def process_live_frame(
    image_b64: str = Form(...),
    style: str = Form("auto"),
    include_redesign: bool = Form(False)
):
    """
    Robust HTTP endpoint for live camera frames.
    """
    t_start = time.time()
    try:
        # 1. Decode Image
        print(f"[Live-HTTP] Processing frame... (Redesign: {include_redesign})")
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else (None, image_b64)
        image_data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"[Live-HTTP] Image decoded: {img.size}")
        
        analyzer = get_analyzer()
        loop = asyncio.get_event_loop()
        
        # 2. Fast Analysis
        print("[Live-HTTP] Running SceneNet...")
        scene_result = await loop.run_in_executor(None, analyzer.analyze_frame_fast, img)
        print("[Live-HTTP] Running YOLO...")
        detections = await loop.run_in_executor(None, detect_objects, img)
        
        # Determine target style (use predicted style if 'auto')
        target_style = style if style != "auto" else scene_result.get("style", "scandinavian")
        
        response_data = {
            "detections": detections,
            "scene_state": scene_result, 
            "is_new_frame": scene_result.get("is_new_frame", True),
            "redesign_frame": None,
            "processing_time": 0
        }
        
        # 3. Optional Redesign
        if include_redesign:
            print(f"[Live-HTTP] Running Redesign (Style: {target_style})...")
            depth_map = await loop.run_in_executor(None, estimate_depth, img)
            
            print("[Live-HTTP] Building Mask...")
            h, w = img.size[1], img.size[0]
            mask = np.zeros((h, w), dtype=np.uint8)
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
            
            print("[Live-HTTP] Running Custom Generator...")
            image_generator._load_custom_gen()
            def run_gen():
                return image_generator._custom_gen.generate_redesign(
                    original_img=img,
                    depth_map=depth_map,
                    masks=[mask]
                )
            result_pil = await loop.run_in_executor(None, run_gen)
            
            print("[Live-HTTP] Encoding Result...")
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG", quality=70)
            res_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            response_data["redesign_frame"] = res_b64

        response_data["processing_time"] = round(time.time() - t_start, 3)
        print(f"[Live-HTTP] Success! Total time: {response_data['processing_time']}s")
        return response_data

    except Exception as e:
        import traceback
        print(f"[Live-HTTP Error] {e}")
        traceback.print_exc()
        raise e
