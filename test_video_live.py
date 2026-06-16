import cv2
import time
import os
import sys
from pathlib import Path
from PIL import Image

# Add backend to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from pipeline.realtime_analyzer import RealtimeAnalyzer

def test_video(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video {video_path} not found.")
        return

    print(f"🎬 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Resolution: {width}x{height} @ {fps}fps | Total frames: {total_frames}")

    # Setup VideoWriter (avc1 = H.264, natively supported on Mac/IDE)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize our RealtimeAnalyzer
    print("🧠 Initializing RealtimeAnalyzer (SceneNet + YOLO)...")
    analyzer = RealtimeAnalyzer(ema_alpha=0.3)
    
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV uses BGR, convert to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # ─── RUN ANALYSIS ───
        # Simulate our live mode: Full analysis (YOLO+SceneNet) every 15 frames, Fast (SceneNet only) on others
        if frame_idx % 15 == 0:
            state = analyzer.analyze_frame_full(pil_img)
        else:
            state = analyzer.analyze_frame_fast(pil_img)
            
        # ─── DRAW OVERLAYS ───
        
        # 1. Draw Bounding Boxes (from YOLO)
        for det in state.get('detections', []):
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['label']} {det['confidence']:.2f}"
            
            # Draw semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 100), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (255, 100, 100), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
        # 2. Draw SceneNet Info Panel (Top Left)
        overlay_text = [
            f"Room: {state['room_type'].replace('_', ' ').title()} ({state['room_type_confidence']:.1%})",
            f"Style: {state['style'].replace('_', ' ').title()} ({state['style_confidence']:.1%})",
            f"Lighting: {state['lighting'].title()}",
            f"Palette: RGB{state['palette'][0]}",
            "",
            f"SceneNet latency: {state['last_scenenet_ms']}ms",
            f"YOLO latency: {state['last_yolo_ms']}ms",
            f"Frames skipped: {analyzer.get_dedup_stats()['seen'] - analyzer.get_dedup_stats()['processed']}/{analyzer.get_dedup_stats()['seen']}"
        ]
        
        y_offset = 40
        for text in overlay_text:
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # Black background for text
                cv2.rectangle(frame, (20, y_offset - th - 10), (20 + tw + 20, y_offset + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            
        # Write the processed frame to the output video
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processing... {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Done! Saved output to: {output_path}")
    print(f"⏱️  Took {elapsed:.1f}s ({total_frames/elapsed:.1f} fps processing speed)")

if __name__ == "__main__":
    video_in = "/Users/snehapatel/visionary/mixkit-hotel-room-with-breakfast-served-4019-hd-ready.mp4"
    video_out = "/Users/snehapatel/visionary/outputs/live_mode_demo.mp4"
    
    os.makedirs("/Users/snehapatel/visionary/outputs", exist_ok=True)
    test_video(video_in, video_out)
