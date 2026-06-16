import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Keep MPS fallback enabled for Apple Silicon stability.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.object_detector import detect_objects
from pipeline.depth_estimator import estimate_depth
from src.redesign.custom_generator import CustomNeuralGenerator


def _build_redesign_masks(frame_shape, sam_segments, detections, full_room_threshold=0.22):
    """
    Convert SAM segments to mask list for the custom generator.
    If furniture coverage is too small, fall back to a whole-room mask
    so we redesign the entire scene look and feel.
    """
    h, w = frame_shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # Prefer SAM masks when available.
    for seg in sam_segments or []:
        m = np.array(seg.get("mask", []), dtype=bool)
        if m.size == 0:
            continue
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        combined[m] = 255

    # Fallback to detection bboxes (fast path when SAM weights are unavailable).
    if not combined.any():
        for det in detections or []:
            x1, y1, x2, y2 = det["bbox"]
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            if x2 > x1 and y2 > y1:
                combined[y1:y2, x1:x2] = 255

    # Expand masked region for smoother whole-furniture transformations.
    if combined.any():
        combined = cv2.dilate(combined, np.ones((21, 21), np.uint8), iterations=1)
        combined = cv2.GaussianBlur(combined, (0, 0), sigmaX=3.0, sigmaY=3.0)
        combined = (combined > 20).astype(np.uint8) * 255

    coverage = float((combined > 0).mean())
    if coverage < full_room_threshold:
        # Force full room redesign when detections are sparse.
        combined = np.full((h, w), 255, dtype=np.uint8)

    return [combined.astype(np.uint8)]


def run_video_redesign(
    video_path: str,
    output_path: str,
    frame_skip: int = 1,
    temporal_alpha: float = 0.18,
    process_width: int = 640,
    depth_interval: int = 3,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Using mp4v for better OpenCV compatibility on various systems.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    print("Initializing redesign models (YOLO + MiDaS + Custom Generator)...")
    generator = CustomNeuralGenerator(
        weights_path=str(PROJECT_ROOT / "models/redesign_generator.pth")
    )
    sam_checkpoint = PROJECT_ROOT / "models/sam_vit_h_4b8939.pth"
    sam_available = sam_checkpoint.exists()
    if not sam_available:
        print("SAM weights not found. Using detection-based masks + full-room fallback.")

    print("🚀 WARMING UP MODELS...")
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    detect_objects(dummy)
    estimate_depth(dummy)
    print("✅ Warmup complete.")

    frame_idx = 0
    processed = 0
    prev_redesign = None
    prev_depth_map = None
    t0 = time.time()
    print(f"Processing {total_frames} frames @ {width}x{height}, fps={fps:.2f}")

    try:
        while processed < 150:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Optional skipping for speed (write previous redesigned frame when skipped).
            if frame_skip > 1 and frame_idx % frame_skip != 0 and prev_redesign is not None:
                writer.write(prev_redesign)
                frame_idx += 1
                continue

            print(f"DEBUG: Processing frame {frame_idx}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run heavy CV analysis on a smaller frame for speed.
            scale = min(1.0, float(process_width) / float(width)) if process_width > 0 else 1.0
            if scale < 1.0:
                proc_w = int(width * scale)
                proc_h = int(height * scale)
                small_rgb = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                small_rgb = frame_rgb
                proc_h, proc_w = height, width

            pil_small = Image.fromarray(small_rgb)
            print("DEBUG: Running YOLO...")
            detections_small = detect_objects(pil_small)
            print(f"DEBUG: YOLO found {len(detections_small)} objects")

            # Scale detections to original frame.
            if scale < 1.0:
                detections = []
                inv = 1.0 / scale
                for det in detections_small:
                    x1, y1, x2, y2 = det["bbox"]
                    det2 = dict(det)
                    det2["bbox"] = [
                        int(np.clip(round(x1 * inv), 0, width - 1)),
                        int(np.clip(round(y1 * inv), 0, height - 1)),
                        int(np.clip(round(x2 * inv), 0, width - 1)),
                        int(np.clip(round(y2 * inv), 0, height - 1)),
                    ]
                    detections.append(det2)
            else:
                detections = detections_small

            sam_segments = []
            if sam_available:
                from pipeline.segmenter import segment_objects
                # Use original resolution for precise masks when SAM is available.
                pil_full = Image.fromarray(frame_rgb)
                sam_segments = segment_objects(pil_full, detections)

            if prev_depth_map is None or processed % max(1, depth_interval) == 0:
                print("DEBUG: Estimating Depth...")
                depth_small = estimate_depth(pil_small)
                depth_map = cv2.resize(depth_small, (width, height), interpolation=cv2.INTER_LINEAR)
                prev_depth_map = depth_map
            else:
                depth_map = prev_depth_map

            redesign_masks = _build_redesign_masks(frame_bgr.shape, sam_segments, detections)

            print("DEBUG: Generating Redesign...")
            redesigned_pil = generator.generate_redesign(
                original_img=Image.fromarray(frame_rgb),
                depth_map=depth_map,
                masks=redesign_masks,
            )
            print("DEBUG: Redesign DONE")
            redesigned_bgr = cv2.cvtColor(np.array(redesigned_pil), cv2.COLOR_RGB2BGR)

            # Temporal smoothing to reduce flicker.
            if prev_redesign is not None:
                redesigned_bgr = cv2.addWeighted(
                    redesigned_bgr, 1.0 - temporal_alpha, prev_redesign, temporal_alpha, 0
                )

            writer.write(redesigned_bgr)
            prev_redesign = redesigned_bgr
            frame_idx += 1
            processed += 1

            if processed % 10 == 0:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                print(f"Processed {processed}/{total_frames} redesign frames ({rate:.2f} fps)")

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    print(f"\nDone. Redesigned video saved to:\n{output_path}")
    print(f"Total time: {elapsed:.1f}s | Effective redesign fps: {processed / max(elapsed, 1e-6):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-room video redesign runner")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "mixkit-hotel-room-with-breakfast-served-4019-hd-ready.mp4"),
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/hotel_room_redesigned.mp4"),
        help="Output video path",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (1 = all frames)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Internal analysis width for YOLO/MiDaS speedup",
    )
    parser.add_argument(
        "--depth-interval",
        type=int,
        default=3,
        help="Recompute depth every N processed frames",
    )
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.18,
        help="Temporal smoothing factor [0..1], higher = smoother but more ghosting",
    )
    args = parser.parse_args()

    run_video_redesign(
        video_path=args.input,
        output_path=args.output,
        frame_skip=max(1, args.frame_skip),
        temporal_alpha=float(np.clip(args.temporal_alpha, 0.0, 0.95)),
        process_width=max(0, args.process_width),
        depth_interval=max(1, args.depth_interval),
    )
