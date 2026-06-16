import time
import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path("/Users/snehapatel/visionary")))

from src.pipeline import VisionaryPipeline

def benchmark():
    test_img = "data/datasets/raw/small_bedroom_design/yW7Yx2CHNSI.jpg"
    if not os.path.exists(test_img):
        print("Test image not found.")
        return

    print("--- Visionary Pipeline Benchmark ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Warm up / Load
    start_load = time.time()
    pipeline = VisionaryPipeline(visualize=True, device=device)
    print(f"Model Load Time: {time.time() - start_load:.3f}s")

    # Benchmark Loop
    iterations = 3
    timings = []

    for i in range(iterations):
        print(f"\nIteration {i+1}...")
        
        # We'll manually time the steps inside a modified version of process_room or just track internally
        # Since I can't easily modify the class for a one-off test without re-editing,
        # I'll just time the whole call and separate it.
        
        t0 = time.time()
        result = pipeline.process_room(test_img, user_prompt="auto layout", generate_image=True)
        total_time = time.time() - t0
        timings.append(total_time)
        print(f"  Total Latency: {total_time:.3f}s")

    avg_time = sum(timings) / iterations
    print(f"\nAverage Latency: {avg_time:.3f}s")
    
    if avg_time < 1.0:
        print("✅ Goal ACHIEVED: < 1 second latency!")
    else:
        print(f"❌ Goal NOT YET ACHIEVED: {avg_time - 1.0:.3f}s over budget.")

if __name__ == "__main__":
    benchmark()
