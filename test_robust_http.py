import requests
import base64
import os
import json

# Configuration
URL = "http://localhost:8000/api/live/frame"
IMG_PATH = "/Users/snehapatel/visionary/outputs/hotel_redesign_luxury.jpg"

def test_robust_http():
    if not os.path.exists(IMG_PATH):
        print(f"❌ Sample image not found: {IMG_PATH}")
        return

    # 1. Encode Image
    print(f"📸 Loading sample image: {os.path.basename(IMG_PATH)}")
    with open(IMG_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 2. Prepare Payload
    data = {
        "image_b64": f"data:image/jpeg;base64,{img_b64}",
        "style": "scandinavian",
        "include_redesign": "true"
    }

    # 3. Send Request
    print(f"🚀 Sending frame to Robust HTTP Pipeline...")
    try:
        t0 = os.times().elapsed
        response = requests.post(URL, data=data)
        t1 = os.times().elapsed
        
        if response.status_code == 200:
            res = response.json()
            print("\n" + "="*40)
            print("✅ TEST SUCCESSFUL — STABLE RESPONSE")
            print("="*40)
            print(f"🔍 Detections: {len(res['detections'])} objects identified")
            print(f"🎨 Style Analysis: {res['scene_state']['style'].upper()}")
            print(f"🛋️ Room Type: {res['scene_state']['room_type'].upper()}")
            
            if res['redesign_frame']:
                print(f"✨ Magic Vision: Redesigned frame received ({len(res['redesign_frame'])} bytes)")
                output_path = "/Users/snehapatel/visionary/outputs/test_http_redesign.jpg"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(res['redesign_frame']))
                print(f"🖼️  Output saved to: {output_path}")
            
            print(f"⏱️ Backend Latency: {res['processing_time']}s")
            print(f"🌐 Total Round Trip: {round(t1 - t0, 3)}s")
            print("="*40)
            print("\nConnection remained 100% stable (No WebSockets used!)")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            print("Ensure the backend server is running on port 8000.")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Please run 'python start_live_system.py' first.")

if __name__ == "__main__":
    test_robust_http()
