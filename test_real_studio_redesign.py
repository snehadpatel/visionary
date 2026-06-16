import requests
import time
import os
import base64

# Configuration
BASE_URL = "http://localhost:8000/api"
IMG_PATH = "/Users/snehapatel/visionary/outputs/hotel_redesign_luxury.jpg"

def run_studio_redesign():
    if not os.path.exists(IMG_PATH):
        print(f"❌ Sample image not found: {IMG_PATH}")
        return

    # 1. Start Redesign Job
    print("📸 Sending image to Studio Redesign Pipeline (High Fidelity)...")
    files = {"image": open(IMG_PATH, "rb")}
    data = {
        "prompt": "Modern luxury hotel room with warm wood accents and velvet furniture",
        "budget_inr": "50000",
        "style": "luxury"
    }
    
    response = requests.post(f"{BASE_URL}/redesign", files=files, data=data)
    if response.status_code != 200:
        print(f"❌ Failed to start job: {response.text}")
        return
    
    job_id = response.json()["job_id"]
    print(f"✅ Job Started! ID: {job_id}")
    print("⏳ This takes about 20-30 seconds as it runs Stable Diffusion on your Mac...")

    # 2. Poll for Completion
    start_time = time.time()
    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}")
        status_data = status_resp.json()
        status = status_data["status"]
        step = status_data.get("step", "")
        
        if status == "done":
            print("\n" + "="*40)
            print("✨ REAL REDESIGN COMPLETE! ✨")
            print("="*40)
            
            # The result URL is relative
            result_url = status_data["result"]["result_url"]
            print(f"🖼️  High-Fidelity Output saved to: {result_url}")
            
            # Show Analysis Details
            vlm = status_data["result"]["vlm_analysis"]
            print(f"🔍 VLM Analysis: {vlm.get('current_style', 'Unknown')}")
            print(f"💰 Budget Plan: {len(status_data['result']['budget_plan'])} items matched")
            print("="*40)
            break
        elif status == "error":
            print(f"\n❌ Pipeline Error: {step}")
            break
        else:
            print(f" ⌛ {step}...", end="\r")
            time.sleep(2)
        
        if time.time() - start_time > 120:
            print("\n❌ Timeout waiting for redesign.")
            break

if __name__ == "__main__":
    run_studio_redesign()
