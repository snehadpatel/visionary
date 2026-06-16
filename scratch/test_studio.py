import requests
import base64
import os
import json

URL = "http://localhost:8000/api/redesign"
IMG_PATH = "/Users/snehapatel/visionary/outputs/hotel_redesign_luxury.jpg"

def test_studio():
    files = {
        'image': ('hotel_redesign_luxury.jpg', open(IMG_PATH, 'rb'), 'image/jpeg')
    }
    data = {
        'style': 'luxury',
        'budget_inr': 50000
    }
    
    print("🚀 Triggering Studio Redesign...")
    response = requests.post(URL, files=files, data=data)
    job_id = response.json()["job_id"]
    print(f"✅ Job started: {job_id}")
    
    import time
    while True:
        status = requests.get(f"http://localhost:8000/api/status/{job_id}").json()
        print(f"Step: {status.get('step')} | Status: {status.get('status')}")
        if status["status"] == "done":
            print("--- VLM ANALYSIS ---")
            print(json.dumps(status["result"]["vlm_analysis"], indent=2))
            print(f"--- RESULT URL ---")
            print(status["result"]["result_url"])
            break
        if status["status"] == "failed":
            print("Failed!")
            break
        time.sleep(5)

if __name__ == "__main__":
    test_studio()
