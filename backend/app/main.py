from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
from src.pipeline import VisionaryPipeline
from pathlib import Path

app = FastAPI(title="Visionary - Custom CV Room Redesign")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path("/Users/snehapatel/visionary")
UPLOAD_DIR = PROJECT_ROOT / "data/uploads"
REDESIGNED_DIR = UPLOAD_DIR / "redesigned"
os.makedirs(REDESIGNED_DIR, exist_ok=True)

# In-memory job store
jobs = {}

# Lazy load pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        # Initialize with visualization enabled
        pipeline = VisionaryPipeline(visualize=True)
    return pipeline

async def run_redesign_task(job_id: str, img_path: Path, style: str, prompt: str):
    try:
        jobs[job_id]["status"] = "processing"
        pipe = get_pipeline()
        
        # Run full pipeline with image generation
        result = pipe.process_room(str(img_path), style, user_prompt=prompt, generate_image=True)
        
        # Update job data
        redesign_data = result['redesign']
        visualized_path = redesign_data.get('visualized_image')
        visualized_url = None
        if visualized_path:
            visualized_url = f"/uploads/redesigned/{Path(visualized_path).name}"

        jobs[job_id].update({
            "status": "done",
            "result": {
                "id": job_id,
                "style": style,
                "replacements": redesign_data['replacements'],
                "inspiration": {
                    "palette": redesign_data['recommended_palette'],
                    "floor_color": redesign_data['floor_color'],
                    "wall_color": redesign_data['wall_color']
                },
                "image_url": f"/uploads/{img_path.name}",
                "visualized_url": visualized_url
            }
        })
    except Exception as e:
        print(f"ERROR processing job {job_id}: {e}")
        jobs[job_id].update({
            "status": "error",
            "error": str(e)
        })

@app.post("/redesign")
async def redesign(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form("scandinavian"),
    prompt: str = Form("")
):
    job_id = str(uuid.uuid4())
    img_ext = file.filename.split(".")[-1]
    img_name = f"{job_id}.{img_ext}"
    img_path = UPLOAD_DIR / img_name
    
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = {"status": "queued", "result": None}
    
    # Offload to background
    background_tasks.add_task(run_redesign_task, job_id, img_path, style, prompt)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}
    return jobs[job_id]

@app.get("/health")
def health():
    return {"status": "ok"}

# Serve uploads and redesigned images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 to allow connections from mobile devices on the same network
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
