"""
Visionary API — Redesign Router.
Handles room redesign pipeline, status polling, and chat refinement.
"""
import uuid
import traceback
import gc
import os
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image

from pipeline.preprocessor import preprocess_image
from pipeline.object_detector import detect_objects
from pipeline.segmenter import segment_objects
from pipeline.depth_estimator import estimate_depth
from pipeline.scene_graph import build_scene_graph
from pipeline.budget_engine import allocate_budget
from pipeline.sd_prompt_builder import build_sd_prompt
from pipeline.image_generator import generate_redesigned_image
from products.matcher import find_products_for_plan
from vlm.visionary_vlm import analyse_room, chat_with_room
from utils.storage import save_image
from utils.pointcloud_utils import export_pointcloud_to_json
from src.geometry.backprojector import estimate_intrinsics, depth_to_pointcloud_np, clean_pointcloud_np

router = APIRouter(prefix="/api", tags=["redesign"])

# In-memory job store
jobs = {}
conversations = {}  # job_id -> conversation history


def prune_old_jobs():
    """Remove jobs and conversations older than 1 hour to free RAM (PIL Image objects)."""
    import time
    now = time.time()
    expired = [jid for jid, job in jobs.items() if now - job.get("created_at", 0) > 3600]
    for jid in expired:
        jobs.pop(jid, None)
        conversations.pop(jid, None)
    if expired:
        import gc
        gc.collect()


@router.post("/redesign")
async def redesign_room(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(""),
    budget_inr: float = Form(25000),
    style: str = Form("auto"),
):
    """
    Start a new room redesign job.
    Returns a job_id for polling status.
    """
    import time
    prune_old_jobs()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "step": "", "result": None, "created_at": time.time()}
    conversations[job_id] = []
    image_bytes = await image.read()
    background_tasks.add_task(
        run_full_pipeline, job_id, image_bytes, prompt, budget_inr, style
    )
    return {"job_id": job_id}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll the status of a redesign job."""
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    
    # Don't send PIL images over JSON
    result = job.get("result")
    if result and "original_img" in result:
        safe_result = {k: v for k, v in result.items() if k != "original_img"}
        return {"status": job["status"], "step": job.get("step", ""), "result": safe_result}
    
    return job


@router.post("/chat/{job_id}")
async def refine_design(job_id: str, message: str = Form(...)):
    """User can chat to refine - 'remove plants', 'make it darker' etc."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse({"error": "Job not ready"}, status_code=400)

    history = conversations.get(job_id, [])
    original_img = job["result"].get("original_img")
    
    if original_img is None:
        return JSONResponse({"error": "Original image not available"}, status_code=400)
    
    response = chat_with_room(original_img, history, message)

    conversations[job_id].append({"role": "user", "content": message})
    conversations[job_id].append({"role": "assistant", "content": response})

    return {"response": response}


def run_full_pipeline(
    job_id: str,
    image_bytes: bytes,
    prompt: str,
    budget_inr: float,
    style: str,
):
    """
    Execute the complete Visionary pipeline:
    1. Preprocess image
    2. VLM room analysis
    3. Object detection (YOLO)
    4. Segmentation (SAM)
    5. Depth estimation (MiDaS)
    6. Scene graph assembly
    7. Budget allocation
    8. Product matching
    9. SD prompt building
    10. Image generation
    """
    def update(step: str):
        jobs[job_id].update({"status": "processing", "step": step})

    try:
        update("Initializing & loading models")
        update("Preprocessing image")
        img = preprocess_image(image_bytes)

        update("Detecting furniture and objects")
        detections = detect_objects(img)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        update("VLM analysing your room")
        vlm_analysis = analyse_room(img, detections)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        update("Segmenting objects")
        masks = segment_objects(img, detections)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        update("Estimating room depth")
        depth_map = estimate_depth(img)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        update("Building scene graph")
        scene_graph = build_scene_graph(
            vlm_analysis, detections, masks, depth_map, img.size
        )

        # Resolve target style
        target_style = style if style != "auto" else vlm_analysis.get(
            "current_style", "scandinavian"
        )

        update("Allocating your budget smartly")
        budget_plan = allocate_budget(budget_inr, scene_graph, target_style)

        update("Finding real products within your budget")
        matched_products = find_products_for_plan(budget_plan, target_style)

        update("Building redesign specification")
        sd_prompt, negative_prompt = build_sd_prompt(
            target_style=target_style,
            budget_plan=budget_plan,
            wall_treatment="matte white paint with warm undertones",
            flooring="light oak hardwood",
            lighting="soft warm ambient lighting",
            accessories=["potted plants", "throw cushions", "decorative vase"],
        )

        update("Generating your redesigned room")
        result_img = generate_redesigned_image(
            img, depth_map, masks, sd_prompt, negative_prompt
        )

        update("Exporting 3D geometry")
        # Generate Point Cloud
        h, w = depth_map.shape
        K = estimate_intrinsics(w, h)
        
        # MiDaS outputs 0-255 (uint8, inverse depth: 0=far, 255=near)
        # Rescale to metric-like depth range (0.3m – 5.0m) for proper 3D geometry
        depth_float = depth_map.astype(np.float32)
        depth_float = depth_float / 255.0
        depth_float = np.clip(depth_float, 0.01, 1.0)
        depth_metric = 0.3 + (1.0 - depth_float) * 4.7
        
        points, colors = depth_to_pointcloud_np(np.array(img), depth_metric, K)
        points, colors = clean_pointcloud_np(points, colors, voxel_size=0.02)
        
        pcd_filename = f"pcd_{job_id}.json"
        pcd_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", pcd_filename)
        export_pointcloud_to_json(points, colors, pcd_path)

        result_url = save_image(result_img, job_id)

        jobs[job_id] = {
            "status": "done",
            "step": "Complete",
            "result": {
                "result_url": result_url,
                "original_img": img,  # keep for chat refinement (not serialized)
                "vlm_analysis": vlm_analysis,
                "budget_plan": budget_plan,
                "matched_products": matched_products,
                "target_style": target_style,
                "sd_prompt": sd_prompt,
                "pcd_url": f"/outputs/{pcd_filename}"
            },
        }

    except Exception as e:
        jobs[job_id] = {"status": "error", "step": str(e), "result": None}
        print(f"[Pipeline Error] Job {job_id}: {e}")
        print(traceback.format_exc())
