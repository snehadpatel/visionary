"""
Main WebSocket router — handles all real-time client communication for Visionary v3.0.
Includes Live Mode for real-time camera feed analysis.
"""

import asyncio
import base64
import io
import json
import os
import time
import uuid
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image
import torch

from websocket_manager import manager
from pipeline.preprocessor import preprocess_image
from pipeline.object_detector import detect_objects
from pipeline.segmenter import segment_objects
from pipeline.depth_estimator import estimate_depth
from pipeline.scene_graph import build_scene_graph
from pipeline.budget_engine import allocate_budget
from pipeline.sd_prompt_builder import build_sd_prompt
from pipeline.image_generator import generate_redesigned_image_streaming
from pipeline.realtime_analyzer import RealtimeAnalyzer
from products.matcher import find_products_for_plan
from utils.storage import save_image
from utils.pointcloud_utils import export_pointcloud_to_json
from src.geometry.backprojector import estimate_intrinsics, depth_to_pointcloud_np, clean_pointcloud_np
from pipeline.image_generator import _load_custom_gen, _custom_gen, generate_custom_redesign_streaming

router = APIRouter()

# Store session state per client
sessions: dict[str, dict] = {}

# Live mode analyzers and locks per client
live_analyzers: dict[str, RealtimeAnalyzer] = {}
live_redesign_locks: dict[str, bool] = {}
last_live_redesign_time: dict[str, float] = {}
last_vlm_time: dict[str, float] = {}

def safe_serialize(obj):
    """Recursively convert tensors/numpy to serializable types."""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist() if obj.numel() < 20 else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size < 20 else None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    sessions[client_id] = {
        "image": None,
        "scene_graph": None,
        "budget_plan": None,
        "result": None,
        "conversation": [],
        "detections": [],
        "analysis": {}, # Replaced 'vlm'
        "live_mode": False,
        "live_frame_count": 0,
    }

    try:
        while True:
            raw = await websocket.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")
            data = message.get("data", {})

            if msg_type == "upload_image":
                await handle_upload(client_id, data)

            elif msg_type == "budget_changed":
                await handle_budget_preview(client_id, data)

            elif msg_type == "submit_redesign":
                # Run full pipeline in background — non-blocking
                asyncio.create_task(run_full_pipeline(client_id, data))

            elif msg_type == "chat_message":
                asyncio.create_task(handle_chat(client_id, data))

            # ─── Live Mode Messages ───

            elif msg_type == "start_live_session":
                await handle_start_live(client_id)

            elif msg_type == "live_frame":
                asyncio.create_task(handle_live_frame(client_id, data))

            elif msg_type == "voice_input":
                asyncio.create_task(handle_voice_input(client_id, data))

            elif msg_type == "stop_live_session":
                await handle_stop_live(client_id)

            elif msg_type == "ping":
                await manager.send(client_id, "pong", {})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        sessions.pop(client_id, None)
        live_analyzers.pop(client_id, None)
    except Exception as e:
        print(f"WS Error for {client_id}: {e}")
        manager.disconnect(client_id)
        sessions.pop(client_id, None)
        live_analyzers.pop(client_id, None)

async def handle_upload(client_id: str, data: dict):
    """
    VLM-Free Upload Handler.
    Runs YOLO + SceneNet immediately.
    """
    try:
        img_b64 = data.get("image_b64", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        img = preprocess_image(img_bytes)
        sessions[client_id]["image"] = img

        loop = asyncio.get_event_loop()

        # 1. Detect objects
        detections = await loop.run_in_executor(None, detect_objects, img)
        sessions[client_id]["detections"] = detections

        await manager.send(client_id, "detection_result", {
            "objects": [
                {
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                    "is_structural": d.get("is_structural", False)
                }
                for d in detections
            ],
            "object_count": len(detections),
        })

        # 2. SceneNet analysis (VLM Replacement)
        from pipeline.realtime_analyzer import RealtimeAnalyzer
        analyzer = RealtimeAnalyzer()
        res = await loop.run_in_executor(None, analyzer.analyze_frame_fast, img)
        sessions[client_id]["analysis"] = res
        
        # Strip heavy features and strictly serialize the rest
        ui_res = {k: v for k, v in res.items() if k != "features"}
        await manager.send(client_id, "vlm_stream", safe_serialize(ui_res))

    except Exception as e:
        print(f"Upload error: {e}")
        await manager.send(client_id, "error", {"message": str(e), "step": "upload"})

async def handle_budget_preview(client_id: str, data: dict):
    """Called every time user changes the budget number."""
    session = sessions.get(client_id, {})
    analysis = session.get("analysis", {})
    detections = session.get("detections", [])

    try:
        budget_inr = float(data.get("budget_inr", 0))
        style = data.get("style", "auto")
        prompt = data.get("prompt", "")

        if budget_inr < 500:
            return

        # Build minimal scene graph from detection if full one not built yet
        scene_graph = {
            "cv_objects": detections,
            "redesign_priority": ["furniture", "decor"], # Fallback
            "current_style": analysis.get("style", "unknown"),
        }

        target_style = style if style != "auto" else analysis.get("style", "scandinavian")

        plan = allocate_budget(
            total_budget_inr=budget_inr,
            scene_graph=scene_graph,
            target_style=target_style,
            user_prompt=prompt,
        )

        await manager.send(client_id, "budget_preview", safe_serialize({
            "allocation": [
                {
                    "item": item["item"],
                    "amount_inr": item["allocated_inr"],
                    "percentage": round((item["allocated_inr"] / budget_inr) * 100, 1),
                    "tier": item["tier"],
                }
                for item in plan["items"]
            ],
            "total_allocated": plan["total_allocated_inr"],
            "buffer": plan["buffer_inr"],
            "budget_tier": plan["budget_tier"],
        }))
    except Exception as e:
        print(f"Budget preview error: {e}")

async def run_full_pipeline(client_id: str, data: dict):
    """Full redesign pipeline with real-time progress streaming."""
    session = sessions.get(client_id, {})
    img = session.get("image")
    analysis = session.get("analysis", {})
    detections = session.get("detections", [])
    
    if not img:
        await manager.send(client_id, "error", {"message": "No image uploaded", "step": "pipeline"})
        return

    loop = asyncio.get_event_loop()
    budget_inr = float(data.get("budget_inr", 30000))
    style_override = data.get("style", "auto")
    user_prompt = data.get("prompt", "")
    t0 = time.time()

    async def step(name: str, index: int, total: int = 10):
        await manager.send(client_id, "pipeline_step", safe_serialize({
            "step": name,
            "step_index": index,
            "total_steps": total,
            "progress": int((index / total) * 100),
            "elapsed_seconds": round(time.time() - t0, 1),
        }))

    try:
        await step("Segmenting detected objects", 1)
        masks = await loop.run_in_executor(None, segment_objects, img, detections)

        await step("Estimating room depth and layout", 2)
        depth_map = await loop.run_in_executor(None, estimate_depth, img)

        await step("Building scene understanding", 3)
        scene_graph = build_scene_graph(analysis, detections, masks, depth_map, img.size)
        sessions[client_id]["scene_graph"] = scene_graph

        target_style = style_override if style_override != "auto" else analysis.get("style", "scandinavian")

        await step("Allocating budget intelligently", 4)
        budget_plan = allocate_budget(budget_inr, scene_graph, target_style, user_prompt)
        sessions[client_id]["budget_plan"] = budget_plan

        await step("Finding real products within budget", 5)
        matched_products = await loop.run_in_executor(
            None, find_products_for_plan, budget_plan, target_style
        )

        await manager.send(client_id, "products_found", safe_serialize({
            "products": [
                {
                    "item": p["item"],
                    "title": p["product"]["title"] if p["product"] else None,
                    "price_inr": p["product"]["price_inr"] if p["product"] else None,
                    "url": p["product"]["url"] if p["product"] else None,
                    "image_url": p["product"]["image_url"] if p["product"] else None,
                    "source": p["product"]["source"] if p["product"] else None,
                    "allocated": p["allocated_budget_inr"],
                    "tier": p["tier"],
                }
                for p in matched_products
            ]
        }))

        await step("Building redesign specification", 6)
        sd_prompt, negative_prompt = build_sd_prompt(
            target_style=target_style,
            budget_plan=budget_plan,
            scene_graph=scene_graph,
            user_prompt=user_prompt,
        )

        use_sd = data.get("use_sd", False)
        
        await step("Generating redesigned room image", 7)
        if use_sd:
            # Using the high-quality streaming generator
            result_img = await generate_redesigned_image_streaming(
                original_img=img,
                depth_map=depth_map,
                sd_prompt=sd_prompt,
                negative_prompt=negative_prompt,
                client_id=client_id,
                manager=manager,
            )
        else:
            # Using the high-speed custom generator (distilled model)
            from pipeline.image_generator import generate_custom_redesign_streaming
            result_img = await generate_custom_redesign_streaming(
                original_img=img,
                depth_map=depth_map,
                masks=masks,
                client_id=client_id,
                manager=manager,
                target_style=target_style,
            )

        # Save result
        result_url = save_image(result_img, client_id, suffix=f"v3_redesign_{int(time.time())}")
        sessions[client_id]["result"] = result_url

        await step("Exporting 3D geometry", 8)
        # Generate Point Cloud
        h, w = depth_map.shape
        K = estimate_intrinsics(w, h)
        
        # MiDaS outputs 0-255 (uint8, inverse depth: 0=far, 255=near)
        # Rescale to metric-like depth range (0.3m – 5.0m) for proper 3D geometry
        depth_float = depth_map.astype(np.float32)
        depth_float = depth_float / 255.0              # normalize 0-1
        depth_float = np.clip(depth_float, 0.01, 1.0)  # avoid division by zero
        depth_metric = 0.3 + (1.0 - depth_float) * 4.7 # invert: near=low, far=high → 0.3m to 5.0m
        
        points, colors = depth_to_pointcloud_np(np.array(img), depth_metric, K)
        points, colors = clean_pointcloud_np(points, colors, voxel_size=0.02)
        
        pcd_filename = f"pcd_{client_id}.json"
        
        # Save Point Cloud to the project-root outputs directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        pcd_path = os.path.join(outputs_dir, pcd_filename)
        export_pointcloud_to_json(points, colors, pcd_path)

        # 3. NEW: Extract individual 3D Objects for Immersive Walkthrough
        await step("Building 3D Scene Components", 9)
        from utils.spatial_utils import extract_3d_objects
        objects_3d = extract_3d_objects(result_img, depth_map, masks, K, outputs_dir, client_id)

        # Save Depth Map for Dashboard
        depth_pil = Image.fromarray(depth_map.astype(np.uint8))
        depth_url = save_image(depth_pil, client_id, suffix="depth_map")

        # Generate Neural Reasoning (The "Why")
        await step("Generating spatial reasoning", 10)
        reasoning = "Architectural redesign optimized for spatial constraints and style consistency."
        try:
            from vlm.decoder import generate_response
            reasoning_prompt = f"Explain briefly why you redesigned this {analysis.get('room_type')} in {target_style} style, focusing on architectural choices."
            reasoning = generate_response(reasoning_prompt, analysis.get("features"))
        except Exception as e:
            print(f"Reasoning error: {e}")

        # Map SceneNet fields to legacy VLM keys for frontend compatibility
        ui_analysis = {
            "room_type": analysis.get("room_type", "unknown"),
            "current_style": analysis.get("style", "unknown"),
            "natural_light": analysis.get("lighting", "medium"),
            "room_size_estimate": "medium", # Fallback for now
            "condition": analysis.get("condition", "fair"),
        }

        await manager.send(client_id, "result_ready", safe_serialize({
            "result_url": result_url,
            "depth_url": depth_url,
            "reasoning": reasoning,
            "vlm_analysis": ui_analysis,
            "budget_plan": budget_plan,
            "matched_products": matched_products,
            "target_style": target_style,
            "sd_prompt": sd_prompt,
            "pcd_url": f"/outputs/{pcd_filename}",
            "objects_3d": objects_3d,
            "total_time_seconds": round(time.time() - t0, 1),
        }))

    except Exception as e:
        print(f"Pipeline error: {e}")
        await manager.send(client_id, "error", {"message": str(e), "step": "pipeline"})

async def handle_chat(client_id: str, data: dict):
    """Real-Time VLM Chat: Structured design reasoning."""
    message = data.get("message", "")
    session = sessions.get(client_id, {})
    analysis = session.get("analysis", {})
    features = analysis.get("features")
    
    if features is None:
        import torch
        features = torch.zeros(1, 1280)

    try:
        from vlm.decoder import stream_generate_response
        full_response = ""
        for token in stream_generate_response(message, features):
            full_response += token
            await manager.stream_chat_token(client_id, token, done=False)
        await manager.stream_chat_token(client_id, "", done=True, full_response=full_response)
    except Exception as e:
        print(f"VLM Chat Error for {client_id}: {e}")
        await manager.stream_chat_token(client_id, "I encountered a neural synchronization error. Please try again.", done=True)


# ───────────────────────────────────────────────────────────
# LIVE MODE HANDLERS
# ───────────────────────────────────────────────────────────

async def handle_start_live(client_id: str):
    """Initialize a live camera session."""
    analyzer = RealtimeAnalyzer(ema_alpha=0.3)
    live_analyzers[client_id] = analyzer
    sessions[client_id]["live_mode"] = True

async def handle_live_frame(client_id: str, data: dict):
    """Processes a live frame via WebSocket."""
    analyzer = live_analyzers.get(client_id)
    if not analyzer: return

    img_b64 = data.get("image_b64", "")
    if not img_b64: return
    if "," in img_b64: img_b64 = img_b64.split(",")[1]
    
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        loop = asyncio.get_event_loop()
        
        # 1. SceneNet (Fast)
        res = await loop.run_in_executor(None, analyzer.analyze_frame_fast, img)
        
        # 2. YOLO (Every 20 frames)
        sessions[client_id]["live_frame_count"] += 1
        if sessions[client_id]["live_frame_count"] % 20 == 0:
            res = await loop.run_in_executor(None, analyzer.analyze_frame_full, img)
            
        # Strip raw tensors and strictly serialize
        ui_res = {k: v for k, v in res.items() if k != "features"}
        await manager.send(client_id, "live_analysis", safe_serialize(ui_res))
    except Exception as e:
        print(f"Live frame error: {e}")

async def handle_voice_input(client_id: str, data: dict):
    """VLM-Free Voice Input."""
    text = data.get("text", "").lower()
    if not text: return
    
    # Intent mapping
    if "modern" in text: response = "Updating to modern design."
    elif "rustic" in text: response = "Setting rustic theme."
    elif "redesign" in text: response = "Starting real-time redesign."
    else: response = "I'm listening and analyzing your room live."

    await manager.send_narration_token(client_id, response, done=True)


async def handle_stop_live(client_id: str):
    """Clean up a live session."""
    analyzer = live_analyzers.pop(client_id, None)
    sessions[client_id]["live_mode"] = False

    await manager.send(client_id, "live_session_stopped", {})

