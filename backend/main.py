"""
Visionary — FastAPI Backend Entry Point.
Room redesign powered by custom VLM + CV pipeline + budget intelligence.
All models run locally on Apple MPS. Zero paid APIs.
"""
import sys
import os
from pathlib import Path
import torch

# Optimize CPU usage (User has 10 cores)
torch.set_num_threads(10)

# Ensure backend directory is in Python path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import routers
from routers.redesign import router as redesign_router
from routers.products import router as products_router
from routers.ws import router as ws_router
from routers.live_http import router as live_http_router
from routers.interaction_http import router as interaction_router

app = FastAPI(
    title="Visionary — AI Room Redesign",
    description="Smart AI-powered room redesign with custom VLM + budget intelligence",
    version="3.0.0",
)

@app.on_event("startup")
async def startup_event():
    """Warm up all models on startup to avoid first-frame latency spikes."""
    print("🚀 Warming up CV models (YOLO, MiDaS, SceneNet, UNet) on MPS...")
    try:
        from pipeline.object_detector import _get_model
        from pipeline.depth_estimator import estimate_depth
        from pipeline.image_generator import _load_custom_gen
        from pipeline.realtime_analyzer import RealtimeAnalyzer
        from PIL import Image
        import numpy as np
        
        # Trigger lazy loads and force MPS
        _get_model()
        _load_custom_gen()
        
        # Create a dummy image for a quick dry run
        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        estimate_depth(dummy_img)
        
        # Warmup analyzer
        from pipeline.realtime_analyzer import RealtimeAnalyzer
        analyzer = RealtimeAnalyzer()
        analyzer.analyze_frame_fast(dummy_img)

        # Warmup HQ Generator (SD v1.5 + ControlNet)
        from pipeline.image_generator import _load_pipeline
        _load_pipeline()

        # Warmup VLM Decoder (TinyLlama)
        from vlm.decoder import _load_model
        _load_model()
        
        print("✅ All CV models warmed up and ready.")
    except Exception as e:
        print(f"⚠ Model warm-up error: {e}")

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(redesign_router)
app.include_router(products_router)
app.include_router(ws_router)
app.include_router(live_http_router)
app.include_router(interaction_router)

# Static file serving for outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "visionary",
        "version": "2.0.0",
        "gpu": "Apple MPS",
    }


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Visionary API — AI Room Redesign",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        reload_dirs=[str(BACKEND_DIR), str(PROJECT_ROOT / "src")],
    )
