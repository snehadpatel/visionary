"""
Visionary — Storage utilities for saving/loading generated images.
"""
import os
import uuid
from pathlib import Path
from PIL import Image

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def save_image(img: Image.Image, job_id: str, suffix: str = "redesign") -> str:
    """
    Save a PIL Image to the outputs directory.
    Returns a URL path that can be served by FastAPI's static files.
    """
    filename = f"{job_id}_{suffix}.png"
    filepath = OUTPUTS_DIR / filename
    img.save(filepath, "PNG", quality=95)
    return f"/outputs/{filename}"


def get_output_path(job_id: str, suffix: str = "redesign") -> Path:
    """Get the filesystem path for a job output."""
    return OUTPUTS_DIR / f"{job_id}_{suffix}.png"


def save_upload(image_bytes: bytes, job_id: str) -> Path:
    """Save uploaded image bytes to outputs directory."""
    upload_dir = OUTPUTS_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    filepath = upload_dir / f"{job_id}_original.png"
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath
