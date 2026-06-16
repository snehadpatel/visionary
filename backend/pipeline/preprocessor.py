"""
Visionary Pipeline — Image Preprocessor.
Normalizes uploaded images to a consistent 1024x1024 canvas.
"""
from PIL import Image
import io

TARGET = (1024, 1024)


def preprocess_image(image_bytes: bytes) -> Image.Image:
    """
    Convert raw bytes to a normalized 1024x1024 RGB image.
    The original aspect ratio is preserved with letterboxing on a white canvas.
    
    Args:
        image_bytes: Raw image file bytes
    
    Returns:
        PIL Image normalized to 1024x1024
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(TARGET, Image.LANCZOS)
    canvas = Image.new("RGB", TARGET, (255, 255, 255))
    offset = ((TARGET[0] - img.width) // 2, (TARGET[1] - img.height) // 2)
    canvas.paste(img, offset)
    return canvas
