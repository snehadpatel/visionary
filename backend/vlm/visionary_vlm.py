"""
Main VLM entry point.
Given a room image + user query, returns structured JSON understanding.
"""
import json
import re
from PIL import Image
import torch
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.encoder import encode_image
from vlm.projection import load_projection
from vlm.decoder import generate_response, stream_generate_response, device



async def stream_chat_response(img: Image.Image, conversation_history: list, user_message: str):
    """
    Streaming version of chat_with_room.
    Yields tokens as they are generated.
    """
    proj = _get_projection()
    img_embedding = encode_image(img)
    projected = proj(img_embedding.to(device))

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Visionary'}: {m['content']}"
        for m in conversation_history[-6:]
    )

    prompt = f"""
Previous conversation:
{history_text}

User now says: "{user_message}"

You are redesigning their room. Respond naturally and helpfully.
"""
    # stream_generate_response is a generator, we yield from it
    for token in stream_generate_response(prompt, projected, max_new_tokens=256):
        yield token

_projection = None


def _get_projection():
    global _projection
    if _projection is None:
        _projection = load_projection()
        _projection.to(device)
    return _projection


ANALYSIS_PROMPT = """
Analyse this room image and respond ONLY with a JSON object containing:
{
  "detected_items": [
    {"item": "sofa", "condition": "good/fair/poor", "style": "modern/rustic/...", "keep": true/false}
  ],
  "room_type": "bedroom/living room/kitchen/...",
  "current_style": "scandinavian/industrial/bohemian/...",
  "color_palette": "warm/cool/neutral/...",
  "room_size_estimate": "small/medium/large",
  "natural_light": "low/medium/high",
  "major_issues": ["cluttered", "poor lighting"],
  "redesign_priority": ["sofa", "wall color"]
}
Respond ONLY with JSON. No extra text.
"""


def analyse_room(img: Image.Image, detections: list = None) -> dict:
    """Full VLM room analysis - returns structured JSON"""
    proj = _get_projection()
    img_embedding = encode_image(img)                          # (1, 768)
    # Ensure float16 for MPS compatibility with TinyLlama
    projected = proj(img_embedding.to(device)).to(torch.float16) 
    
    hint = ""
    if detections:
        items = [d['label'] for d in detections]
        hint = f"\nNote: I have already detected these physical items in the room: {', '.join(items)}. Use this to inform your analysis.\n"

    raw_response = generate_response(ANALYSIS_PROMPT + hint, projected)

    # Extract JSON safely
    try:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Fallback structured response if JSON parsing fails
    return {
        "detected_items": [],
        "room_type": "living room",
        "current_style": "unknown",
        "color_palette": "neutral",
        "room_size_estimate": "medium",
        "natural_light": "medium",
        "major_issues": [],
        "redesign_priority": ["sofa", "walls", "flooring"],
    }


def chat_with_room(img: Image.Image, conversation_history: list, user_message: str) -> str:
    """
    Conversational refinement - user can say things like
    'remove the plants' or 'make it darker' and VLM will reason about the room.
    """
    proj = _get_projection()
    img_embedding = encode_image(img)
    projected = proj(img_embedding.to(device))

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Visionary'}: {m['content']}"
        for m in conversation_history[-6:]  # last 3 turns
    )

    prompt = f"""
Previous conversation:
{history_text}

User now says: "{user_message}"

You are redesigning their room. Respond naturally and also output an updated redesign instruction.
"""
    return generate_response(prompt, projected, max_new_tokens=256)
