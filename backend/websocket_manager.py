"""
WebSocket Connection Manager
Handles all real-time communication between backend and frontend.
Every pipeline step streams events back as it completes.
"""

from fastapi import WebSocket
from typing import Dict, Any
import asyncio
import json
import base64
from io import BytesIO
from PIL import Image

class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, ws: WebSocket):
        await ws.accept()
        self.active[client_id] = ws
        await self.send(client_id, "connected", {"client_id": client_id})

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)

    async def send(self, client_id: str, event_type: str, data: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps({
                    "type": event_type,
                    "data": data
                }))
            except Exception as e:
                print(f"[WebSocket] Error sending to {client_id}: {e}")
                self.disconnect(client_id)

    async def send_image_preview(self, client_id: str, img: Image.Image, step: int, total: int):
        """Stream intermediate SD generation preview to frontend."""
        buf = BytesIO()
        # Downscale preview to reduce data size for smooth streaming
        preview = img.copy()
        preview.thumbnail((512, 512))
        preview.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode()
        await self.send(client_id, "sd_preview", {
            "step": step,
            "total_steps": total,
            "preview_base64": b64,
        })

    async def stream_chat_token(self, client_id: str, token: str, done: bool = False, full_response: str = ""):
        """Stream each chat response token like ChatGPT."""
        await self.send(client_id, "chat_token" if not done else "chat_done", {
            "token": token,
            "done": done,
            "full_response": full_response if done else "",
        })

    # ─── Live Mode Methods ───

    async def send_live_detections(self, client_id: str, detections: list, object_count: int):
        """Send YOLO detection results for live overlay bounding boxes."""
        await self.send(client_id, "live_detections", {
            "objects": detections,
            "object_count": object_count,
        })

    async def send_scene_state(self, client_id: str, scene_state: dict):
        """Send SceneNet analysis results for real-time info cards."""
        await self.send(client_id, "live_scene_update", scene_state)

    async def send_narration_token(self, client_id: str, token: str, done: bool = False):
        """Stream AI narration tokens for live mode voice readout."""
        await self.send(client_id, "live_narration", {
            "token": token,
            "done": done,
            "full_text": token if done else "",
        })

    async def send_quick_redesign(self, client_id: str, preview_b64: str, style: str):
        """Send a fast style-transfer preview for live mode."""
        await self.send(client_id, "quick_redesign_preview", {
            "preview_base64": preview_b64,
            "style": style,
        })

manager = ConnectionManager()
