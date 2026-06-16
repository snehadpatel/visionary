"""
Visionary — Unified HTTP API for Voice and Chat.
Replacement for flaky WebSockets.
"""
from fastapi import APIRouter, Form
import time

router = APIRouter(prefix="/api/interaction", tags=["interaction"])

# Global session store for chat history (simplified for now)
chat_histories = {}

@router.post("/voice")
async def handle_voice_input(
    client_id: str = Form(...),
    text: str = Form(...)
):
    """Handle voice commands via HTTP instead of WebSocket."""
    print(f"[Interaction] Voice from {client_id}: {text}")
    
    # Simple logic: If they say 'redesign', we could trigger it
    # For now, just return a confirmation
    return {
        "status": "received",
        "transcription": text,
        "response": "I heard you! I'm analyzing the room now."
    }

@router.post("/chat")
async def handle_chat_message(
    client_id: str = Form(...),
    message: str = Form(...)
):
    """Handle chat refinement via HTTP."""
    print(f"[Interaction] Chat from {client_id}: {message}")
    
    # Get history
    history = chat_histories.get(client_id, [])
    
    # For live mode, we'd ideally have the latest image
    # But since we're non-persistent, we'll keep it simple
    response = "That's a great idea! I'll update the redesign plan."
    
    chat_histories[client_id] = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    
    return {
        "response": response,
        "history_count": len(chat_histories[client_id])
    }
