import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://127.0.0.1:8000/ws/test_client"
    try:
        async with websockets.connect(uri) as websocket:
            print("Successfully connected to WebSocket!")
            # Wait for the 'connected' message
            response = await websocket.recv()
            print(f"Received from server: {response}")
            
            # Send a ping
            await websocket.send(json.dumps({"type": "ping", "data": {}}))
            pong = await websocket.recv()
            print(f"Received pong: {pong}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
