import cv2
import asyncio
from backend.services.guard_service import run_guard
import time

async def stream_frames(sio, factory_id, line_id, rtsp_url):
    """
    Reads frames from RTSP (or webcam), runs them through the Guard Service,
    and emits the results to the specific factory/line room via Socket.IO.
    """
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Failed to open video stream: {rtsp_url}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame to bytes for the guard service
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Run prediction
            result = run_guard(frame_bytes)
            
            # Add line context
            result["line_id"] = line_id
            result["timestamp"] = time.time()

            # Emit to connected clients
            await sio.emit("detection", result, room=factory_id)
            
            # Target 15 FPS
            await asyncio.sleep(1/15)
            
    finally:
        cap.release()
