"""
sockets/stream.py
-----------------
RTSP / webcam frame streamer → Guard Service → Socket.IO emit.

FIX: run_guard() expects a file path (str), NOT raw bytes.
     Frames are now saved to a temp file, guard runs on path, file is cleaned up.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

import cv2

from backend.services.guard_service import run_guard

logger = logging.getLogger(__name__)

# Temp dir for frame dumps
_TMP_DIR = Path("uploads/stream_tmp")
_TMP_DIR.mkdir(parents=True, exist_ok=True)


async def stream_frames(sio, factory_id: str, line_id: str, rtsp_url: str):
    """
    Reads frames from RTSP (or webcam index), runs them through the Guard Service,
    and emits results to the specific factory/line room via Socket.IO.

    FIX: frame bytes saved to tmp .jpg → run_guard(path) → tmp file deleted.
    """
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("[stream] Failed to open video stream: %s", rtsp_url)
        await sio.emit("stream_error", {"error": f"Cannot open stream: {rtsp_url}"}, room=factory_id)
        return

    logger.info("[stream] Started streaming %s → factory=%s line=%s", rtsp_url, factory_id, line_id)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[stream] Stream ended or frame read failed: %s", rtsp_url)
                break

            # FIX: save frame to tmp file — run_guard expects a path, not bytes
            tmp_path = _TMP_DIR / f"frame_{uuid.uuid4().hex}.jpg"
            try:
                cv2.imwrite(str(tmp_path), frame)

                result        = run_guard(str(tmp_path))
                result["line_id"]   = line_id
                result["factory_id"] = factory_id
                result["timestamp"] = time.time()

                await sio.emit("detection", result, room=factory_id)

            except Exception as exc:
                logger.error("[stream] Guard error on frame: %s", exc)
            finally:
                # Always clean up tmp frame
                tmp_path.unlink(missing_ok=True)

            # Target ~15 FPS
            await asyncio.sleep(1 / 15)

    finally:
        cap.release()
        logger.info("[stream] Stream closed: %s", rtsp_url)