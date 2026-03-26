"""
routes/predict.py
-----------------
POST /predict       — single image prediction
POST /predict/batch — up to 10 images
GET  /predict/health — model + guard status

FIX 1: UPLOAD_DIR now from settings (was hardcoded "uploads")
FIX 2: tmp filenames use UUID (was time-based — collision risk under load)
FIX 3: 10MB file size limit added
FIX 4: file.content_type check kept but secondary — magic byte check added
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ml.guard import GuardConfig, GuardResult, build_detector
from ml.rag   import get_verifier
from backend.config import settings

router = APIRouter(prefix="/predict", tags=["predict"])

# FIX: use settings instead of hardcoded path
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

# FIX: 10MB max upload size
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Allowed image magic bytes (first few bytes of file)
_IMAGE_MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG":      "png",
    b"GIF8":         "gif",
    b"RIFF":         "webp",  # RIFF....WEBP
    b"BM":           "bmp",
}

_model    = None
_detector = None


def _validate_image_magic(data: bytes) -> bool:
    """Check actual file magic bytes — content_type can be spoofed."""
    for magic in _IMAGE_MAGIC:
        if data[:len(magic)] == magic:
            return True
    return False


def get_detector():
    global _model, _detector
    if _detector is None:
        cfg = GuardConfig(
            n_passes      = settings.MC_PASSES,
            threshold     = settings.UNCERTAINTY_THRESHOLD,
            low_conf_gate = 0.25,
        )
        _model, _detector = build_detector(cfg=cfg)
    return _detector


def _build_response(
    result:     GuardResult,
    rag_result: Optional[dict],
    factory_id: Optional[str],
    line_id:    Optional[str],
    elapsed_ms: float,
) -> dict:
    return {
        "status":           "ok",
        "factory_id":       factory_id,
        "line_id":          line_id,
        "verdict":          result.verdict,
        "is_hallucination": result.is_hallucination,
        "mean_confidence":  result.mean_confidence,
        "uncertainty":      result.uncertainty,
        "n_detections":     result.n_detections,
        "passes_used":      result.passes_used,
        "detections": [
            {
                "class":      result.classes[i],
                "confidence": result.confidences[i],
                "box":        result.boxes[i],
            }
            for i in range(result.n_detections)
        ],
        "rag":        rag_result,
        "latency_ms": round(elapsed_ms, 1),
    }


@router.post("")
async def predict(
    file:       UploadFile     = File(...),
    factory_id: Optional[str] = Form(None),
    line_id:    Optional[str] = Form(None),
    part_id:    Optional[str] = Form(None),
    use_rag:    bool          = Form(True),
):
    """
    Main prediction endpoint.
    - YOLOv8s (mAP 0.83, 17 classes)
    - MC Dropout guard (10 passes)
    - Optional RAG cross-check if prediction is uncertain
    """
    # FIX: content_type check
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    # FIX: file size limit
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # FIX: magic byte validation
    if not _validate_image_magic(contents):
        raise HTTPException(status_code=400, detail="Invalid image file (magic bytes mismatch)")

    # FIX: UUID-based tmp filename — no collision under concurrent load
    suffix   = Path(file.filename or "image.jpg").suffix or ".jpg"
    tmp_path = UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(contents)

    t0 = time.time()
    try:
        detector = get_detector()
        result   = detector.predict(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed_ms = (time.time() - t0) * 1000

    # RAG cross-check (only if uncertain + detections exist)
    rag_result = None
    if use_rag and result.is_hallucination and result.n_detections > 0:
        top_class = result.classes[0] if result.classes else "unknown"
        verifier  = get_verifier()
        rag       = verifier.verify(
            defect_type = top_class,
            part_id     = part_id,
            line_id     = line_id,
        )
        rag_result = rag.to_dict()

        if rag.plausible and result.is_hallucination:
            result.verdict          = "⚠️  Uncertain — routed to manual review"
            result.is_hallucination = False

    return JSONResponse(_build_response(result, rag_result, factory_id, line_id, elapsed_ms))


@router.post("/batch")
async def predict_batch(
    files:      list[UploadFile] = File(...),
    factory_id: Optional[str]   = Form(None),
    line_id:    Optional[str]   = Form(None),
):
    """Batch prediction — up to 10 images."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch")

    results  = []
    detector = get_detector()

    for file in files:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            results.append({"file": file.filename, "error": "File too large (max 10MB)"})
            continue

        if not _validate_image_magic(contents):
            results.append({"file": file.filename, "error": "Invalid image file"})
            continue

        suffix   = Path(file.filename or "image.jpg").suffix or ".jpg"
        tmp_path = UPLOAD_DIR / f"tmp_{uuid.uuid4().hex}{suffix}"
        tmp_path.write_bytes(contents)

        t0 = time.time()
        try:
            result     = detector.predict(str(tmp_path))
            elapsed_ms = (time.time() - t0) * 1000
            r          = _build_response(result, None, factory_id, line_id, elapsed_ms)
            r["file"]  = file.filename
            results.append(r)
        except Exception as exc:
            results.append({"file": file.filename, "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)

    return JSONResponse({"status": "ok", "results": results, "count": len(results)})


@router.get("/health")
async def predict_health():
    """Model + guard status check."""
    try:
        detector = get_detector()
        return {
            "status":     "ok",
            "model":      "YOLOv8s — negi3961/factory-defect-guard",
            "map50":      0.83,
            "classes":    17,
            "mc_dropout": detector._has_dropout,
            "n_passes":   detector.cfg.n_passes,
            "threshold":  detector.cfg.threshold,
        }
    except Exception as exc:
        return JSONResponse(
            status_code = 503,
            content     = {"status": "error", "detail": str(exc)},
        )