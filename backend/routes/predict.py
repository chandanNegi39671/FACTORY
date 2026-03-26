"""
routes/predict.py
-----------------
POST /predict — accepts image, runs YOLOv8 + Hallucination Guard + RAG
POST /predict/batch — multiple images
GET  /predict/health — model status
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ── ML imports ─────────────────────────────────────────────────────────────────
from ml.guard import GuardConfig, GuardResult, build_detector
from ml.rag   import get_verifier

router = APIRouter(prefix="/predict", tags=["predict"])

# ── Global model + detector (loaded once at startup) ──────────────────────────
_model    = None
_detector = None

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_detector():
    global _model, _detector
    if _detector is None:
        cfg = GuardConfig(n_passes=10, threshold=0.05, low_conf_gate=0.25)
        _model, _detector = build_detector(cfg=cfg)
    return _detector


# ── Schemas ────────────────────────────────────────────────────────────────────
def _build_response(
    result: GuardResult,
    rag_result: Optional[dict],
    factory_id: Optional[str],
    line_id:    Optional[str],
    elapsed_ms: float,
) -> dict:
    return {
        "status":     "ok",
        "factory_id": factory_id,
        "line_id":    line_id,
        # Guard output
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
        # RAG output (None if not triggered)
        "rag": rag_result,
        # Meta
        "latency_ms": round(elapsed_ms, 1),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("")
async def predict(
    file:       UploadFile = File(...),
    factory_id: Optional[str] = Form(None),
    line_id:    Optional[str] = Form(None),
    part_id:    Optional[str] = Form(None),
    use_rag:    bool          = Form(True),
):
    """
    Main prediction endpoint.

    - Runs YOLOv8s (mAP 0.83, 17 classes)
    - MC Dropout guard (10 passes, σ threshold = 0.05)
    - Optional RAG cross-check if prediction is uncertain
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save upload temporarily
    tmp_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{file.filename}"
    contents = await file.read()
    tmp_path.write_bytes(contents)

    t0 = time.time()
    try:
        detector = get_detector()
        result   = detector.predict(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed_ms = (time.time() - t0) * 1000

    # ── RAG cross-check (only if uncertain + detections exist) ────────────────
    rag_result = None
    if use_rag and result.is_hallucination and result.n_detections > 0:
        top_class  = result.classes[0] if result.classes else "unknown"
        verifier   = get_verifier()
        rag        = verifier.verify(
            defect_type = top_class,
            part_id     = part_id,
            line_id     = line_id,
        )
        rag_result = rag.to_dict()

        # If RAG says implausible → keep hallucination flag
        # If RAG says plausible → downgrade to uncertain (not outright hallucination)
        if rag.plausible and result.is_hallucination:
            result.verdict          = "⚠️  Uncertain — routed to manual review"
            result.is_hallucination = False   # don't auto-scrap

    return JSONResponse(_build_response(result, rag_result, factory_id, line_id, elapsed_ms))


@router.post("/batch")
async def predict_batch(
    files:      list[UploadFile] = File(...),
    factory_id: Optional[str]    = Form(None),
    line_id:    Optional[str]    = Form(None),
):
    """Batch prediction — up to 10 images."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Max 10 images per batch")

    results = []
    detector = get_detector()

    for file in files:
        tmp_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{file.filename}"
        contents = await file.read()
        tmp_path.write_bytes(contents)

        t0 = time.time()
        try:
            result = detector.predict(str(tmp_path))
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
            continue
        finally:
            tmp_path.unlink(missing_ok=True)

        elapsed_ms = (time.time() - t0) * 1000
        r = _build_response(result, None, factory_id, line_id, elapsed_ms)
        r["file"] = file.filename
        results.append(r)

    return JSONResponse({"status": "ok", "results": results, "count": len(results)})


@router.get("/health")
async def predict_health():
    """Model + guard status check."""
    try:
        detector = get_detector()
        return {
            "status":       "ok",
            "model":        "YOLOv8s — negi3961/factory-defect-guard",
            "map50":        0.83,
            "classes":      17,
            "mc_dropout":   detector._has_dropout,
            "n_passes":     detector.cfg.n_passes,
            "threshold":    detector.cfg.threshold,
        }
    except Exception as e:
        return JSONResponse(
            status_code = 503,
            content     = {"status": "error", "detail": str(e)},
        )