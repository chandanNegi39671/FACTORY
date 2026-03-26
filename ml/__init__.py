"""
ml/models/__init__.py
─────────────────────
Auto-downloads best.pt from HuggingFace on first use.
Caches locally so subsequent restarts are instant.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID   = os.getenv("HF_REPO_ID",   "negi3961/factory-defect-guard")
HF_FILENAME  = os.getenv("HF_FILENAME",  "best.pt")
HF_TOKEN     = os.getenv("HF_TOKEN",     None)          # optional for private repos

# Local cache: project_root/ml/models/best.pt
_MODEL_DIR   = Path(__file__).parent
MODEL_PATH   = _MODEL_DIR / HF_FILENAME


def get_model_path() -> Path:
    """
    Returns the local path to best.pt.
    Downloads from HuggingFace if not already cached.
    """
    if MODEL_PATH.exists():
        logger.info("[models] Using cached model: %s", MODEL_PATH)
        return MODEL_PATH

    logger.info("[models] Model not found locally — downloading from HF (%s / %s) …",
                HF_REPO_ID, HF_FILENAME)

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = HF_FILENAME,
            token     = HF_TOKEN,
            local_dir = str(_MODEL_DIR),
        )
        logger.info("[models] Downloaded to: %s", downloaded)
        return Path(downloaded)

    except Exception as exc:
        logger.error("[models] HF download failed: %s", exc)
        raise RuntimeError(
            f"Could not load model '{HF_FILENAME}' from '{HF_REPO_ID}'. "
            "Set HF_TOKEN env var if the repo is private."
        ) from exc


def load_yolo():
    """
    Returns a loaded YOLO model (ultralytics).
    Call this anywhere you need the model object.

    Usage:
        from ml.models import load_yolo
        model = load_yolo()
    """
    from ultralytics import YOLO

    path = get_model_path()
    logger.info("[models] Loading YOLO from %s", path)
    return YOLO(str(path))


# ── Singleton (lazy) ──────────────────────────────────────────────────────────
_yolo_instance = None


def get_yolo():
    """
    Returns a module-level singleton YOLO model.
    First call downloads + loads; subsequent calls return cached instance.

    Usage:
        from ml.models import get_yolo
        model = get_yolo()   # fast after first call
    """
    global _yolo_instance
    if _yolo_instance is None:
        _yolo_instance = load_yolo()
    return _yolo_instance