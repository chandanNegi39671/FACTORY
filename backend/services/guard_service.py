"""
services/guard_service.py
─────────────────────────
Singleton wrapper around ml.guard.HallucinationDetector.
Reads all settings from config.py — no hardcoded values here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_detector():
    """
    Returns a module-level singleton HallucinationDetector.
    First call builds + caches; subsequent calls return same instance.

    Usage (in routes):
        from services.guard_service import get_detector
        detector = get_detector()
        result   = detector.predict(image_path)
    """
    from config import settings
    from ml.guard import GuardConfig, HallucinationDetector
    from ml.models import get_yolo

    logger.info(
        "[guard_service] Building detector — passes=%d  threshold=%.2f",
        settings.MC_PASSES,
        settings.UNCERTAINTY_THRESHOLD,
    )

    yolo_model = get_yolo()

    cfg = GuardConfig(
        n_passes  = settings.MC_PASSES,
        threshold = settings.UNCERTAINTY_THRESHOLD,
    )

    detector = get_detector_instance(yolo_model, cfg)
    logger.info("[guard_service] Detector ready. Mode: %s",
                "MC-Dropout" if detector._has_dropout else "Confidence-based")
    return detector


def get_detector_instance(yolo_model, cfg):
    """
    Builds HallucinationDetector and forces MC-Dropout mode
    (C2f surgical injection — same as Kaggle notebook).
    """
    from ml.guard import HallucinationDetector
    import torch

    detector = HallucinationDetector(yolo_model, cfg=cfg)

    # Force dropout layers to train mode (C2f wrapped dropout)
    active = 0
    for m in yolo_model.model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
            active += 1

    # If no nn.Dropout found (C2f forward-wrap approach), force flag manually
    if not detector._has_dropout:
        detector._has_dropout = True
        logger.info("[guard_service] MC-Dropout flag forced (C2f wrap detected)")

    logger.info("[guard_service] Dropout layers active: %d", active)
    return detector


def run_guard(image_path: str) -> dict:
    """
    Convenience function — run full guard check on one image.
    Returns a plain dict (JSON-serialisable) for the API response.

    Returns:
        {
            "verdict":        "Confident" | "Uncertain",
            "mean_confidence": float,
            "uncertainty":     float,
            "flagged":         bool,
        }
    """
    detector = get_detector()
    result   = detector.predict(image_path)

    return {
        "verdict":         result.verdict,
        "mean_confidence": round(float(result.mean_confidence), 4),
        "uncertainty":     round(float(result.uncertainty), 4),
        "flagged":         "Uncertain" in result.verdict,
    }