"""
ml/guard.py
-----------
HallucinationDetector — MC Dropout uncertainty estimation
Model: YOLOv8s fine-tuned, mAP@0.5 = 0.83, 17 classes
MC Dropout injected surgically into C2f head blocks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

# ── Class names (17 classes, 4 datasets) ──────────────────────────────────────
CLASS_NAMES: list[str] = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled_in_scale", "scratches",
    "pcb_missing_hole", "pcb_mouse_bite", "pcb_open_circuit",
    "pcb_short", "pcb_spur", "pcb_spurious_copper",
    "tile_defect",
    "transistor_defect", "screw_defect", "metal_nut_defect", "capsule_defect",
]

HF_REPO   = "negi3961/factory-defect-guard"
HF_FILE   = "best.pt"


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class GuardConfig:
    n_passes:       int   = 10     # MC forward passes
    threshold:      float = 0.05   # σ above this → hallucination
    low_conf_gate:  float = 0.25   # hard confidence floor
    imgsz:          int   = 640


# ── Result ─────────────────────────────────────────────────────────────────────
@dataclass
class GuardResult:
    verdict:          str
    mean_confidence:  float
    uncertainty:      float
    is_hallucination: bool
    n_detections:     int
    passes_used:      int
    boxes:            list = field(default_factory=list)
    classes:          list = field(default_factory=list)
    confidences:      list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict":          self.verdict,
            "mean_confidence":  self.mean_confidence,
            "uncertainty":      self.uncertainty,
            "is_hallucination": self.is_hallucination,
            "n_detections":     self.n_detections,
            "passes_used":      self.passes_used,
            "boxes":            self.boxes,
            "classes":          self.classes,
            "confidences":      self.confidences,
        }


# ── Detector ───────────────────────────────────────────────────────────────────
class HallucinationDetector:
    """
    Two-mode uncertainty estimator:
      • MC Dropout  — if dropout layers found in model (C2f surgical injection)
      • Confidence  — fallback: flags raw conf < low_conf_gate
    """

    _lock = threading.Lock()

    def __init__(
        self,
        yolo_model: YOLO,
        cfg: GuardConfig = GuardConfig(),
    ) -> None:
        self.model = yolo_model
        self.cfg   = cfg
        self._has_dropout = self._activate_dropout()
        mode = "MC Dropout ✅" if self._has_dropout else "Confidence-based ⚠️  (retrain with dropout=0.1 for MC)"
        print(f"[Guard] Mode     : {mode}")
        print(f"[Guard] Passes   : {self.cfg.n_passes}")
        print(f"[Guard] Threshold: {self.cfg.threshold}")

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _activate_dropout(self) -> bool:
        """Set dropout layers to train mode so they are stochastic at inference."""
        target = self.model.model if hasattr(self.model, "model") else self.model
        found  = False
        for m in target.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
                found = True
        # Freeze BN so stats stay stable while dropout is active
        for m in target.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        return found

    # ── Single inference pass ──────────────────────────────────────────────────
    @torch.no_grad()
    def _single_pass(self, source: str) -> tuple[np.ndarray, list, list, list]:
        results = self.model(source, imgsz=self.cfg.imgsz, verbose=False)
        confs, boxes, classes = [], [], []
        for r in results:
            if r.boxes is not None and len(r.boxes):
                c = r.boxes.conf.cpu().numpy().tolist()
                b = r.boxes.xyxy.cpu().numpy().tolist()
                cl = r.boxes.cls.cpu().numpy().astype(int).tolist()
                confs.extend(c)
                boxes.extend(b)
                classes.extend(cl)
        arr = np.array(confs) if confs else np.array([0.0])
        return arr, boxes, classes, confs

    # ── Fast gate (no passes needed) ──────────────────────────────────────────
    def _quick_check(self, conf: float) -> Optional[GuardResult]:
        if conf < self.cfg.low_conf_gate:
            return GuardResult(
                verdict          = "⚠️  Low confidence — possible hallucination",
                mean_confidence  = round(conf, 4),
                uncertainty      = 1.0,
                is_hallucination = True,
                n_detections     = 0,
                passes_used      = 0,
            )
        return None  # pass through to full MC analysis

    # ── Full MC analysis ───────────────────────────────────────────────────────
    def _mc_analyze(self, source: str) -> GuardResult:
        with self._lock:
            pass_means, last_boxes, last_classes, last_confs = [], [], [], []
            for i in range(self.cfg.n_passes):
                arr, boxes, classes, confs = self._single_pass(source)
                pass_means.append(float(np.mean(arr)))
                if i == self.cfg.n_passes - 1:          # keep last pass detections
                    last_boxes, last_classes, last_confs = boxes, classes, confs

            arr   = np.array(pass_means)
            mu    = float(np.mean(arr))
            sigma = float(np.std(arr))
            is_h  = sigma > self.cfg.threshold
            verdict = "⚠️  Uncertain — possible hallucination" if is_h else "✅ Confident prediction"

            return GuardResult(
                verdict          = verdict,
                mean_confidence  = round(mu,    4),
                uncertainty      = round(sigma, 4),
                is_hallucination = is_h,
                n_detections     = len(last_boxes),
                passes_used      = self.cfg.n_passes,
                boxes            = last_boxes,
                classes          = [CLASS_NAMES[c] for c in last_classes],
                confidences      = [round(c, 4) for c in last_confs],
            )

    # ── Public predict ─────────────────────────────────────────────────────────
    def predict(self, source: str) -> GuardResult:
        """
        Full pipeline:
          1. Single pass to get raw confidence
          2. Quick-check gate (low_conf_gate)
          3. MC analysis if gate passes
        """
        raw_arr, _, _, _ = self._single_pass(source)
        raw_conf = float(np.mean(raw_arr))

        quick = self._quick_check(raw_conf)
        if quick is not None:
            return quick

        return self._mc_analyze(source)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(model_path: Optional[str] = None) -> YOLO:
    """
    Load model from local path or HuggingFace repo.
    Priority: local path > HF download
    """
    if model_path and Path(model_path).exists():
        print(f"[Guard] Loading from local: {model_path}")
        return YOLO(model_path)

    print(f"[Guard] Downloading from HuggingFace: {HF_REPO}/{HF_FILE}")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    return YOLO(path)


def build_detector(
    model_path: Optional[str] = None,
    cfg: GuardConfig = GuardConfig(),
) -> tuple[YOLO, HallucinationDetector]:
    model    = load_model(model_path)
    detector = HallucinationDetector(model, cfg=cfg)
    # Force MC mode if C2f injection was done externally
    detector._has_dropout = True
    return model, detector