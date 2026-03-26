"""
ml/rag.py
---------
RAG Cross-Verifier — FAISS + sentence-transformers + IoT logs
Queries historical IoT sensor data to cross-check uncertain predictions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy imports — only load if RAG is actually used
_faiss       = None
_SentenceTransformer = None

IOT_LOGS_PATH = Path(__file__).parent / "data" / "iot_logs.json"


def _import_deps():
    global _faiss, _SentenceTransformer
    if _faiss is None:
        import faiss
        _faiss = faiss
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer


# ── Result ─────────────────────────────────────────────────────────────────────
@dataclass
class RAGResult:
    plausible:        bool
    verdict:          str
    matched_log:      Optional[dict]
    similarity_score: float
    physics_check:    str

    def to_dict(self) -> dict:
        return {
            "plausible":        self.plausible,
            "verdict":          self.verdict,
            "matched_log":      self.matched_log,
            "similarity_score": round(self.similarity_score, 4),
            "physics_check":    self.physics_check,
        }


# ── Physics rules ──────────────────────────────────────────────────────────────
def _physics_check(log: dict, defect_type: str) -> tuple[bool, str]:
    """
    Simple rule engine:
    - If torque_delta < 5% AND no vibration spike → misalignment unlikely
    - If temperature_ok AND no vibration → surface defect less likely
    """
    torque_delta = abs(log.get("torque_delta_pct", 0))
    vibration    = log.get("vibration_spike", False)
    temp_ok      = log.get("temperature_normal", True)

    misalignment_types = {"misalignment", "pitted_surface", "inclusion"}
    surface_types      = {"crazing", "scratches", "rolled_in_scale", "patches"}

    if defect_type in misalignment_types:
        if torque_delta < 5.0 and not vibration:
            return False, f"Physics FAIL: torque_delta={torque_delta:.1f}% < 5%, no vibration → {defect_type} unlikely"
        return True, f"Physics OK: torque_delta={torque_delta:.1f}%, vibration={vibration}"

    if defect_type in surface_types:
        if temp_ok and not vibration:
            return True, f"Physics OK: temp normal, no vibration → {defect_type} plausible"
        return True, "Physics OK: conditions consistent"

    # PCB / MVTec defects — no specific physics rule yet
    return True, "Physics OK: no specific rule for this class"


# ── RAG Verifier ───────────────────────────────────────────────────────────────
class RAGVerifier:
    """
    Loads iot_logs.json, builds FAISS index on first use,
    and cross-checks uncertain predictions.
    """

    def __init__(self, logs_path: str = str(IOT_LOGS_PATH), model_name: str = "all-MiniLM-L6-v2"):
        self.logs_path  = logs_path
        self.model_name = model_name
        self._logs      = []
        self._index     = None
        self._encoder   = None
        self._ready     = False

    def _load(self):
        if self._ready:
            return
        _import_deps()

        # Load IoT logs
        with open(self.logs_path, "r") as f:
            self._logs = json.load(f)

        # Build text representations for embedding
        texts = [self._log_to_text(log) for log in self._logs]

        # Encode
        self._encoder = _SentenceTransformer(self.model_name)
        embeddings    = self._encoder.encode(texts, convert_to_numpy=True)
        embeddings    = embeddings.astype("float32")

        # FAISS index
        dim          = embeddings.shape[1]
        self._index  = _faiss.IndexFlatL2(dim)
        self._index.add(embeddings)

        self._ready = True
        print(f"[RAG] Index built: {len(self._logs)} IoT logs, dim={dim}")

    def _log_to_text(self, log: dict) -> str:
        return (
            f"part_id={log.get('part_id', 'unknown')} "
            f"line={log.get('line_id', '?')} "
            f"torque_delta={log.get('torque_delta_pct', 0):.1f}pct "
            f"vibration={'yes' if log.get('vibration_spike') else 'no'} "
            f"temp={'normal' if log.get('temperature_normal', True) else 'high'} "
            f"defect_history={log.get('recent_defect_type', 'none')}"
        )

    def verify(
        self,
        defect_type: str,
        part_id: Optional[str] = None,
        line_id: Optional[str] = None,
        top_k: int = 1,
    ) -> RAGResult:
        """
        Cross-check a defect prediction against IoT history.
        Returns RAGResult with plausibility verdict.
        """
        try:
            self._load()
        except Exception as e:
            # Graceful fallback if FAISS/sentence-transformers not installed
            return RAGResult(
                plausible        = True,
                verdict          = f"RAG unavailable ({e}) — defaulting to plausible",
                matched_log      = None,
                similarity_score = 0.0,
                physics_check    = "skipped",
            )

        # Build query text
        query = (
            f"defect={defect_type} "
            f"part_id={part_id or 'unknown'} "
            f"line={line_id or 'unknown'}"
        )
        q_emb = self._encoder.encode([query], convert_to_numpy=True).astype("float32")

        distances, indices = self._index.search(q_emb, top_k)
        best_idx  = int(indices[0][0])
        best_dist = float(distances[0][0])

        # Convert L2 distance → similarity score (0-1)
        similarity = float(1 / (1 + best_dist))
        matched    = self._logs[best_idx]

        # Physics rule check
        plausible, physics_msg = _physics_check(matched, defect_type)

        verdict = (
            f"✅ RAG: {defect_type} plausible — {physics_msg}"
            if plausible
            else f"⚠️  RAG: {defect_type} unlikely — {physics_msg}"
        )

        return RAGResult(
            plausible        = plausible,
            verdict          = verdict,
            matched_log      = matched,
            similarity_score = similarity,
            physics_check    = physics_msg,
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
_verifier: Optional[RAGVerifier] = None

def get_verifier() -> RAGVerifier:
    global _verifier
    if _verifier is None:
        _verifier = RAGVerifier()
    return _verifier