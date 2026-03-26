"""
ml/rag.py
---------
RAG Cross-Verifier — FAISS + sentence-transformers + IoT logs
Queries historical IoT sensor data to cross-check uncertain predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_faiss               = None
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
    Rule engine based on iot_logs.json fields:
    temperature_c, humidity_pct, action, sensor
    """
    temp      = float(log.get("temperature_c", 25))
    humidity  = float(log.get("humidity_pct",  50))
    action    = log.get("action", "")

    # High temp + high humidity → crazing / patches plausible
    heat_types = {"crazing", "patches", "pitted_surface", "rolled_in_scale"}
    pcb_types  = {"mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "pcb_missing_hole",
                  "pcb_mouse_bite", "pcb_open_circuit", "pcb_short", "pcb_spur", "pcb_spurious_copper"}

    if defect_type in heat_types:
        if temp > 70 and humidity > 60:
            return True, f"Physics OK: temp={temp}°C, humidity={humidity}% — {defect_type} plausible"
        if temp < 50:
            return False, f"Physics FAIL: temp={temp}°C too low for {defect_type}"
        return True, f"Physics OK: temp/humidity within range"

    if defect_type in pcb_types:
        # PCB defects don't depend on temperature
        return True, f"Physics OK: PCB defect — no thermal constraint"

    return True, "Physics OK: no specific rule for this class"


# ── RAG Verifier ───────────────────────────────────────────────────────────────
class RAGVerifier:
    """
    Loads iot_logs.json, builds FAISS index on first use,
    and cross-checks uncertain predictions.
    """

    def __init__(
        self,
        logs_path:  str = str(IOT_LOGS_PATH),
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.logs_path  = logs_path
        self.model_name = model_name
        self._logs      = []
        self._index     = None
        self._encoder   = None
        self._ready     = False

    # ── FIX: _log_to_text now matches actual iot_logs.json fields ──────────────
    def _log_to_text(self, log: dict) -> str:
        """Convert a log entry to a searchable text string."""
        return (
            f"defect={log.get('defect_class', 'unknown')} "
            f"sensor={log.get('sensor', 'unknown')} "
            f"temperature={log.get('temperature_c', 0):.0f}c "
            f"humidity={log.get('humidity_pct', 0):.0f}pct "
            f"action={log.get('action', 'unknown')} "
            f"notes={log.get('notes', '')}"
        )

    def _load(self):
        if self._ready:
            return
        _import_deps()

        with open(self.logs_path, "r", encoding="utf-8") as f:
            self._logs = json.load(f)

        texts         = [self._log_to_text(log) for log in self._logs]
        self._encoder = _SentenceTransformer(self.model_name)
        embeddings    = self._encoder.encode(texts, convert_to_numpy=True).astype("float32")

        dim          = embeddings.shape[1]
        self._index  = _faiss.IndexFlatL2(dim)
        self._index.add(embeddings)

        self._ready = True
        logger.info("[RAG] Index built: %d IoT logs, dim=%d", len(self._logs), dim)

    # ── FIX: _build_index added — routes/iot.py calls this after ingesting ─────
    def _build_index(self):
        """Force a full index rebuild (called after new data is ingested)."""
        self._ready = False
        self._load()
        logger.info("[RAG] Index rebuilt with %d logs", len(self._logs))

    def verify(
        self,
        defect_type: str,
        part_id:    Optional[str] = None,
        line_id:    Optional[str] = None,
        top_k:      int           = 1,
    ) -> RAGResult:
        """
        Cross-check a defect prediction against IoT history.
        Returns RAGResult with plausibility verdict.
        """
        try:
            self._load()
        except Exception as exc:
            logger.warning("[RAG] Unavailable: %s — defaulting to plausible", exc)
            return RAGResult(
                plausible        = True,
                verdict          = f"RAG unavailable ({exc}) — defaulting to plausible",
                matched_log      = None,
                similarity_score = 0.0,
                physics_check    = "skipped",
            )

        query = (
            f"defect={defect_type} "
            f"sensor={line_id or 'unknown'} "
            f"notes={defect_type} defect detected"
        )
        q_emb              = self._encoder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self._index.search(q_emb, top_k)
        best_idx           = int(indices[0][0])
        best_dist          = float(distances[0][0])
        similarity         = float(1 / (1 + best_dist))
        matched            = self._logs[best_idx]

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


# ── FIX: expose rag_verifier alias — routes/iot.py imports this ───────────────
rag_verifier = get_verifier()