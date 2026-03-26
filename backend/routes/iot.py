"""
routes/iot.py
-------------
POST /iot/ingest — ingest real IoT sensor data into RAG index.
FIX: correct import (rag_verifier), auth added, _build_index() call fixed.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
import json
import logging

from ml.rag import rag_verifier
from backend.auth.jwt import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/iot")


class IoTRecord(BaseModel):
    id:             str   = Field(..., description="Unique log ID")
    timestamp:      str
    defect_class:   str   = Field(..., description="Defect class name")
    confidence:     float = Field(..., ge=0.0, le=1.0)
    sensor:         str
    temperature_c:  float
    humidity_pct:   float
    action:         str   = Field(..., description="scrap | manual_review | pass")
    notes:          str   = ""


@router.post("/ingest")
async def ingest_iot_data(
    records: List[IoTRecord],
    # FIX: auth required — only authenticated users can poison the RAG index
    current_user: dict = Depends(get_current_user),
):
    """
    Ingest real factory IoT sensor data to populate the FAISS index for the RAG Guard.
    Requires valid JWT token.

    FIX 1: auth added — unauthenticated users cannot write to RAG index
    FIX 2: rag_verifier imported correctly (was missing from old rag.py)
    FIX 3: _build_index() called correctly (was missing method in old rag.py)
    """
    if not records:
        raise HTTPException(status_code=400, detail="No records provided")

    if len(records) > 500:
        raise HTTPException(status_code=400, detail="Max 500 records per batch")

    new_data = [r.dict() for r in records]

    # Read existing logs
    try:
        with open(rag_verifier.logs_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # Dedup by id
    existing_ids = {entry.get("id") for entry in existing_data}
    added = [r for r in new_data if r.get("id") not in existing_ids]

    existing_data.extend(added)

    with open(rag_verifier.logs_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    # FIX: _build_index() now exists in RAGVerifier
    rag_verifier._build_index()

    logger.info("[iot] Ingested %d new records by user=%s", len(added), current_user.get("username"))

    return {
        "message":        f"Ingested {len(added)} new records (skipped {len(new_data) - len(added)} duplicates).",
        "total_in_index": len(existing_data),
    }