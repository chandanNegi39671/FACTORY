from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from ml.rag import rag_verifier
import json

router = APIRouter(prefix="/iot")

class IoTRecord(BaseModel):
    part_id: str
    timestamp: str
    torque_delta: float
    vibration_g: float
    anomaly_flag: bool
    context: str

@router.post("/ingest")
async def ingest_iot_data(records: List[IoTRecord]):
    """
    Ingest real factory IoT sensor data to populate the FAISS index for the RAG Guard.
    """
    # For MVP/Phase 2, we update the local JSON and rebuild the FAISS index.
    # In production, this would write to PostgreSQL and trigger an async FAISS rebuild.
    
    new_data = [record.dict() for record in records]
    
    # Read existing
    try:
        with open(rag_verifier.logs_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
        
    # Append and save
    existing_data.extend(new_data)
    
    with open(rag_verifier.logs_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
        
    # Rebuild index dynamically
    rag_verifier._build_index()
    
    return {"message": f"Successfully ingested {len(records)} records into the FAISS index."}
