from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from backend.db.database import get_db
from backend.db.models import Detection
from backend.auth.jwt import get_factory_id
from backend.config import settings

router = APIRouter()

@router.get("/defects")
async def get_defects(
    line_id: Optional[str] = None,
    status: Optional[str] = None,
    defect_type: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    factory_id: str = Depends(get_factory_id),
    db: Session = Depends(get_db)
):
    """
    Get paginated defect history, strictly scoped to the user's factory_id.
    """
    query = db.query(Detection).filter(Detection.factory_id == factory_id)
    
    if line_id:
        query = query.filter(Detection.line_id == line_id)
    if status:
        query = query.filter(Detection.status == status)
    if defect_type:
        query = query.filter(Detection.defect_type == defect_type)
        
    total = query.count()
    detections = query.order_by(Detection.timestamp.desc()).offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": [
            {
                "id": d.id,
                "part_id": d.part_id,
                "line_id": d.line_id,
                "defect_type": d.defect_type,
                "confidence": d.confidence,
                "status": d.status,
                "image_url": f"{settings.BASE_URL}/{d.image_path}" if d.image_path else None,
                "timestamp": d.timestamp
            } for d in detections
        ]
    }
