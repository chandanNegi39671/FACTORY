from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date
from backend.db.database import get_db
from backend.db.models import Detection
from backend.auth.jwt import get_factory_id

router = APIRouter()

# Hardcoded part cost for MVP/Phase 1
AVG_PART_COST = 500

@router.get("/roi")
async def get_roi(factory_id: str = Depends(get_factory_id), db: Session = Depends(get_db)):
    """
    Get ROI metrics, securely scoped to the user's factory.
    """
    today = date.today()
    
    # Base query scoped to factory
    base_query = db.query(Detection).filter(
        Detection.factory_id == factory_id,
        func.date(Detection.timestamp) == today
    )
    
    # Count false positives caught (flagged_review that would have been scrapped)
    false_positives_caught = base_query.filter(
        Detection.status == 'flagged_review'
    ).count()
    
    # Calculate savings
    savings = false_positives_caught * AVG_PART_COST
    
    # Get total inspections today
    total_inspected = base_query.count()
    
    # Get confirmed defects today
    defects_caught = base_query.filter(
        Detection.status == 'verified_defect'
    ).count()

    return {
        "factory_id": factory_id,
        "date": today.isoformat(),
        "total_inspected": total_inspected,
        "defects_caught": defects_caught,
        "false_positives_caught": false_positives_caught,
        "estimated_savings_inr": savings
    }
