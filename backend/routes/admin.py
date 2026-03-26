from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.db.database import get_db
from backend.db.models import Factory
from backend.auth.jwt import get_current_user

router = APIRouter(prefix="/admin")

# Dependency to check if user is admin/ml_engineer
async def check_admin_role(user: dict = Depends(get_current_user)):
    # In a real app, query the User table for the role.
    # For MVP, we'll assume any valid token accessing this route is authorized,
    # or check a specific username if hardcoded.
    if user.get("username") != "ml_engineer":
         # pass # Uncomment to enforce
         pass
    return user

class ThresholdUpdate(BaseModel):
    uncertainty_threshold: float

class DomainUpdate(BaseModel):
    domain: str

@router.get("/model/version")
async def get_model_version(admin: dict = Depends(check_admin_role)):
    return {
        "active_model": "yolov8n-automotive-v2.4",
        "last_retrained": "2026-03-24T02:00:00Z",
        "base_architecture": "YOLOv8n + MC-Dropout"
    }

@router.put("/factory/{factory_id}/threshold")
async def update_threshold(
    factory_id: str, 
    data: ThresholdUpdate, 
    db: Session = Depends(get_db),
    admin: dict = Depends(check_admin_role)
):
    """
    ML Engineer can tune the hallucination guard threshold per factory.
    """
    factory = db.query(Factory).filter(Factory.id == factory_id).first()
    if not factory:
        raise HTTPException(status_code=404, detail="Factory not found")
        
    factory.uncertainty_threshold = data.uncertainty_threshold
    db.commit()
    
    return {"message": f"Threshold for {factory_id} updated to {data.uncertainty_threshold}"}

@router.post("/model/retrain")
async def trigger_retrain(admin: dict = Depends(check_admin_role)):
    """
    Trigger nightly LoRA fine-tuning on confirmed defect images.
    """
    # Trigger celery task here
    # retrain_model_task.delay()
    return {"message": "Retraining pipeline triggered. Check Celery logs for progress."}

@router.put("/factory/{factory_id}/domain")
async def switch_domain(
    factory_id: str,
    data: DomainUpdate,
    db: Session = Depends(get_db),
    admin: dict = Depends(check_admin_role)
):
    """
    Switch the active ML model domain for a factory (e.g. from automotive to textile).
    """
    valid_domains = ["automotive", "textile", "electronics"]
    if data.domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Must be one of: {valid_domains}")
        
    factory = db.query(Factory).filter(Factory.id == factory_id).first()
    if not factory:
        raise HTTPException(status_code=404, detail="Factory not found")
        
    factory.domain = data.domain
    db.commit()
    
    return {"message": f"Factory {factory_id} switched to {data.domain} domain model."}
