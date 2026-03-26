"""
routes/admin.py
---------------
Admin + ML Engineer routes.

FIX: check_admin_role was a no-op (pass instead of raise).
     Now properly enforces 'admin' or 'ml_engineer' role from DB.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.db.database import get_db
from backend.db.models import Factory, User
from backend.auth.jwt import get_current_user

router = APIRouter(prefix="/admin")

# ── Role enforcement ──────────────────────────────────────────────────────────
async def check_admin_role(
    current_user: dict = Depends(get_current_user),
    db: Session        = Depends(get_db),
):
    """
    FIX: Previously had `pass` — any valid token could access admin routes.
    Now queries DB for actual user role.
    Allows: 'admin', 'ml_engineer'
    """
    username = current_user.get("username")
    user     = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "User not found",
        )

    allowed_roles = {"admin", "ml_engineer"}
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = f"Access denied. Required roles: {allowed_roles}. Your role: {user.role}",
        )

    return current_user


# ── Schemas ───────────────────────────────────────────────────────────────────
class ThresholdUpdate(BaseModel):
    uncertainty_threshold: float

class DomainUpdate(BaseModel):
    domain: str


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/model/version")
async def get_model_version(admin: dict = Depends(check_admin_role)):
    return {
        "active_model":      "yolov8s-factory-defect-v6-mc",
        "map50":             0.83,
        "classes":           17,
        "last_retrained":    "2026-03-26",
        "base_architecture": "YOLOv8s + MC-Dropout (C2f surgical injection)",
        "hf_repo":           "negi3961/factory-defect-guard",
    }


@router.put("/factory/{factory_id}/threshold")
async def update_threshold(
    factory_id: str,
    data:       ThresholdUpdate,
    db:         Session = Depends(get_db),
    admin:      dict    = Depends(check_admin_role),
):
    """ML Engineer can tune the hallucination guard threshold per factory."""
    if not (0.0 < data.uncertainty_threshold < 1.0):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    factory = db.query(Factory).filter(Factory.id == factory_id).first()
    if not factory:
        raise HTTPException(status_code=404, detail="Factory not found")

    factory.uncertainty_threshold = data.uncertainty_threshold
    db.commit()

    return {"message": f"Threshold for {factory_id} updated to {data.uncertainty_threshold}"}


@router.post("/model/retrain")
async def trigger_retrain(admin: dict = Depends(check_admin_role)):
    """Trigger nightly LoRA fine-tuning on confirmed defect images."""
    from backend.tasks.retrain import nightly_retrain
    nightly_retrain.delay()
    return {"message": "Retraining pipeline triggered. Check Celery logs for progress."}


@router.put("/factory/{factory_id}/domain")
async def switch_domain(
    factory_id: str,
    data:       DomainUpdate,
    db:         Session = Depends(get_db),
    admin:      dict    = Depends(check_admin_role),
):
    valid_domains = {"automotive", "textile", "electronics"}
    if data.domain not in valid_domains:
        raise HTTPException(status_code=400, detail=f"Invalid domain. Must be one of: {valid_domains}")

    factory = db.query(Factory).filter(Factory.id == factory_id).first()
    if not factory:
        raise HTTPException(status_code=404, detail="Factory not found")

    factory.domain = data.domain
    db.commit()

    return {"message": f"Factory {factory_id} switched to {data.domain} domain model."}