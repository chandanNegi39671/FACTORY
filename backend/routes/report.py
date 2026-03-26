from fastapi import APIRouter, Depends, BackgroundTasks
from backend.auth.jwt import get_factory_id
from backend.tasks.shift_report import generate_and_send_report

router = APIRouter()

@router.post("/report/shift")
async def trigger_shift_report(
    background_tasks: BackgroundTasks,
    factory_id: str = Depends(get_factory_id)
):
    """
    Manually trigger end-of-shift report generation and WhatsApp delivery.
    """
    # Run the celery task asynchronously
    generate_and_send_report.delay(factory_id)
    
    return {"message": f"Shift report generation triggered for factory {factory_id}. You will receive a WhatsApp message shortly."}
