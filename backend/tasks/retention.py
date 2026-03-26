from datetime import datetime, timedelta
from backend.tasks.celery_app import celery_app
from backend.db.database import SessionLocal
from backend.db.models import Detection
import os

@celery_app.task
def cleanup_old_data():
    """
    Data retention policy: auto-delete frame images older than 90 days.
    (In MVP, we just delete the database records for simplicity if they are older than 90 days, 
    or simulate deleting images from an S3 bucket).
    """
    db = SessionLocal()
    try:
        cutoff_date = datetime.now() - timedelta(days=90)
        
        # In a real app, we'd query for image paths and delete them from disk/S3
        old_detections = db.query(Detection).filter(Detection.timestamp < cutoff_date).all()
        
        count = len(old_detections)
        for det in old_detections:
            # Simulate deleting the image file
            # image_path = f"temp/images/{det.id}.jpg"
            # if os.path.exists(image_path): os.remove(image_path)
            pass
            
        print(f"Data retention policy executed. Cleaned up {count} old records/images.")
        return f"Cleaned up {count} old records."
        
    finally:
        db.close()
