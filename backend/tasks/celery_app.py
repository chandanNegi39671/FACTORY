from celery import Celery
from backend.config import settings

celery_app = Celery(
    "factory_defect_predictor",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
)

# Schedule for end-of-shift reports
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'generate-shift-report-morning': {
        'task': 'backend.tasks.shift_report.generate_and_send_report',
        'schedule': crontab(hour=8, minute=0), # End of night shift
    },
    'generate-shift-report-evening': {
        'task': 'backend.tasks.shift_report.generate_and_send_report',
        'schedule': crontab(hour=16, minute=0), # End of morning shift
    },
    'generate-shift-report-night': {
        'task': 'backend.tasks.shift_report.generate_and_send_report',
        'schedule': crontab(hour=0, minute=0), # End of evening shift
    },
    'nightly-lora-retrain': {
        'task': 'backend.tasks.retrain.nightly_retrain',
        'schedule': crontab(hour=2, minute=0), # Run at 2 AM
    },
    'data-retention-cleanup': {
        'task': 'backend.tasks.retention.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0), # Run at 3 AM
    }
}
