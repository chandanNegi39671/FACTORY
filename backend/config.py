import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model Settings
    MODEL_PATH: str = "ml/models/best.pt"
    MC_PASSES: int = 10
    UNCERTAINTY_THRESHOLD: float = 0.3

    # Database
    DATABASE_URL: str = "sqlite:///./defects.db"
    
    # Storage
    UPLOAD_DIR: str = "uploads"
    BASE_URL: str = "http://localhost:8000"

    # Auth
    JWT_SECRET: str = "changeme_in_production"
    JWT_EXPIRE_HOURS: int = 24

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_WHATSAPP_FROM: str = "whatsapp:+14155238886"

    # Tasks (Celery)
    # Using filesystem/SQLAlchemy for Windows setup without Docker
    # In production, swap to Redis: redis://localhost:6379 
    CELERY_BROKER_URL: str = "sqla+sqlite:///./celery_broker.sqlite"
    CELERY_RESULT_BACKEND: str = "db+sqlite:///./celery_results.sqlite"

    class Config:
        env_file = ".env"

settings = Settings()
