from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    # HuggingFace
    HF_REPO_ID: str = "negi3961/factory-defect-guard"
    HF_FILENAME: str = "best.pt"
    HF_TOKEN: str = ""

    # Auth
    SECRET_KEY: str = "changeme"
    ALGORITHM: str = "HS256"

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

    # Celery
    CELERY_BROKER_URL: str = "sqla+sqlite:///./celery_broker.sqlite"
    CELERY_RESULT_BACKEND: str = "db+sqlite:///./celery_results.sqlite"

    class Config:
        env_file = ".env"
        extra = "ignore"        
settings = Settings()
