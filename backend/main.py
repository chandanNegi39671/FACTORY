from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes import health, predict, roi, defects, report, admin, iot
from backend.sockets.server import sio
from backend.db.database import engine, Base
import socketio
from routes.predict import router as predict_router
app.include_router(predict_router)

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Create database tables
Base.metadata.create_all(bind=engine)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="Factory Defect Predictor API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for MVP
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, tags=["predict"])
app.include_router(roi.router, tags=["roi"])
app.include_router(defects.router, tags=["defects"])
app.include_router(report.router, tags=["reports"])
app.include_router(admin.router, tags=["admin"])
app.include_router(iot.router, tags=["iot"])

# Mount static files for images
import os
from backend.config import settings
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Mount Socket.IO app
sio_app = socketio.ASGIApp(sio, other_asgi_app=app)

if __name__ == "__main__":
    import uvicorn
    # Note: run sio_app instead of app when starting the server
    uvicorn.run("backend.main:sio_app", host="0.0.0.0", port=8000, reload=True)
