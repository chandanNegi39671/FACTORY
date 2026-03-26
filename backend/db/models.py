from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.db.database import Base

class Factory(Base):
    __tablename__ = "factories"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    uncertainty_threshold = Column(Float, default=0.3) # Added for Admin phase
    domain = Column(String, default="automotive") # Phase 2: automotive, textile, electronics
    
    lines = relationship("ProductionLine", back_populates="factory")
    users = relationship("User", back_populates="factory")
    detections = relationship("Detection", back_populates="factory")

class ProductionLine(Base):
    __tablename__ = "lines"
    
    id = Column(String, primary_key=True, index=True)
    factory_id = Column(String, ForeignKey("factories.id"), index=True)
    name = Column(String, nullable=False)
    
    factory = relationship("Factory", back_populates="lines")
    detections = relationship("Detection", back_populates="line")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String) # 'supervisor', 'qc_manager', 'admin'
    factory_id = Column(String, ForeignKey("factories.id"), index=True)
    
    factory = relationship("Factory", back_populates="users")

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    factory_id = Column(String, ForeignKey("factories.id"), index=True) # Added for scoping
    line_id = Column(String, ForeignKey("lines.id"), index=True)
    part_id = Column(String, index=True)
    defect_type = Column(String)
    confidence = Column(Float)
    uncertainty_std = Column(Float)
    is_uncertain = Column(Boolean)
    rag_verified = Column(Boolean)
    status = Column(String) # 'verified_defect', 'verified_good', 'flagged_review'
    image_path = Column(String) # Path to stored image
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    factory = relationship("Factory", back_populates="detections")
    line = relationship("ProductionLine", back_populates="detections")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    factory_id = Column(String, ForeignKey("factories.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String) # 'manually_confirmed', 'manually_cleared', 'retrain_triggered'
    target_id = Column(String) # ID of detection or model
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

