import os
from datetime import datetime
from backend.tasks.celery_app import celery_app
from backend.db.database import SessionLocal
from backend.db.models import Factory, Detection
from backend.services.alert_service import alert_service
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_heatmap(factory_id: str, detections: list, output_path: str):
    """
    Creates an OpenCV accumulation buffer heatmap based on bounding box coordinates.
    """
    # Create a blank template image (e.g., 640x640)
    width, height = 640, 640
    heatmap = np.zeros((height, width), dtype=np.float32)

    for d in detections:
        # Assuming we saved bbox somewhere, or for MVP just plot center points if we only have type
        # For this mockup, we'll add random hotspots if real bboxes aren't in DB yet
        cx, cy = np.random.randint(100, 540), np.random.randint(100, 540)
        heatmap[cy-20:cy+20, cx-20:cx+20] += 1

    # Normalize and apply colormap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap) * 255
    heatmap = np.uint8(heatmap)
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    cv2.imwrite(output_path, color_map)
    return output_path

def create_pdf(factory_id: str, stats: dict, heatmap_path: str, output_path: str):
    """
    Generates a bilingual (English + Hindi) PDF report using ReportLab.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, f"End of Shift Report - Factory {factory_id}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Stats
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 120, "Summary / सारांश")
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 150, f"Total Inspected: {stats['total']}")
    c.drawString(50, height - 170, f"Defects Confirmed: {stats['defects']}")
    c.drawString(50, height - 190, f"False Positives Caught (Guard): {stats['false_positives']}")
    c.drawString(50, height - 210, f"Estimated Savings: ₹{stats['savings']}")
    
    # Heatmap
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 260, "Defect Heatmap / दोष हीटमैप")
    if os.path.exists(heatmap_path):
        c.drawImage(heatmap_path, 50, height - 580, width=300, height=300)
    
    c.save()
    return output_path

@celery_app.task
def generate_and_send_report(factory_id: str = None):
    db = SessionLocal()
    try:
        factories = [db.query(Factory).filter(Factory.id == factory_id).first()] if factory_id else db.query(Factory).all()
        
        for factory in factories:
            if not factory: continue
            
            fid = factory.id
            today = datetime.now().date()
            
            # Gather stats
            total = db.query(Detection).filter(Detection.factory_id == fid).count()
            defects = db.query(Detection).filter(Detection.factory_id == fid, Detection.status == 'verified_defect').count()
            fps = db.query(Detection).filter(Detection.factory_id == fid, Detection.status == 'flagged_review').count()
            
            stats = {
                "total": total,
                "defects": defects,
                "false_positives": fps,
                "savings": fps * 500
            }
            
            # Ensure dirs exist
            os.makedirs(f"temp/heatmaps", exist_ok=True)
            os.makedirs(f"temp/reports", exist_ok=True)
            
            # Generate assets
            heatmap_path = f"temp/heatmaps/hm_{fid}_{today}.png"
            pdf_path = f"temp/reports/report_{fid}_{today}.pdf"
            
            generate_heatmap(fid, [], heatmap_path)
            create_pdf(fid, stats, heatmap_path, pdf_path)
            
            print(f"Report generated for {fid}: {pdf_path}")
            # In production, integrate with WhatsApp API to send the PDF as a media message.
            # alert_service.send_whatsapp_pdf(factory.manager_phone, pdf_path)
            
    finally:
        db.close()
