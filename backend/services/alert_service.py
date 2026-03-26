import time
from twilio.rest import Client
from backend.config import settings

class AlertService:
    def __init__(self):
        self.client = None
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
            self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
            
        # Format: {"factory_id_line_id": last_alert_timestamp}
        self.last_alert_times = {}
        self.throttle_seconds = 30

    def should_throttle(self, factory_id: str, line_id: str) -> bool:
        key = f"{factory_id}_{line_id}"
        now = time.time()
        
        if key in self.last_alert_times:
            time_since_last = now - self.last_alert_times[key]
            if time_since_last < self.throttle_seconds:
                return True # Throttle active
                
        self.last_alert_times[key] = now
        return False

    def send_whatsapp(self, factory_id: str, line_id: str, to_number: str, result: dict):
        if not self.client:
            print("Twilio credentials not configured. Skipping WhatsApp alert.")
            return False
            
        if self.should_throttle(factory_id, line_id):
            print(f"Alert throttled for {factory_id} / {line_id}")
            return False

        defect_type = result.get("defect_type", "Unknown")
        confidence = result.get("confidence", 0)
        
        message_body = (
            f"🚨 *DEFECT ALERT*\n\n"
            f"*Factory:* {factory_id}\n"
            f"*Line:* {line_id}\n"
            f"*Defect Type:* {defect_type}\n"
            f"*Confidence:* {confidence:.1%}\n\n"
            f"Please inspect the line immediately."
        )

        try:
            message = self.client.messages.create(
                from_=settings.TWILIO_WHATSAPP_FROM,
                body=message_body,
                to=f"whatsapp:{to_number}"
            )
            print(f"WhatsApp alert sent. SID: {message.sid}")
            return True
        except Exception as e:
            print(f"Failed to send WhatsApp alert: {e}")
            return False

alert_service = AlertService()
