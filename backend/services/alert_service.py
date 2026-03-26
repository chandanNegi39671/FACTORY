"""
services/alert_service.py
--------------------------
WhatsApp alert service via Twilio.

Improvements:
1. logging instead of print()
2. Phone number sanitization — strip non-digits, validate E.164 format
3. factory_id / line_id sanitized before interpolation into message
4. throttle_seconds from settings (was hardcoded 30)
5. async-safe: blocking Twilio call wrapped in run_in_executor
6. Improved message with uncertainty info
7. send_whatsapp_media added for shift report PDF delivery
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from backend.config import settings

logger = logging.getLogger(__name__)

# Thread pool for blocking Twilio calls
_executor = ThreadPoolExecutor(max_workers=4)

# Phone number validation — E.164: + followed by 7-15 digits
_PHONE_RE = re.compile(r"^\+?[1-9]\d{6,14}$")


def _sanitize_str(value: str, max_len: int = 64) -> str:
    """Remove non-printable chars and truncate."""
    clean = re.sub(r"[^\w\s\-_.,:/()#]", "", str(value))
    return clean[:max_len]


def _validate_phone(number: str) -> str:
    """
    Strip spaces/dashes, ensure E.164 format.
    Raises ValueError if invalid.
    """
    digits_only = re.sub(r"[\s\-()]", "", number)
    if not digits_only.startswith("+"):
        digits_only = "+" + digits_only.lstrip("+")
    if not _PHONE_RE.match(digits_only):
        raise ValueError(f"Invalid phone number format: {number!r}")
    return digits_only


class AlertService:
    def __init__(self):
        self.client = None
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
            try:
                from twilio.rest import Client
                self.client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                logger.info("[alert] Twilio client initialized")
            except Exception as exc:
                logger.error("[alert] Twilio init failed: %s", exc)

        # {factory_id_line_id: last_alert_timestamp}
        self.last_alert_times: dict[str, float] = {}
        # FIX: throttle from settings or default 30s
        self.throttle_seconds: int = getattr(settings, "ALERT_THROTTLE_SECONDS", 30)

    # ── Throttle ───────────────────────────────────────────────────────────────
    def should_throttle(self, factory_id: str, line_id: str) -> bool:
        key = f"{factory_id}_{line_id}"
        now = time.time()
        if key in self.last_alert_times:
            if now - self.last_alert_times[key] < self.throttle_seconds:
                return True
        self.last_alert_times[key] = now
        return False

    # ── Sync send (runs in thread pool) ───────────────────────────────────────
    def _send_sync(self, to_number: str, body: str) -> str:
        """Blocking Twilio call — run in executor, never call from async directly."""
        message = self.client.messages.create(
            from_ = settings.TWILIO_WHATSAPP_FROM,
            body  = body,
            to    = f"whatsapp:{to_number}",
        )
        return message.sid

    # ── Async WhatsApp alert ───────────────────────────────────────────────────
    async def send_whatsapp_async(
        self,
        factory_id:  str,
        line_id:     str,
        to_number:   str,
        result:      dict,
    ) -> bool:
        """
        Async-safe WhatsApp alert.
        FIX: Twilio call runs in thread pool — does not block event loop.
        FIX: phone number validated before use.
        FIX: factory_id / line_id sanitized before message interpolation.
        """
        if not self.client:
            logger.warning("[alert] Twilio not configured — skipping alert")
            return False

        if self.should_throttle(factory_id, line_id):
            logger.info("[alert] Throttled for %s / %s", factory_id, line_id)
            return False

        # FIX: validate phone number
        try:
            safe_number = _validate_phone(to_number)
        except ValueError as exc:
            logger.error("[alert] Invalid phone number: %s", exc)
            return False

        # FIX: sanitize interpolated values
        safe_factory    = _sanitize_str(factory_id)
        safe_line       = _sanitize_str(line_id)
        defect_type     = _sanitize_str(result.get("defect_type", "Unknown"))
        confidence      = float(result.get("confidence", 0))
        uncertainty     = result.get("uncertainty", None)
        verdict         = _sanitize_str(result.get("verdict", ""))

        unc_line = f"*Uncertainty:* {uncertainty:.4f}\n" if uncertainty is not None else ""

        body = (
            f"🚨 *DEFECT ALERT*\n\n"
            f"*Factory:* {safe_factory}\n"
            f"*Line:* {safe_line}\n"
            f"*Defect:* {defect_type}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"{unc_line}"
            f"*Guard Verdict:* {verdict}\n\n"
            f"Please inspect the line immediately."
        )

        try:
            loop = asyncio.get_event_loop()
            sid  = await loop.run_in_executor(_executor, self._send_sync, safe_number, body)
            logger.info("[alert] WhatsApp sent to %s. SID: %s", safe_number, sid)
            return True
        except Exception as exc:
            logger.error("[alert] Failed to send WhatsApp alert: %s", exc)
            return False

    # ── Sync wrapper (for Celery tasks) ───────────────────────────────────────
    def send_whatsapp(
        self,
        factory_id: str,
        line_id:    str,
        to_number:  str,
        result:     dict,
    ) -> bool:
        """Sync wrapper — use from Celery tasks or non-async contexts."""
        if not self.client:
            logger.warning("[alert] Twilio not configured — skipping alert")
            return False

        if self.should_throttle(factory_id, line_id):
            logger.info("[alert] Throttled for %s / %s", factory_id, line_id)
            return False

        try:
            safe_number  = _validate_phone(to_number)
        except ValueError as exc:
            logger.error("[alert] Invalid phone number: %s", exc)
            return False

        safe_factory = _sanitize_str(factory_id)
        safe_line    = _sanitize_str(line_id)
        defect_type  = _sanitize_str(result.get("defect_type", "Unknown"))
        confidence   = float(result.get("confidence", 0))

        body = (
            f"🚨 *DEFECT ALERT*\n\n"
            f"*Factory:* {safe_factory}\n"
            f"*Line:* {safe_line}\n"
            f"*Defect:* {defect_type}\n"
            f"*Confidence:* {confidence:.1%}\n\n"
            f"Please inspect the line immediately."
        )

        try:
            sid = self._send_sync(safe_number, body)
            logger.info("[alert] WhatsApp sent to %s. SID: %s", safe_number, sid)
            return True
        except Exception as exc:
            logger.error("[alert] Failed to send WhatsApp: %s", exc)
            return False

    # ── PDF media message (for shift reports) ─────────────────────────────────
    def send_whatsapp_media(
        self,
        to_number:   str,
        media_url:   str,
        caption:     str = "End-of-shift report",
    ) -> bool:
        """Send a WhatsApp message with a media attachment (PDF report)."""
        if not self.client:
            logger.warning("[alert] Twilio not configured — skipping media alert")
            return False

        try:
            safe_number = _validate_phone(to_number)
            message = self.client.messages.create(
                from_     = settings.TWILIO_WHATSAPP_FROM,
                body      = caption,
                media_url = [media_url],
                to        = f"whatsapp:{safe_number}",
            )
            logger.info("[alert] Media message sent. SID: %s", message.sid)
            return True
        except Exception as exc:
            logger.error("[alert] Failed to send media message: %s", exc)
            return False


alert_service = AlertService()