"""
PumpIQ SMS Service — Twilio Verify Integration
================================================
Sends real OTP SMS to phone numbers via Twilio Verify API.
Twilio Verify handles OTP generation, delivery, and verification.

Setup:
  1. Sign up at https://www.twilio.com/
  2. Get Account SID + Auth Token from Console
  3. Create a Verify Service (Console → Verify → Services)
  4. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SID in .env
"""

import os
import logging

logger = logging.getLogger(__name__)

_client = None


def _get_config():
    """Read Twilio config from env (called at runtime, after load_dotenv)."""
    return (
        os.getenv("TWILIO_ACCOUNT_SID", ""),
        os.getenv("TWILIO_AUTH_TOKEN", ""),
        os.getenv("TWILIO_VERIFY_SID", ""),
    )


def _get_client():
    """Lazy-init Twilio client."""
    global _client
    sid, token, _ = _get_config()
    if _client is None and sid and token:
        from twilio.rest import Client
        _client = Client(sid, token)
    return _client


def is_configured() -> bool:
    """Check if Twilio Verify is properly configured."""
    sid, token, verify_sid = _get_config()
    return bool(sid and token and verify_sid)


def _format_phone(phone_number: str) -> str:
    """Format phone to E.164 (+91XXXXXXXXXX for India)."""
    clean = phone_number.strip().replace(" ", "").replace("-", "")
    if clean.startswith("+"):
        return clean
    if clean.startswith("91") and len(clean) == 12:
        return "+" + clean
    if len(clean) == 10 and clean.isdigit():
        return "+91" + clean
    return "+" + clean


def send_otp(phone_number: str) -> dict:
    """
    Send OTP via Twilio Verify. Twilio generates and sends the code itself.

    Args:
        phone_number: Phone number (10-digit Indian or E.164 format)

    Returns:
        {"success": True/False, "message": "...", "phone_last4": "..."}
    """
    if not is_configured():
        logger.warning("Twilio not configured — falling back to debug mode")
        return {
            "success": False,
            "error": "SMS service not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SID in .env",
            "fallback": True,
        }

    e164 = _format_phone(phone_number)
    phone_last4 = e164[-4:]
    _, _, verify_sid = _get_config()

    try:
        client = _get_client()
        verification = client.verify.v2 \
            .services(verify_sid) \
            .verifications \
            .create(to=e164, channel="sms")

        logger.info("Twilio Verify sent to ****%s, status: %s, sid: %s",
                     phone_last4, verification.status, verification.sid)

        return {
            "success": True,
            "message": f"OTP sent to ••••••{phone_last4}",
            "phone_last4": phone_last4,
            "status": verification.status,  # "pending"
        }

    except Exception as e:
        error_msg = str(e)
        logger.error("Twilio Verify error for ****%s: %s", phone_last4, error_msg)
        return {"success": False, "error": f"SMS delivery failed: {error_msg}"}


def verify_otp(phone_number: str, code: str) -> dict:
    """
    Verify an OTP code via Twilio Verify.

    Args:
        phone_number: Same phone the OTP was sent to
        code: 6-digit OTP code entered by user

    Returns:
        {"success": True/False, "message": "...", "status": "approved"/"pending"}
    """
    if not is_configured():
        return {
            "success": False,
            "error": "SMS service not configured",
            "fallback": True,
        }

    e164 = _format_phone(phone_number)
    _, _, verify_sid = _get_config()

    try:
        client = _get_client()
        verification_check = client.verify.v2 \
            .services(verify_sid) \
            .verification_checks \
            .create(to=e164, code=code)

        logger.info("Twilio Verify check for ****%s: status=%s",
                     e164[-4:], verification_check.status)

        if verification_check.status == "approved":
            return {"success": True, "message": "OTP verified successfully", "status": "approved"}
        else:
            return {"success": False, "error": "Invalid or expired OTP code", "status": verification_check.status}

    except Exception as e:
        error_msg = str(e)
        logger.error("Twilio Verify check error: %s", error_msg)
        return {"success": False, "error": f"Verification failed: {error_msg}"}
