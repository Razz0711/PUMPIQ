"""
PumpIQ SMTP Email Service
============================
Handles email verification, password reset, and notifications via SMTP.
Supports Gmail, Outlook, SendGrid, or any SMTP server.

Required env vars:
    SMTP_HOST          – e.g. smtp.gmail.com
    SMTP_PORT          – e.g. 587
    SMTP_EMAIL         – sender email address
    SMTP_PASSWORD      – sender password or app password
    SMTP_USE_TLS       – true/false (default true)
    APP_BASE_URL       – e.g. http://localhost:8000

Optional:
    SMTP_FROM_NAME     – display name (default "PumpIQ")
"""

from __future__ import annotations

import os
import smtplib
import secrets
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── SMTP Config ─────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "PumpIQ")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")


def is_configured() -> bool:
    """Check if SMTP credentials are set."""
    return bool(SMTP_HOST and SMTP_EMAIL and SMTP_PASSWORD)


def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    if not is_configured():
        logger.warning("SMTP not configured — skipping email to %s", to_email)
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = subject

    # Plain-text fallback
    plain = html_body.replace("<br>", "\n").replace("</p>", "\n")
    import re
    plain = re.sub(r"<[^>]+>", "", plain)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        if SMTP_USE_TLS:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10)

        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, msg.as_string())
        server.quit()
        logger.info("Email sent to %s: %s", to_email, subject)
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to_email, e)
        return False


# ── Token Generation ────────────────────────────────────────────

def generate_verification_token() -> str:
    """Generate a 64-char hex token for email verification."""
    return secrets.token_hex(32)


def generate_reset_token() -> str:
    """Generate a 64-char hex token for password reset."""
    return secrets.token_hex(32)


# ── Email Templates ─────────────────────────────────────────────

def _base_template(title: str, content: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                 background: #0a0a0f; color: #e0e0e0; padding: 40px 20px;">
        <div style="max-width: 500px; margin: 0 auto; background: #14141f; border-radius: 16px;
                    padding: 40px; border: 1px solid #2a2a3a;">
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #7c5cff; font-size: 28px; margin: 0;">🚀 PumpIQ</h1>
                <p style="color: #888; font-size: 14px; margin-top: 4px;">Smart Crypto Intelligence</p>
            </div>
            <h2 style="color: #fff; font-size: 20px; margin-bottom: 16px;">{title}</h2>
            {content}
            <hr style="border: none; border-top: 1px solid #2a2a3a; margin: 30px 0 16px;">
            <p style="color: #666; font-size: 11px; text-align: center;">
                &copy; PumpIQ — This is an automated email. Do not reply.
            </p>
        </div>
    </body>
    </html>
    """


def send_verification_email(to_email: str, username: str, token: str) -> bool:
    """Send email verification link."""
    verify_url = f"{APP_BASE_URL}/verify-email?token={token}"
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Welcome to PumpIQ! Please verify your email address to activate your account.
    </p>
    <div style="text-align: center; margin: 28px 0;">
        <a href="{verify_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            Verify Email Address
        </a>
    </div>
    <p style="color: #888; font-size: 13px;">
        Or copy this link into your browser:<br>
        <span style="color: #7c5cff; word-break: break-all;">{verify_url}</span>
    </p>
    <p style="color: #888; font-size: 12px;">This link expires in 24 hours.</p>
    """
    return _send_email(to_email, "Verify your PumpIQ email", _base_template("Verify Your Email", content))


def send_password_reset_email(to_email: str, username: str, token: str) -> bool:
    """Send password reset link."""
    reset_url = f"{APP_BASE_URL}/reset-password?token={token}"
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        We received a request to reset your password. Click the button below to set a new one.
    </p>
    <div style="text-align: center; margin: 28px 0;">
        <a href="{reset_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            Reset Password
        </a>
    </div>
    <p style="color: #888; font-size: 13px;">
        Or copy this link into your browser:<br>
        <span style="color: #7c5cff; word-break: break-all;">{reset_url}</span>
    </p>
    <p style="color: #888; font-size: 12px;">This link expires in 1 hour. If you didn't request this, ignore this email.</p>
    """
    return _send_email(to_email, "Reset your PumpIQ password", _base_template("Reset Your Password", content))


def send_welcome_email(to_email: str, username: str) -> bool:
    """Send welcome email after verification."""
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Your email is verified and your PumpIQ account is fully active! 🎉
    </p>
    <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0;">
        <p style="color: #aaa; margin: 0 0 10px;">Here's what you can do now:</p>
        <ul style="color: #ccc; padding-left: 20px; line-height: 2;">
            <li>🔗 Connect your crypto wallets</li>
            <li>📊 Get AI-powered token analysis</li>
            <li>⭐ Build your watchlist</li>
            <li>🔔 Set price & token alerts</li>
        </ul>
    </div>
    <div style="text-align: center; margin: 20px 0;">
        <a href="{APP_BASE_URL}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            Go to PumpIQ
        </a>
    </div>
    """
    return _send_email(to_email, "Welcome to PumpIQ! 🚀", _base_template("Welcome to PumpIQ!", content))


def send_price_alert_email(to_email: str, username: str, coin_name: str, symbol: str,
                           price: float, alert_type: str, threshold: float) -> bool:
    """Send a price alert notification."""
    direction = "above" if alert_type == "above" else "below"
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Your price alert for <strong>{coin_name} ({symbol})</strong> has triggered!
    </p>
    <div style="background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 20px 0; text-align: center;">
        <p style="color: #888; margin: 0; font-size: 13px;">Current Price</p>
        <p style="color: #00d4aa; font-size: 28px; font-weight: 700; margin: 8px 0;">${price:,.6f}</p>
        <p style="color: #aaa; font-size: 13px;">
            Price went <strong>{direction}</strong> your threshold of <strong>${threshold:,.6f}</strong>
        </p>
    </div>
    <div style="text-align: center; margin: 20px 0;">
        <a href="{APP_BASE_URL}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            View on PumpIQ
        </a>
    </div>
    """
    return _send_email(
        to_email,
        f"🔔 Price Alert: {symbol} is ${price:,.6f}",
        _base_template(f"Price Alert — {symbol}", content),
    )
