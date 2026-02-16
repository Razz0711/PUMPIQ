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
    APP_BASE_URL       – e.g. https://pumpiq.com

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


# ── SMTP Config (lazy — reads env at call time so load_dotenv() works) ──

def _cfg(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def is_configured() -> bool:
    """Check if SMTP credentials are set."""
    return bool(_cfg("SMTP_HOST") and _cfg("SMTP_EMAIL") and _cfg("SMTP_PASSWORD"))


def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Send an email via SMTP. Returns True on success."""
    if not is_configured():
        logger.warning("SMTP not configured — skipping email to %s", to_email)
        return False

    smtp_host = _cfg("SMTP_HOST")
    smtp_port = int(_cfg("SMTP_PORT", "587"))
    smtp_email = _cfg("SMTP_EMAIL")
    smtp_password = _cfg("SMTP_PASSWORD")
    smtp_from_name = _cfg("SMTP_FROM_NAME", "PumpIQ")
    use_tls = _cfg("SMTP_USE_TLS", "true").lower() == "true"

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{smtp_from_name} <{smtp_email}>"
    msg["To"] = to_email
    msg["Subject"] = subject

    # Plain-text fallback
    plain = html_body.replace("<br>", "\n").replace("</p>", "\n")
    import re
    plain = re.sub(r"<[^>]+>", "", plain)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
            server.ehlo()
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10)

        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, to_email, msg.as_string())
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
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")
    verify_url = f"{base_url}/verify-email?token={token}"
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


def send_registration_email(to_email: str, username: str, password: str) -> bool:
    """Send professional registration confirmation with credentials."""
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")
    masked_pw = password[:2] + "*" * (len(password) - 3) + password[-1] if len(password) > 3 else "***"
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Dear <strong>{username}</strong>,<br><br>
        Thank you for registering with <strong>PumpIQ</strong> &mdash; your smart crypto intelligence platform.
        Your account has been successfully created.
    </p>

    <div style="background: #1a1a2e; border-radius: 12px; padding: 24px; margin: 24px 0; border: 1px solid #2a2a3a; overflow: hidden;">
        <p style="color: #7c5cff; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 16px;">Your Account Details</p>
        <table style="width: 100%; border-collapse: collapse; table-layout: fixed;">
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; width: 90px;">Username</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; font-weight: 600; text-align: right; word-break: break-all; overflow-wrap: break-word;">{username}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a; width: 90px;">Email</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a; word-break: break-all; overflow-wrap: break-word;">{to_email}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a; width: 90px;">Password</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a; word-break: break-all; overflow-wrap: break-word;">{masked_pw}</td>
            </tr>
        </table>
    </div>

    <p style="color: #aaa; font-size: 13px; line-height: 1.6;">
        For your security, we recommend keeping your credentials safe and not sharing them with anyone.
    </p>

    <div style="text-align: center; margin: 28px 0;">
        <a href="{base_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            Go to PumpIQ
        </a>
    </div>

    <p style="color: #888; font-size: 13px; line-height: 1.6;">
        If you did not create this account, please disregard this email or
        <a href="mailto:{_cfg('SMTP_EMAIL')}" style="color: #7c5cff;">contact support</a>.
    </p>
    """
    return _send_email(
        to_email,
        "Welcome to PumpIQ \u2014 Registration Successful \U0001f680",
        _base_template("Registration Successful", content),
    )


def send_password_reset_email(to_email: str, username: str, token: str) -> bool:
    """Send password reset link."""
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")
    reset_url = f"{base_url}/reset-password?token={token}"
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
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")
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
        <a href="{base_url}"
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
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")
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
        <a href="{base_url}"
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


def send_login_alert_email(to_email: str, username: str, ip_address: str, user_agent: str) -> bool:
    """Send a security alert when someone logs in."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%B %d, %Y at %I:%M %p UTC")

    # Parse browser/OS from user-agent for a cleaner display
    device = "Unknown device"
    if "Windows" in user_agent:
        device = "Windows PC"
    elif "Macintosh" in user_agent or "Mac OS" in user_agent:
        device = "Mac"
    elif "iPhone" in user_agent:
        device = "iPhone"
    elif "iPad" in user_agent:
        device = "iPad"
    elif "Android" in user_agent:
        device = "Android device"
    elif "Linux" in user_agent:
        device = "Linux PC"

    browser = "Unknown browser"
    if "Chrome" in user_agent and "Edg" not in user_agent:
        browser = "Chrome"
    elif "Firefox" in user_agent:
        browser = "Firefox"
    elif "Safari" in user_agent and "Chrome" not in user_agent:
        browser = "Safari"
    elif "Edg" in user_agent:
        browser = "Microsoft Edge"
    elif "Opera" in user_agent or "OPR" in user_agent:
        browser = "Opera"

    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Dear <strong>{username}</strong>,<br><br>
        We detected a new login to your PumpIQ account. If this was you, no action is needed.
    </p>

    <div style="background: #1a1a2e; border-radius: 12px; padding: 24px; margin: 24px 0; border: 1px solid #2a2a3a;">
        <p style="color: #ff9900; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 16px;">
            &#x1F6E1; Login Activity
        </p>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px;">Date &amp; Time</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; text-align: right;">{now}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">IP Address</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a;">{ip_address}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Device</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">{device}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Browser</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">{browser}</td>
            </tr>
        </table>
    </div>

    <p style="color: #ff6b6b; font-size: 13px; line-height: 1.6; background: #2a1a1a; padding: 12px 16px; border-radius: 8px; border: 1px solid #3a2020;">
        &#x26A0;&#xFE0F; <strong>If this wasn't you</strong>, please change your password immediately and
        <a href="mailto:{_cfg('SMTP_EMAIL')}" style="color: #7c5cff;">contact support</a>.
    </p>
    """
    return _send_email(
        to_email,
        f"\U0001F6E1 PumpIQ Security Alert — New Login Detected",
        _base_template("New Login Detected", content),
    )


# ── Trade Notification Emails ───────────────────────────────────

def send_trade_email(
    to_email: str,
    username: str,
    action: str,
    symbol: str,
    coin_name: str,
    price: float,
    quantity: float,
    amount: float,
    ai_reasoning: str,
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    wallet_balance: float = 0.0,
) -> bool:
    """Send a detailed trade notification email for every BUY or SELL."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%B %d, %Y at %I:%M %p UTC")
    base_url = _cfg("APP_BASE_URL", "https://pumpiq.com")

    is_buy = action.upper() == "BUY"
    action_label = "BUY" if is_buy else "SELL"
    action_color = "#10b981" if is_buy else "#ef4444"
    action_emoji = "\U0001f7e2" if is_buy else "\U0001f534"
    action_icon = "\U0001f4c8" if is_buy else "\U0001f4c9"

    # Build trade details rows
    details_rows = f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px;">Action</td>
            <td style="color: {action_color}; padding: 10px 0; font-size: 14px; font-weight: 700; text-align: right;">
                {action_emoji} {action_label}
            </td>
        </tr>
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Coin</td>
            <td style="color: #fff; padding: 10px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a;">
                {symbol.upper()} <span style="color:#888;font-weight:400">({coin_name})</span>
            </td>
        </tr>
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Price</td>
            <td style="color: #fff; padding: 10px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a;">
                ${price:,.6f}
            </td>
        </tr>
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Quantity</td>
            <td style="color: #fff; padding: 10px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">
                {quantity:,.6f}
            </td>
        </tr>
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Total Amount</td>
            <td style="color: #fff; padding: 10px 0; font-size: 16px; font-weight: 700; text-align: right; border-top: 1px solid #2a2a3a;">
                ${amount:,.2f}
            </td>
        </tr>
    """

    # For SELL, add P&L row
    if not is_buy:
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        pnl_sign = "+" if pnl >= 0 else ""
        details_rows += f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Profit / Loss</td>
            <td style="color: {pnl_color}; padding: 10px 0; font-size: 16px; font-weight: 700; text-align: right; border-top: 1px solid #2a2a3a;">
                {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)
            </td>
        </tr>
        """

    # For BUY, add stop-loss & take-profit
    if is_buy and stop_loss > 0:
        details_rows += f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Stop Loss</td>
            <td style="color: #ef4444; padding: 10px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">
                ${stop_loss:,.6f}
            </td>
        </tr>
        """
    if is_buy and take_profit > 0:
        details_rows += f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Take Profit</td>
            <td style="color: #10b981; padding: 10px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">
                ${take_profit:,.6f}
            </td>
        </tr>
        """

    # Wallet balance row
    details_rows += f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Wallet Balance</td>
            <td style="color: #7c5cff; padding: 10px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a;">
                ${wallet_balance:,.2f}
            </td>
        </tr>
    """

    # Date row
    details_rows += f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">Date &amp; Time</td>
            <td style="color: #ccc; padding: 10px 0; font-size: 13px; text-align: right; border-top: 1px solid #2a2a3a;">{now}</td>
        </tr>
    """

    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Your PumpIQ Auto Trader has executed a <strong style="color:{action_color}">{action_label}</strong> order.
        Here are the details:
    </p>

    <div style="background: #1a1a2e; border-radius: 12px; padding: 24px; margin: 24px 0; border: 1px solid #2a2a3a;">
        <p style="color: {action_color}; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 16px;">
            {action_icon} Trade Executed &mdash; {action_label}
        </p>
        <table style="width: 100%; border-collapse: collapse;">
            {details_rows}
        </table>
    </div>

    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 20px 0; border: 1px solid #2a2a3a;">
        <p style="color: #7c5cff; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px;">
            \U0001f916 AI Analysis &amp; Reasoning
        </p>
        <p style="color: #ccc; font-size: 14px; line-height: 1.7; margin: 0;">
            {ai_reasoning}
        </p>
    </div>

    <div style="text-align: center; margin: 28px 0;">
        <a href="{base_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            View Auto Trader Dashboard
        </a>
    </div>

    <p style="color: #888; font-size: 12px; line-height: 1.6; text-align: center;">
        This trade was executed automatically by PumpIQ Auto Trader.<br>
        You can manage your settings or turn off auto-trading from your dashboard.
    </p>
    """

    pnl_str = f" | P&L: {'+' if pnl >= 0 else ''}{pnl_pct:.1f}%" if not is_buy else ""
    subject = f"{action_emoji} PumpIQ Auto Trade: {action_label} {symbol.upper()} @ ${price:,.4f}{pnl_str}"

    return _send_email(to_email, subject, _base_template(f"Trade Executed — {action_label} {symbol.upper()}", content))
