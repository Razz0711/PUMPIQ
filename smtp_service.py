"""
NEXYPHER SMTP Email Service
============================
Handles email verification, password reset, and notifications via SMTP.
Supports Gmail, Outlook, SendGrid, or any SMTP server.

Required env vars:
    SMTP_HOST          – e.g. smtp.gmail.com
    SMTP_PORT          – e.g. 587
    SMTP_EMAIL         – sender email address
    SMTP_PASSWORD      – sender password or app password
    SMTP_USE_TLS       – true/false (default true)
    APP_BASE_URL       – e.g. https://NEXYPHER.vercel.app

Optional:
    SMTP_FROM_NAME     – display name (default "NEXYPHER")
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
    smtp_from_name = _cfg("SMTP_FROM_NAME", "NEXYPHER")
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
                <h1 style="color: #7c5cff; font-size: 28px; margin: 0;">🚀 NEXYPHER</h1>
                <p style="color: #888; font-size: 14px; margin-top: 4px;">Smart Crypto Intelligence</p>
            </div>
            <h2 style="color: #fff; font-size: 20px; margin-bottom: 16px;">{title}</h2>
            {content}
            <hr style="border: none; border-top: 1px solid #2a2a3a; margin: 30px 0 16px;">
            <p style="color: #666; font-size: 11px; text-align: center;">
                &copy; NEXYPHER — This is an automated email. Do not reply.
            </p>
        </div>
    </body>
    </html>
    """


def send_verification_email(to_email: str, username: str, token: str) -> bool:
    """Send email verification link."""
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
    verify_url = f"{base_url}/verify-email?token={token}"
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Welcome to NEXYPHER! Please verify your email address to activate your account.
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
    return _send_email(to_email, "Verify your NEXYPHER email", _base_template("Verify Your Email", content))


def send_registration_email(to_email: str, username: str) -> bool:
    """Send professional registration confirmation (no credentials included)."""
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Dear <strong>{username}</strong>,<br><br>
        Thank you for registering with <strong>NEXYPHER</strong> &mdash; your smart crypto intelligence platform.
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
        </table>
    </div>

    <p style="color: #aaa; font-size: 13px; line-height: 1.6;">
        For your security, we never include passwords in emails. If you forget your password,
        you can reset it from the login page at any time.
    </p>

    <div style="text-align: center; margin: 28px 0;">
        <a href="{base_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            Go to NEXYPHER
        </a>
    </div>

    <p style="color: #888; font-size: 13px; line-height: 1.6;">
        If you did not create this account, please disregard this email or
        <a href="mailto:{_cfg('SMTP_EMAIL')}" style="color: #7c5cff;">contact support</a>.
    </p>
    """
    return _send_email(
        to_email,
        "Welcome to NEXYPHER \u2014 Registration Successful \U0001f680",
        _base_template("Registration Successful", content),
    )


def send_password_reset_email(to_email: str, username: str, token: str) -> bool:
    """Send password reset link."""
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
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
    return _send_email(to_email, "Reset your NEXYPHER password", _base_template("Reset Your Password", content))


def send_welcome_email(to_email: str, username: str) -> bool:
    """Send welcome email after verification."""
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        Your email is verified and your NEXYPHER account is fully active! 🎉
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
            Go to NEXYPHER
        </a>
    </div>
    """
    return _send_email(to_email, "Welcome to NEXYPHER! 🚀", _base_template("Welcome to NEXYPHER!", content))


def send_price_alert_email(to_email: str, username: str, coin_name: str, symbol: str,
                           price: float, alert_type: str, threshold: float) -> bool:
    """Send a price alert notification."""
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
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
            View on NEXYPHER
        </a>
    </div>
    """
    return _send_email(
        to_email,
        f"🔔 Price Alert: {symbol} is ${price:,.6f}",
        _base_template(f"Price Alert — {symbol}", content),
    )


def _get_ip_geolocation(ip_address: str) -> dict:
    """Get approximate location from IP address using free ip-api.com."""
    try:
        import urllib.request, json
        # Skip private/localhost IPs
        if ip_address in ("127.0.0.1", "localhost", "::1", "unknown") or ip_address.startswith(("192.168.", "10.", "172.")):
            return {"city": "Local Network", "region": "", "country": "India", "isp": "Local", "query": ip_address, "lat": 0, "lon": 0, "offset": 19800, "timezone": "Asia/Kolkata"}
        url = f"http://ip-api.com/json/{ip_address}?fields=status,message,country,regionName,city,isp,query,lat,lon,timezone,offset"
        req = urllib.request.Request(url, headers={"User-Agent": "NEXYPHER/3.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "success":
                return {
                    "city": data.get("city", "Unknown"),
                    "region": data.get("regionName", ""),
                    "country": data.get("country", ""),
                    "isp": data.get("isp", ""),
                    "query": data.get("query", ip_address),
                    "lat": data.get("lat", 0),
                    "lon": data.get("lon", 0),
                    "offset": data.get("offset", 0),
                    "timezone": data.get("timezone", "UTC"),
                }
    except Exception as e:
        logger.debug("IP geolocation failed for %s: %s", ip_address, e)
    return {"city": "Unknown", "region": "", "country": "", "isp": "", "query": ip_address, "lat": 0, "lon": 0, "offset": 0, "timezone": "UTC"}


def send_login_alert_email(to_email: str, username: str, ip_address: str, user_agent: str) -> bool:
    """Send a security alert when someone logs in, including IP geolocation & DIGIPIN."""
    from datetime import datetime, timezone as tz, timedelta as td

    # Get IP geolocation
    geo = _get_ip_geolocation(ip_address)

    # Use the IP's timezone offset for accurate local time
    offset_seconds = geo.get("offset", 0)
    tz_name = geo.get("timezone", "UTC")
    local_tz = tz(td(seconds=offset_seconds))
    now = datetime.now(local_tz).strftime("%B %d, %Y at %I:%M %p") + f" ({tz_name})"

    location_parts = [p for p in [geo.get("city"), geo.get("region"), geo.get("country")] if p]
    location_str = ", ".join(location_parts) if location_parts else "Unknown location"
    isp_str = geo.get("isp", "")

    # Generate DIGIPIN for Indian locations (lat 1.5–39.0, lon 63.5–99.0)
    digipin_str = ""
    lat, lon = geo.get("lat", 0), geo.get("lon", 0)
    if 1.5 <= lat <= 39.0 and 63.5 <= lon <= 99.0:
        try:
            from digipin import encode as digipin_encode
            digipin_str = digipin_encode(lat, lon)
        except Exception as e:
            logger.debug("DIGIPIN generation failed: %s", e)

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
        We detected a new login to your NEXYPHER account. If this was you, no action is needed.
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
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">&#x1F4CD; Location</td>
                <td style="color: #00d4aa; padding: 8px 0; font-size: 14px; font-weight: 600; text-align: right; border-top: 1px solid #2a2a3a;">{location_str}</td>
            </tr>
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">ISP / Network</td>
                <td style="color: #fff; padding: 8px 0; font-size: 14px; text-align: right; border-top: 1px solid #2a2a3a;">{isp_str}</td>
            </tr>{''.join([f'''
            <tr>
                <td style="color: #888; padding: 8px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">&#x1F4EC; DIGIPIN</td>
                <td style="color: #ff9900; padding: 8px 0; font-size: 14px; font-weight: 700; text-align: right; border-top: 1px solid #2a2a3a; letter-spacing: 1.5px;">
                    <a href="https://www.indiapost.gov.in/VAS/Pages/FindDigipin.aspx" style="color: #ff9900; text-decoration: none;">{digipin_str}</a>
                </td>
            </tr>'''] if digipin_str else [])}
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
        f"\U0001F6E1 NEXYPHER Security Alert — New Login from {location_str}",
        _base_template("New Login Detected", content),
    )


# ── Trade Notification Emails ───────────────────────────────────


def _row(label: str, value: str, color: str = "#fff", bold: bool = False) -> str:
    """Helper: generate a single table row for trade email."""
    weight = "700" if bold else "400"
    return f"""
        <tr>
            <td style="color: #888; padding: 10px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">{label}</td>
            <td style="color: {color}; padding: 10px 0; font-size: 14px; font-weight: {weight}; text-align: right; border-top: 1px solid #2a2a3a;">
                {value}
            </td>
        </tr>"""


def _format_market_cap(mc: float) -> str:
    if mc >= 1e12: return f"${mc/1e12:.2f}T"
    if mc >= 1e9:  return f"${mc/1e9:.2f}B"
    if mc >= 1e6:  return f"${mc/1e6:.1f}M"
    return f"${mc:,.0f}"


def _score_bar(score: int) -> str:
    """Visual score bar 0-100."""
    fill_pct = max(0, min(100, score))
    bar_color = "#10b981" if score >= 60 else "#f59e0b" if score >= 35 else "#ef4444"
    return (
        f'<div style="background:#1e1e2e;border-radius:6px;height:10px;width:100%;margin:4px 0;">'
        f'<div style="background:{bar_color};border-radius:6px;height:10px;width:{fill_pct}%;"></div>'
        f'</div>'
        f'<span style="color:{bar_color};font-weight:700;font-size:16px;">{score}/100</span>'
    )


def _signal_section(title: str, items: list, icon: str = "📊") -> str:
    """Build a styled signal section with bullet items."""
    if not items:
        return ""
    bullets = "".join(
        f'<li style="color: #ccc; padding: 3px 0; font-size: 13px; line-height: 1.5;">{item}</li>'
        for item in items if item
    )
    return f"""
    <div style="background: #1a1a2e; border-radius: 10px; padding: 16px 20px; margin: 12px 0; border: 1px solid #2a2a3a;">
        <p style="color: #7c5cff; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px;">
            {icon} {title}
        </p>
        <ul style="margin: 0; padding-left: 18px;">{bullets}</ul>
    </div>
    """


def send_trade_email(
    to_email: str,
    username: str,
    action: str,          # BUY, SHORT, SELL, COVER
    symbol: str,
    coin_name: str,
    price: float,
    quantity: float,
    amount: float,
    ai_reasoning: str = "",
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    wallet_balance: float = 0.0,
    # ── Enhanced fields (all optional for backward compat) ──
    trade_metadata: dict = None,
    # Close-specific fields
    entry_price: float = 0.0,
    close_reason: str = "",
    hold_duration: str = "",
    side: str = "",
) -> bool:
    """Send a comprehensive trade notification email with full analysis.

    For BUY/SHORT: includes strategy, ML signals, backtest, risk params, reasoning.
    For SELL/COVER: includes close reason, hold duration, entry→exit, P&L breakdown.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%B %d, %Y at %I:%M %p UTC")
    base_url = _cfg("APP_BASE_URL", "https://NEXYPHER.vercel.app")
    meta = trade_metadata or {}

    # Determine action type
    action_upper = action.upper()
    is_entry = action_upper in ("BUY", "SHORT")
    is_short = action_upper in ("SHORT", "COVER")
    if not side:
        side = "short" if is_short else "long"

    if is_entry:
        action_label = "SHORT" if is_short else "BUY (LONG)"
        action_color = "#f59e0b" if is_short else "#10b981"
        action_emoji = "🟡" if is_short else "🟢"
        action_icon = "📉" if is_short else "📈"
        header_title = f"Trade Opened — {action_label}"
    else:
        action_label = "COVER (Close Short)" if is_short else "SELL (Close Long)"
        action_color = "#ef4444" if pnl < 0 else "#10b981"
        action_emoji = "🔴" if pnl < 0 else "🟢"
        action_icon = "📉" if pnl < 0 else "📈"
        header_title = f"Trade Closed — {'COVER' if is_short else 'SELL'}"

    # ═══════════════════════════════════════════
    # SECTION 1: Core Trade Details
    # ═══════════════════════════════════════════
    direction_str = meta.get("direction", side.upper())
    details_rows = f"""
        <tr>
            <td style="color: #888; padding: 12px 0; font-size: 14px;">Action</td>
            <td style="color: {action_color}; padding: 12px 0; font-size: 15px; font-weight: 700; text-align: right;">
                {action_emoji} {action_label}
            </td>
        </tr>"""

    details_rows += _row("Coin", f"{symbol.upper()} <span style='color:#888'>({coin_name})</span>", "#fff", True)
    details_rows += _row("Direction", f"{'🐻 SHORT' if side == 'short' else '🐂 LONG'}", "#f59e0b" if side == "short" else "#10b981", True)
    details_rows += _row("Entry Price" if is_entry else "Exit Price", f"${price:,.6f}", "#fff", True)
    details_rows += _row("Quantity", f"{quantity:,.6f}")
    details_rows += _row("Amount", f"${amount:,.2f}", "#fff", True)

    # ═══════════════════════════════════════════
    # SECTION: Entry-specific (BUY/SHORT)
    # ═══════════════════════════════════════════
    if is_entry:
        # 24h change
        change_24h = meta.get("change_24h", 0)
        if change_24h:
            ch_color = "#10b981" if change_24h >= 0 else "#ef4444"
            details_rows += _row("24h Change", f"{change_24h:+.1f}%", ch_color, True)

        # SL & TP with percentages
        sl_pct = meta.get("stop_loss_pct", 0)
        tp_pct = meta.get("take_profit_pct", 0)
        if stop_loss > 0:
            details_rows += _row("Stop Loss", f"${stop_loss:,.6f} <span style='color:#888'>(-{sl_pct:.1f}%)</span>", "#ef4444", True)
        if take_profit > 0:
            details_rows += _row("Take Profit", f"${take_profit:,.6f} <span style='color:#888'>(+{tp_pct:.1f}%)</span>", "#10b981", True)

        # Max hold & auto-exit
        max_hold = meta.get("max_hold_hours", 0)
        if max_hold:
            details_rows += _row("Max Hold Time", f"{max_hold}h (auto-exit)", "#f59e0b")

        # Portfolio allocation
        max_alloc = meta.get("max_alloc_pct", 0)
        slots_used = meta.get("portfolio_slots_used", 0)
        slots_max = meta.get("portfolio_slots_max", 0)
        if slots_max:
            details_rows += _row("Portfolio Slot", f"{slots_used}/{slots_max}", "#7c5cff")
        if max_alloc:
            details_rows += _row("Max Allocation", f"{max_alloc:.0f}% of portfolio", "#7c5cff")

    # ═══════════════════════════════════════════
    # SECTION: Close-specific (SELL/COVER)
    # ═══════════════════════════════════════════
    if not is_entry:
        # Entry → Exit comparison
        if entry_price > 0:
            price_change = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            ch_color = "#10b981" if price_change >= 0 else "#ef4444"
            details_rows += _row("Entry Price", f"${entry_price:,.6f}")
            details_rows += _row("Price Change", f"{price_change:+.1f}%", ch_color, True)

        # P&L
        pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_label = "PROFIT" if pnl >= 0 else "LOSS"
        details_rows += f"""
        <tr>
            <td style="color: #888; padding: 12px 0; font-size: 14px; border-top: 1px solid #2a2a3a;">{pnl_label}</td>
            <td style="color: {pnl_color}; padding: 12px 0; font-size: 18px; font-weight: 700; text-align: right; border-top: 1px solid #2a2a3a;">
                {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)
            </td>
        </tr>"""

        # Close reason
        if close_reason:
            # Clean up the reason for display
            reason_display = close_reason
            reason_color = "#f59e0b"
            cr_upper = close_reason.upper()
            if "STOP-LOSS" in cr_upper or "STOP_LOSS" in cr_upper:
                reason_display = "🛑 Stop-Loss Hit"
                reason_color = "#ef4444"
            elif "TAKE-PROFIT" in cr_upper or "TAKE_PROFIT" in cr_upper:
                reason_display = "🎯 Take-Profit Hit"
                reason_color = "#10b981"
            elif "AUTO-EXIT" in cr_upper or "AUTO_EXIT" in cr_upper:
                reason_display = "⏰ Auto-Exit (Max Hold Time)"
                reason_color = "#f59e0b"
            elif "MANUAL" in cr_upper:
                reason_display = "👤 Manual Close"
                reason_color = "#7c5cff"
            details_rows += _row("Close Reason", reason_display, reason_color, True)

        # Hold duration
        if hold_duration:
            details_rows += _row("Hold Duration", hold_duration, "#ccc")

    # Common bottom rows
    details_rows += _row("Wallet Balance", f"${wallet_balance:,.2f}", "#7c5cff", True)
    details_rows += _row("Date & Time", now, "#999")

    # ═══════════════════════════════════════════
    # BUILD EMAIL BODY
    # ═══════════════════════════════════════════
    intro = f"Your NEXYPHER Auto Trader has opened a new <strong style='color:{action_color}'>{direction_str}</strong> position." if is_entry else f"Your NEXYPHER Auto Trader has closed a <strong style='color:{action_color}'>{side.upper()}</strong> position."

    content = f"""
    <p style="color: #ccc; line-height: 1.6;">
        Hey <strong>{username}</strong>,<br><br>
        {intro}
    </p>

    <!-- Trade Details Table -->
    <div style="background: #1a1a2e; border-radius: 12px; padding: 24px; margin: 24px 0; border: 1px solid #2a2a3a;">
        <p style="color: {action_color}; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 16px;">
            {action_icon} {header_title}
        </p>
        <table style="width: 100%; border-collapse: collapse;">
            {details_rows}
        </table>
    </div>
    """

    # ═══════════════════════════════════════════
    # SECTION 2: AI Confidence Score (entry only)
    # ═══════════════════════════════════════════
    ai_score = meta.get("ai_score", 0)
    if is_entry and ai_score:
        content += f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #2a2a3a;">
        <p style="color: #7c5cff; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 10px;">
            🎯 AI Confidence Score
        </p>
        {_score_bar(ai_score)}
        <p style="color: #888; font-size: 12px; margin: 6px 0 0;">
            Market Regime: <strong style="color:#ccc">{meta.get('market_regime', 'N/A')}</strong>
        </p>
    </div>
    """

    # ═══════════════════════════════════════════
    # SECTION 3: Why This Trade (entry only)
    # ═══════════════════════════════════════════
    if is_entry:
        # Collect the reasons/signals
        direction_reasons = meta.get("reasons", [])
        if direction_reasons:
            content += _signal_section(
                f"Why {direction_str}? — Scoring Breakdown",
                direction_reasons,
                "🐻" if side == "short" else "🐂"
            )

    # ═══════════════════════════════════════════
    # SECTION 4: ML & AI Model Signals (entry only)
    # ═══════════════════════════════════════════
    if is_entry:
        model_signals = []

        # XGBoost ML signal
        ml_signal = meta.get("ml_signal", "")
        ml_prob = meta.get("ml_buy_probability")
        ml_acc = meta.get("ml_model_accuracy")
        if ml_signal:
            model_signals.append(ml_signal)
        elif ml_prob is not None:
            model_signals.append(f"🤖 XGBoost ML: {ml_prob*100:.0f}% up-probability (model accuracy: {ml_acc*100:.0f}%)" if ml_acc else f"🤖 XGBoost ML: {ml_prob*100:.0f}% up-probability")

        # Pretrained 38-feature model
        pt_signal = meta.get("pt_signal", "")
        pt_verdict = meta.get("pt_verdict", "")
        pt_prob = meta.get("pt_prob_7d")
        pt_conf = meta.get("pt_confidence")
        if pt_signal:
            model_signals.append(pt_signal)
        elif pt_verdict:
            extra = f" — {pt_prob:.0f}% 7d up-prob, confidence {pt_conf}" if pt_prob else ""
            model_signals.append(f"🧠 Pretrained Model: {pt_verdict}{extra}")

        # Gemini AI analysis
        ai_analysis = meta.get("ai_analysis", "")
        if ai_analysis:
            model_signals.append(f"💬 Gemini AI: {ai_analysis[:200]}")

        if model_signals:
            content += _signal_section("AI & ML Model Signals", model_signals, "🤖")

    # ═══════════════════════════════════════════
    # SECTION 5: Backtest Verification (entry only)
    # ═══════════════════════════════════════════
    if is_entry:
        bt_verified = meta.get("backtest_verified", False)
        bt_stats = meta.get("backtest_stats", {})
        bt_status = meta.get("backtest_status", "none")
        if bt_stats or bt_status != "none":
            bt_color = "#10b981" if bt_verified else "#ef4444"
            bt_icon = "✅" if bt_verified else "⚠️"
            bt_items = [
                f"{bt_icon} Status: <strong style='color:{bt_color}'>{bt_status.upper()}</strong>"
            ]

            bt_dir = meta.get("backtest_strategy_direction", "")
            bt_trend = meta.get("backtest_detected_trend", "")
            if bt_dir:
                bt_items.append(f"Strategy Direction: {bt_dir}")
            if bt_trend:
                bt_items.append(f"Detected Trend: {bt_trend}")

            if bt_stats:
                wr = bt_stats.get("win_rate", 0)
                tr = bt_stats.get("total_return", 0)
                sr = bt_stats.get("sharpe_ratio", 0)
                md = bt_stats.get("max_drawdown", 0)
                tt = bt_stats.get("total_trades", 0)
                period = bt_stats.get("period", "")
                wr_color = "#10b981" if wr >= 50 else "#ef4444"
                bt_items.append(f"Win Rate: <strong style='color:{wr_color}'>{wr:.1f}%</strong> ({tt} trades)")
                bt_items.append(f"Total Return: {tr:+.1f}%")
                bt_items.append(f"Sharpe Ratio: {sr:.2f}")
                bt_items.append(f"Max Drawdown: {md:.1f}%")
                if period:
                    bt_items.append(f"Period: {period}")

            strategies = meta.get("backtest_strategies_tested", [])
            if strategies:
                bt_items.append(f"Strategies Tested: {', '.join(strategies)}")

            bt_rec = meta.get("backtest_recommendation", "")
            if bt_rec:
                bt_items.append(f"Recommendation: {bt_rec}")

            content += _signal_section("Backtest Verification", bt_items, "📊")

    # ═══════════════════════════════════════════
    # SECTION 6: Market Context (entry only)
    # ═══════════════════════════════════════════
    if is_entry:
        market_items = []
        mc = meta.get("market_cap", 0)
        vol = meta.get("volume_24h", 0)
        cap_tier = meta.get("cap_tier") or meta.get("cap_label", "")
        is_trending = meta.get("is_trending", False)

        if cap_tier:
            market_items.append(f"Market Cap Tier: {cap_tier}" + (f" ({_format_market_cap(mc)})" if mc else ""))
        elif mc:
            market_items.append(f"Market Cap: {_format_market_cap(mc)}")
        if vol:
            market_items.append(f"24h Volume: {_format_market_cap(vol)}")
        if is_trending:
            market_items.append("🔥 Trending on CoinGecko")

        regime = meta.get("market_regime", "")
        if regime:
            market_items.append(f"Market Regime: {regime}")

        if market_items:
            content += _signal_section("Market Context", market_items, "🌍")

    # ═══════════════════════════════════════════
    # SECTION 7: Risk Parameters (entry only)
    # ═══════════════════════════════════════════
    if is_entry:
        risk_items = []
        sl_pct = meta.get("stop_loss_pct", 0)
        tp_pct = meta.get("take_profit_pct", 0)
        max_alloc = meta.get("max_alloc_pct", 0)
        max_hold = meta.get("max_hold_hours", 0)
        if sl_pct:
            risk_items.append(f"Stop Loss: -{sl_pct:.1f}% from entry")
        if tp_pct:
            risk_items.append(f"Take Profit: +{tp_pct:.1f}% from entry")
        if max_alloc:
            risk_items.append(f"Max Position Size: {max_alloc:.0f}% of portfolio")
        if max_hold:
            risk_items.append(f"Auto-Exit After: {max_hold}h (trailing stop active)")
        risk_items.append("Trailing stop adjusts as price moves in your favor")
        content += _signal_section("Risk Management", risk_items, "🛡️")

    # ═══════════════════════════════════════════
    # SECTION 8: AI Reasoning (always)
    # ═══════════════════════════════════════════
    if ai_reasoning:
        content += f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #2a2a3a;">
        <p style="color: #7c5cff; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px;">
            🤖 AI Reasoning
        </p>
        <p style="color: #ccc; font-size: 13px; line-height: 1.7; margin: 0;">
            {ai_reasoning}
        </p>
    </div>
    """

    # ═══════════════════════════════════════════
    # SECTION: Close details (SELL/COVER only)
    # ═══════════════════════════════════════════
    if not is_entry and close_reason and not any(k in close_reason.upper() for k in ["MANUAL"]):
        content += f"""
    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #2a2a3a;">
        <p style="color: #f59e0b; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px;">
            📋 Close Details
        </p>
        <p style="color: #ccc; font-size: 13px; line-height: 1.7; margin: 0;">
            {close_reason}
        </p>
    </div>
    """

    # CTA button + footer
    content += f"""
    <div style="text-align: center; margin: 28px 0;">
        <a href="{base_url}"
           style="display: inline-block; background: linear-gradient(135deg, #7c5cff, #00d4aa);
                  color: #fff; text-decoration: none; padding: 14px 36px; border-radius: 10px;
                  font-weight: 600; font-size: 15px;">
            View Auto Trader Dashboard
        </a>
    </div>

    <p style="color: #888; font-size: 12px; line-height: 1.6; text-align: center;">
        This trade was executed automatically by NEXYPHER Auto Trader.<br>
        You can manage your settings or turn off auto-trading from your dashboard.
    </p>
    """

    # Subject line
    if is_entry:
        subject = f"{action_emoji} NEXYPHER: {direction_str} {symbol.upper()} @ ${price:,.4f} | Score: {ai_score}/100"
    else:
        pnl_sign = "+" if pnl >= 0 else ""
        result_word = "PROFIT" if pnl >= 0 else "LOSS"
        subject = f"{action_emoji} NEXYPHER: Closed {symbol.upper()} ({side.upper()}) | {result_word}: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.1f}%)"

    return _send_email(to_email, subject, _base_template(header_title, content))


# ── Contact Form Email ─────────────────────────────────────────────

def send_contact_email(
    from_name: str,
    from_email: str,
    subject: str,
    message: str,
    to_email: str,
) -> bool:
    """Forward a contact form submission to the admin inbox."""
    content = f"""
    <p style="color: #ccc; font-size: 14px; line-height: 1.7;">
        You received a new message from the <strong>NexYpher Contact Form</strong>.
    </p>

    <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
        <tr>
            <td style="padding:10px 14px; color:#888; font-size:13px; border-bottom:1px solid #2a2a2a; width:100px;">Name</td>
            <td style="padding:10px 14px; color:#fff; font-size:14px; border-bottom:1px solid #2a2a2a;">{from_name}</td>
        </tr>
        <tr>
            <td style="padding:10px 14px; color:#888; font-size:13px; border-bottom:1px solid #2a2a2a;">Email</td>
            <td style="padding:10px 14px; color:#fff; font-size:14px; border-bottom:1px solid #2a2a2a;">
                <a href="mailto:{from_email}" style="color:#7c5cff;">{from_email}</a>
            </td>
        </tr>
        <tr>
            <td style="padding:10px 14px; color:#888; font-size:13px; border-bottom:1px solid #2a2a2a;">Subject</td>
            <td style="padding:10px 14px; color:#fff; font-size:14px; border-bottom:1px solid #2a2a2a;">{subject}</td>
        </tr>
    </table>

    <div style="background: #1a1b23; border-radius: 10px; padding: 18px 20px; margin: 16px 0;">
        <p style="color: #ccc; font-size: 14px; line-height: 1.7; white-space: pre-wrap; margin: 0;">{message}</p>
    </div>

    <p style="color: #666; font-size: 12px; margin-top: 20px;">
        You can reply directly to <a href="mailto:{from_email}" style="color:#7c5cff;">{from_email}</a>.
    </p>
    """
    email_subject = f"[NexYpher Contact] {subject} — from {from_name}"
    return _send_email(to_email, email_subject, _base_template("New Contact Message", content))
