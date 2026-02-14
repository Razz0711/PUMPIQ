"""
Vercel Serverless Entry Point for PumpIQ
"""
from web_app import app

# Export the FastAPI app for Vercel
handler = app
