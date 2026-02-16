"""
PumpIQ – Launch the web application.

Usage:
    python run_web.py

Live:    https://pumpiq.com
Local:   http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  PumpIQ – AI Crypto Intelligence")
    print("  Live:  https://pumpiq.com")
    print("  Local: http://localhost:8000")
    print("=" * 50 + "\n")

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[".", "src", "web"],
        reload_excludes=[".venv/*", "__pycache__/*", "*.pyc", "node_modules/*"],
        log_level="info",
    )
