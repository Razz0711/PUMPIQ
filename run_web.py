"""
NEXYPHER – Launch the web application.

Usage:
    python run_web.py

Live:    https://NEXYPHER.vercel.app
Local:   http://localhost:8000
"""

import uvicorn
import sys

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  NEXYPHER – AI Crypto Intelligence")
    print("  Live:  https://NEXYPHER.vercel.app")
    print("  Local: http://localhost:8000")
    print("=" * 50 + "\n")

    # Disable reload on Windows to avoid multiprocessing permission errors
    is_windows = sys.platform.startswith("win")
    
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_windows,  # Disable reload on Windows
        reload_dirs=[".", "src", "web"] if not is_windows else None,
        reload_excludes=[".venv/*", "__pycache__/*", "*.pyc", "node_modules/*"] if not is_windows else None,
        log_level="info",
    )
