"""Root Entrypoint for Nationwide Demand Forecasting & Inventory Optimization Dashboard.

Run with:
    python app.py [--port 8050] [--host 127.0.0.1] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on Python search path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import create_dash_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("App")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Nationwide Demand Forecasting & Inventory Optimization Dashboard."
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash live hot-reloading and debug mode")
    args = parser.parse_args()

    logger.info("Initializing Demand Forecasting & Inventory Optimization Platform...")
    app = create_dash_app()

    logger.info(f"Serving dashboard at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
