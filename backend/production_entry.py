"""PyInstaller entry point for the packaged DataPilot backend service.

The Electron shell and frontend API client use ``DATAPILOT_BACKEND_PORT`` with
8011 as the public default.  The packaged backend must honour the same contract
or the desktop UI reports every backend-dependent service (including Ollama) as
offline even when the local Ollama daemon is healthy.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import uvicorn


def configure_packaged_logging() -> None:
    """Write durable backend logs outside the read-only Electron package."""
    log_path = os.getenv("DATAPILOT_LOG_FILE", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    except OSError:
        # Logging must never prevent the local analytical service from booting.
        pass


configure_packaged_logging()
from backend.main import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("DATAPILOT_BACKEND_PORT", "8011")),
        log_level="info",
        access_log=False,
    )
