"""
DataPilot OS — Main Entry Server Wrapper.
Forwards to backend.main FastAPI app instance.
"""
import os
import uvicorn
from backend.main import app

if __name__ == '__main__':
    # Auto-reload spawns a supervisor plus a child process.  It is useful only
    # while developing, and is unsafe as a normal desktop entry point because
    # it can leave a second backend owning the API port after the window exits.
    # Packaged production starts ``backend/production_entry.py`` instead.
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=int(os.getenv("DATAPILOT_BACKEND_PORT", "8011")),
        reload=os.getenv("DATAPILOT_DEV_RELOAD", "0") == "1",
    )
