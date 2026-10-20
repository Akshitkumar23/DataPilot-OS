"""
DataPilot OS — Backend API Server.
Industry-grade no-fallback data science workspace backend.
All requests go through typed validation and explicit error states.
"""
import secrets
import os
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

def _configure_backend_logging() -> None:
    """Configure one durable backend log sink when the desktop shell requests it.

    The Electron application may intentionally detach the scientific Python
    process so it survives a window restart.  In that mode stdout is not a
    dependable diagnostic channel.  A bounded rotating file is therefore the
    authoritative process log for both packaged and local launches.
    """
    log_path = os.getenv("DATAPILOT_LOG_FILE", "").strip()
    if not log_path:
        logging.basicConfig(level=logging.INFO)
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        root_logger = logging.getLogger()
        absolute_path = os.path.abspath(log_path)
        already_configured = any(
            isinstance(handler, RotatingFileHandler)
            and os.path.abspath(getattr(handler, "baseFilename", "")) == absolute_path
            for handler in root_logger.handlers
        )
        if not already_configured:
            handler = RotatingFileHandler(
                absolute_path,
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    except OSError:
        # Diagnostics must never stop the local analytical service from
        # starting. stdout/stderr remain available for a non-detached launch.
        logging.basicConfig(level=logging.INFO)


_configure_backend_logging()
logger = logging.getLogger(__name__)

# Configure logging before importing routers. Router imports bootstrap the
# registry, so doing this afterwards loses the most valuable startup evidence.
from backend.api.ollama import router as ollama_router
from backend.api.datasets import router as datasets_router
from backend.api.code import router as code_router
from backend.api.results import router as results_router
from backend.api.production import router as production_router
from backend.api.chat import router as chat_router
from backend.api.graph import router as graph_router
from backend.api.workflows import router as workflows_router
from backend.api.operations import router as operations_router
from backend.api.governance import router as governance_router
from backend.api.eda import router as eda_router
from backend.api.ml import router as ml_router
from backend.api.unsupervised import router as unsupervised_router
from backend.api.timeseries import router as timeseries_router
from backend.api.sql import router as sql_router
from backend.api.reports import router as reports_router
from backend.api.agent import router as agent_router

try:
    from backend.intelligence.ollama_service import ollama_service
except ImportError:
    ollama_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DataPilot OS Backend v2.0 starting...")
    try:
        from backend.persistence.workflow_store import workflow_store
        workflow_store._load_from_disk()
        logger.info("Workflow store loaded from disk.")
    except Exception as e:
        logger.warning(f"Could not load workflow store: {e}")
    try:
        from backend.persistence.provenance_store import provenance_store
        provenance_store._load_from_disk()
        logger.info("Provenance store loaded from disk.")
    except Exception as e:
        logger.warning(f"Could not load provenance store: {e}")
    yield
    logger.info("DataPilot OS Backend shutting down.")

app = FastAPI(
    title="DataPilot OS Backend",
    description="Industry-grade no-fallback data science workspace backend.",
    version="2.0",
    lifespan=lifespan
)

# Hardened CORS: allow local Electron renderer and dev origins only
_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8011",
    "http://127.0.0.1:8011",
    "null",
]
_extra_origins = os.getenv("DATAPILOT_ALLOWED_ORIGINS", "").strip()
if _extra_origins:
    _allowed_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Authentication verification helper
def verify_token(request: Request):
    api_token = os.getenv('DATAPILOT_API_TOKEN', '')
    if not api_token:
        return True
    auth = request.headers.get('Authorization', '').strip()
    if not auth or not auth.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Unauthorized: Missing or invalid Bearer token')
    token = auth[7:].strip()
    if not secrets.compare_digest(token, api_token):
        raise HTTPException(status_code=401, detail='Invalid authorization token')
    return True

# Public unauthenticated health check endpoints
@app.get("/livez")
def liveness_check():
    """Fast process liveness probe: no Ollama/network/model work is allowed."""
    # Electron uses this opaque value to prove that a warm local service was
    # started from the same packaged backend binary as the visible desktop
    # shell.  It is supplied by Electron at spawn time, not derived from a
    # dataset, query, or user input.
    return {
        "status": "live",
        "version": "2.0",
        "backend": "DataPilot OS",
        "release_id": os.getenv("DATAPILOT_RELEASE_ID"),
    }


@app.get("/")
@app.get("/health")
@app.get("/api/v2/health")
def health_check():
    """Readiness endpoint, including the local Ollama planner dependency."""
    readiness_stat = {"ready": False, "error_code": "OLLAMA_SERVICE_NOT_LOADED", "error_message": "OllamaService not loaded"}
    if ollama_service:
        try:
            readiness = ollama_service.verify_ready()
            readiness_stat = readiness.model_dump()
        except Exception as e:
            readiness_stat = {"ready": False, "error_code": "OLLAMA_CHECK_ERROR", "error_message": str(e)}

    is_online = readiness_stat.get("ready", False)
    
    # Verify ToolSpec and driver bootstrap readiness
    from modules.capability_registry import capability_registry
    toolspec_count = len(capability_registry.list_tool_specs())
    toolspec_ready = toolspec_count >= 12

    return {
        "status": "online" if (is_online and toolspec_ready) else "unavailable",
        "version": "2.0",
        "backend": "DataPilot OS",
        "release_id": os.getenv("DATAPILOT_RELEASE_ID"),
        "ollama_ready": readiness_stat.get("ready", False),
        "model_ready": readiness_stat.get("model_available", False),
        "configured_model": os.getenv("OLLAMA_MODEL") or readiness_stat.get("model") or "gemma4:e2b",
        "intent_planner_ready": is_online,
        "toolspec_ready": toolspec_ready,
        "toolspec_count": toolspec_count,
        "ollama_status": readiness_stat
    }

app.include_router(chat_router, dependencies=[Depends(verify_token)])
app.include_router(ollama_router, dependencies=[Depends(verify_token)])
app.include_router(datasets_router, dependencies=[Depends(verify_token)])
app.include_router(code_router, dependencies=[Depends(verify_token)])
app.include_router(results_router, dependencies=[Depends(verify_token)])
app.include_router(production_router, dependencies=[Depends(verify_token)])
app.include_router(graph_router, dependencies=[Depends(verify_token)])
app.include_router(workflows_router, dependencies=[Depends(verify_token)])
app.include_router(operations_router, dependencies=[Depends(verify_token)])
app.include_router(governance_router, dependencies=[Depends(verify_token)])
app.include_router(eda_router, dependencies=[Depends(verify_token)])
app.include_router(ml_router, dependencies=[Depends(verify_token)])
app.include_router(unsupervised_router, dependencies=[Depends(verify_token)])
app.include_router(timeseries_router, dependencies=[Depends(verify_token)])
app.include_router(sql_router, dependencies=[Depends(verify_token)])
app.include_router(reports_router, dependencies=[Depends(verify_token)])
app.include_router(agent_router, dependencies=[Depends(verify_token)])


def _quarantined_endpoint(path_family: str) -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={
            "error": True,
            "code": "QUARANTINED_ENDPOINT",
            "message": (
                f"The legacy {path_family} API is quarantined. "
                "Analytical requests must use the typed /api/chat pipeline."
            ),
            "required_action": "Use /api/chat or a registered deterministic capability.",
        },
    )


@app.api_route(
    "/api/execution/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    dependencies=[Depends(verify_token)],
    include_in_schema=False,
)
async def quarantined_execution_api(path: str):
    return _quarantined_endpoint("arbitrary-code execution")


@app.api_route(
    "/api/intent/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    dependencies=[Depends(verify_token)],
    include_in_schema=False,
)
async def quarantined_intent_api(path: str):
    return _quarantined_endpoint("duplicate intent")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail, "detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "DataPilotError", "message": str(exc)}
    )

if __name__ == '__main__':
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=int(os.getenv("DATAPILOT_BACKEND_PORT", "8011")),
        # Explicit opt-in only.  Reload supervises an extra child process and
        # is a development convenience, not a desktop-runtime default.
        reload=os.getenv("DATAPILOT_DEV_RELOAD", "0") == "1",
    )
