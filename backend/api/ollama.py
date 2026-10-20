"""
DataPilot OS — Ollama API Router.
Exposes Ollama connectivity and model management endpoints.
No fallback: if Ollama is unavailable, returns explicit OLLAMA_UNAVAILABLE status.
"""
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    from backend.intelligence.ollama_service import ollama_service
    OLLAMA_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.error(f"OllamaService import failed: {e}")
    ollama_service = None
    OLLAMA_SERVICE_AVAILABLE = False

router = APIRouter(prefix="/api/ollama", tags=["ollama"])


class TestRequest(BaseModel):
    prompt: str = "Respond with exactly: OK"
    model: Optional[str] = None


class ModelSelectRequest(BaseModel):
    model_name: str


@router.get("/status")
def get_ollama_status():
    """Returns current Ollama server status. Never raises — always explicit."""
    if not OLLAMA_SERVICE_AVAILABLE or ollama_service is None:
        return {
            "is_running": False,
            "error_code": "OLLAMA_UNAVAILABLE",
            "message": "OllamaService module could not be loaded.",
            "required_action": "Check backend/intelligence/ollama_service.py for import errors."
        }
    try:
        status = ollama_service.detect_server()
        return status.model_dump()
    except Exception as e:
        logger.error(f"Error detecting Ollama server: {e}", exc_info=True)
        return {
            "is_running": False,
            "error_code": "OLLAMA_UNAVAILABLE",
            "message": f"Failed to detect Ollama server: {str(e)}",
            "required_action": "Ensure Ollama server is running at localhost:11434."
        }


@router.get("/models")
def get_models():
    """Returns list of available Ollama models. Empty list if Ollama offline."""
    if not OLLAMA_SERVICE_AVAILABLE or ollama_service is None:
        return []
    try:
        models = ollama_service.list_models()
        return [m.model_dump() for m in models]
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}", exc_info=True)
        return []


@router.post("/test")
def test_ollama(request: TestRequest):
    """Sends a test prompt to Ollama to verify connectivity and model responsiveness."""
    if not OLLAMA_SERVICE_AVAILABLE or ollama_service is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "error_code": "OLLAMA_UNAVAILABLE",
                "message": "OllamaService not loaded.",
                "recoverable": True,
                "required_action": "Ensure Ollama is running: run 'ollama serve' in a terminal."
            }
        )
    try:
        from backend.intelligence.ollama_service import OllamaRequest
        req = OllamaRequest(
            prompt=request.prompt,
            model=request.model,
            timeout_sec=10.0
        )
        response = ollama_service.structured_request(req)
        return response.model_dump()
    except Exception as e:
        logger.error(f"Error during Ollama test: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "error": True,
                "error_code": "OLLAMA_TEST_FAILED",
                "message": f"Ollama test request failed: {str(e)}",
                "recoverable": True,
                "required_action": "Check Ollama server logs and model availability."
            }
        )


@router.post("/model/select")
def select_model(request: ModelSelectRequest):
    """Sets the active Ollama model for all DataPilot operations."""
    if not OLLAMA_SERVICE_AVAILABLE or ollama_service is None:
        return JSONResponse(status_code=503, content={"error": "OllamaService not available"})
    try:
        success = ollama_service.set_active_model(request.model_name)
        if success:
            return {"status": "success", "active_model": request.model_name}
        return JSONResponse(
            status_code=404,
            content={
                "error": True,
                "error_code": "OLLAMA_MODEL_UNAVAILABLE",
                "message": f"Model '{request.model_name}' not found in available models.",
                "recoverable": True,
                "required_action": f"Pull the model first: ollama pull {request.model_name}"
            }
        )
    except Exception as e:
        logger.error(f"Error selecting model {request.model_name}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": "OLLAMA_SELECT_ERROR",
                "message": f"Error selecting model: {str(e)}"
            }
        )
