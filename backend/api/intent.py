"""
DataPilot OS — Intent API Router.
Converts natural language queries into structured intents via Ollama.
No fallback: OLLAMA_UNAVAILABLE if Ollama is down.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from backend.intelligence.intent_engine import intent_engine
from backend.intelligence.clarification_engine import clarification_engine
from backend.core.error_states import DataPilotError

# Assuming dataset_registry and workflow_store are available
try:
    from backend.core.dataset_registry import dataset_registry
except ImportError:
    dataset_registry = None

try:
    from backend.persistence.workflow_store import workflow_store
except ImportError:
    workflow_store = None

router = APIRouter(prefix="/api/intent", tags=["intent"])

_active_clarifications = {}

class IntentAnalyzeRequest(BaseModel):
    query: str
    dataset_id: str
    conversation_history: List[Dict[str, Any]] = []
    workflow_id: Optional[str] = None

class IntentClarifyRequest(BaseModel):
    request_id: str
    answers: Dict[str, Any]
    workflow_id: str

@router.post("/analyze")
async def analyze_intent(request: IntentAnalyzeRequest):
    dataset = dataset_registry.get(request.dataset_id) if dataset_registry else None
    if dataset and hasattr(dataset, "get_context"):
        dataset_context = dataset.get_context()
    elif dataset:
        dataset_context = getattr(dataset, "knowledge_model", {"columns": []})
    else:
        dataset_context = {"columns": []}
    
    result = intent_engine.analyze(request.query, dataset_context, request.conversation_history)
    
    if isinstance(result, DataPilotError):
        return JSONResponse(status_code=400, content={"error": result.code, "message": result.message})
        
    if result.requires_clarification:
        clarification = clarification_engine.generate_clarification(result, result.ambiguities, dataset_context)
        _active_clarifications[clarification.request_id] = clarification
        return {"intent": result.model_dump(), "clarification_request": clarification.model_dump()}
        
    if workflow_store and request.workflow_id:
        workflow = workflow_store.get(request.workflow_id)
        if workflow:
            workflow.intent = result
            workflow_store.save(workflow)
            
    return result.model_dump()

@router.post("/clarify")
async def clarify_intent(request: IntentClarifyRequest):
    clarification_req = _active_clarifications.get(request.request_id)
    if not clarification_req:
        return JSONResponse(status_code=404, content={"error": "CLARIFICATION_NOT_FOUND", "message": "Request ID not found"})
        
    updated_intent = clarification_engine.resolve_clarification(clarification_req, request.answers, clarification_req.partial_intent)
    
    if workflow_store:
        workflow = workflow_store.get(request.workflow_id)
        if workflow:
            workflow.intent = updated_intent
            workflow_store.save(workflow)
            
    return updated_intent.model_dump()
