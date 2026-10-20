"""
DataPilot OS — Results API Router.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.results.interpreter import result_interpreter
from backend.core.error_states import DataPilotError

# Assuming execution_runner is available
try:
    from backend.execution.runner import execution_runner
except ImportError:
    execution_runner = None

router = APIRouter(prefix="/api/results", tags=["results"])

class ResultsValidateRequest(BaseModel):
    execution_id: str
    intent: Dict[str, Any]
    code_version: Optional[str] = "1.0"

class ResultsInterpretRequest(BaseModel):
    execution_id: str
    user_query: str
    intent: Dict[str, Any]

@router.post("/validate")
async def validate_result(request: ResultsValidateRequest):
    if not execution_runner:
        return JSONResponse(status_code=500, content={"error": "NO_RUNNER"})
        
    record = execution_runner.get_status(request.execution_id)
    if not record or not record.result:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND"})
        
    report = result_validator.validate(record.result, request.intent, request.code_version)
    return report.model_dump()

@router.post("/interpret")
async def interpret_result(request: ResultsInterpretRequest):
    if not execution_runner:
        return JSONResponse(status_code=500, content={"error": "NO_RUNNER"})
        
    record = execution_runner.get_status(request.execution_id)
    if not record or not record.result:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND"})
        
    report = result_validator.validate(record.result, request.intent, "1.0")
    
    result = result_interpreter.interpret(report, record.result, record.artifacts, request.user_query, request.intent)
    if isinstance(result, DataPilotError):
        return JSONResponse(status_code=400, content={"error": result.code, "message": result.message})
        
    return result.model_dump()

@router.get("/{execution_id}")
async def get_execution_result(execution_id: str):
    if not execution_runner:
        return JSONResponse(status_code=500, content={"error": "NO_RUNNER"})
        
    record = execution_runner.get_status(execution_id)
    if not record:
        return JSONResponse(status_code=404, content={"error": "NOT_FOUND"})
        
    return {
        "execution_record": record.model_dump() if hasattr(record, "model_dump") else record,
        "artifacts": [a.model_dump() if hasattr(a, "model_dump") else a for a in record.artifacts] if hasattr(record, "artifacts") else [],
        "validation_report": None # Can be generated if needed
    }
