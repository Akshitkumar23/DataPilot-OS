"""
DataPilot OS — Code API Router.
Code generation, validation, repair, and user-edit versioning.
Generated code is NEVER executed without passing the 7-step validator first.
"""
import logging
import os
import uuid
from typing import Optional, Dict, Any, List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/code", tags=["code"])

try:
    from backend.ds_engine.code_generator import code_generator, CodeVersion
    from backend.ds_engine.code_validator import code_validator
    from backend.ds_engine.repair_engine import repair_engine
    from backend.core.dataset_registry import dataset_registry
    DS_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"DS Engine import failed: {e}")
    DS_ENGINE_AVAILABLE = False

# In-memory code version store (until persistence layer takes over)
_code_versions: Dict[str, Any] = {}


class CodeGenerateRequest(BaseModel):
    workflow_id: str
    node_id: str
    capability: str
    intent: Dict[str, Any]
    dataset_id: str


class CodeValidateRequest(BaseModel):
    code_version_id: Optional[str] = None
    code: Optional[str] = None
    dataset_id: str


class CodeRepairRequest(BaseModel):
    code_version_id: str
    error_output: str
    dataset_id: str
    attempt_number: int = 1


class CodeUserEditRequest(BaseModel):
    original_version_id: str
    new_code: str


@router.post("/generate")
async def generate_code(request: CodeGenerateRequest):
    """Generate Python code from capability + intent + dataset context. Returns CodeVersion."""
    if os.getenv("DATAPILOT_ENABLE_LEGACY_CODEGEN", "0").strip() != "1":
        return JSONResponse(status_code=403, content={
            "error": True,
            "code": "LEGACY_CODEGEN_DISABLED",
            "message": "Free-form code generation is disabled in the production runtime.",
            "required_action": "Use /api/chat so a registered typed operator performs the analysis."
        })
    if not DS_ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={
            "error": True, "error_code": "CODE_INVALID",
            "message": "DS Engine not available.",
            "required_action": "Check backend/ds_engine/ for import errors."
        })

    entry = dataset_registry.get(request.dataset_id)
    if not entry:
        return JSONResponse(status_code=404, content={
            "error": True, "error_code": "DATASET_NOT_CONNECTED",
            "message": f"Dataset '{request.dataset_id}' not connected.",
            "required_action": "Connect a dataset first via /api/datasets/connect"
        })

    dataset_context = {
        "file_path": entry.file_path,
        "format": entry.format,
        "columns": entry.knowledge_model.get("columns", []) if entry.knowledge_model else []
    }

    result = code_generator.generate(
        node_id=request.node_id,
        capability=request.capability,
        intent=request.intent,
        dataset_context=dataset_context,
        workflow_id=request.workflow_id
    )

    if isinstance(result, CodeVersion):
        _code_versions[result.version_id] = result
        return result.model_dump()
    else:  # DataPilotError
        return JSONResponse(status_code=422, content=result.to_response())


@router.post("/validate")
async def validate_code(request: CodeValidateRequest):
    """Run the 7-step code validation pipeline. Code that fails NEVER reaches execution."""
    if not DS_ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "DS Engine not available"})

    # Resolve code
    code_to_validate = None
    code_ver = None
    if request.code_version_id and request.code_version_id in _code_versions:
        code_ver = _code_versions[request.code_version_id]
        code_to_validate = code_ver.code
    elif request.code:
        code_to_validate = request.code
        # Create ephemeral CodeVersion for validation
        from backend.ds_engine.code_generator import CodeVersion
        from datetime import datetime, timezone
        code_ver = CodeVersion(
            version_id=str(uuid.uuid4()),
            workflow_id="ephemeral",
            node_id="validate",
            purpose="User validation",
            code=code_to_validate,
            dependencies=[],
            expected_outputs=[],
            dataset_columns_referenced=[],
            validation_requirements=[],
            created_at=datetime.now(timezone.utc).isoformat()
        )
    else:
        return JSONResponse(status_code=400, content={
            "error": True, "error_code": "CODE_INVALID",
            "message": "Provide either code_version_id or code string.",
            "required_action": "Include code or code_version_id in request."
        })

    entry = dataset_registry.get(request.dataset_id)
    dataset_context = {}
    if entry and entry.knowledge_model:
        dataset_context = {
            "columns": [c.get("name") for c in entry.knowledge_model.get("columns", [])]
        }

    result = code_validator.validate(code_ver, dataset_context)
    return result.model_dump()


@router.post("/repair")
async def repair_code(request: CodeRepairRequest):
    """Attempt code repair with diff-check. Returns LOOP_NO_PROGRESS if identical code proposed."""
    if not DS_ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={"error": "DS Engine not available"})

    code_ver = _code_versions.get(request.code_version_id)
    if not code_ver:
        return JSONResponse(status_code=404, content={
            "error": True, "error_code": "CODE_INVALID",
            "message": f"Code version '{request.code_version_id}' not found.",
            "required_action": "Generate code first via /api/code/generate"
        })

    entry = dataset_registry.get(request.dataset_id)
    dataset_context = {}
    if entry and entry.knowledge_model:
        dataset_context = {
            "columns": [c.get("name") for c in entry.knowledge_model.get("columns", [])],
            "file_path": entry.file_path
        }

    result = repair_engine.attempt_repair(
        failed_code_version=code_ver,
        error_output=request.error_output,
        dataset_context=dataset_context,
        attempt_number=request.attempt_number
    )

    if isinstance(result, CodeVersion):
        _code_versions[result.version_id] = result
        return result.model_dump()
    else:  # DataPilotError
        status = 422
        if result.code in ("LOOP_NO_PROGRESS", "LOOP_BUDGET_EXCEEDED"):
            status = 409  # Conflict — loop exhausted
        return JSONResponse(status_code=status, content=result.to_response())


@router.post("/user-edit")
async def user_edit_code(request: CodeUserEditRequest):
    """Save a user-modified version of generated code. Marks as user-edited."""
    original = _code_versions.get(request.original_version_id)
    if not original:
        return JSONResponse(status_code=404, content={
            "error": True, "error_code": "CODE_INVALID",
            "message": f"Original code version '{request.original_version_id}' not found.",
            "required_action": "Generate code first via /api/code/generate"
        })

    if DS_ENGINE_AVAILABLE:
        new_ver = code_generator.create_user_edit_version(original, request.new_code)
    else:
        from backend.ds_engine.code_generator import CodeVersion
        import datetime
        new_ver = CodeVersion(
            version_id=str(uuid.uuid4()),
            workflow_id=original.workflow_id,
            node_id=original.node_id,
            purpose=original.purpose,
            code=request.new_code,
            dependencies=original.dependencies,
            expected_outputs=original.expected_outputs,
            dataset_columns_referenced=original.dataset_columns_referenced,
            validation_requirements=original.validation_requirements,
            created_at=datetime.datetime.utcnow().isoformat(),
            is_user_edited=True,
            parent_version_id=original.version_id
        )

    _code_versions[new_ver.version_id] = new_ver
    return new_ver.model_dump()


@router.get("/version/{version_id}")
async def get_code_version(version_id: str):
    """Retrieve a specific code version by ID."""
    ver = _code_versions.get(version_id)
    if not ver:
        raise __import__('fastapi').HTTPException(status_code=404, detail=f"Code version '{version_id}' not found")
    return ver.model_dump()
