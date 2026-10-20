"""Production validation for persisted workflows.

Generated-code execution is default-deny until workflows are compiled to the
same typed operator graph used by the canonical query path.
"""
import os
from typing import Any, Dict, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.dataset_registry import dataset_registry
from backend.persistence.workflow_store import workflow_store

router = APIRouter(prefix="/api/production", tags=["production"])


class ProductionValidateRequest(BaseModel):
    workflow_id: str
    production_dataset_id: str


class ProductionExecuteRequest(ProductionValidateRequest):
    user_approved: bool


def _load_workflow_and_dataset(workflow_id: str, dataset_id: str) -> Tuple[Any, Any, Any]:
    workflow = workflow_store.load(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    dataset = dataset_registry.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Production dataset '{dataset_id}' not found")
    dataframe = dataset_registry.get_dataframe(dataset_id)
    if dataframe is None:
        raise HTTPException(status_code=422, detail="The selected production dataset has no readable dataframe")
    return workflow, dataset, dataframe


def _validate(workflow: Any, dataframe: Any) -> Dict[str, Any]:
    columns = {str(column) for column in dataframe.columns}
    required_columns = set(workflow.required_columns or [])
    missing_columns = sorted(required_columns - columns)
    checks = [
        {
            "check_name": "schema_compatibility",
            "passed": not missing_columns,
            "message": "All workflow columns are present" if not missing_columns else f"Missing columns: {', '.join(missing_columns)}",
        },
        {
            "check_name": "dataset_not_empty",
            "passed": len(dataframe.index) > 0,
            "message": f"Dataset contains {len(dataframe.index):,} rows",
        },
        {
            "check_name": "validated_code_available",
            "passed": bool(workflow.code),
            "message": "Validated generated code is available" if workflow.code else "This workflow has no executable validated code",
        },
    ]
    compatible = all(check["passed"] for check in checks)
    return {
        "is_compatible": compatible,
        "checks": checks,
        "warnings": [] if compatible else ["Production execution remains blocked until every check passes."],
    }


@router.post("/validate")
async def validate_production(request: ProductionValidateRequest):
    workflow, _, dataframe = _load_workflow_and_dataset(request.workflow_id, request.production_dataset_id)
    return _validate(workflow, dataframe)


@router.post("/preview")
async def preview_production(request: ProductionValidateRequest):
    workflow, _, dataframe = _load_workflow_and_dataset(request.workflow_id, request.production_dataset_id)
    validation = _validate(workflow, dataframe)
    return {
        "is_compatible": validation["is_compatible"],
        "row_count": len(dataframe.index),
        "artifacts_expected": [],
        "warnings": validation["warnings"],
    }


@router.post("/execute")
async def execute_production(request: ProductionExecuteRequest):
    if os.getenv("DATAPILOT_ENABLE_LEGACY_PRODUCTION_EXECUTION", "0").strip() != "1":
        raise HTTPException(
            status_code=403,
            detail={
                "code": "LEGACY_PRODUCTION_EXECUTION_DISABLED",
                "message": "Generated-code production execution is quarantined.",
                "required_action": "Migrate this workflow to a typed operator graph before execution.",
            },
        )
    if not request.user_approved:
        raise HTTPException(status_code=400, detail="PRODUCTION_BLOCKED: explicit user approval is required")

    workflow, _, dataframe = _load_workflow_and_dataset(request.workflow_id, request.production_dataset_id)
    validation = _validate(workflow, dataframe)
    if not validation["is_compatible"]:
        raise HTTPException(status_code=422, detail={"code": "PRODUCTION_BLOCKED", **validation})

    try:
        from modules.execution_engine import ModularExecutionEngine

        result = ModularExecutionEngine().execute_code_subprocess(workflow.code, df=dataframe)
    except Exception as exc:
        workflow_store.update_status(workflow.workflow_id, "FAILED")
        raise HTTPException(status_code=500, detail=f"Production execution failed: {exc}") from exc

    succeeded = bool(result.get("success", True))
    workflow_store.update_status(workflow.workflow_id, "SUCCEEDED" if succeeded else "FAILED")
    return {
        "status": "SUCCEEDED" if succeeded else "FAILED",
        "workflow_id": workflow.workflow_id,
        "output": result.get("stdout") or result.get("output"),
        "plot_image": result.get("plot_image"),
        "error": result.get("stderr") or result.get("error"),
    }
