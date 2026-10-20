"""Workflow history API backed by the local persistent workflow store."""
from fastapi import APIRouter, HTTPException

from backend.persistence.workflow_store import workflow_store

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.get("")
@router.get("/")
def list_workflows(limit: int = 50):
    return [record.model_dump() for record in workflow_store.list_recent(limit=limit)]


@router.get("/{workflow_id}")
def get_workflow(workflow_id: str):
    record = workflow_store.load(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return record.model_dump()


@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: str):
    if not workflow_store.delete(workflow_id):
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return {"status": "deleted", "workflow_id": workflow_id}
