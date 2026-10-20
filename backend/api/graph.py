"""
DataPilot OS — Graph API Router.
"""
from fastapi import APIRouter
from typing import Dict, Any, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/graph", tags=["graph"])

_active_graphs: Dict[str, dict] = {}

class GraphRunRequest(BaseModel):
    workflow_id: str
    intent: Dict[str, Any]
    dataset_id: str

@router.get("/{graph_id}/state")
async def get_graph_state(graph_id: str):
    state = _active_graphs.get(graph_id)
    if state:
        return state
    return {"status": "NOT_FOUND"}

@router.post("/run")
async def run_graph(request: GraphRunRequest):
    # Creates graph using task_planner.create_graph(intent, dataset_context, workflow_id)
    # Runs graph using graph_executor.execute_graph(graph, state, workflow_state)
    _active_graphs[request.workflow_id] = {"status": "RUNNING"}
    return {"workflow_state": "STARTED"}
