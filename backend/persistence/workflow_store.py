"""
DataPilot OS — Workflow Persistence Store.
Persists workflow history, versions, and audit records.
Every workflow is versioned and reproducible.
"""
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class WorkflowRecord(BaseModel):
    workflow_id: str
    session_id: str
    user_query: str
    dataset_ids: List[str]
    graph_versions: List[str]
    code_versions: List[str]
    execution_ids: List[str]
    insight_summary: str
    status: str
    created_at: str
    updated_at: str
    dataset_name: str = ""
    intent: Dict[str, Any] = Field(default_factory=dict)
    code: Optional[str] = None
    required_columns: List[str] = Field(default_factory=list)

class WorkflowStore:
    def __init__(self, data_path: str = "backend/data/workflows.json"):
        runtime_data_dir = os.getenv("DATAPILOT_DATA_DIR")
        self.data_path = (
            os.path.join(runtime_data_dir, "workflows.json")
            if runtime_data_dir else os.path.join(os.getcwd(), data_path)
        )
        self._workflows: Dict[str, WorkflowRecord] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._workflows[k] = WorkflowRecord(**v)
            except Exception:
                pass

    def _persist_to_disk(self):
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self._workflows.items()}, f, indent=2)

    def save(self, workflow_state_dict: Dict[str, Any]) -> str:
        w_id = workflow_state_dict.get("workflow_id") or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        existing = self._workflows.get(w_id)
        
        record = WorkflowRecord(
            workflow_id=w_id,
            session_id=workflow_state_dict.get("session_id", existing.session_id if existing else ""),
            user_query=workflow_state_dict.get("user_query", existing.user_query if existing else ""),
            dataset_ids=workflow_state_dict.get("dataset_ids", existing.dataset_ids if existing else []),
            graph_versions=workflow_state_dict.get("graph_versions", existing.graph_versions if existing else []),
            code_versions=workflow_state_dict.get("code_versions", existing.code_versions if existing else []),
            execution_ids=workflow_state_dict.get("execution_ids", existing.execution_ids if existing else []),
            insight_summary=workflow_state_dict.get("insight_summary", existing.insight_summary if existing else ""),
            status=workflow_state_dict.get("status", existing.status if existing else "running"),
            created_at=workflow_state_dict.get("created_at", existing.created_at if existing else now),
            updated_at=now,
            dataset_name=workflow_state_dict.get("dataset_name", existing.dataset_name if existing else ""),
            intent=workflow_state_dict.get("intent", existing.intent if existing else {}),
            code=workflow_state_dict.get("code", existing.code if existing else None),
            required_columns=workflow_state_dict.get("required_columns", existing.required_columns if existing else []),
        )
        self._workflows[w_id] = record
        self._persist_to_disk()
        return w_id

    def load(self, workflow_id: str) -> Optional[WorkflowRecord]:
        return self._workflows.get(workflow_id)

    def get(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Compatibility alias used by API routers."""
        return self.load(workflow_id)

    def list_recent(self, limit: int = 20) -> List[WorkflowRecord]:
        sorted_w = sorted(self._workflows.values(), key=lambda x: x.created_at, reverse=True)
        return sorted_w[:limit]

    def update_status(self, workflow_id: str, status: str):
        if workflow_id in self._workflows:
            self._workflows[workflow_id].status = status
            self._workflows[workflow_id].updated_at = datetime.utcnow().isoformat()
            self._persist_to_disk()

    def delete(self, workflow_id: str) -> bool:
        if workflow_id not in self._workflows:
            return False
        del self._workflows[workflow_id]
        self._persist_to_disk()
        return True

workflow_store = WorkflowStore()
