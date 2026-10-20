"""
DataPilot OS — Workflow State Management.
Full state machine representation for DataPilot OS workflows.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime
import uuid

class WorkflowState(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_query: str
    dataset_ids: List[str] = Field(default_factory=list)
    dataset_context: Dict[str, Any] = Field(default_factory=dict)
    intent: Optional[Dict[str, Any]] = None
    clarification: Optional[Dict[str, Any]] = None
    graph: Optional[Dict[str, Any]] = None
    graph_version: int = 0
    graph_version_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_node_id: Optional[str] = None
    completed_node_ids: List[str] = Field(default_factory=list)
    failed_node_ids: List[str] = Field(default_factory=list)
    generated_code_versions: List[Dict[str, Any]] = Field(default_factory=list)
    execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    insights: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    loop_state: Dict[str, Any] = Field(default_factory=lambda: {
        "iteration": 0,
        "max_iterations": 3,
        "reason": "",
        "history": []
    })
    production: Dict[str, Any] = Field(default_factory=lambda: {
        "validated": False,
        "approved": False,
        "audit_record": None
    })
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    error: Optional[Dict[str, Any]] = None

    def increment_graph_version(self, reason: str, added_nodes: list) -> None:
        """Increment graph version and store history."""
        history_entry = {
            "version_id": self.graph_version,
            "reason": reason,
            "added_nodes": added_nodes,
            "parent_version": max(0, self.graph_version - 1),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        self.graph_version_history.append(history_entry)
        self.graph_version += 1
        self.updated_at = datetime.datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        """Serialize fully to dict."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        """Deserialize from dict."""
        return cls(**data)
        
    def is_complete(self) -> bool:
        """Check if workflow is completely finished without errors."""
        if not self.graph or self.error:
            return False
        return True
        
    def has_error(self) -> bool:
        """Check if state has an error."""
        return self.error is not None
