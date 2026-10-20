"""
DataPilot OS — Graph Execution State.
Shared mutable context passed through all nodes during graph execution.
This is the authoritative state — frontend reads this via API.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
from .node import NodeStatus

class GraphExecutionPhase(str, Enum):
    INITIALIZING = "INITIALIZING"
    DATASET_PROFILING = "DATASET_PROFILING"
    INTENT_ANALYSIS = "INTENT_ANALYSIS"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    CODE_GENERATION = "CODE_GENERATION"
    VALIDATION = "VALIDATION"
    SANDBOX_EXECUTION = "SANDBOX_EXECUTION"
    RESULT_VALIDATION = "RESULT_VALIDATION"
    INSIGHT_GENERATION = "INSIGHT_GENERATION"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PRODUCTION_EXECUTION = "PRODUCTION_EXECUTION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"

class LoopState(BaseModel):
    loop_type: str
    iteration: int
    max_iterations: int
    reason: str
    history: List[Dict[str, Any]]
    last_action_hash: Optional[str] = None

class GraphExecutionState:
    def __init__(self, graph_id: str, workflow_id: str):
        self.graph_id = graph_id
        self.workflow_id = workflow_id
        self.phase = GraphExecutionPhase.INITIALIZING
        self.node_states: Dict[str, NodeStatus] = {}
        self.active_dataframe_id: Optional[str] = None
        self.shared_context: Dict[str, Any] = {}
        self.loop_states: Dict[str, LoopState] = {}
        self.error: Optional[str] = None
        self.artifacts: List[Dict[str, Any]] = []
        self.execution_log: List[Dict[str, Any]] = []

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "workflow_id": self.workflow_id,
            "phase": self.phase.value,
            "node_states": {k: v.value for k, v in self.node_states.items()},
            "active_dataframe_id": self.active_dataframe_id,
            "shared_context": self.shared_context,
            "loop_states": {k: v.model_dump() for k, v in self.loop_states.items()},
            "error": self.error,
            "artifacts": self.artifacts,
            "execution_log": self.execution_log,
        }

    def log_transition(self, from_phase: GraphExecutionPhase, to_phase: GraphExecutionPhase, reason: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "from_phase": from_phase.value,
            "to_phase": to_phase.value,
            "reason": reason
        }
        self.execution_log.append(log_entry)
        self.phase = to_phase

    def set_node_status(self, node_id: str, status: NodeStatus):
        self.node_states[node_id] = status

    def get_ui_state(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "workflow_id": self.workflow_id,
            "phase": self.phase.value,
            "node_states": {k: v.value for k, v in self.node_states.items()},
            "active_dataframe_id": self.active_dataframe_id,
            "error": self.error,
            "latest_log": self.execution_log[-1] if self.execution_log else None
        }
