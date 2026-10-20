"""
DataPilot OS — Graph Node Base Class.
Every analytical capability is represented as a typed graph node
with explicit input/output schemas and validation contracts.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from abc import ABC, abstractmethod

class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    REPLANNED = "REPLANNED"
    SKIPPED = "SKIPPED"

class NodeInputSchema(BaseModel):
    required_fields: List[str]
    optional_fields: List[str]
    accepts_dataframe: bool
    requires_numeric_columns: bool
    requires_datetime_column: bool

class NodeOutputSchema(BaseModel):
    produces_dataframe: bool
    produces_table: bool
    produces_plot: bool
    produces_metrics: bool
    produces_insight: bool
    output_keys: List[str]

class NodePrecondition(BaseModel):
    condition_type: str
    parameters: Dict[str, Any]
    error_if_fails: str

class GraphNode(ABC):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        input_schema: NodeInputSchema,
        output_schema: NodeOutputSchema,
        preconditions: List[NodePrecondition]
    ):
        self.id = id
        self.name = name
        self.description = description
        self.status = NodeStatus.PENDING
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.preconditions = preconditions
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None

    @abstractmethod
    def can_run(self, state: dict) -> Tuple[bool, Optional[str]]:
        pass

    @abstractmethod
    def execute(self, state: dict) -> Dict[str, Any]:
        pass

    def validate_output(self, output: dict) -> Tuple[bool, Optional[str]]:
        for key in self.output_schema.output_keys:
            if key not in output:
                return False, f"Missing required output key: {key}"
        return True, None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "input_schema": self.input_schema.model_dump(),
            "output_schema": self.output_schema.model_dump(),
            "preconditions": [p.model_dump() for p in self.preconditions],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
        }

    def to_ui_state(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }
