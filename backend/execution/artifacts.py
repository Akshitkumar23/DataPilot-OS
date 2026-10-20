"""
DataPilot OS — Artifact Manager.
Manages all outputs from execution: plots, tables, files.
Artifacts are immutable after creation. New executions create new artifacts.
"""
from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel
import uuid
import time

class ArtifactType(str, Enum):
    PLOT = "PLOT"
    TABLE = "TABLE"
    FILE = "FILE"
    METRIC = "METRIC"
    INSIGHT = "INSIGHT"

class Artifact(BaseModel):
    artifact_id: str
    execution_id: str
    workflow_id: str
    artifact_type: ArtifactType
    name: str
    content: Any
    created_at: str
    metadata: Dict[str, Any]

class ArtifactManager:
    def __init__(self):
        self._artifacts: Dict[str, Artifact] = {}

    def store_plot(self, execution_id: str, workflow_id: str, plot_b64: str, index: int) -> Artifact:
        a_id = str(uuid.uuid4())
        art = Artifact(
            artifact_id=a_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            artifact_type=ArtifactType.PLOT,
            name=f"plot_{index}",
            content=plot_b64,
            created_at=str(time.time()),
            metadata={}
        )
        self._artifacts[a_id] = art
        return art

    def store_table(self, execution_id: str, workflow_id: str, df_json: dict, name: str) -> Artifact:
        a_id = str(uuid.uuid4())
        art = Artifact(
            artifact_id=a_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            artifact_type=ArtifactType.TABLE,
            name=name,
            content=df_json,
            created_at=str(time.time()),
            metadata={}
        )
        self._artifacts[a_id] = art
        return art

    def store_metric(self, execution_id: str, workflow_id: str, name: str, value: Any) -> Artifact:
        a_id = str(uuid.uuid4())
        art = Artifact(
            artifact_id=a_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            artifact_type=ArtifactType.METRIC,
            name=name,
            content=str(value),
            created_at=str(time.time()),
            metadata={}
        )
        self._artifacts[a_id] = art
        return art

    def get_for_execution(self, execution_id: str) -> List[Artifact]:
        return [a for a in self._artifacts.values() if a.execution_id == execution_id]

    def get_for_workflow(self, workflow_id: str) -> List[Artifact]:
        return [a for a in self._artifacts.values() if a.workflow_id == workflow_id]

artifact_manager = ArtifactManager()
