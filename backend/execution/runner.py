"""
DataPilot OS — Execution Coordinator.
Manages execution lifecycle: state machine, queuing, cancellation.
Coordinates between CodeValidator, Sandbox, and ResultValidator.
"""
from typing import Dict, Optional, Any
from pydantic import BaseModel
import uuid
import time
from .sandbox import python_sandbox, SandboxResult, ExecutionStatus

class ExecutionRecord(BaseModel):
    execution_id: str
    workflow_id: str
    node_id: str
    code_version_id: str
    status: ExecutionStatus
    sandbox_result: Optional[SandboxResult] = None
    validation_passed: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_state: Optional[str] = None

class ExecutionRunner:
    def __init__(self):
        self._executions: Dict[str, ExecutionRecord] = {}

    def run(self, code_version: str, dataset_context: dict, workflow_id: str, node_id: str) -> ExecutionRecord:
        execution_id = str(uuid.uuid4())
        record = ExecutionRecord(
            execution_id=execution_id,
            workflow_id=workflow_id,
            node_id=node_id,
            code_version_id=code_version,
            status=ExecutionStatus.QUEUED
        )
        self._executions[execution_id] = record
        
        if not code_version:
            record.status = ExecutionStatus.FAILED
            record.error_state = "CODE_INVALID"
            return record
            
        record.status = ExecutionStatus.RUNNING
        record.started_at = str(time.time())
        
        res = python_sandbox.execute(code_version, dataset_context)
        
        record.sandbox_result = res
        record.status = res.status
        record.completed_at = str(time.time())
        
        self._executions[execution_id] = record
        return record

    def get_status(self, execution_id: str) -> Optional[ExecutionRecord]:
        return self._executions.get(execution_id)

    def cancel(self, execution_id: str) -> bool:
        record = self._executions.get(execution_id)
        if record and record.status in [ExecutionStatus.QUEUED, ExecutionStatus.RUNNING]:
            python_sandbox.cancel(execution_id)
            record.status = ExecutionStatus.CANCELLED
            return True
        return False

    def get_artifacts(self, execution_id: str) -> Dict:
        record = self._executions.get(execution_id)
        if record and record.sandbox_result:
            return {
                "plots": record.sandbox_result.plots,
                "tables": record.sandbox_result.output_tables
            }
        return {}

execution_runner = ExecutionRunner()
