"""
DataPilot OS — Audit Store.
Tracks production execution audit trail.
"""
import uuid
from datetime import datetime
from typing import List, Any
from pydantic import BaseModel

class AuditRecord(BaseModel):
    audit_id: str
    workflow_id: str
    user_approved_at: str
    production_dataset_id: str
    row_count_processed: int
    execution_time_sec: float
    status: str
    artifacts: List[str]
    timestamp: str

class AuditStore:
    def __init__(self):
        self._audits: List[AuditRecord] = []

    def save_audit(self, record: AuditRecord) -> str:
        # If audit_id is empty, generate one
        if not record.audit_id:
            record.audit_id = str(uuid.uuid4())
        self._audits.append(record)
        return record.audit_id

    def get_audits_for_workflow(self, workflow_id: str) -> List[AuditRecord]:
        return [a for a in self._audits if a.workflow_id == workflow_id]

audit_store = AuditStore()
