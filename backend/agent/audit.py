"""
DataPilot OS — Agent Audit Logging Engine.
Maintains an append-only, structured audit log of all agent operations,
approvals, command runs, and verification statuses with complete secret redaction.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.agent.security import redact_secrets


class AgentAuditRecord(BaseModel):
    task_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operation: str
    path: Optional[str] = None
    command: Optional[str] = None
    approval_status: str = "NOT_REQUIRED"  # APPROVED | AUTO_APPROVED | REJECTED | NOT_REQUIRED
    exit_code: Optional[int] = None
    status: str = "SUCCESS"  # SUCCESS | FAILED | HALTED | CANCELLED
    error_code: Optional[str] = None
    verification_status: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentAuditLogger:
    def __init__(self, log_path: Optional[str] = None):
        if not log_path:
            base_dir = os.getenv("DATAPILOT_DATA_DIR", os.getcwd())
            target_dir = os.path.join(base_dir, ".system_generated")
            os.makedirs(target_dir, exist_ok=True)
            self._log_path = os.path.join(target_dir, "agent_audit.jsonl")
        else:
            self._log_path = log_path
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

        self._lock = threading.Lock()
        self._in_memory_records: List[AgentAuditRecord] = []

    def log(
        self,
        task_id: str,
        operation: str,
        path: Optional[str] = None,
        command: Optional[str] = None,
        approval_status: str = "NOT_REQUIRED",
        exit_code: Optional[int] = None,
        status: str = "SUCCESS",
        error_code: Optional[str] = None,
        verification_status: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AgentAuditRecord:
        """Records an operation with automatic secret redaction."""
        clean_command = redact_secrets(command) if command else None
        clean_details = {}
        if details:
            for k, v in details.items():
                if isinstance(v, str):
                    clean_details[k] = redact_secrets(v)
                else:
                    clean_details[k] = v

        record = AgentAuditRecord(
            task_id=task_id,
            operation=operation,
            path=path,
            command=clean_command,
            approval_status=approval_status,
            exit_code=exit_code,
            status=status,
            error_code=error_code,
            verification_status=verification_status,
            details=clean_details,
        )

        with self._lock:
            self._in_memory_records.append(record)
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")
            except Exception:
                pass

        return record

    def get_records(self, limit: int = 100, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = self._in_memory_records
            if task_id:
                records = [r for r in records if r.task_id == task_id]
            return [r.model_dump() for r in records[-limit:]]


# Singleton audit logger
agent_audit = AgentAuditLogger()
