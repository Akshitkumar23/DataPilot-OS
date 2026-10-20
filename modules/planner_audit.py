"""
DataPilot OS — Planner Audit & Telemetry Contract.
Records structured metadata about capability retrieval, candidate plans,
validation failures, repair loop attempts, and final decisions.
Zero sensitive row/cell leakage.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PlannerAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str
    query_fingerprint: str
    model_name: str
    retrieved_tools: List[str]
    locked_tool: Optional[str] = None
    initial_raw_response: Optional[str] = None
    validation_failures: List[str] = Field(default_factory=list)
    repair_attempts: int = 0
    repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    final_decision: str = "execute"
    final_tool_call: Optional[Dict[str, Any]] = None
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlannerAuditor:
    """
    In-memory, bounded thread-safe audit store for semantic planner operations.
    Excludes sensitive dataset values.
    """

    def __init__(self, max_records: int = 500):
        self._records: deque[PlannerAuditRecord] = deque(maxlen=max_records)

    @staticmethod
    def fingerprint_query(query: str) -> str:
        """Compute SHA-256 fingerprint of normalized query text."""
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def record(self, audit: PlannerAuditRecord) -> None:
        """Store an audit record safely."""
        self._records.append(audit)
        logger.info(
            "PLANNER_AUDIT_LOG | Request: %s | Retrieved: %s | Locked: %s | Repairs: %d | Decision: %s",
            audit.request_id,
            audit.retrieved_tools,
            audit.locked_tool,
            audit.repair_attempts,
            audit.final_decision,
        )

    def get_recent(self, limit: int = 50) -> List[PlannerAuditRecord]:
        """Return the most recent audit records."""
        records = list(self._records)
        return records[-limit:]

    def get_history(self, request_id: Optional[str] = None) -> List[PlannerAuditRecord]:
        """Return audit records, optionally filtered by request_id."""
        if request_id:
            return [r for r in self._records if r.request_id == request_id]
        return list(self._records)

    def clear(self) -> None:
        """Clear audit history."""
        self._records.clear()


planner_auditor = PlannerAuditor()
