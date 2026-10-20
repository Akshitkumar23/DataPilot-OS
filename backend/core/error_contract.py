"""
DataPilot OS — Canonical Error Contract & Envelopes.
Defines the authoritative error schema shared across backend APIs and frontend UI.
Eliminates divergent error shapes, normalizes HTTP statuses, and sanitizes raw stack traces.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class CanonicalStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"
    UNSAFE = "UNSAFE"
    FAILED = "FAILED"


class CanonicalDecision(str, Enum):
    EXECUTE = "execute"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"
    UNSAFE = "unsafe"


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    error: bool = Field(default=True, description="Always true for error and clarification envelopes")
    code: str = Field(..., description="Machine-readable invariant error or status code")
    error_code: Optional[str] = Field(None, description="Compatibility alias for code")
    message: str = Field(..., description="User-facing explanation without raw tracebacks")
    required_action: str = Field(default="", description="Concrete next step or guidance for user")
    stage: str = Field(default="execution", description="Lifecycle stage where failure occurred")
    recoverable: bool = Field(default=False, description="Whether client or user can retry/recover")
    candidate_columns: List[str] = Field(default_factory=list, description="Candidate columns for clarification")
    suggested_options: List[str] = Field(default_factory=list, description="Actionable options for clarification")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Diagnostics for development and audit")
    request_id: Optional[str] = Field(None, description="Correlation ID for tracing")

    def model_post_init(self, __context: Any) -> None:
        if not self.error_code:
            self.error_code = self.code

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def sanitize_traceback(raw_text: str) -> str:
    """Remove Python internal traces, memory addresses, and file paths for user safety."""
    if not raw_text:
        return "An internal system error occurred."
    
    # Strip Traceback header and subsequent lines
    if "Traceback (most recent call last)" in raw_text:
        parts = raw_text.split("Traceback (most recent call last)")
        raw_text = parts[0].strip() or "Execution halted due to an internal exception."
    
    # Remove file paths like File "C:\..." line ...
    cleaned = re.sub(r'File\s+"[^"]+",\s+line\s+\d+.*', '', raw_text)
    # Remove memory addresses like <... at 0x0000012345678>
    cleaned = re.sub(r'<[^>]+at\s+0x[0-9a-fA-F]+>', '<internal_object>', cleaned)
    
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    final_text = " ".join(lines)
    return final_text or "An internal error occurred during analytical execution."


def create_error_envelope(
    code: str,
    message: str,
    required_action: str = "",
    stage: str = "execution",
    recoverable: bool = False,
    candidate_columns: Optional[List[str]] = None,
    suggested_options: Optional[List[str]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorEnvelope:
    safe_msg = sanitize_traceback(message)
    return ErrorEnvelope(
        error=True,
        code=code,
        error_code=code,
        message=safe_msg,
        required_action=required_action,
        stage=stage,
        recoverable=recoverable,
        candidate_columns=candidate_columns or [],
        suggested_options=suggested_options or [],
        diagnostics=diagnostics or {},
        request_id=request_id
    )


def map_status_to_http_code(status: str, error_code: Optional[str] = None) -> int:
    """Canonical mapping from business status to HTTP status code."""
    status_upper = (status or "").upper().strip()

    if status_upper == CanonicalStatus.SUCCEEDED.value:
        return 200
    if status_upper == CanonicalStatus.NEEDS_CLARIFICATION.value:
        return 422
    if status_upper in [
        CanonicalStatus.UNSUPPORTED.value,
        CanonicalStatus.UNSAFE.value,
    ]:
        return 400

    # Status is FAILED: inspect error code
    code_upper = (error_code or "").upper().strip()
    if code_upper in [
        "INVALID_QUERY", "EMPTY_QUERY", "INVALID_REQUEST", "INVALID_ARGUMENT", "BAD_REQUEST",
        "UNSUPPORTED_OPERATION", "TARGET_LEAKAGE_DETECTED", "SCIENTIFIC_FEASIBILITY_VIOLATION",
    ]:
        return 400
    if code_upper in [
        "OLLAMA_UNAVAILABLE", "OLLAMA_MODEL_NOT_FOUND", "OLLAMA_MODEL_UNAVAILABLE",
        "OLLAMA_TIMEOUT", "INTENT_PLANNER_UNAVAILABLE", "SERVICE_UNAVAILABLE",
    ]:
        return 503
    if code_upper in [
        "INTENT_PLAN_INVALID", "PLAN_VALIDATION_FAILED", "AMBIGUOUS_INTENT",
        "AMBIGUOUS_COLUMN", "COLUMN_NOT_FOUND", "INVALID_COLUMN",
        "INVALID_COLUMN_TYPE", "NO_TEMPORAL_COLUMN", "NO_DATE_COLUMN",
        "INVALID_OPERATOR", "INVALID_AGGREGATION",
        "MISSING_REQUIRED_INFORMATION", "CODE_VALIDATION_FAILED", "SECURITY_VIOLATION",
    ]:
        return 422

    return 500
