"""
DataPilot OS — Explicit Error State System.
Every failure is explicit, diagnosable, and recoverable.
No silent fallbacks. No template substitutions.
"""
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class ErrorState(str, Enum):
    OLLAMA_UNAVAILABLE = "OLLAMA_UNAVAILABLE"
    OLLAMA_MODEL_UNAVAILABLE = "OLLAMA_MODEL_UNAVAILABLE"
    OLLAMA_INVALID_RESPONSE = "OLLAMA_INVALID_RESPONSE"
    OLLAMA_TIMEOUT = "OLLAMA_TIMEOUT"
    INTENT_INCOMPLETE = "INTENT_INCOMPLETE"
    INTENT_AMBIGUOUS = "INTENT_AMBIGUOUS"
    DATASET_INVALID = "DATASET_INVALID"
    DATASET_NOT_CONNECTED = "DATASET_NOT_CONNECTED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    CAPABILITY_NOT_SUPPORTED = "CAPABILITY_NOT_SUPPORTED"
    CAPABILITY_INCOMPATIBLE = "CAPABILITY_INCOMPATIBLE"
    GRAPH_INVALID = "GRAPH_INVALID"
    CODE_INVALID = "CODE_INVALID"
    CODE_UNSAFE = "CODE_UNSAFE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    RESULT_INVALID = "RESULT_INVALID"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    PRODUCTION_INCOMPATIBLE = "PRODUCTION_INCOMPATIBLE"
    PRODUCTION_BLOCKED = "PRODUCTION_BLOCKED"
    LOOP_NO_PROGRESS = "LOOP_NO_PROGRESS"
    LOOP_BUDGET_EXCEEDED = "LOOP_BUDGET_EXCEEDED"
    # FW-04 Relational and Semantic Governance Error States
    OLLAMA_SCHEMA_VALIDATION_ERROR = "OLLAMA_SCHEMA_VALIDATION_ERROR"
    OLLAMA_UNSUPPORTED_INTENT = "OLLAMA_UNSUPPORTED_INTENT"
    RELATION_NOT_FOUND = "RELATION_NOT_FOUND"
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    INCOMPATIBLE_JOIN_TYPES = "INCOMPATIBLE_JOIN_TYPES"
    RELATIONSHIP_NOT_VERIFIED = "RELATIONSHIP_NOT_VERIFIED"
    JOIN_REQUIRES_CLARIFICATION = "JOIN_REQUIRES_CLARIFICATION"

class DataPilotError(Exception):
    def __init__(
        self,
        code: ErrorState,
        message: str,
        stage: str = "execution",
        recoverable: bool = False,
        required_action: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.stage = stage
        self.recoverable = recoverable
        self.required_action = required_action
        self.details = details or {}
        super().__init__(self.message)

    def to_response(self) -> dict:
        """Convert to API-ready response dict."""
        code_val = self.code.value if hasattr(self.code, "value") else str(self.code)
        return {
            "error": True,
            "code": code_val,
            "error_code": code_val,
            "message": self.message,
            "stage": self.stage,
            "recoverable": self.recoverable,
            "required_action": self.required_action,
            "details": self.details
        }

    def to_envelope(self):
        from backend.core.error_contract import create_error_envelope
        code_val = self.code.value if hasattr(self.code, "value") else str(self.code)
        return create_error_envelope(
            code=code_val,
            message=self.message,
            required_action=self.required_action,
            stage=self.stage,
            recoverable=self.recoverable,
            diagnostics=self.details
        )

# Pre-built error factories for common cases
def ollama_unavailable_error() -> DataPilotError:
    return DataPilotError(
        code=ErrorState.OLLAMA_UNAVAILABLE,
        message="Ollama is not running. DataPilot requires a local Ollama instance for AI analysis.",
        stage="intent_engine",
        recoverable=True,
        required_action="Start Ollama: run 'ollama serve' in a terminal, then pull a model with 'ollama pull qwen2.5-coder:7b'",
        details={"ollama_url": "http://localhost:11434", "docs": "https://ollama.com"}
    )

def ollama_timeout_error(timeout_sec: float) -> DataPilotError:
    return DataPilotError(
        code=ErrorState.OLLAMA_TIMEOUT,
        message=f"Ollama did not respond within {timeout_sec}s. The model may be loading or the system is under load.",
        stage="ollama_service",
        recoverable=True,
        required_action="Wait for the model to fully load, then retry. Or switch to a smaller/faster model.",
        details={"timeout_sec": timeout_sec}
    )

def capability_incompatible_error(capability: str, reason: str, available_columns: list) -> DataPilotError:
    return DataPilotError(
        code=ErrorState.CAPABILITY_INCOMPATIBLE,
        message=f"Capability '{capability}' cannot run on the connected dataset: {reason}",
        stage="capability_engine",
        recoverable=True,
        required_action="Connect a dataset with the required features, or choose a different analysis.",
        details={"capability": capability, "reason": reason, "available_columns": available_columns}
    )

def loop_no_progress_error(iteration: int, reason: str) -> DataPilotError:
    return DataPilotError(
        code=ErrorState.LOOP_NO_PROGRESS,
        message=f"Loop iteration {iteration}: proposed action is identical to previous failed attempt.",
        stage="loop_engine",
        recoverable=False,
        required_action="Modify your query or connect a different dataset. The AI could not find a different approach.",
        details={"iteration": iteration, "reason": reason}
    )

def code_unsafe_error(violation: str) -> DataPilotError:
    return DataPilotError(
        code=ErrorState.CODE_UNSAFE,
        message=f"Generated code contains unsafe operation: {violation}",
        stage="code_validator",
        recoverable=False,
        required_action="The generated code was blocked for security. Try a simpler or more specific query.",
        details={"violation": violation}
    )
