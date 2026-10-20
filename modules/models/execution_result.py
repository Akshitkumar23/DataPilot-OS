from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

class LifecycleState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    PLAN_VALIDATION = "PLAN_VALIDATION"
    EXECUTING = "EXECUTING"
    VALIDATING_RESULT = "VALIDATING_RESULT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    EXECUTION_ERROR = "execution_error"
    EMPTY_RESULT = "empty_result"
    INVALID_RESULT = "invalid_result"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS_INTENT = "ambiguous_intent"

class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    status: ExecutionStatus
    response: str
    code: Optional[str] = None
    plot_image: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    phase: Optional[str] = None
    request_id: Optional[str] = None


class QueryErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    code: str
    error_code: Optional[str] = None
    message: str
    required_action: str = ""
    stage: Optional[str] = None
    phase: Optional[str] = None
    recoverable: bool = False
    request_id: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    suggested_options: List[str] = Field(default_factory=list)
    slot_filling: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.error_code:
            self.error_code = self.code
        if not self.stage and self.phase:
            self.stage = self.phase
        if not self.phase and self.stage:
            self.phase = self.stage
        if self.details and not self.diagnostics:
            self.diagnostics = self.details
        elif self.diagnostics and not self.details:
            self.details = self.diagnostics

CanonicalError = QueryErrorDetail

class CanonicalQueryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    request_id: Optional[str] = None
    status: Any  # "SUCCEEDED", "NEEDS_CLARIFICATION", "UNSUPPORTED", "UNSAFE", "FAILED"
    decision: Optional[str] = None  # "execute", "clarify", "unsupported", "unsafe"
    phase: Optional[str] = None   # "SUCCEEDED", "FAILED", or active LifecycleState
    analysis_spec: Optional[Dict[str, Any]] = None
    execution_graph: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    clarification: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    slot_filling: Optional[Dict[str, Any]] = None
    inferred_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[Any] = None
    workflow_id: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.analysis_spec is None and self.plan is not None:
            self.analysis_spec = self.plan
        elif self.plan is None and self.analysis_spec is not None:
            self.plan = self.analysis_spec

        if self.manifest is None and isinstance(self.result, dict):
            self.manifest = self.result.get("manifest")
        if self.validation is None and isinstance(self.result, dict):
            self.validation = self.result.get("validation")

        if self.decision is None:
            s = (self.status or "").upper().strip()
            if s == "SUCCEEDED":
                self.decision = "execute"
            elif s == "NEEDS_CLARIFICATION":
                self.decision = "clarify"
            elif s == "UNSUPPORTED":
                self.decision = "unsupported"
            elif s == "UNSAFE":
                self.decision = "unsafe"
            elif s == "FAILED":
                code = ""
                if isinstance(self.error, dict):
                    code = self.error.get("code", "")
                elif hasattr(self.error, "code"):
                    code = getattr(self.error, "code", "")
                code_upper = str(code).upper()
                if "UNSUPPORTED" in code_upper:
                    self.decision = "unsupported"
                elif "LEAKAGE" in code_upper or "UNSAFE" in code_upper:
                    self.decision = "unsafe"
                elif "CLARIF" in code_upper or "AMBIGUOUS" in code_upper:
                    self.decision = "clarify"
                else:
                    self.decision = "failed"
