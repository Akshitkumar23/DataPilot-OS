"""
DataPilot OS — Result Validator (Canonical Forwarding Module).
Authoritative implementation resides in backend.ds_engine.result_validator.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from backend.ds_engine.result_validator import (
    ResultValidator,
    SemanticResultValidator,
    ResultValidationResult
)


class ResultValidationCheck(BaseModel):
    check_name: str
    passed: bool
    message: str
    severity: str


class ResultValidationReport(BaseModel):
    is_valid: bool
    evidence_sufficient: bool
    checks: List[ResultValidationCheck] = []
    warnings: List[str] = []
    missing_evidence: List[str] = []
    suggested_additional_analyses: List[str] = []


result_validator = ResultValidator()

__all__ = [
    "ResultValidator",
    "result_validator",
    "SemanticResultValidator",
    "ResultValidationResult",
    "ResultValidationCheck",
    "ResultValidationReport"
]
