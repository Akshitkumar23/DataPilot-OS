"""
DataPilot OS — Result Validator (Canonical Forwarding Module).
Authoritative implementation resides in backend.ds_engine.result_validator.
"""

from backend.ds_engine.result_validator import (
    ResultValidator,
    SemanticResultValidator,
    ResultValidationResult
)

__all__ = ["ResultValidator", "SemanticResultValidator", "ResultValidationResult"]
