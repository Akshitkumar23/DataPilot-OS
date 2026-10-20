"""
DataPilot OS — Analysis Decision Contract.
Defines canonical analytical decision states (execute, clarify, unsupported, unsafe)
and structured decision records linking grounding evidence to runtime execution.
Zero domain-specific heuristics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DecisionType(str, Enum):
    EXECUTE = "execute"
    CLARIFY = "clarify"
    UNSUPPORTED = "unsupported"
    UNSAFE = "unsafe"


# Canonical alias for compatibility
AnalysisDecision = DecisionType


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    decision: DecisionType
    reason: Optional[str] = None
    candidate_columns: List[str] = Field(default_factory=list)
    clarification_prompt: Optional[str] = None
    suggested_columns: Dict[str, List[str]] = Field(default_factory=dict)
    grounded_columns: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.decision == DecisionType.EXECUTE

    @property
    def requires_clarification(self) -> bool:
        return self.decision == DecisionType.CLARIFY

    @property
    def is_unsupported(self) -> bool:
        return self.decision == DecisionType.UNSUPPORTED

    @property
    def is_unsafe(self) -> bool:
        return self.decision == DecisionType.UNSAFE


def make_execute_decision(
    grounded_columns: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Construct an explicit execution decision record."""
    return DecisionRecord(
        decision=DecisionType.EXECUTE,
        grounded_columns=list(grounded_columns or []),
        metadata=dict(metadata or {}),
    )


def make_clarify_decision(
    reason: str,
    candidate_columns: Optional[List[str]] = None,
    suggested_columns: Optional[Dict[str, List[str]]] = None,
    clarification_prompt: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Construct an explicit clarification decision record."""
    prompt = clarification_prompt or reason
    return DecisionRecord(
        decision=DecisionType.CLARIFY,
        reason=reason,
        candidate_columns=list(candidate_columns or []),
        suggested_columns=dict(suggested_columns or {}),
        clarification_prompt=prompt,
        metadata=dict(metadata or {}),
    )


def make_unsupported_decision(
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Construct an explicit unsupported operation decision record."""
    return DecisionRecord(
        decision=DecisionType.UNSUPPORTED,
        reason=reason,
        metadata=dict(metadata or {}),
    )


def make_unsafe_decision(
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> DecisionRecord:
    """Construct an explicit unsafe operation decision record."""
    return DecisionRecord(
        decision=DecisionType.UNSAFE,
        reason=reason,
        metadata=dict(metadata or {}),
    )
