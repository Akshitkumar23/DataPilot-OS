"""
DataPilot OS — Request Context & Lifecycle Metadata.
Provides thread-safe, async-safe request-scoped metadata via contextvars.
Tracks request ID, dataset ID/fingerprint, model version, registry version,
and fine-grained phase latencies across the analytical pipeline.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.ds_engine.repair_budget import RepairBudgetManager


class RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    request_id: str = Field(default="")
    dataset_id: Optional[str] = None
    dataset_hash: Optional[str] = None
    model_version: Optional[str] = None
    registry_version: str = "2.0.0"
    phase_timings: Dict[str, float] = Field(default_factory=dict)
    repair_budget_manager: Optional[RepairBudgetManager] = None
    created_at: float = Field(default_factory=time.time)

    # Internal tracking for active phase starts
    _phase_starts: Dict[str, float] = {}

    def start_phase(self, phase_name: str) -> None:
        """Mark the start timestamp of a pipeline phase."""
        self._phase_starts[phase_name] = time.time()

    def end_phase(self, phase_name: str) -> float:
        """Mark the end timestamp of a pipeline phase and record latency in ms."""
        start = self._phase_starts.pop(phase_name, None)
        if start is not None:
            duration_ms = round((time.time() - start) * 1000, 2)
        else:
            duration_ms = 0.0
        self.phase_timings[phase_name] = duration_ms
        return duration_ms

    def get_repair_budget_manager(self, max_attempts: int = 2) -> RepairBudgetManager:
        """Get or initialize the request-local repair budget manager."""
        if self.repair_budget_manager is None:
            self.repair_budget_manager = RepairBudgetManager(max_attempts=max_attempts)
        return self.repair_budget_manager

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "model_version": self.model_version,
            "registry_version": self.registry_version,
            "phase_timings": dict(self.phase_timings),
        }


_current_request_context: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "current_request_context", default=None
)


def get_request_context() -> RequestContext:
    """Return the active RequestContext, or an ephemeral context if none active."""
    ctx = _current_request_context.get()
    if ctx is None:
        ctx = RequestContext(request_id="ephemeral")
        _current_request_context.set(ctx)
    return ctx


def set_request_context(ctx: RequestContext) -> contextvars.Token:
    """Set the active RequestContext."""
    return _current_request_context.set(ctx)


def reset_request_context(token: Optional[contextvars.Token] = None) -> None:
    """Reset the RequestContext back to the previous token, or clear if None."""
    if token is not None:
        _current_request_context.reset(token)
    else:
        _current_request_context.set(None)


@contextmanager
def request_context_scope(
    request_id: str,
    dataset_id: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    model_version: Optional[str] = None,
    registry_version: str = "2.0.0",
    max_repair_attempts: int = 2,
) -> Generator[RequestContext, None, None]:
    """Context manager for running a block within an isolated RequestContext."""
    ctx = RequestContext(
        request_id=request_id,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        model_version=model_version,
        registry_version=registry_version,
        repair_budget_manager=RepairBudgetManager(max_attempts=max_repair_attempts),
    )
    token = set_request_context(ctx)
    try:
        yield ctx
    finally:
        reset_request_context(token)
