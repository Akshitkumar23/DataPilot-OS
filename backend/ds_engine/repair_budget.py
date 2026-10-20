"""
DataPilot OS — Finite Self-Healing Repair Budget Manager.
Enforces finite self-repair bounds (max 2 attempts), diff-checking for progress,
and zero-hallucination safety guards during automated plan recovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field

from backend.core.error_states import DataPilotError, ErrorState

logger = logging.getLogger(__name__)


class RepairAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_number: int
    original_plan_hash: str
    proposed_plan_hash: str
    is_identical: bool
    error_diagnosed: str
    adjustment_summary: str
    success: bool
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RepairBudgetManager:
    """
    Manages finite self-repair budget with strict progress verification.
    """
    MAX_ATTEMPTS: int = 2

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self.attempts_used = 0
        self.history: List[RepairAttemptRecord] = []

    @staticmethod
    def compute_plan_hash(plan: Any) -> str:
        """Compute canonical JSON SHA-256 hash of a plan structure."""
        if hasattr(plan, "model_dump_json"):
            canonical_str = plan.model_dump_json()
        elif hasattr(plan, "model_dump"):
            canonical_str = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
        elif isinstance(plan, dict):
            canonical_str = json.dumps(plan, sort_keys=True, default=str)
        else:
            canonical_str = str(plan)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def can_attempt_repair(self) -> bool:
        """Return True if repair attempts remain within budget."""
        return self.attempts_used < self.max_attempts

    def check_progress_and_record(
        self,
        original_plan: Any,
        proposed_plan: Any,
        error_diagnosed: str = "",
        adjustment_summary: str = "",
        success: bool = False,
    ) -> Optional[DataPilotError]:
        """
        Record a repair attempt.
        Fails fast with LOOP_NO_PROGRESS if proposed plan is identical to original.
        Fails fast with LOOP_BUDGET_EXCEEDED if maximum attempts exceeded.
        """
        self.attempts_used += 1

        orig_hash = self.compute_plan_hash(original_plan)
        prop_hash = self.compute_plan_hash(proposed_plan)
        is_identical = (orig_hash == prop_hash)

        record = RepairAttemptRecord(
            attempt_number=self.attempts_used,
            original_plan_hash=orig_hash,
            proposed_plan_hash=prop_hash,
            is_identical=is_identical,
            error_diagnosed=error_diagnosed,
            adjustment_summary=adjustment_summary,
            success=success,
        )
        self.history.append(record)

        # 1. No Progress Check
        if is_identical:
            logger.warning(
                f"[RepairBudget] LOOP_NO_PROGRESS on attempt {self.attempts_used}/{self.max_attempts}. "
                "Proposed plan is identical to failed plan."
            )
            return DataPilotError(
                code=ErrorState.LOOP_NO_PROGRESS,
                message=(
                    f"Self-repair attempt #{self.attempts_used}: proposed repair is identical to "
                    "the failed plan. No progress made."
                ),
                stage="repair_engine",
                recoverable=False,
                required_action="Modify query, provide explicit columns, or adjust analysis parameters.",
                details={"attempt": self.attempts_used, "plan_hash": orig_hash},
            )

        # 2. Budget Exceeded Check
        if not success and self.attempts_used >= self.max_attempts:
            logger.warning(
                f"[RepairBudget] LOOP_BUDGET_EXCEEDED after {self.attempts_used} attempts."
            )
            return DataPilotError(
                code=ErrorState.LOOP_BUDGET_EXCEEDED,
                message=(
                    f"Self-repair budget exhausted ({self.max_attempts} attempts reached) without "
                    "producing a valid plan."
                ),
                stage="repair_engine",
                recoverable=False,
                required_action="Refine your query or check dataset schema constraints.",
                details={"max_attempts": self.max_attempts, "attempts_used": self.attempts_used},
            )

        return None
