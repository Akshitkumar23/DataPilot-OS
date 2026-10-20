"""
DataPilot OS — Loop Engine (Graph Control Layer).
The Loop Controller sits INSIDE the Graph Executor as a decision authority.
It decides: continue current graph? replan? repair code? request clarification? stop?

Loops are NOT generic retries. Every loop has:
- A measurable condition
- A reason for re-entry
- A maximum iteration budget  
- Changed action on next iteration (or LOOP_NO_PROGRESS)
- An audit record
"""
import hashlib
import json
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel

from backend.graph_engine.state import GraphExecutionState

class LoopReason(str, Enum):
    INTENT_INCOMPLETE = "INTENT_INCOMPLETE"
    CODE_REPAIR_REQUIRED = "CODE_REPAIR_REQUIRED"
    RESULT_INSUFFICIENT = "RESULT_INSUFFICIENT"
    EVIDENCE_MISSING = "EVIDENCE_MISSING"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VISUALIZATION_INVALID = "VISUALIZATION_INVALID"

class LoopDecision(BaseModel):
    should_continue: bool
    reason: Optional[LoopReason]
    action: str
    iteration: int
    max_iterations: int
    added_nodes: List[str]
    error_state: Optional[str]
    audit_record: Dict[str, Any]

class LoopState(BaseModel):
    loop_id: str
    iteration: int = 0
    history: List[Dict[str, Any]] = []
    last_action_hash: Optional[str] = None

class LoopController:
    def __init__(self):
        self.loop_states: Dict[str, LoopState] = {}

    def _get_action_hash(self, action_params: Dict[str, Any]) -> str:
        s = json.dumps(action_params, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def _assess_result_quality(self, result: Dict[str, Any]) -> Tuple[bool, Optional[LoopReason]]:
        if not result:
            return False, LoopReason.RESULT_INSUFFICIENT
        if result.get("error"):
            return False, LoopReason.VALIDATION_FAILED
        return True, None

    def _check_evidence_sufficiency(self, result: Dict[str, Any], user_objective: str) -> bool:
        if not result.get("evidence"):
            return False
        return True

    def evaluate(self, state: GraphExecutionState, result: Dict[str, Any], loop_id: str, max_iterations: int = 3) -> LoopDecision:
        if loop_id not in self.loop_states:
            self.loop_states[loop_id] = LoopState(loop_id=loop_id)
        
        ls = self.loop_states[loop_id]
        ls.iteration += 1
        
        if ls.iteration > max_iterations:
            return LoopDecision(
                should_continue=False,
                reason=None,
                action="stop_budget",
                iteration=ls.iteration,
                max_iterations=max_iterations,
                added_nodes=[],
                error_state="LOOP_BUDGET_EXCEEDED",
                audit_record={"msg": "Budget exceeded"}
            )
            
        action_params = {"action": result.get("proposed_action", "default")}
        current_hash = self._get_action_hash(action_params)
        
        if ls.last_action_hash == current_hash:
            return LoopDecision(
                should_continue=False,
                reason=None,
                action="stop_no_progress",
                iteration=ls.iteration,
                max_iterations=max_iterations,
                added_nodes=[],
                error_state="LOOP_NO_PROGRESS",
                audit_record={"msg": "No progress"}
            )
            
        is_suff, reason = self._assess_result_quality(result)
        if not is_suff:
            return LoopDecision(
                should_continue=True,
                reason=reason,
                action="replan",
                iteration=ls.iteration,
                max_iterations=max_iterations,
                added_nodes=[],
                error_state=None,
                audit_record={"msg": "Result insufficient"}
            )
            
        return LoopDecision(
            should_continue=False,
            reason=None,
            action="stop_success",
            iteration=ls.iteration,
            max_iterations=max_iterations,
            added_nodes=[],
            error_state=None,
            audit_record={"msg": "Success"}
        )

    def record_iteration(self, state: GraphExecutionState, loop_id: str, action: str, result: Dict[str, Any]) -> None:
        if loop_id not in self.loop_states:
            self.loop_states[loop_id] = LoopState(loop_id=loop_id)
        ls = self.loop_states[loop_id]
        ls.history.append({"action": action, "result": result})
        ls.last_action_hash = self._get_action_hash({"action": action})

loop_controller = LoopController()
