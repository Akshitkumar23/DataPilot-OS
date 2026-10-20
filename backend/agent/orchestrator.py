"""
DataPilot OS — Workspace Agent Task Orchestrator.
Coordinates multi-step task execution, step approvals, file backups for rollback,
cancellation, verification, and honest task summary reporting.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from backend.agent.audit import agent_audit
from backend.agent.config import agent_config
from backend.agent.permissions import (
    AgentAction,
    AgentPolicy,
    PermissionDeniedError,
    workspace_registry,
)
from backend.agent.planner import TaskStep, WorkspaceTaskPlan
from backend.agent.registry import ToolCategory, agent_tool_registry
from backend.agent import tools

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    def __init__(self):
        self._tasks: Dict[str, WorkspaceTaskPlan] = {}
        # Backups for task-scoped rollback: task_id -> { path: original_bytes_or_None }
        self._backups: Dict[str, Dict[str, Optional[bytes]]] = {}

    def register_plan(self, plan: WorkspaceTaskPlan) -> WorkspaceTaskPlan:
        self._tasks[plan.task_id] = plan
        self._backups[plan.task_id] = {}
        if plan.is_ambiguous or plan.status == "NEEDS_CLARIFICATION":
            plan.status = "NEEDS_CLARIFICATION"
            return plan
        # Evaluate initial state
        has_approval_needed = any(s.requires_approval and s.status == "PENDING" for s in plan.steps)
        if has_approval_needed:
            plan.status = "AWAITING_APPROVAL"
        return plan

    def get_task(self, task_id: str) -> Optional[WorkspaceTaskPlan]:
        return self._tasks.get(task_id)

    def execute_task(self, task_id: str) -> WorkspaceTaskPlan:
        """Executes or resumes a registered task plan."""
        plan = self._tasks.get(task_id)
        if not plan:
            raise ValueError(f"Task '{task_id}' not found.")

        if plan.is_ambiguous or plan.status == "NEEDS_CLARIFICATION":
            plan.status = "NEEDS_CLARIFICATION"
            plan.error = plan.clarification_needed or "Task requires clarification before execution."
            return plan

        plan.status = "EXECUTING"
        files_inspected = set(plan.summary.get("files_inspected", []))
        files_changed = set(plan.summary.get("files_changed", []))
        commands_executed = list(plan.summary.get("commands_executed", []))
        unresolved_errors = list(plan.summary.get("unresolved_errors", []))

        for step in plan.steps:
            if step.status in ("SUCCEEDED", "SKIPPED", "CANCELLED"):
                continue

            # If step requires approval and hasn't been approved yet, pause
            if step.requires_approval:
                step.status = "AWAITING_APPROVAL"
                plan.status = "AWAITING_APPROVAL"
                self._update_summary(plan, files_inspected, files_changed, commands_executed, unresolved_errors)
                return plan

            step.status = "RUNNING"
            tool_name = step.tool_name
            args = step.arguments.copy()

            tool_obj = agent_tool_registry.get(tool_name)
            category = tool_obj.category if tool_obj else None

            # Record backup for files before writing/patching/deleting
            target_p = args.get("path")
            if target_p and category in (ToolCategory.WORKSPACE_WRITE, ToolCategory.DESTRUCTIVE):
                if target_p not in self._backups[task_id]:
                    if os.path.exists(target_p) and os.path.isfile(target_p):
                        try:
                            with open(target_p, "rb") as bf:
                                self._backups[task_id][target_p] = bf.read()
                        except Exception:
                            self._backups[task_id][target_p] = None
                    else:
                        self._backups[task_id][target_p] = None  # Did not exist originally

            # Execute tool
            res = self._dispatch_tool(tool_name, args, task_id)

            if res.success:
                step.status = "SUCCEEDED"
                step.result = res.output
                if category == ToolCategory.READ_ONLY:
                    if res.path:
                        files_inspected.add(res.path)
                elif category == ToolCategory.WORKSPACE_WRITE:
                    if res.path:
                        files_changed.add(res.path)
                elif category == ToolCategory.DESTRUCTIVE:
                    if res.path:
                        files_changed.add(f"[DELETED] {res.path}")
                elif category == ToolCategory.EXECUTE_COMMAND:
                    commands_executed.append({
                        "command": args.get("command"),
                        "exit_code": (res.output or {}).get("exit_code"),
                        "duration_ms": (res.output or {}).get("duration_ms"),
                    })
            else:
                step.status = "FAILED"
                step.error = res.error or res.error_code
                unresolved_errors.append(f"Step {step.step_id} ({tool_name}) failed: {step.error}")
                plan.status = "FAILED"
                plan.error = step.error
                self._update_summary(plan, files_inspected, files_changed, commands_executed, unresolved_errors)
                return plan

        # Run verification checks if steps succeeded
        checks_passed = []
        if plan.verification_criteria:
            plan.status = "VERIFYING"
            v_res = tools.run_verification(task_id, plan.verification_criteria)
            checks_passed = (v_res.output or {}).get("checks", [])
            if not v_res.success:
                failures = [c for c in (v_res.output or {}).get("checks", []) if not c.get("passed")]
                repair_attempts = plan.summary.get("repair_attempts", 0)
                if repair_attempts < agent_config.MAX_REPAIR_ATTEMPTS:
                    logger.info(f"Task {task_id}: Verification failed. Attempting bounded repair (attempt {repair_attempts + 1}/{agent_config.MAX_REPAIR_ATTEMPTS})...")
                    plan.summary["repair_attempts"] = repair_attempts + 1
                    try:
                        from backend.agent.planner import generate_repair_steps
                        repair_steps = generate_repair_steps(plan, failures)
                        if repair_steps:
                            plan.steps.extend(repair_steps)
                            plan.status = "EXECUTING"
                            self._update_summary(plan, files_inspected, files_changed, commands_executed, unresolved_errors, checks_passed)
                            has_approval_needed = any(s.requires_approval and s.status == "PENDING" for s in repair_steps)
                            if has_approval_needed:
                                plan.status = "AWAITING_APPROVAL"
                                return plan
                            return self.execute_task(task_id)
                    except Exception as e:
                        logger.warning(f"Repair attempt failed: {e}")
                        unresolved_errors.append(f"Repair proposal failed: {str(e)}")

                plan.status = "FAILED"
                plan.error = "One or more verification checks failed."
                unresolved_errors.append("Verification failed.")
            else:
                plan.status = "COMPLETED"
        else:
            plan.status = "COMPLETED"

        self._update_summary(plan, files_inspected, files_changed, commands_executed, unresolved_errors, checks_passed)
        return plan

    def approve_step(self, task_id: str, step_id: str) -> WorkspaceTaskPlan:
        """Approves a pending step by generating and consuming a valid approval token, then resuming execution."""
        plan = self._tasks.get(task_id)
        if not plan:
            raise ValueError(f"Task '{task_id}' not found.")

        target_step = None
        for step in plan.steps:
            if step.step_id == step_id:
                target_step = step
                break

        if not target_step:
            raise ValueError(f"Step '{step_id}' not found in task '{task_id}'.")

        token = f"approval_{uuid.uuid4().hex}"
        workspace_registry.register_approval_token(token)
        target_step.arguments["approval_token"] = token
        target_step.requires_approval = False
        target_step.status = "PENDING"

        # Resume task execution
        return self.execute_task(task_id)

    def reject_step(self, task_id: str, step_id: str, reason: str = "") -> WorkspaceTaskPlan:
        """Rejects a step, stopping task execution."""
        plan = self._tasks.get(task_id)
        if not plan:
            raise ValueError(f"Task '{task_id}' not found.")

        for step in plan.steps:
            if step.step_id == step_id:
                step.status = "FAILED"
                step.error = f"Rejected by user: {reason or 'Operation not approved.'}"
                break

        plan.status = "FAILED"
        plan.error = f"Task halted: Step '{step_id}' was rejected by user."
        curr_unresolved = list(plan.summary.get("unresolved_errors", []))
        curr_unresolved.append(plan.error)
        plan.summary["unresolved_errors"] = curr_unresolved

        agent_audit.log(
            task_id=task_id,
            operation="reject_step",
            approval_status="REJECTED",
            status="HALTED",
            details={"step_id": step_id, "reason": reason},
        )
        return plan

    def cancel_task(self, task_id: str) -> WorkspaceTaskPlan:
        """Cancels an active task and any subprocess currently running."""
        plan = self._tasks.get(task_id)
        if not plan:
            raise ValueError(f"Task '{task_id}' not found.")

        tools.cancel_running_command(task_id)

        for step in plan.steps:
            if step.status in ("PENDING", "AWAITING_APPROVAL", "RUNNING"):
                step.status = "CANCELLED"

        plan.status = "CANCELLED"
        plan.error = "Task cancelled by user."
        agent_audit.log(task_id=task_id, operation="cancel_task", status="CANCELLED")
        return plan

    def rollback_task(self, task_id: str) -> bool:
        """Rolls back filesystem changes made during this task using captured backups."""
        backups = self._backups.get(task_id)
        if not backups:
            return False

        rolled_back = True
        for path, orig_bytes in backups.items():
            try:
                if orig_bytes is None:
                    # File did not exist originally; remove if it exists now
                    if os.path.exists(path):
                        os.remove(path)
                else:
                    # Restore original content
                    with open(path, "wb") as f:
                        f.write(orig_bytes)
            except Exception as e:
                logger.error(f"Rollback failed for '{path}': {e}")
                rolled_back = False

        agent_audit.log(
            task_id=task_id,
            operation="rollback_task",
            status="SUCCESS" if rolled_back else "FAILED",
        )
        return rolled_back

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any], task_id: str) -> tools.ToolResult:
        """Dispatches tool execution dynamically via the agent tool capability registry."""
        return agent_tool_registry.dispatch(tool_name, args, task_id)

    def _update_summary(
        self,
        plan: WorkspaceTaskPlan,
        files_inspected: set,
        files_changed: set,
        commands_executed: list,
        unresolved_errors: list,
        checks_passed: list = None,
    ) -> None:
        plan.summary = {
            "files_inspected": sorted(list(files_inspected)),
            "files_changed": sorted(list(files_changed)),
            "commands_executed": commands_executed,
            "checks_passed": checks_passed or plan.summary.get("checks_passed", []),
            "unresolved_errors": unresolved_errors,
            "repair_attempts": plan.summary.get("repair_attempts", 0),
        }


# Singleton orchestrator
task_orchestrator = TaskOrchestrator()
