"""
DataPilot OS — Agent API Router.
REST endpoints for workspace management, permission policy, natural language task planning,
step approvals/rejections, task cancellation, direct tool invocation, and audit inspection.
Returns standard ErrorEnvelope contracts on failures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agent.audit import agent_audit
from backend.agent.permissions import (
    AgentAction,
    AgentPolicy,
    PermissionDeniedError,
    workspace_registry,
)
from backend.agent.planner import PlannerError, WorkspaceTaskPlan, generate_task_plan
from backend.agent.orchestrator import task_orchestrator
from backend.agent.security import SecurityContainmentError
from backend.agent import tools
from backend.core.error_contract import create_error_envelope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class AuthorizeWorkspaceRequest(BaseModel):
    path: str = Field(..., description="Absolute local directory path to authorize")


class RevokeWorkspaceRequest(BaseModel):
    path: str = Field(..., description="Directory path to revoke from authorized workspaces")


class PolicyRequest(BaseModel):
    policy: str = Field(..., description="Policy mode: review | auto_edit | autonomous")


class PlanTaskRequest(BaseModel):
    prompt: str = Field(..., description="Natural language task or command objective")
    workspace_path: Optional[str] = Field(None, description="Target workspace path")
    context_files: Optional[List[str]] = Field(default_factory=list, description="Known context files")
    model: Optional[str] = Field(None, description="Optional explicit Ollama model to use for planning")


class ExecuteTaskRequest(BaseModel):
    task_id: str = Field(..., description="Unique ID of task to execute")


class StepApprovalRequest(BaseModel):
    step_id: str = Field(..., description="Step ID to approve")


class StepRejectRequest(BaseModel):
    step_id: str = Field(..., description="Step ID to reject")
    reason: Optional[str] = Field(None, description="Optional explanation for rejection")


class ToolInvokeRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


# ─────────────────────────────────────────────────────────────────────────────
# Workspace & Policy Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/workspaces")
def get_workspaces():
    """Returns currently authorized workspace roots and the active workspace."""
    return {
        "authorized_workspaces": workspace_registry.get_authorized_workspaces(),
        "active_workspace": workspace_registry.get_active_workspace(),
        "policy": workspace_registry.get_policy().value,
    }


@router.post("/workspaces/authorize")
def authorize_workspace(req: AuthorizeWorkspaceRequest):
    """Authorizes a local directory for filesystem agent inspection and operations."""
    try:
        canonical = workspace_registry.authorize_workspace(req.path)
        return {
            "success": True,
            "authorized_path": canonical,
            "workspaces": workspace_registry.get_authorized_workspaces(),
            "active_workspace": workspace_registry.get_active_workspace(),
        }
    except SecurityContainmentError as sce:
        envelope = create_error_envelope(
            code=sce.code,
            message=sce.message,
            required_action="Select a valid existing folder that is not an entire root drive.",
            stage="authorization",
        )
        return JSONResponse(status_code=400, content=envelope.to_dict())
    except Exception as exc:
        envelope = create_error_envelope(code="AUTHORIZE_ERROR", message=str(exc), stage="authorization")
        return JSONResponse(status_code=500, content=envelope.to_dict())


@router.post("/workspaces/revoke")
def revoke_workspace(req: RevokeWorkspaceRequest):
    """Revokes authorization for a directory."""
    revoked = workspace_registry.revoke_workspace(req.path)
    return {
        "success": revoked,
        "revoked_path": req.path,
        "workspaces": workspace_registry.get_authorized_workspaces(),
        "active_workspace": workspace_registry.get_active_workspace(),
    }


@router.get("/policy")
def get_policy():
    """Gets the active agent permission policy."""
    return {"policy": workspace_registry.get_policy().value}


@router.post("/policy")
def set_policy(req: PolicyRequest):
    """Sets the active agent permission policy (review, auto_edit, autonomous)."""
    try:
        updated = workspace_registry.set_policy(req.policy)
        return {"success": True, "policy": updated.value}
    except Exception as exc:
        envelope = create_error_envelope(
            code="INVALID_POLICY",
            message=f"Invalid policy '{req.policy}'. Valid policies: review, auto_edit, autonomous.",
            required_action="Specify one of 'review', 'auto_edit', or 'autonomous'.",
            stage="policy",
        )
        return JSONResponse(status_code=400, content=envelope.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Task Planning & Orchestration
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/plan")
def plan_task(req: PlanTaskRequest):
    """Generates a structured, validated task plan from natural language.
    Model output is treated strictly as an untrusted candidate proposal.
    """
    try:
        plan = generate_task_plan(
            user_prompt=req.prompt,
            workspace_path=req.workspace_path,
            context_files=req.context_files,
            model=req.model,
        )
        task_orchestrator.register_plan(plan)
        return plan.model_dump()
    except PlannerError as pe:
        envelope = create_error_envelope(
            code=pe.code,
            message=pe.message,
            required_action=pe.required_action,
            stage="planning",
            recoverable=True,
        )
        status_code = 503 if pe.code == "OLLAMA_SERVICE_UNAVAILABLE" else 422
        return JSONResponse(status_code=status_code, content=envelope.to_dict())
    except Exception as exc:
        envelope = create_error_envelope(
            code="PLAN_GENERATION_FAILED",
            message=str(exc),
            required_action="Check agent inputs and try again.",
            stage="planning",
        )
        return JSONResponse(status_code=500, content=envelope.to_dict())


@router.post("/execute")
def execute_task(req: ExecuteTaskRequest):
    """Executes or resumes a registered task plan."""
    try:
        plan = task_orchestrator.execute_task(req.task_id)
        return plan.model_dump()
    except ValueError as ve:
        envelope = create_error_envelope(code="TASK_NOT_FOUND", message=str(ve), stage="execution")
        return JSONResponse(status_code=404, content=envelope.to_dict())
    except PermissionDeniedError as pde:
        envelope = create_error_envelope(
            code=pde.code,
            message=pde.message,
            required_action="Review and approve the pending step to proceed.",
            stage="permission",
            recoverable=pde.requires_approval,
            diagnostics=pde.approval_details,
        )
        return JSONResponse(status_code=403, content=envelope.to_dict())
    except Exception as exc:
        envelope = create_error_envelope(code="EXECUTION_ERROR", message=str(exc), stage="execution")
        return JSONResponse(status_code=500, content=envelope.to_dict())


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Returns task status, steps, diffs, outputs, and summary."""
    plan = task_orchestrator.get_task(task_id)
    if not plan:
        envelope = create_error_envelope(code="TASK_NOT_FOUND", message=f"Task '{task_id}' not found.", stage="lookup")
        return JSONResponse(status_code=404, content=envelope.to_dict())
    return plan.model_dump()


@router.post("/tasks/{task_id}/approve")
def approve_step(task_id: str, req: StepApprovalRequest):
    """Approves a sensitive step and resumes execution."""
    try:
        plan = task_orchestrator.approve_step(task_id=task_id, step_id=req.step_id)
        return plan.model_dump()
    except ValueError as ve:
        envelope = create_error_envelope(code="STEP_NOT_FOUND", message=str(ve), stage="approval")
        return JSONResponse(status_code=404, content=envelope.to_dict())
    except Exception as exc:
        envelope = create_error_envelope(code="APPROVAL_ERROR", message=str(exc), stage="approval")
        return JSONResponse(status_code=500, content=envelope.to_dict())


@router.post("/tasks/{task_id}/reject")
def reject_step(task_id: str, req: StepRejectRequest):
    """Rejects a step and halts the task."""
    try:
        plan = task_orchestrator.reject_step(task_id=task_id, step_id=req.step_id, reason=req.reason or "")
        return plan.model_dump()
    except ValueError as ve:
        envelope = create_error_envelope(code="STEP_NOT_FOUND", message=str(ve), stage="rejection")
        return JSONResponse(status_code=404, content=envelope.to_dict())
    except Exception as exc:
        envelope = create_error_envelope(code="REJECTION_ERROR", message=str(exc), stage="rejection")
        return JSONResponse(status_code=500, content=envelope.to_dict())


@router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """Cancels a running task and terminates any active subprocess."""
    try:
        plan = task_orchestrator.cancel_task(task_id)
        return plan.model_dump()
    except ValueError as ve:
        envelope = create_error_envelope(code="TASK_NOT_FOUND", message=str(ve), stage="cancellation")
        return JSONResponse(status_code=404, content=envelope.to_dict())


@router.post("/tasks/{task_id}/rollback")
def rollback_task(task_id: str):
    """Rolls back filesystem changes made during this task using captured file backups."""
    success = task_orchestrator.rollback_task(task_id)
    return {"success": success, "task_id": task_id}


# ─────────────────────────────────────────────────────────────────────────────
# Direct Tool Invocation (Filesystem Exploration & Inspection)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/tools/invoke")
def invoke_tool(req: ToolInvokeRequest):
    """Directly invokes an inspection or browsing tool governed by workspace permissions."""
    name = req.tool_name.strip()
    args = req.arguments

    res = task_orchestrator._dispatch_tool(name, args, task_id="direct_invoke")
    if not res.success:
        status_code = 403 if res.error_code in ("PATH_NOT_AUTHORIZED", "APPROVAL_REQUIRED", "EXPLICITLY_DENIED") else 400
        envelope = create_error_envelope(
            code=res.error_code or "TOOL_ERROR",
            message=res.error or f"Tool '{name}' failed.",
            required_action="Verify tool arguments and workspace authorization.",
            stage="tool_invocation",
            recoverable=True,
            diagnostics=res.metadata,
        )
        return JSONResponse(status_code=status_code, content=envelope.to_dict())

    return res.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Audit Log Inspection
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/audit")
def get_audit_log(limit: int = Query(default=100, ge=1, le=1000), task_id: Optional[str] = None):
    """Retrieves redacted audit log records."""
    records = agent_audit.get_records(limit=limit, task_id=task_id)
    return {"records": records, "count": len(records)}
