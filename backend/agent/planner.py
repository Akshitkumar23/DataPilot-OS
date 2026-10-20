"""
DataPilot OS — Agent Task Planner & Intent Engine.
Translates natural language user tasks into structured, validated WorkspaceTaskPlans
using local Ollama models. Model output is strictly treated as an untrusted candidate proposal.
Fails cleanly with structured error envelopes without ungrounded regex fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.agent.audit import agent_audit
from backend.agent.config import agent_config
from backend.agent.permissions import AgentAction, AgentPolicy, workspace_registry
from backend.agent.registry import ToolCategory, agent_tool_registry
from backend.agent.security import is_path_contained, resolve_canonical_path

logger = logging.getLogger(__name__)

PROMPT_VERSION = agent_config.PROMPT_VERSION
SCHEMA_VERSION = agent_config.SCHEMA_VERSION


class _ValidToolsProxy(set):
    """Dynamic proxy for registered tools ensuring backward compatibility with sets."""
    def __contains__(self, item: object) -> bool:
        return agent_tool_registry.is_valid_tool(str(item))

    def __iter__(self):
        return iter(agent_tool_registry.get_valid_tool_names())

    def __len__(self):
        return len(agent_tool_registry.get_valid_tool_names())


# Registered valid tools dynamically backed by agent_tool_registry
VALID_TOOL_NAMES = _ValidToolsProxy()

WORKSPACE_TASK_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "is_ambiguous": {"type": "boolean"},
        "clarification_needed": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "explanation": {"type": "string"},
                },
                "required": ["tool_name", "arguments"],
            },
        },
        "verification_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["file_exists", "file_contains", "command_exit_zero"]},
                    "description": {"type": "string"},
                    "path": {"type": "string"},
                    "expected": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["type", "description"],
            },
        },
    },
    "required": ["objective", "steps"],
}

REPAIR_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "repair_explanation": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "explanation": {"type": "string"},
                },
                "required": ["tool_name", "arguments"],
            },
        },
    },
    "required": ["repair_explanation", "steps"],
}


class TaskStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:6]}")
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    requires_approval: bool = False
    status: str = "PENDING"  # PENDING, AWAITING_APPROVAL, RUNNING, SUCCEEDED, FAILED, CANCELLED
    result: Optional[Any] = None
    proposed_diff: Optional[str] = None
    error: Optional[str] = None


class WorkspaceTaskPlan(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    objective: str
    workspace_path: str
    is_ambiguous: bool = False
    clarification_needed: Optional[str] = None
    steps: List[TaskStep] = Field(default_factory=list)
    verification_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "PLANNED"  # PLANNED, NEEDS_CLARIFICATION, AWAITING_APPROVAL, EXECUTING, COMPLETED, FAILED, CANCELLED
    error: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)


class PlannerError(Exception):
    def __init__(self, code: str, message: str, required_action: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.required_action = required_action


def _prepare_step_arguments(tool: str, raw_args: Any, active_ws: str) -> Dict[str, Any]:
    """Validates and prepares step arguments, populating default workspace and optimistic concurrency hashes.

    If the caller or model passes apply_file_patch:
    - If a valid 64-character SHA-256 hex string is present, normalize and preserve it.
    - If a placeholder (e.g. 'expected...', 'null') or missing hash is provided, attempt to obtain
      the trusted current SHA-256 from file metadata if the file exists within the workspace.
    - If no trusted hash is available, cleanly omit the optional field so no malformed placeholder reaches
      the trusted tool boundary.
    """
    args = dict(raw_args) if isinstance(raw_args, dict) else {}

    # Fill default workspace for tools that need it
    if "workspace_path" in args and not args["workspace_path"]:
        args["workspace_path"] = active_ws
    elif tool in ("search_files", "search_file_contents") and "workspace_path" not in args:
        args["workspace_path"] = active_ws

    # Concurrency control for apply_file_patch
    if tool == "apply_file_patch":
        target_path = args.get("path")
        raw_hash = args.get("expected_hash")
        is_valid_sha256 = isinstance(raw_hash, str) and bool(re.fullmatch(r"^[0-9a-fA-F]{64}$", raw_hash.strip()))

        if is_valid_sha256:
            args["expected_hash"] = raw_hash.strip().lower()
        else:
            # Query trusted metadata for existing file
            real_hash = None
            if target_path:
                try:
                    from backend.agent import tools
                    meta = tools.get_file_metadata(target_path)
                    if meta.success:
                        out = meta.output if isinstance(meta.output, dict) else (meta.metadata or {})
                        real_hash = out.get("sha256")
                except Exception:
                    pass
            if real_hash:
                args["expected_hash"] = real_hash
            else:
                args.pop("expected_hash", None)

    return args


def generate_task_plan(
    user_prompt: str,
    workspace_path: Optional[str] = None,
    context_files: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> WorkspaceTaskPlan:
    """Uses the local Ollama model to synthesize a structured task plan.
    
    If the semantic model is unavailable or produces an invalid JSON schema,
    fails closed with a structured PlannerError. Does not fall back to regex parsing.
    """
    clean_prompt = user_prompt.strip()
    if not clean_prompt:
        raise PlannerError(
            code="EMPTY_PROMPT",
            message="User task prompt cannot be empty.",
            required_action="Provide a specific instruction or question for the workspace agent.",
        )

    # Resolve active workspace
    active_ws = workspace_path or workspace_registry.get_active_workspace()
    if not active_ws:
        authorized = workspace_registry.get_authorized_workspaces()
        if authorized:
            active_ws = authorized[0]
        else:
            raise PlannerError(
                code="NO_AUTHORIZED_WORKSPACE",
                message="No workspace directory has been authorized.",
                required_action="Select and authorize a workspace folder before planning agent tasks.",
            )

    try:
        from backend.intelligence.ollama_service import ollama_service, OllamaRequest, OllamaResponse
    except ImportError as ie:
        raise PlannerError(
            code="OLLAMA_SERVICE_UNAVAILABLE",
            message=f"Local Ollama service module could not be imported: {ie}",
            required_action="Ensure the local Ollama background service is running on http://localhost:11434.",
        ) from ie

    readiness = ollama_service.verify_ready()
    if not readiness.ready:
        raise PlannerError(
            code=readiness.error_code or "OLLAMA_SERVICE_UNAVAILABLE",
            message=f"Ollama server is unavailable: {readiness.error_message}",
            required_action="Start Ollama locally ('ollama serve') and verify model availability.",
        )

    available_tools_spec = agent_tool_registry.format_system_prompt_tools()
    system_prompt = (
        f"You are the DataPilot OS Local Workspace Agent (prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}). "
        "Your task is to plan the necessary filesystem and command operations to fulfill the user's objective.\n"
        f"Active authorized workspace path: {active_ws}\n\n"
        f"Available Tools:\n{available_tools_spec}\n\n"
        "Tool Selection Rules:\n"
        "- When asked to read, view, inspect, or check file contents, ALWAYS use 'read_file'.\n"
        "- When asked to list or explore directory contents, ALWAYS use 'list_directory'.\n"
        "- Only use 'execute_command' when explicitly instructed to run a command or script.\n\n"
        "Verification Criteria Guidelines:\n"
        "- For read-only tasks (read_file, list_directory), use 'file_exists' for verification criteria. Never guess or hallucinate unread file contents in 'file_contains' or 'expected'.\n"
        "- For file modification or creation tasks, use 'file_exists' or verify the exact known text requested by the user.\n\n"
        "Ambiguity Handling:\n"
        '- If the user request is ambiguous, under-specified, or missing essential file paths or actions (such as "fix the error", "fix the issue", "make it work", "update the project"), you MUST set "is_ambiguous": true, set "clarification_needed": "<concise question asking for details>", and return empty steps [].\n\n'
        "Return ONLY a valid JSON object matching the required schema with NO markdown wrapping."
    )

    prompt = f"User Request: {clean_prompt}\nTarget Workspace: {active_ws}"
    if context_files:
        prompt += f"\nKnown Context Files: {json.dumps(context_files)}"

    req = OllamaRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        response_schema=WORKSPACE_TASK_PLAN_SCHEMA,
        model=model or readiness.model,
        temperature=agent_config.TEMPERATURE,
        timeout_sec=agent_config.PLANNER_TIMEOUT_SEC,
    )

    try:
        resp: OllamaResponse = ollama_service.structured_request(req)
    except Exception as exc:
        raise PlannerError(
            code="PLANNER_INFERENCE_FAILED",
            message=f"Semantic planning failed during model inference: {exc}",
            required_action="Verify that local Ollama has sufficient RAM/VRAM to generate responses.",
        ) from exc

    if not resp.success:
        err_code = resp.error_code or "PLANNER_INFERENCE_FAILED"
        if "timeout" in str(resp.error_message).lower() or err_code == "TIMEOUT":
            err_code = "PLANNER_TIMEOUT"
        raise PlannerError(
            code=err_code,
            message=f"Semantic planning failed: {resp.error_message}",
            required_action="Verify Ollama server status or try again with a refined prompt.",
        )

    # Parse JSON from model reply
    parsed = resp.parsed_json or _extract_json(resp.content or "")
    if not parsed or not isinstance(parsed, dict) or "steps" not in parsed:
        raise PlannerError(
            code="INVALID_TASK_PLAN",
            message="The semantic planner model returned an unparseable or incomplete plan structure.",
            required_action="Refine your task description with explicit filenames or actions.",
        )

    is_ambig = bool(parsed.get("is_ambiguous")) or bool(parsed.get("clarification_needed"))
    clarification_msg = parsed.get("clarification_needed") or (
        "The request is ambiguous or under-specified. Please clarify what files or actions are needed."
        if is_ambig else None
    )

    # Validate and construct steps
    task_plan = WorkspaceTaskPlan(
        objective=parsed.get("objective", clean_prompt),
        workspace_path=active_ws,
        is_ambiguous=is_ambig,
        clarification_needed=clarification_msg,
        verification_criteria=parsed.get("verification_criteria", []),
        status="NEEDS_CLARIFICATION" if is_ambig else "PLANNED",
    )

    policy = workspace_registry.get_policy()

    for idx, s in enumerate(parsed.get("steps", []), start=1):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool_name", "").strip()
        if not agent_tool_registry.is_valid_tool(tool):
            raise PlannerError(
                code="INVALID_TOOL_IN_PLAN",
                message=f"Proposed step {idx} references unregistered tool '{tool}'.",
                required_action="The agent cannot execute unknown tools.",
            )

        args = _prepare_step_arguments(tool, s.get("arguments", {}), active_ws)

        # Evaluate approval requirement strictly at tool boundary via capability registry
        requires_approval = agent_tool_registry.evaluate_step_approval(tool, policy)

        step = TaskStep(
            step_id=f"step_{idx}_{uuid.uuid4().hex[:4]}",
            tool_name=tool,
            arguments=args,
            explanation=s.get("explanation", ""),
            requires_approval=requires_approval,
            status="PENDING",
        )
        task_plan.steps.append(step)

    # Sanitize verification criteria path fallbacks
    valid_checks = []
    for c in parsed.get("verification_criteria", []):
        if not isinstance(c, dict):
            continue
        c_path = c.get("path")
        if not c_path or not str(c_path).strip():
            target = active_ws
            for st in task_plan.steps:
                if st.arguments.get("path"):
                    target = st.arguments["path"]
                    break
            c["path"] = target
        valid_checks.append(c)
    task_plan.verification_criteria = valid_checks

    if not task_plan.steps:
        if is_ambig:
            agent_audit.log(
                task_id=task_plan.task_id,
                operation="plan_task",
                status="NEEDS_CLARIFICATION",
                details={
                    "prompt_version": PROMPT_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "model_used": resp.model_used or readiness.model,
                    "latency_ms": resp.latency_ms,
                    "load_duration_ms": resp.load_duration_ms,
                    "prompt_eval_duration_ms": resp.prompt_eval_duration_ms,
                    "eval_duration_ms": resp.eval_duration_ms,
                    "is_ambiguous": True,
                    "clarification_needed": clarification_msg,
                },
            )
            return task_plan
        raise PlannerError(
            code="EMPTY_TASK_PLAN",
            message="Planner produced zero valid steps for the requested task.",
            required_action="Provide a more concrete task prompt.",
        )

    # Record planning telemetry in agent audit
    agent_audit.log(
        task_id=task_plan.task_id,
        operation="plan_task",
        status="SUCCESS",
        details={
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "model_used": resp.model_used or readiness.model,
            "latency_ms": resp.latency_ms,
            "load_duration_ms": resp.load_duration_ms,
            "prompt_eval_duration_ms": resp.prompt_eval_duration_ms,
            "eval_duration_ms": resp.eval_duration_ms,
            "prompt_eval_count": resp.prompt_eval_count,
            "eval_count": resp.eval_count,
            "step_count": len(task_plan.steps),
            "verification_criteria_count": len(task_plan.verification_criteria),
        },
    )

    return task_plan


def generate_repair_steps(
    plan: WorkspaceTaskPlan,
    verification_failures: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> List[TaskStep]:
    """Generates remedial candidate steps when verification fails.
    Candidate steps are strictly untrusted and undergo full validation,
    containment enforcement, and approval checks under active policy.
    """
    try:
        from backend.intelligence.ollama_service import ollama_service, OllamaRequest
    except ImportError:
        return []

    readiness = ollama_service.verify_ready()
    if not readiness.ready:
        return []

    active_ws = plan.workspace_path
    available_tools_spec = agent_tool_registry.format_system_prompt_tools()
    system_prompt = (
        f"You are the DataPilot OS Local Workspace Agent (prompt_version={PROMPT_VERSION}, schema_version={SCHEMA_VERSION}). "
        "A previous execution plan failed verification checks. Propose remedial steps to fix the failures.\n"
        f"Active authorized workspace path: {active_ws}\n\n"
        f"Available Tools:\n{available_tools_spec}\n\n"
        "Return ONLY a valid JSON object matching the required schema with NO markdown wrapping."
    )

    prompt = (
        f"Original Objective: {plan.objective}\n"
        f"Target Workspace: {active_ws}\n"
        f"Verification Failures: {json.dumps(verification_failures)}\n"
        "Propose minimal corrective steps to resolve the verification failure."
    )

    req = OllamaRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        response_schema=REPAIR_PLAN_SCHEMA,
        model=model or readiness.model,
        temperature=agent_config.TEMPERATURE,
        timeout_sec=agent_config.PLANNER_TIMEOUT_SEC,
    )

    try:
        resp = ollama_service.structured_request(req)
        if not resp.success:
            return []
    except Exception:
        return []

    parsed = resp.parsed_json or _extract_json(resp.content or "")
    if not parsed or not isinstance(parsed, dict) or "steps" not in parsed:
        return []

    policy = workspace_registry.get_policy()
    remedial_steps: List[TaskStep] = []

    for idx, s in enumerate(parsed.get("steps", []), start=1):
        if not isinstance(s, dict):
            continue
        tool = s.get("tool_name", "").strip()
        if not agent_tool_registry.is_valid_tool(tool):
            continue

        args = _prepare_step_arguments(tool, s.get("arguments", {}), active_ws)

        requires_approval = agent_tool_registry.evaluate_step_approval(tool, policy)

        step = TaskStep(
            step_id=f"step_repair_{idx}_{uuid.uuid4().hex[:4]}",
            tool_name=tool,
            arguments=args,
            explanation=s.get("explanation", parsed.get("repair_explanation", "Remedial repair step")),
            requires_approval=requires_approval,
            status="PENDING",
        )
        remedial_steps.append(step)

    if remedial_steps:
        agent_audit.log(
            task_id=plan.task_id,
            operation="repair_plan",
            status="SUCCESS",
            details={
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "model_used": resp.model_used or readiness.model,
                "latency_ms": resp.latency_ms,
                "repair_explanation": parsed.get("repair_explanation", ""),
                "remedial_steps_count": len(remedial_steps),
            },
        )

    return remedial_steps


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts a JSON object from raw LLM output."""
    if not text:
        return None
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Search for first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
        return None
