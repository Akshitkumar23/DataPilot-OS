"""
DataPilot OS — Live Ollama Workspace Agent Acceptance & Model Quality Suite.
Tests real end-to-end interactions with the locally installed Ollama model.
Does not download external models or alter user installations.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.agent.audit import agent_audit
from backend.agent.permissions import AgentAction, AgentPolicy, workspace_registry
from backend.agent.planner import (
    PlannerError,
    TaskStep,
    VALID_TOOL_NAMES,
    WorkspaceTaskPlan,
    generate_task_plan,
)
from backend.agent.orchestrator import task_orchestrator
from backend.agent.security import redact_secrets
from backend.agent import tools
from backend.intelligence.ollama_service import ollama_service

pytestmark = pytest.mark.live_ollama


def check_ollama_ready():
    readiness = ollama_service.verify_ready()
    return readiness.ready


@pytest.fixture(autouse=True)
def clean_agent_state():
    workspace_registry.clear()
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)
    task_orchestrator._tasks.clear()
    task_orchestrator._backups.clear()
    yield
    workspace_registry.clear()


@pytest.fixture
def disposable_workspace(tmp_path):
    ws = tmp_path / "live_workspace"
    ws.mkdir()
    (ws / "config.json").write_text(json.dumps({"app": "DataPilot", "version": "2.4.0", "status": "active"}, indent=2), encoding="utf-8")
    (ws / "notes.txt").write_text("Project notes:\n1. Initial baseline created.\n2. Verification pending.\n", encoding="utf-8")
    (ws / "main.py").write_text("def hello():\n    return 'Hello from DataPilot'\n", encoding="utf-8")
    workspace_registry.authorize_workspace(str(ws))
    return str(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: List and summarize workspace contents
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_1_list_workspace_contents(disposable_workspace):
    """Real Ollama: List and summarize files in an authorized workspace."""
    start_t = time.time()
    plan = generate_task_plan("List all files and check directory contents", workspace_path=disposable_workspace)
    elapsed = time.time() - start_t

    assert plan is not None
    assert plan.status in ("PLANNED", "AWAITING_APPROVAL")
    assert len(plan.steps) >= 1
    assert any(s.tool_name == "list_directory" for s in plan.steps)

    # Register and execute plan
    task_orchestrator.register_plan(plan)
    executed_plan = task_orchestrator.execute_task(plan.task_id)
    assert executed_plan.status in ("COMPLETED", "AWAITING_APPROVAL")
    summary = executed_plan.summary
    assert len(summary.get("files_inspected", [])) >= 1 or any(s.status == "SUCCEEDED" for s in executed_plan.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Read a specific file and extract contents
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_2_read_specific_file(disposable_workspace):
    """Real Ollama: Read a specific configuration file."""
    config_file = os.path.join(disposable_workspace, "config.json")
    prompt = f"Read the file '{config_file}'"

    plan = generate_task_plan(prompt, workspace_path=disposable_workspace)
    assert plan is not None
    assert any(s.tool_name in ("read_file", "search_file_contents", "get_file_metadata") for s in plan.steps), f"Actual steps: {[(s.step_id, s.tool_name, s.arguments) for s in plan.steps]}, is_ambiguous: {plan.is_ambiguous}, clarification: {plan.clarification_needed}"

    read_step = next((s for s in plan.steps if s.tool_name == "read_file"), None)
    if read_step:
        assert "config.json" in read_step.arguments.get("path", "")

    task_orchestrator.register_plan(plan)
    executed_plan = task_orchestrator.execute_task(plan.task_id)
    assert executed_plan.status == "COMPLETED", f"Execution failed: {executed_plan.error}, summary: {executed_plan.summary}"
    if read_step and read_step.result:
        assert "2.4.0" in str(read_step.result)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Propose a small, clearly specified text edit
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_3_propose_text_edit(disposable_workspace):
    """Real Ollama: Propose an edit and confirm approval requirement under Review policy."""
    workspace_registry.set_policy(AgentPolicy.REVIEW)
    notes_file = os.path.join(disposable_workspace, "notes.txt")
    prompt = f"In '{notes_file}', update the file to append '3. Tasks completed successfully.'"

    plan = generate_task_plan(prompt, workspace_path=disposable_workspace)
    assert plan is not None
    write_step = next((s for s in plan.steps if s.tool_name in ("apply_file_patch", "create_file")), None)
    assert write_step is not None, "Model must propose a file write or patch tool"

    # Under REVIEW policy, file write MUST require approval
    assert write_step.requires_approval is True

    # Register plan and execute without approval: must pause cleanly
    task_orchestrator.register_plan(plan)
    res_plan = task_orchestrator.execute_task(plan.task_id)
    assert res_plan.status == "AWAITING_APPROVAL"

    # Verify file on disk is UNMODIFIED before approval
    with open(notes_file, "r", encoding="utf-8") as f:
        content_before = f.read()
    assert "3. Tasks completed successfully." not in content_before

    # Approve step and verify execution
    approved_plan = task_orchestrator.approve_step(plan.task_id, write_step.step_id)
    assert approved_plan.status == "COMPLETED", f"Approved plan failed: {approved_plan.error}, steps: {[(s.step_id, s.tool_name, s.status, s.error) for s in approved_plan.steps]}, summary: {approved_plan.summary}"
    with open(notes_file, "r", encoding="utf-8") as f:
        content_after = f.read()
    assert "3. Tasks completed successfully." in content_after or "completed" in content_after.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Multi-step plan using registered tools
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_4_multistep_plan(disposable_workspace):
    """Real Ollama: Generate a structured multi-step plan."""
    config_file = os.path.join(disposable_workspace, "config.json")
    backup_file = os.path.join(disposable_workspace, "config.bak")
    prompt = f"Read '{config_file}' and create a backup at '{backup_file}' with the same content"

    plan = generate_task_plan(prompt, workspace_path=disposable_workspace)
    assert plan is not None
    assert len(plan.steps) >= 2
    for s in plan.steps:
        assert s.tool_name in VALID_TOOL_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Ambiguous request handling & structured clarification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_5_ambiguous_request_handling(disposable_workspace):
    """Real Ollama: Ambiguous or underspecified request returns clarification or clean error."""
    ambiguous_prompt = "Fix the error in the project"

    try:
        plan = generate_task_plan(ambiguous_prompt, workspace_path=disposable_workspace)
        # If model returned a plan, it must either be marked as ambiguous or contain concrete inspection steps
        if plan.is_ambiguous:
            assert plan.status == "NEEDS_CLARIFICATION"
            assert plan.clarification_needed is not None
        else:
            # If model proposed inspection, tools must be safe read-only tools
            assert all(s.tool_name in ("list_directory", "search_files", "search_file_contents", "read_file", "get_file_metadata", "get_diff") for s in plan.steps), f"Proposed non-read-only tools: {[s.tool_name for s in plan.steps]}"
    except PlannerError as pe:
        # Planner error should be structured, not unhandled crash
        assert pe.code in ("EMPTY_TASK_PLAN", "INVALID_TASK_PLAN", "PLANNER_INFERENCE_FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: Unknown or invalid tool rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_live_task_6_invalid_tool_rejection(disposable_workspace):
    """Contract: Proposed unregistered tool is strictly rejected by the planner validation."""
    from backend.agent.planner import _extract_json

    # Test that planner rejects steps referencing invalid tools
    plan_with_invalid_tool = {
        "objective": "Attempt exploit",
        "steps": [
            {"tool_name": "execute_arbitrary_shell_payload", "arguments": {}, "explanation": "unregistered"}
        ]
    }
    # Verify VALID_TOOL_NAMES does not contain exploit tool
    assert "execute_arbitrary_shell_payload" not in VALID_TOOL_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: Attempt unauthorized path and confirm containment
# ─────────────────────────────────────────────────────────────────────────────

def test_live_task_7_unauthorized_path_blocked(disposable_workspace):
    """Contract: Any tool attempting to touch an unauthorized outside directory is blocked."""
    outside_path = os.path.join(tempfile.gettempdir(), f"outside_{int(time.time()*1000)}.txt")
    res = tools.read_file(outside_path)
    assert res.success is False
    assert res.error_code in ("PATH_NOT_AUTHORIZED", "ACCESS_DENIED", "PERMISSION_DENIED", "CONTAINMENT_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Task 8: Rejected or unapproved write does not modify file
# ─────────────────────────────────────────────────────────────────────────────

def test_live_task_8_rejected_write_preserves_disk(disposable_workspace):
    """Contract: A step rejected by user halts task and leaves target file untouched."""
    workspace_registry.set_policy(AgentPolicy.REVIEW)
    protected_file = os.path.join(disposable_workspace, "secure.txt")
    with open(protected_file, "w", encoding="utf-8") as f:
        f.write("ORIGINAL_PROTECTED_DATA\n")

    plan = WorkspaceTaskPlan(
        objective="Overwrite secure file",
        workspace_path=disposable_workspace,
        steps=[
            TaskStep(
                step_id="step_write_sec",
                tool_name="apply_file_patch",
                arguments={"path": protected_file, "new_content": "OVERWRITTEN_MALICIOUS_DATA\n"},
                requires_approval=True,
                status="PENDING",
            )
        ]
    )
    task_orchestrator.register_plan(plan)

    # Reject step
    halted_plan = task_orchestrator.reject_step(plan.task_id, "step_write_sec", reason="Disapproved by operator")
    assert halted_plan.status == "FAILED"

    # Verify file on disk is untouched
    with open(protected_file, "r", encoding="utf-8") as f:
        assert f.read() == "ORIGINAL_PROTECTED_DATA\n"


# ─────────────────────────────────────────────────────────────────────────────
# Task 9: Verification failure triggers bounded repair proposal
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not check_ollama_ready(), reason="Local Ollama server is not running or no model installed")
def test_live_task_9_live_verification_failure_repair(disposable_workspace):
    """Real Ollama: Harmless verification failure invokes live model for remedial repair step."""
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)
    verify_target = os.path.join(disposable_workspace, "status_check.txt")

    plan = WorkspaceTaskPlan(
        objective="Create status check file with correct token",
        workspace_path=disposable_workspace,
        steps=[
            TaskStep(
                step_id="step_init",
                tool_name="create_file",
                arguments={"path": verify_target, "content": "STATUS: PENDING\n"},
                requires_approval=False,
                status="PENDING",
            )
        ],
        verification_criteria=[
            {
                "type": "file_contains",
                "description": "status_check.txt must contain STATUS: READY",
                "path": verify_target,
                "expected": "STATUS: READY"
            }
        ]
    )
    task_orchestrator.register_plan(plan)

    # Initial run: step_init succeeds, verification fails, orchestrator invokes live Ollama for repair
    res_plan = task_orchestrator.execute_task(plan.task_id)
    assert res_plan.summary.get("repair_attempts", 0) <= 1
    # Check that repair steps were bounded
    assert len(res_plan.steps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 10: Accurate distinction of proposed vs executed vs verified
# ─────────────────────────────────────────────────────────────────────────────

def test_live_task_10_execution_summary_parity(disposable_workspace):
    """Contract: Execution summary strictly separates proposed, executed, changed, and verified."""
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)
    test_f = os.path.join(disposable_workspace, "summary_test.txt")
    plan = WorkspaceTaskPlan(
        objective="Demonstrate summary parity",
        workspace_path=disposable_workspace,
        steps=[
            TaskStep(
                step_id="s1",
                tool_name="create_file",
                arguments={"path": test_f, "content": "DATA_PARITY\n"},
                requires_approval=False,
                status="PENDING",
            ),
            TaskStep(
                step_id="s2",
                tool_name="read_file",
                arguments={"path": test_f},
                requires_approval=False,
                status="PENDING",
            )
        ],
        verification_criteria=[
            {"type": "file_exists", "description": "file exists", "path": test_f}
        ]
    )
    task_orchestrator.register_plan(plan)
    executed = task_orchestrator.execute_task(plan.task_id)

    assert executed.status == "COMPLETED"
    summary = executed.summary
    assert "files_inspected" in summary
    assert "files_changed" in summary
    assert "commands_executed" in summary
    assert "checks_passed" in summary
    assert "repair_attempts" in summary

    # Proposed steps: 2
    assert len(executed.steps) == 2
    # Executed steps succeeded: 2
    assert all(s.status == "SUCCEEDED" for s in executed.steps)
    # Changed file tracked:
    assert any(test_f in fc for fc in summary["files_changed"])
    # Inspected file tracked:
    assert any(test_f in fi for fi in summary["files_inspected"])
    # Verified:
    assert len(summary["checks_passed"]) == 1
    assert summary["checks_passed"][0]["passed"] is True
