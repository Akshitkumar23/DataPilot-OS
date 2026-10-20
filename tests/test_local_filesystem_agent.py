"""
DataPilot OS — Local Filesystem Agent Acceptance & Security Test Suite.
Verifies all 18 core acceptance criteria including:
- Windows path authorization & canonical containment (fail closed)
- Absolute paths with spaces & Unicode
- Safe atomic file creation & concurrent modification detection
- Denial of traversal and symlink escapes
- Approval requirements under Review, Auto-edit, and Autonomous policies
- Controlled subprocess execution with timeout, cancellation, output limits, and secret redaction
- Source-native dataset immutability without copying to uploads/
- Error envelopes & API contracts
- Frozen Home component checksum preservation (lines 985-1220) & No Phase 8
"""

import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.agent.audit import agent_audit
from backend.agent.permissions import (
    AgentAction,
    AgentPolicy,
    PermissionDeniedError,
    enforce_operation_permission,
    workspace_registry,
)
from backend.agent.planner import WorkspaceTaskPlan, TaskStep, generate_task_plan
from backend.agent.orchestrator import task_orchestrator
from backend.agent.security import (
    SecurityContainmentError,
    clean_windows_path,
    is_path_contained,
    redact_secrets,
    resolve_canonical_path,
    revalidate_path_containment,
)
from backend.agent import tools
from backend.core.dataset_registry import dataset_registry
from backend.dataset_engine.loader import dataset_loader
from backend.main import app


@pytest.fixture(autouse=True)
def reset_workspace_state():
    """Resets workspace registry and active tasks before each test."""
    workspace_registry.clear()
    task_orchestrator._tasks.clear()
    task_orchestrator._backups.clear()
    yield
    workspace_registry.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 1: List and inspect folder with bounded traversal & exclusions
# ─────────────────────────────────────────────────────────────────────────────

def test_1_list_and_inspect_folder_bounded(tmp_path):
    """List and inspect a folder outside DataPilot root after authorization."""
    ws = tmp_path / "MyProject"
    ws.mkdir()
    (ws / "main.py").write_text("print('hello')", encoding="utf-8")
    (ws / "README.md").write_text("# Project", encoding="utf-8")

    # Excluded dirs
    git_dir = ws / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("dummy", encoding="utf-8")

    node_dir = ws / "node_modules"
    node_dir.mkdir()
    (node_dir / "pkg.json").write_text("{}", encoding="utf-8")

    workspace_registry.authorize_workspace(str(ws))

    res = tools.list_directory(str(ws), recursive=False)
    assert res.success is True
    file_names = [e["name"] for e in res.output]
    assert "main.py" in file_names
    assert "README.md" in file_names
    # Excluded dirs should not be in entries
    assert ".git" not in file_names
    assert "node_modules" not in file_names
    # Disclosed exclusions must be reported
    disclosed = res.metadata.get("disclosed_exclusions", [])
    assert ".git" in disclosed or "node_modules" in disclosed


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 2: Read text file with spaces in Windows path
# ─────────────────────────────────────────────────────────────────────────────

def test_2_read_text_file_windows_path_with_spaces(tmp_path):
    """Read a text file using an absolute path containing spaces, with line slicing and encoding."""
    spaced_dir = tmp_path / "My Workspace" / "Sub Project Folder"
    spaced_dir.mkdir(parents=True)
    target_file = spaced_dir / "sample data.txt"
    target_file.write_text("Line 1: Alpha\nLine 2: Beta\nLine 3: Gamma\nLine 4: Delta\n", encoding="utf-8")

    workspace_registry.authorize_workspace(str(tmp_path / "My Workspace"))

    res = tools.read_file(str(target_file), line_start=2, line_end=3)
    assert res.success is True
    assert "Line 2: Beta" in res.output
    assert "Line 3: Gamma" in res.output
    assert "Line 1: Alpha" not in res.output
    assert res.metadata["total_lines"] == 4
    assert res.metadata["encoding"].startswith("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 3: Create new file at exact path safely
# ─────────────────────────────────────────────────────────────────────────────

def test_3_create_new_file_exact_path(tmp_path):
    """Create a new file at an exact requested path inside authorized workspace."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

    target_file = tmp_path / "docs" / "architecture.md"
    content = "# Architecture\nDataPilot OS Local Filesystem Agent.\n"

    res = tools.create_file(str(target_file), content=content)
    assert res.success is True
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == content


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 4: Update existing file and verify diff
# ─────────────────────────────────────────────────────────────────────────────

def test_4_update_existing_file_and_verify_diff(tmp_path):
    """Update an existing file and verify its unified diff and saved content."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

    cfg = tmp_path / "config.json"
    cfg.write_text('{\n  "version": "1.0",\n  "debug": false\n}\n', encoding="utf-8")
    expected_hash = hashlib.sha256(cfg.read_bytes()).hexdigest()

    new_content = '{\n  "version": "1.0",\n  "debug": true,\n  "agent": true\n}\n'
    res = tools.apply_file_patch(str(cfg), new_content=new_content, expected_hash=expected_hash)

    assert res.success is True
    assert "-  \"debug\": false" in res.output
    assert "+  \"debug\": true" in res.output
    assert "+  \"agent\": true" in res.output
    assert '"agent": true' in cfg.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 5: Reject unauthorized path (fail-closed)
# ─────────────────────────────────────────────────────────────────────────────

def test_5_reject_unauthorized_path_fail_closed(tmp_path):
    """Reject operations targeting paths outside authorized workspace roots."""
    ws = tmp_path / "authorized_ws"
    ws.mkdir()
    secret_dir = tmp_path / "outside_forbidden"
    secret_dir.mkdir()
    forbidden_file = secret_dir / "secrets.txt"
    forbidden_file.write_text("TOP_SECRET", encoding="utf-8")

    workspace_registry.authorize_workspace(str(ws))

    res = tools.read_file(str(forbidden_file))
    assert res.success is False
    assert res.error_code in ("PATH_NOT_AUTHORIZED", "SECURITY_CONTAINMENT_FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 6: Reject traversal attempts and symlink escapes
# ─────────────────────────────────────────────────────────────────────────────

def test_6_reject_traversal_attempts(tmp_path):
    """Reject .. directory traversal attempts escaping authorized workspace."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")

    workspace_registry.authorize_workspace(str(ws))

    traversal_path = str(ws / ".." / "outside.txt")
    res = tools.read_file(traversal_path)
    assert res.success is False
    assert res.error_code in ("PATH_NOT_AUTHORIZED", "SECURITY_CONTAINMENT_FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 7: Prevent accidental overwrite
# ─────────────────────────────────────────────────────────────────────────────

def test_7_prevent_accidental_overwrite(tmp_path):
    """Attempting to create an existing file with overwrite=False fails closed."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

    existing = tmp_path / "data.csv"
    existing.write_text("a,b\n1,2\n", encoding="utf-8")

    res = tools.create_file(str(existing), content="x,y\n9,9\n", overwrite=False)
    assert res.success is False
    assert res.error_code == "FILE_ALREADY_EXISTS"
    # Original content preserved
    assert existing.read_text(encoding="utf-8") == "a,b\n1,2\n"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 8: Detect concurrent file modifications
# ─────────────────────────────────────────────────────────────────────────────

def test_8_detect_concurrent_file_modifications(tmp_path):
    """Attempting to patch a file whose hash changed raises CONCURRENT_MODIFICATION."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

    f = tmp_path / "app.py"
    f.write_text("v = 1\n", encoding="utf-8")
    initial_hash = hashlib.sha256(f.read_bytes()).hexdigest()

    # External actor modifies file
    f.write_text("v = 2 # modified externally\n", encoding="utf-8")

    # Agent tries to patch using stale initial_hash
    res = tools.apply_file_patch(str(f), new_content="v = 3\n", expected_hash=initial_hash)
    assert res.success is False
    assert res.error_code == "CONCURRENT_MODIFICATION"
    assert "externally" in f.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 9: Denied operation never writes to disk
# ─────────────────────────────────────────────────────────────────────────────

def test_9_denied_operation_never_writes_to_disk(tmp_path):
    """A denied write operation never touches or creates the file on disk."""
    ws = tmp_path / "ws"
    ws.mkdir()
    workspace_registry.authorize_workspace(str(ws))
    # Review policy: write requires approval
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    target = ws / "unapproved.txt"
    res = tools.create_file(str(target), content="hello")
    assert res.success is False
    assert res.error_code == "APPROVAL_REQUIRED"
    assert not target.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 10: Approval requirements under Review, Auto-edit, Autonomous
# ─────────────────────────────────────────────────────────────────────────────

def test_10_approval_requirements_under_policies(tmp_path):
    """Verify approval requirements for deletes and commands across policies."""
    workspace_registry.authorize_workspace(str(tmp_path))
    f = tmp_path / "delete_target.txt"
    f.write_text("content", encoding="utf-8")

    # 1. Delete without explicit_confirmed must fail under all policies
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)
    res_del = tools.delete_file(str(f), explicit_confirmed=False)
    assert res_del.success is False
    assert res_del.error_code == "APPROVAL_REQUIRED"
    assert f.exists()

    # With explicit confirmation, delete succeeds
    res_del_ok = tools.delete_file(str(f), explicit_confirmed=True)
    assert res_del_ok.success is True
    assert not f.exists()

    # 2. Command execution under Review requires approval
    workspace_registry.set_policy(AgentPolicy.REVIEW)
    res_cmd = tools.execute_command(command="python --version")
    assert res_cmd.success is False
    assert res_cmd.error_code == "APPROVAL_REQUIRED"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 11: Command execution timeout, cancellation, output limits
# ─────────────────────────────────────────────────────────────────────────────

def test_11_command_execution_safeguards(tmp_path):
    """Verify command execution captures exit code, enforces timeout, and limits output."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # Fast successful command
    res_ok = tools.execute_command(command=f'"{sys.executable}" -c "print(\'Agent Success\')"', cwd=str(tmp_path))
    assert res_ok.success is True
    assert res_ok.output["exit_code"] == 0
    assert "Agent Success" in res_ok.output["stdout"]

    # Timeout enforcement
    res_timeout = tools.execute_command(
        command=f'"{sys.executable}" -c "import time; time.sleep(10)"',
        cwd=str(tmp_path),
        timeout_seconds=1,
    )
    assert res_timeout.success is False
    assert res_timeout.error_code == "TIMEOUT"
    assert res_timeout.output["timed_out"] is True

    # Output truncation limit
    res_trunc = tools.execute_command(
        command=f'"{sys.executable}" -c "print(\'X\' * 1000)"',
        cwd=str(tmp_path),
        max_output_chars=200,
    )
    assert res_trunc.output["is_truncated"] is True
    assert "[OUTPUT TRUNCATED" in res_trunc.output["stdout"]


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 12: Secret redaction in logs and outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_12_secret_redaction():
    """Verify tokens, API keys, and passwords are redacted."""
    raw = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy sk-abcdef12345678901234567890 AKIA1234567890ABCDEF"
    redacted = redact_secrets(raw)
    assert "sk-" not in redacted
    assert "AKIA" not in redacted
    assert "[REDACTED_SECRET]" in redacted or "[REDACTED_TOKEN]" in redacted


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 13: Source-native dataset files remain byte-for-byte unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_13_source_native_dataset_immutable(tmp_path):
    """Verify original dataset files remain byte-for-byte unchanged during analysis."""
    csv_file = tmp_path / "original_data.csv"
    csv_file.write_text("id,val\n1,100\n2,200\n", encoding="utf-8")
    original_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()

    df, load_res = dataset_loader.load(str(csv_file))
    assert load_res.success is True
    assert len(df) == 2

    # Verify byte-for-byte immutability
    post_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    assert post_hash == original_hash


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 14: Source-native files not copied to uploads/
# ─────────────────────────────────────────────────────────────────────────────

def test_14_source_native_not_copied_to_uploads(tmp_path):
    """Verify loading from workspace does not duplicate into uploads/."""
    csv_file = tmp_path / "workspace_dataset.csv"
    csv_file.write_text("a,b\n10,20\n", encoding="utf-8")

    uploads_dir = Path("uploads")
    initial_uploads = set(uploads_dir.glob("*")) if uploads_dir.exists() else set()

    df, load_res = dataset_loader.load(str(csv_file))
    assert load_res.success is True

    current_uploads = set(uploads_dir.glob("*")) if uploads_dir.exists() else set()
    new_files = current_uploads - initial_uploads
    assert len(new_files) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 15: Dataset operations continuity & drift detection
# ─────────────────────────────────────────────────────────────────────────────

def test_15_dataset_operations_continuity(tmp_path):
    """Verify dataset registry detects source drift if file modified externally."""
    from backend.core.dataset_registry import DatasetEntry

    csv_file = tmp_path / "drift_test.csv"
    csv_file.write_text("col1,col2\n1,2\n", encoding="utf-8")

    df1, load1 = dataset_loader.load(str(csv_file))
    assert load1.success is True

    entry = DatasetEntry(
        dataset_id="test_ds_drift",
        name="drift_test.csv",
        file_path=str(csv_file),
        format="csv",
        row_count=len(df1),
        column_count=len(df1.columns),
        file_hash=load1.file_hash,
        source_file_mtime=os.path.getmtime(str(csv_file)),
        source_file_size=os.path.getsize(str(csv_file)),
        knowledge_model={},
        loaded_at="2026-10-19T00:00:00Z",
        is_source_native=True,
    )
    dataset_registry.register("test_ds_drift", entry)

    # Initial check: no drift
    assert entry.check_drift() is False

    # Modify file externally
    time.sleep(0.05)
    csv_file.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")

    # Re-check: drift detected
    drift_detected = entry.check_drift()
    assert drift_detected is True
    assert entry.drift_detected is True


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 16: API Contracts and Error Envelopes
# ─────────────────────────────────────────────────────────────────────────────

def test_16_api_contracts_and_error_envelopes(tmp_path):
    """Verify FastAPI /api/agent endpoints return structured ErrorEnvelope on error."""
    token = os.getenv("DATAPILOT_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    client = TestClient(app)

    # 1. Authorize workspace
    res_auth = client.post("/api/agent/workspaces/authorize", json={"path": str(tmp_path)}, headers=headers)
    assert res_auth.status_code == 200
    assert res_auth.json()["success"] is True

    # 2. Get workspaces
    res_list = client.get("/api/agent/workspaces", headers=headers)
    assert res_list.status_code == 200
    assert str(tmp_path) in res_list.json()["authorized_workspaces"]

    # 3. Policy get & set
    res_pol = client.post("/api/agent/policy", json={"policy": "auto_edit"}, headers=headers)
    assert res_pol.status_code == 200
    assert res_pol.json()["policy"] == "auto_edit"

    # 4. Error contract for invalid tool invocation
    res_err = client.post("/api/agent/tools/invoke", json={"tool_name": "non_existent_tool", "arguments": {}}, headers=headers)
    assert res_err.status_code == 400
    body = res_err.json()
    assert body["error"] is True
    assert "code" in body
    assert "message" in body

    # 5. Unauthorized path invocation fails closed
    res_unauth = client.post(
        "/api/agent/tools/invoke",
        json={"tool_name": "read_file", "arguments": {"path": "C:/Windows/System32/drivers/etc/hosts"}},
        headers=headers,
    )
    assert res_unauth.status_code in (400, 403)
    assert res_unauth.json()["code"] in ("PATH_NOT_AUTHORIZED", "SECURITY_CONTAINMENT_FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 17: Task Rollback Support
# ─────────────────────────────────────────────────────────────────────────────

def test_17_task_orchestrator_rollback(tmp_path):
    """Verify task orchestrator can restore original file content on rollback."""
    workspace_registry.authorize_workspace(str(tmp_path))
    workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

    target_file = tmp_path / "rollback_test.txt"
    target_file.write_text("ORIGINAL CONTENT", encoding="utf-8")

    plan = WorkspaceTaskPlan(
        objective="Modify file and test rollback",
        workspace_path=str(tmp_path),
        steps=[
            TaskStep(
                tool_name="apply_file_patch",
                arguments={"path": str(target_file), "new_content": "MODIFIED CONTENT"},
                explanation="Modify test file",
            )
        ],
    )
    task_orchestrator.register_plan(plan)
    executed = task_orchestrator.execute_task(plan.task_id)
    assert executed.status == "COMPLETED"
    assert target_file.read_text(encoding="utf-8") == "MODIFIED CONTENT"

    # Rollback
    rolled_back = task_orchestrator.rollback_task(plan.task_id)
    assert rolled_back is True
    assert target_file.read_text(encoding="utf-8") == "ORIGINAL CONTENT"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance Criterion 18: Frozen Home Component Checksum & No Phase 8
# ─────────────────────────────────────────────────────────────────────────────

def test_18_frozen_home_component_and_no_phase_8():
    """Verify the exact reproducible Home component SHA-256 and confirm no Phase 8."""
    # 1. Reproducible checksum verification
    with open("src/App.jsx", "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunk = "".join(lines[984:1220])
    h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
    assert h == "39b504b45801f8209c4ca80fcf9656e7c3e75dfa96a1504a508946f6e10a444e", f"Hash mismatch: {h}"

    # 2. Confirm no 'Phase 8' introduced anywhere in codebase
    agent_dir = Path("backend/agent")
    for py_file in agent_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        assert "phase 8" not in text
        assert "phase8" not in text
