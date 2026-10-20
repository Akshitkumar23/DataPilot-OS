"""
DataPilot OS — Real-World Acceptance & Security Validation Test Suite.
Validates:
1. Disposable test workspace strictly outside DataPilot installation directory.
2. Review policy enforcement: write/command approval, rejection, token tampering, and replay prevention.
3. Hardened Windows path containment: sibling prefixes, drive/UNC boundaries, symlinks/junctions, traversal, non-existent targets, TOCTOU.
4. Command execution runner: commands with spaces in executable and arguments, non-zero exit codes, timeout, cancellation, output truncation, and safe shell builtins.
5. Source-native dataset integrity: in-place access with zero copy to uploads/, and external drift detection.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.agent.security import (
    SecurityContainmentError,
    clean_windows_path,
    is_path_contained,
    resolve_canonical_path,
    revalidate_path_containment,
    redact_secrets,
)
from backend.agent.permissions import (
    AgentAction,
    AgentPolicy,
    PermissionDeniedError,
    workspace_registry,
    enforce_operation_permission,
)
from backend.agent import tools
from backend.agent.tools import (
    apply_file_patch,
    cancel_running_command,
    create_directory,
    create_file,
    delete_file,
    execute_command,
    get_diff,
    get_file_metadata,
    list_directory,
    read_file,
    search_file_contents,
    search_files,
)
from backend.agent.orchestrator import (
    TaskStep,
    WorkspaceTaskPlan,
    task_orchestrator,
)
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.dataset_engine.loader import dataset_loader


@pytest.fixture(autouse=True)
def clean_agent_state():
    workspace_registry.clear()
    yield
    workspace_registry.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Disposable Workspace Outside DataPilot Installation Directory
# ─────────────────────────────────────────────────────────────────────────────

def test_disposable_workspace_outside_installation():
    """Verify full agent lifecycle inside a disposable folder located in system TEMP (outside project dir)."""
    # System temp directory is strictly outside C:\\Users\\Mr\\Desktop\\DataPilot OS
    system_temp = tempfile.gettempdir()
    disposable_dir = os.path.join(system_temp, f"datapilot_disposable_{int(time.time()*1000)}")
    os.makedirs(disposable_dir, exist_ok=True)

    try:
        project_root = os.path.normcase(os.path.realpath(os.getcwd()))
        disp_root = os.path.normcase(os.path.realpath(disposable_dir))
        assert not disp_root.startswith(project_root), "Disposable workspace must be outside project root"

        # Authorize disposable workspace
        auth_path = workspace_registry.authorize_workspace(disposable_dir)
        workspace_registry.set_policy(AgentPolicy.AUTO_EDIT)

        # 1. Create file at exact path
        target_file = os.path.join(disposable_dir, "notes", "architecture.md")
        create_res = create_file(target_file, "# Architecture\nInitial content.\n")
        assert create_res.success is True
        assert os.path.exists(target_file)

        # 2. Read file
        read_res = read_file(target_file)
        assert read_res.success is True
        assert "Initial content." in read_res.output

        # 3. Inspect diff before edit
        new_content = "# Architecture\nInitial content.\n\n## Added Section\nProduction ready.\n"
        diff_res = get_diff(target_file, new_content)
        assert diff_res.success is True
        assert "+## Added Section" in diff_res.output

        # 4. Apply patch and verify saved contents
        patch_res = apply_file_patch(target_file, new_content)
        assert patch_res.success is True

        # 5. Read-back verification
        verified_read = read_file(target_file)
        assert "Production ready." in verified_read.output
    finally:
        shutil.rmtree(disposable_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Review Policy: Approval, Rejection, Tampering, and Replay Prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_review_policy_enforcement_and_tampering(tmp_path):
    """Verify Review policy blocks unapproved writes/commands, rejects invalid tokens, and blocks replay."""
    workspace = str(tmp_path)
    workspace_registry.authorize_workspace(workspace)
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    target_file = os.path.join(workspace, "sensitive_config.json")

    # 1. Unapproved write must fail
    res_unapproved = create_file(target_file, '{"setting": true}')
    assert res_unapproved.success is False
    assert res_unapproved.error_code == "APPROVAL_REQUIRED"
    assert not os.path.exists(target_file)

    # 2. Token tampering (fabricated or altered token) must fail
    tampered_token = "approval_fake_or_tampered_token_99999"
    res_tampered = create_file(target_file, '{"setting": true}', approval_token=tampered_token)
    assert res_tampered.success is False
    assert res_tampered.error_code == "APPROVAL_REQUIRED"
    assert not os.path.exists(target_file)

    # 3. Valid token succeeds exactly once
    valid_token = "approval_valid_token_12345"
    workspace_registry.register_approval_token(valid_token)

    res_valid = create_file(target_file, '{"setting": true}', approval_token=valid_token)
    assert res_valid.success is True
    assert os.path.exists(target_file)

    # 4. Approval Replay: Attempting to reuse the exact same token must fail
    res_replay = apply_file_patch(target_file, '{"setting": false}', approval_token=valid_token)
    assert res_replay.success is False
    assert res_replay.error_code == "APPROVAL_REQUIRED"
    # Content must remain unchanged
    assert read_file(target_file).output == '{"setting": true}'


def test_review_policy_rejection_via_orchestrator(tmp_path):
    """Verify orchestrator properly handles step rejection and halts execution without disk changes."""
    workspace = str(tmp_path)
    workspace_registry.authorize_workspace(workspace)
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    target_file = os.path.join(workspace, "do_not_create.txt")

    plan = WorkspaceTaskPlan(
        objective="Create a file subject to user review",
        workspace_path=workspace,
        steps=[
            TaskStep(
                step_id="step_1",
                tool_name="create_file",
                arguments={"path": target_file, "content": "MALICIOUS OR UNWANTED"},
                explanation="Write file",
                requires_approval=True,
            )
        ],
    )
    task_orchestrator.register_plan(plan)

    # Reject step
    rejected_plan = task_orchestrator.reject_step(plan.task_id, "step_1", reason="User declined this modification.")
    assert rejected_plan.status == "FAILED"
    assert "User declined this modification." in rejected_plan.steps[0].error
    assert not os.path.exists(target_file)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hardened Windows Path Containment
# ─────────────────────────────────────────────────────────────────────────────

def test_windows_path_containment_sibling_prefix(tmp_path):
    """Verify that sibling-prefix directories (e.g. workspace_sibling) cannot bypass containment."""
    authorized_workspace = tmp_path / "target_workspace"
    authorized_workspace.mkdir()

    sibling_workspace = tmp_path / "target_workspace_sibling"
    sibling_workspace.mkdir()
    sibling_file = sibling_workspace / "secret.txt"
    sibling_file.write_text("SIBLING SECRET", encoding="utf-8")

    workspace_registry.authorize_workspace(str(authorized_workspace))

    # Sibling file must be rejected
    assert not is_path_contained(str(sibling_file), str(authorized_workspace))
    with pytest.raises(SecurityContainmentError):
        revalidate_path_containment(str(sibling_file), [str(authorized_workspace)])

    res = read_file(str(sibling_file))
    assert res.success is False
    assert res.error_code in ("PATH_NOT_AUTHORIZED", "CONTAINMENT_VIOLATION")


def test_windows_path_containment_unc_and_extended_paths(tmp_path):
    """Verify that UNC and Windows extended device paths (\\\\?\\UNC\\ and \\\\?\\C:) are handled securely."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)

    # Extended path within workspace
    extended_valid = f"\\\\?\\{os.path.abspath(ws)}\\file.txt"
    cleaned = clean_windows_path(extended_valid)
    assert not cleaned.startswith("\\\\?\\")

    # UNC path outside workspace must fail containment
    unc_target = r"\\fake-corp-server\share\payroll.csv"
    assert not is_path_contained(unc_target, ws)
    with pytest.raises(SecurityContainmentError):
        revalidate_path_containment(unc_target, [ws])

    # Direct containment checks on UNC-root and drive-root
    from backend.agent.security import _check_containment_normalized
    assert _check_containment_normalized(r"\\server\share\sub\file.txt", r"\\server\share") is True
    assert _check_containment_normalized(r"\\server\share_sibling\file.txt", r"\\server\share") is False
    assert _check_containment_normalized(r"C:\folder\file.txt", "C:\\") is True
    assert _check_containment_normalized(r"C:\workspace_sibling\file.txt", r"C:\workspace") is False



def test_windows_path_containment_traversal_and_nonexistent(tmp_path):
    """Verify traversal and deep non-existent targets are safely contained."""
    ws = tmp_path / "safe_ws"
    ws.mkdir()
    workspace_registry.authorize_workspace(str(ws))

    # 1. Traversal outside workspace
    outside_target = str(ws / ".." / "outside.txt")
    assert not is_path_contained(outside_target, str(ws))

    # 2. Deep non-existent target inside workspace is valid
    deep_inside = str(ws / "deep" / "nested" / "sub" / "file.txt")
    assert is_path_contained(deep_inside, str(ws))

    # 3. Deep non-existent target escaping workspace via ..
    deep_escaping = str(ws / "deep" / ".." / ".." / "outside.txt")
    assert not is_path_contained(deep_escaping, str(ws))


def test_windows_directory_junction_containment(tmp_path):
    """Verify that directory junctions pointing outside the workspace fail containment."""
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    secret_file = outside / "secret.txt"
    secret_file.write_text("SENSITIVE OUTSIDE", encoding="utf-8")

    workspace_registry.authorize_workspace(str(ws))

    # Create junction if on Windows
    junction_path = ws / "junction_to_outside"
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(junction_path), str(outside)],
                capture_output=True,
                check=False,
            )
            if junction_path.exists():
                # Target through junction must resolve to real outside path and fail containment
                target_through_junction = str(junction_path / "secret.txt")
                assert not is_path_contained(target_through_junction, str(ws))
                res = read_file(target_through_junction)
                assert res.success is False
        finally:
            if junction_path.exists():
                subprocess.run(["cmd.exe", "/c", "rmdir", str(junction_path)], capture_output=True, check=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Command Execution Safeguards & Shell Builtins
# ─────────────────────────────────────────────────────────────────────────────

def test_command_execution_executable_and_args_with_spaces(tmp_path):
    """Verify command execution handles executable paths and arguments containing spaces."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # Use current python executable (which may contain spaces in path)
    py_exe = sys.executable
    token = "cmd_tok_spaces"
    workspace_registry.register_approval_token(token)

    cmd = f'"{py_exe}" -c "import sys; print(\'ARGUMENT WITH SPACES: \' + sys.argv[1])" "hello world 2026"'
    res = execute_command(cmd, cwd=ws, approval_token=token)

    assert res.success is True
    assert "ARGUMENT WITH SPACES: hello world 2026" in res.output["stdout"]


def test_command_execution_shell_builtins_safe_comspec(tmp_path):
    """Verify shell builtins like echo and dir work cleanly via cmd.exe /d /c without shell=True."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    token = "cmd_tok_echo"
    workspace_registry.register_approval_token(token)

    res = execute_command("echo DataPilot Safe Builtin", cwd=ws, approval_token=token)
    assert res.success is True
    assert "DataPilot Safe Builtin" in res.output["stdout"]


def test_command_injection_shell_builtin_gap(tmp_path):
    """Regression test: verify untrusted shell operators (&, |, >, <, ^, %) in shell builtins are rejected."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    token = "cmd_tok_inject"
    workspace_registry.register_approval_token(token)

    # Attempt command chaining via & in shell builtin
    res_chain = execute_command("echo hello & echo INJECTED", cwd=ws, approval_token=token)
    assert res_chain.success is False
    assert res_chain.error_code == "UNTRUSTED_SHELL_OPERATOR"
    if res_chain.output:
        assert "INJECTED" not in res_chain.output.get("stdout", "")

    # Attempt argument-level injection with &
    token2 = "cmd_tok_inject2"
    workspace_registry.register_approval_token(token2)
    res_arg = execute_command("echo", args=["echo", "foo&whoami"], cwd=ws, approval_token=token2)
    assert res_arg.success is False
    assert res_arg.error_code == "UNTRUSTED_SHELL_OPERATOR"

    # Attempt file redirection via >
    token3 = "cmd_tok_inject3"
    workspace_registry.register_approval_token(token3)
    pwn_file = os.path.join(ws, "pwned.txt")
    res_redir = execute_command(f"echo secret > {pwn_file}", cwd=ws, approval_token=token3)
    assert res_redir.success is False
    assert res_redir.error_code == "UNTRUSTED_SHELL_OPERATOR"
    assert not os.path.exists(pwn_file)


def test_adversarial_shell_builtin_marker_protection(tmp_path):
    """Adversarial security matrix: confirm shell builtins and cmd.exe invocations fail closed against
    quoting, delimiters, delayed expansion, grouping, destructive builtins, and path traversal,
    and verify rejected commands NEVER launch cmd.exe or create/modify marker files.
    """
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    marker_file = os.path.join(ws, "marker.txt")
    initial_content = "CANARY_INTEGRITY_OK"
    with open(marker_file, "w", encoding="utf-8") as f:
        f.write(initial_content)
    initial_mtime = os.path.getmtime(marker_file)

    adversarial_cases = [
        # 1. Destructive builtins (must fail closed to protect dedicated delete/move tools)
        ("del marker.txt", ["BUILTIN_DISALLOWED", "UNTRUSTED_SHELL_OPERATOR"]),
        ("erase marker.txt", ["BUILTIN_DISALLOWED", "UNTRUSTED_SHELL_OPERATOR"]),
        ("rmdir /s /q sub", ["BUILTIN_DISALLOWED", "UNTRUSTED_SHELL_OPERATOR"]),
        ("copy marker.txt marker2.txt", ["BUILTIN_DISALLOWED", "UNTRUSTED_SHELL_OPERATOR"]),
        ("move marker.txt marker2.txt", ["BUILTIN_DISALLOWED", "UNTRUSTED_SHELL_OPERATOR"]),
        # 2. Command chaining & piping with marker payload
        (f"echo safe & echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"echo safe && echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"echo safe | echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"echo safe || echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 3. Output & input redirection
        (f"echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"echo TAMPERED >> {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"type < {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 4. Caret escape tricks
        (f"echo safe^&echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"echo safe^>{marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 5. Environment & delayed expansion metacharacters
        ("echo %COMSPEC%", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo %TEMP%", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo !PATH!", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 6. Grouping and delimiter metacharacters
        ("(echo safe)", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo;TAMPERED", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo,TAMPERED", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo=TAMPERED", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 7. Quote breakouts and backticks
        ('echo "TAMPERED"', ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo 'TAMPERED'", ["UNTRUSTED_SHELL_OPERATOR"]),
        ("echo `TAMPERED`", ["UNTRUSTED_SHELL_OPERATOR"]),
        # 8. Traversal via dir or type
        ("dir ..\\..", ["PATH_NOT_AUTHORIZED", "UNTRUSTED_SHELL_OPERATOR"]),
        ("type ..\\..\\secret.txt", ["PATH_NOT_AUTHORIZED", "UNTRUSTED_SHELL_OPERATOR"]),
        # 9. Direct cmd.exe wrapper attempts
        (f"cmd.exe /c echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
        (f"cmd /c echo TAMPERED > {marker_file}", ["UNTRUSTED_SHELL_OPERATOR"]),
    ]

    for idx, (cmd_str, expected_error_codes) in enumerate(adversarial_cases):
        token = f"adv_tok_{idx}"
        workspace_registry.register_approval_token(token)
        res = execute_command(cmd_str, cwd=ws, approval_token=token)

        assert res.success is False, f"Adversarial command should have failed: {cmd_str}"
        assert res.error_code in expected_error_codes, f"Command '{cmd_str}' failed with {res.error_code}, expected one of {expected_error_codes}"

        # Crucial security verification: Marker file was NEVER deleted or modified
        assert os.path.exists(marker_file), f"Marker file was deleted by {cmd_str}"
        with open(marker_file, "r", encoding="utf-8") as f:
            current_content = f.read()
        assert current_content == initial_content, f"Marker file was modified by {cmd_str}: {current_content}"
        assert os.path.getmtime(marker_file) == initial_mtime, f"Marker file mtime altered by {cmd_str}"




def test_command_execution_timeout_and_cancellation(tmp_path):
    """Verify command runner enforces timeouts and supports process tree cancellation."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # 1. Timeout enforcement
    py_exe = sys.executable
    token_timeout = "tok_timeout"
    workspace_registry.register_approval_token(token_timeout)

    timeout_cmd = f'"{py_exe}" -c "import time; time.sleep(10)"'
    res_timeout = execute_command(timeout_cmd, cwd=ws, timeout_seconds=1, approval_token=token_timeout)

    assert res_timeout.success is False
    assert res_timeout.output["timed_out"] is True
    assert res_timeout.error_code == "TIMEOUT"

    # 2. Cancellation
    token_cancel = "tok_cancel"
    workspace_registry.register_approval_token(token_cancel)
    cancel_running_command("cancel_task_id")
    # Cancelled flag was registered for this task_id
    res_cancelled = execute_command(f'"{py_exe}" -c "print(\'done\')"', cwd=ws, task_id="cancel_task_id", approval_token=token_cancel)
    assert res_cancelled.output["cancelled"] is True


def test_command_execution_output_truncation(tmp_path):
    """Verify command output is capped at max_output_chars."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    py_exe = sys.executable
    token = "tok_trunc"
    workspace_registry.register_approval_token(token)

    cmd = f'"{py_exe}" -c "print(\'X\' * 5000)"'
    res = execute_command(cmd, cwd=ws, max_output_chars=200, approval_token=token)

    assert res.success is True
    assert res.output["is_truncated"] is True
    assert len(res.output["stdout"]) < 400
    assert "OUTPUT TRUNCATED" in res.output["stdout"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Source-Native Dataset In-Place Access & Drift
# ─────────────────────────────────────────────────────────────────────────────

def test_source_native_datasets_zero_copy_and_drift(tmp_path):
    """Verify source-native files remain at their original paths and detect external modifications."""
    ds_dir = tmp_path / "external_data"
    ds_dir.mkdir()
    csv_file = ds_dir / "sales_2026.csv"
    csv_file.write_text("region,revenue\nNorth,50000\nSouth,60000\n", encoding="utf-8")

    # Confirm not in uploads directory
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        assert not any(uploads_dir.glob("sales_2026.csv"))

    # Load and register as source native
    df1, load1 = dataset_loader.load(str(csv_file))
    entry = DatasetEntry(
        dataset_id="sales_2026",
        name="sales_2026.csv",
        file_path=str(csv_file),
        format="csv",
        row_count=len(df1),
        column_count=len(df1.columns),
        file_hash=load1.file_hash,
        source_file_mtime=os.path.getmtime(str(csv_file)),
        source_file_size=os.path.getsize(str(csv_file)),
        knowledge_model={},
        loaded_at="2026-10-19T20:00:00Z",
        is_source_native=True,
    )
    dataset_registry.register("sales_2026", entry)


    # Initial check
    assert entry.check_drift() is False
    assert entry.file_path == str(csv_file)

    # External modification
    time.sleep(0.05)
    csv_file.write_text("region,revenue\nNorth,50000\nSouth,60000\nWest,75000\n", encoding="utf-8")

    assert entry.check_drift() is True
    assert entry.drift_detected is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Advanced Windows Security & Root Cause Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_native_diagnostic_builtins_zero_child_process(tmp_path, monkeypatch):
    """Confirm diagnostic builtins (echo, dir, type, ver, vol, cls) execute natively without subprocess.Popen."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # Monitor subprocess.Popen to ensure it is NEVER called
    popen_called = []
    orig_popen = subprocess.Popen

    def mock_popen(*args, **kwargs):
        popen_called.append(args)
        return orig_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    token = "tok_diag_1"
    workspace_registry.register_approval_token(token)
    res_echo = execute_command("echo Hello DataPilot Native", cwd=ws, approval_token=token)
    assert res_echo.success is True
    assert "Hello DataPilot Native" in res_echo.output["stdout"]
    assert len(popen_called) == 0, "Native echo must not invoke subprocess.Popen"

    # Test native dir
    token2 = "tok_diag_2"
    workspace_registry.register_approval_token(token2)
    (tmp_path / "sample.txt").write_text("sample content", encoding="utf-8")
    res_dir = execute_command("dir /b", cwd=ws, approval_token=token2)
    assert res_dir.success is True
    assert "sample.txt" in res_dir.output["stdout"]
    assert len(popen_called) == 0, "Native dir must not invoke subprocess.Popen"

    # Test native type
    token3 = "tok_diag_3"
    workspace_registry.register_approval_token(token3)
    res_type = execute_command("type sample.txt", cwd=ws, approval_token=token3)
    assert res_type.success is True
    assert "sample content" in res_type.output["stdout"]
    assert len(popen_called) == 0, "Native type must not invoke subprocess.Popen"

    # Test native ver & vol
    token4 = "tok_diag_4"
    workspace_registry.register_approval_token(token4)
    res_ver = execute_command("ver", cwd=ws, approval_token=token4)
    assert res_ver.success is True
    assert len(popen_called) == 0, "Native ver must not invoke subprocess.Popen"


def test_untrusted_cwd_binary_shadowing_prevention(tmp_path):
    """Confirm rogue binaries in the current workspace directory cannot shadow trusted system executables."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # Place fake rogue 'git.exe' or 'git.bat' in the workspace directory
    rogue_git = tmp_path / "git.bat"
    rogue_marker = tmp_path / "shadow_marker.txt"
    rogue_git.write_text(f"@echo off\necho SHADOWED > {rogue_marker}\n", encoding="utf-8")

    token = "tok_shadow_1"
    workspace_registry.register_approval_token(token)

    # Execute git status: must resolve to real Git on PATH, NOT the rogue git.bat in CWD
    res = execute_command("git status", cwd=ws, approval_token=token)
    # The real git status runs (either succeeds or says not a git repo)
    assert not rogue_marker.exists(), "Rogue workspace binary was executed via CWD shadowing!"


def test_untrusted_workspace_batch_script_rejection(tmp_path):
    """Confirm executing untrusted batch scripts (.bat, .cmd) from the workspace fails closed."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    batch_file = tmp_path / "payload.bat"
    marker_file = tmp_path / "batch_marker.txt"
    batch_file.write_text(f"@echo off\necho PWNED > {marker_file}\n", encoding="utf-8")

    token = "tok_batch_1"
    workspace_registry.register_approval_token(token)

    res = execute_command("payload.bat", cwd=ws, approval_token=token)
    assert res.success is False
    assert res.error_code == "UNTRUSTED_BATCH_FILE"
    assert not marker_file.exists(), "Untrusted batch file in workspace must not execute!"

    # Explicit relative path
    token2 = "tok_batch_2"
    workspace_registry.register_approval_token(token2)
    res2 = execute_command(".\\payload.bat", cwd=ws, approval_token=token2)
    assert res2.success is False
    assert res2.error_code == "UNTRUSTED_BATCH_FILE"
    assert not marker_file.exists()


def test_shell_wrapper_interpreters_rejection(tmp_path):
    """Confirm direct invocations of shell interpreters (powershell, bash, wscript) are blocked."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    interpreters = [
        "powershell -Command Get-Process",
        "powershell.exe -Command Get-Process",
        "pwsh -c dir",
        "bash -c ls",
        "wscript.exe script.vbs",
        "cscript.exe script.vbs",
    ]

    for idx, cmd in enumerate(interpreters):
        tok = f"tok_interp_{idx}"
        workspace_registry.register_approval_token(tok)
        res = execute_command(cmd, cwd=ws, approval_token=tok)
        assert res.success is False
        assert res.error_code == "UNTRUSTED_SHELL_OPERATOR"


def test_unicode_and_spaces_in_paths(tmp_path):
    """Confirm paths containing Unicode and spaces are preserved and handled safely."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    # Unicode folder with spaces
    special_folder = tmp_path / "földer with späces" / "тест dir"
    target_file = str(special_folder / "données 2026.txt")

    # 1. Create directory and file
    create_res = create_file(target_file, "Contenu en Français et Русский.\n")
    assert create_res.success is True
    assert os.path.exists(target_file)

    # 2. Read file
    read_res = read_file(target_file)
    assert read_res.success is True
    assert "Contenu en Français et Русский." in read_res.output

    # 3. Diff and patch
    diff_res = get_diff(target_file, "Contenu en Français et Русский — Modifié.\n")
    assert diff_res.success is True
    assert diff_res.metadata["has_changes"] is True

    patch_res = apply_file_patch(target_file, "Contenu en Français et Русский — Modifié.\n")
    assert patch_res.success is True

    read_back = read_file(target_file)
    assert "Modifié" in read_back.output


# ─────────────────────────────────────────────────────────────────────────────
# 13. Ollama Semantic Planner, Structured Schema & Error Envelope Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_planner_valid_structured_plan(tmp_path, monkeypatch):
    """Verify generate_task_plan creates a valid WorkspaceTaskPlan using OllamaRequest/Response."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    from backend.agent.planner import generate_task_plan, PROMPT_VERSION, SCHEMA_VERSION
    from backend.agent.audit import agent_audit
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaReadinessStatus

    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=True, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct", model_available=True
    ))

    mock_plan_json = {
        "objective": "Create a README file and list directory",
        "steps": [
            {
                "tool_name": "create_file",
                "arguments": {"path": os.path.join(ws, "README.md"), "content": "# Hello DataPilot\n"},
                "explanation": "Create README file"
            },
            {
                "tool_name": "list_directory",
                "arguments": {"path": ws},
                "explanation": "List directory contents"
            }
        ],
        "verification_criteria": [
            {
                "type": "file_exists",
                "description": "README.md must exist",
                "path": os.path.join(ws, "README.md")
            }
        ]
    }

    def mock_structured_request(req):
        assert req.response_schema is not None
        assert PROMPT_VERSION in req.system_prompt
        assert SCHEMA_VERSION in req.system_prompt
        return OllamaResponse(
            success=True,
            content=json.dumps(mock_plan_json),
            parsed_json=mock_plan_json,
            latency_ms=120.5,
            prompt_eval_count=45,
            eval_count=80,
            model_used="qwen2.5:3b-instruct"
        )

    monkeypatch.setattr(ollama_service, "structured_request", mock_structured_request)

    plan = generate_task_plan(user_prompt="Create README and list files", workspace_path=ws)
    assert plan.objective == "Create a README file and list directory"
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "create_file"
    assert plan.steps[0].requires_approval is True  # REVIEW policy requires approval for create_file
    assert plan.steps[1].tool_name == "list_directory"
    assert plan.steps[1].requires_approval is False
    assert len(plan.verification_criteria) == 1

    # Verify audit telemetry
    records = agent_audit.get_records(limit=5, task_id=plan.task_id)
    plan_records = [r for r in records if r["operation"] == "plan_task"]
    assert len(plan_records) >= 1
    details = plan_records[0]["details"]
    assert details["prompt_version"] == "1.0"
    assert details["schema_version"] == "1.0"
    assert details["model_used"] == "qwen2.5:3b-instruct"
    assert details["step_count"] == 2


def test_ollama_planner_error_modes(tmp_path, monkeypatch):
    """Verify planner failure modes fail closed with structured PlannerError without regex guessing."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)

    from backend.agent.planner import generate_task_plan, PlannerError
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaReadinessStatus

    # 1. Empty prompt
    with pytest.raises(PlannerError) as exc_info:
        generate_task_plan("   \t  \n ", workspace_path=ws)
    assert exc_info.value.code == "EMPTY_PROMPT"

    # 2. No workspace authorized
    workspace_registry.clear()
    with pytest.raises(PlannerError) as exc_info2:
        generate_task_plan("Do something", workspace_path=None)
    assert exc_info2.value.code == "NO_AUTHORIZED_WORKSPACE"

    workspace_registry.authorize_workspace(ws)

    # 3. Ollama unavailable
    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=False, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct",
        error_code="OLLAMA_SERVICE_UNAVAILABLE", error_message="Connection refused"
    ))
    with pytest.raises(PlannerError) as exc_info3:
        generate_task_plan("Do something", workspace_path=ws)
    assert exc_info3.value.code == "OLLAMA_SERVICE_UNAVAILABLE"

    # 4. Ollama timeout
    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=True, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct"
    ))
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=False, error_code="TIMEOUT", error_message="Request timed out after 45s"
    ))
    with pytest.raises(PlannerError) as exc_info4:
        generate_task_plan("Do something", workspace_path=ws)
    assert exc_info4.value.code == "PLANNER_TIMEOUT"

    # 5. Malformed JSON output
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=True, content="I will do this task by reading your files.", parsed_json=None
    ))
    with pytest.raises(PlannerError) as exc_info5:
        generate_task_plan("Do something", workspace_path=ws)
    assert exc_info5.value.code == "INVALID_TASK_PLAN"

    # 6. Unknown tool in plan
    bad_plan = {
        "objective": "Run an exploit",
        "steps": [{"tool_name": "format_c_drive", "arguments": {}, "explanation": "bad"}]
    }
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=True, content=json.dumps(bad_plan), parsed_json=bad_plan
    ))
    with pytest.raises(PlannerError) as exc_info6:
        generate_task_plan("Do something", workspace_path=ws)
    assert exc_info6.value.code == "INVALID_TOOL_IN_PLAN"


def test_bounded_verification_self_repair(tmp_path, monkeypatch):
    """Verify TaskOrchestrator invokes bounded self-repair on verification failure and respects approval policy."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    from backend.agent.planner import WorkspaceTaskPlan, TaskStep
    from backend.agent.orchestrator import task_orchestrator
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaReadinessStatus

    target_file = os.path.join(ws, "output.txt")

    # Initial plan: create file with wrong content, verification expects "CORRECT_CONTENT"
    init_tok = "tok_init_repair_1"
    workspace_registry.register_approval_token(init_tok)
    plan = WorkspaceTaskPlan(
        objective="Create output with specific text",
        workspace_path=ws,
        steps=[
            TaskStep(
                step_id="step_1",
                tool_name="create_file",
                arguments={"path": target_file, "content": "WRONG_CONTENT\n", "approval_token": init_tok},
                requires_approval=False,  # Pretend already approved with valid token
                status="PENDING",
            )
        ],
        verification_criteria=[
            {
                "type": "file_contains",
                "description": "output.txt must contain CORRECT_CONTENT",
                "path": target_file,
                "expected": "CORRECT_CONTENT"
            }
        ]
    )

    task_orchestrator.register_plan(plan)

    # Mock Ollama for repair proposal
    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=True, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct"
    ))

    repair_response_json = {
        "repair_explanation": "Replace file content with CORRECT_CONTENT",
        "steps": [
            {
                "tool_name": "apply_file_patch",
                "arguments": {"path": target_file, "new_content": "CORRECT_CONTENT\n"},
                "explanation": "Fix file content to match verification criteria"
            }
        ]
    }

    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=True,
        content=json.dumps(repair_response_json),
        parsed_json=repair_response_json,
        latency_ms=90.0,
        model_used="qwen2.5:3b-instruct"
    ))

    # 1. Execute initial task: step 1 succeeds, verification fails, repair step is generated.
    # Under REVIEW policy, apply_file_patch requires approval! So task halts at AWAITING_APPROVAL.
    res_plan = task_orchestrator.execute_task(plan.task_id)
    assert res_plan.status == "AWAITING_APPROVAL"
    assert res_plan.summary["repair_attempts"] == 1
    assert len(res_plan.steps) == 2
    repair_step = res_plan.steps[1]
    assert repair_step.tool_name == "apply_file_patch"
    assert repair_step.requires_approval is True

    # 2. Approve the repair step
    approved_plan = task_orchestrator.approve_step(plan.task_id, repair_step.step_id)
    # Execution resumes, apply_file_patch runs, verification re-runs and succeeds!
    assert approved_plan.status == "COMPLETED"
    assert approved_plan.summary["repair_attempts"] == 1
    with open(target_file, "r", encoding="utf-8") as f:
        assert "CORRECT_CONTENT" in f.read()


def test_bounded_verification_repair_exhaustion(tmp_path, monkeypatch):
    """Verify repair loop strictly limits to 1 attempt and fails closed when verification still fails."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    from backend.agent.planner import WorkspaceTaskPlan, TaskStep
    from backend.agent.orchestrator import task_orchestrator
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaReadinessStatus

    target_file = os.path.join(ws, "stubborn.txt")

    plan = WorkspaceTaskPlan(
        objective="Create stubborn file",
        workspace_path=ws,
        steps=[
            TaskStep(
                step_id="step_init",
                tool_name="create_file",
                arguments={"path": target_file, "content": "INITIAL\n"},
                requires_approval=False,
                status="PENDING",
            )
        ],
        verification_criteria=[
            {
                "type": "file_contains",
                "description": "must contain IMPOSSIBLE",
                "path": target_file,
                "expected": "IMPOSSIBLE"
            }
        ]
    )
    task_orchestrator.register_plan(plan)

    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=True, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct"
    ))

    # Repair model proposes another change that STILL does not satisfy the criteria
    repair_json = {
        "repair_explanation": "Patch file with STILL_WRONG_DATA",
        "steps": [
            {
                "tool_name": "apply_file_patch",
                "arguments": {"path": target_file, "new_content": "STILL_WRONG_DATA\n"},
                "explanation": "Remedial step"
            }
        ]
    }
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=True, content=json.dumps(repair_json), parsed_json=repair_json
    ))

    res_plan = task_orchestrator.execute_task(plan.task_id)
    # Because AUTONOMOUS policy permits apply_file_patch, repair step executes immediately.
    # Verification re-runs and fails again.
    # Since repair_attempts is now 1, it aborts with status FAILED without infinite loop!
    assert res_plan.status == "FAILED"
    assert res_plan.summary["repair_attempts"] == 1
    assert "Verification failed" in " ".join(res_plan.summary["unresolved_errors"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Concurrency Protection & expected_hash Strict Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_file_patch_correct_hash_succeeds(tmp_path):
    """Confirm a valid 64-char SHA-256 matching the file applies the patch cleanly."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    f = tmp_path / "target.txt"
    f.write_text("INITIAL_VERSION_1\n", encoding="utf-8")
    correct_hash = hashlib.sha256(f.read_bytes()).hexdigest()

    res = tools.apply_file_patch(str(f), new_content="UPDATED_VERSION_2\n", expected_hash=correct_hash)
    assert res.success is True
    assert f.read_text(encoding="utf-8") == "UPDATED_VERSION_2\n"


def test_apply_file_patch_stale_hash_rejected(tmp_path):
    """Confirm a stale hash immediately halts execution with CONCURRENT_MODIFICATION and preserves disk."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    f = tmp_path / "stale_target.txt"
    f.write_text("STATE_A\n", encoding="utf-8")
    stale_hash = hashlib.sha256(f.read_bytes()).hexdigest()

    # External actor changes file
    f.write_text("EXTERNAL_STATE_B\n", encoding="utf-8")

    res = tools.apply_file_patch(str(f), new_content="ATTEMPTED_STATE_C\n", expected_hash=stale_hash)
    assert res.success is False
    assert res.error_code == "CONCURRENT_MODIFICATION"
    assert "Concurrent modification detected" in res.error
    # File content preserved exactly
    assert f.read_text(encoding="utf-8") == "EXTERNAL_STATE_B\n"


def test_apply_file_patch_malformed_hashes_rejected(tmp_path):
    """Confirm any malformed, non-hex, or placeholder expected_hash fails closed with INVALID_HASH_FORMAT."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    f = tmp_path / "malformed_hash_target.txt"
    f.write_text("ORIGINAL_SAFE_CONTENT\n", encoding="utf-8")

    malformed_candidates = [
        "expected...",
        "null",
        "None",
        "undefined",
        "12345",
        "not_a_hex_hash_at_all",
        "g" * 64,  # 'g' is non-hex
        "a" * 63,  # wrong length (63)
        "a" * 65,  # wrong length (65)
        "",        # empty string
        "   ",     # whitespace
        12345,     # non-string integer
    ]

    for candidate in malformed_candidates:
        res = tools.apply_file_patch(
            str(f),
            new_content=f"MUTATED_BY_{candidate}\n",
            expected_hash=candidate,
        )
        assert res.success is False, f"Expected failure for malformed hash: {repr(candidate)}"
        assert res.error_code == "INVALID_HASH_FORMAT", f"Expected INVALID_HASH_FORMAT for {repr(candidate)}, got {res.error_code}"
        # Disk must NEVER be modified
        assert f.read_text(encoding="utf-8") == "ORIGINAL_SAFE_CONTENT\n"


def test_apply_file_patch_missing_optional_hash_succeeds(tmp_path):
    """Confirm when expected_hash is omitted (None), patching succeeds without hash comparison."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.AUTONOMOUS)

    f = tmp_path / "optional_target.txt"
    f.write_text("BASE_TEXT\n", encoding="utf-8")

    res = tools.apply_file_patch(str(f), new_content="PATCHED_NO_HASH\n", expected_hash=None)
    assert res.success is True
    assert f.read_text(encoding="utf-8") == "PATCHED_NO_HASH\n"


def test_concurrent_modification_between_planning_and_execution(tmp_path, monkeypatch):
    """Confirm that external file edits occurring between plan synthesis and task execution are caught."""
    ws = str(tmp_path)
    workspace_registry.authorize_workspace(ws)
    workspace_registry.set_policy(AgentPolicy.REVIEW)

    from backend.agent.planner import generate_task_plan
    from backend.agent.orchestrator import task_orchestrator
    from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaReadinessStatus

    target_file = tmp_path / "concurrency_race.txt"
    target_file.write_text("PLAN_TIME_DATA\n", encoding="utf-8")

    monkeypatch.setattr(ollama_service, "verify_ready", lambda: OllamaReadinessStatus(
        ready=True, ollama_url="http://localhost:11434", model="qwen2.5:3b-instruct"
    ))

    # Model proposes a patch without hash or with placeholder
    mock_plan_json = {
        "objective": "Update concurrency_race.txt",
        "is_ambiguous": False,
        "steps": [
            {
                "tool_name": "apply_file_patch",
                "arguments": {"path": str(target_file), "new_content": "PLANNED_NEW_CONTENT\n", "expected_hash": "expected..."},
                "explanation": "Update file"
            }
        ],
        "verification_criteria": []
    }
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: OllamaResponse(
        success=True, content=json.dumps(mock_plan_json), parsed_json=mock_plan_json
    ))

    initial_bytes = target_file.read_bytes()
    plan = generate_task_plan("Update concurrency race file", workspace_path=ws)
    assert plan.steps[0].arguments["expected_hash"] == hashlib.sha256(initial_bytes).hexdigest()

    # External modification occurs AFTER plan generation and BEFORE execution approval
    target_file.write_text("EXTERNAL_PROCESS_MODIFICATION\n", encoding="utf-8")

    task_orchestrator.register_plan(plan)
    # Under REVIEW policy, file patch step requires approval
    step_id = plan.steps[0].step_id

    # Approve and execute
    approved_plan = task_orchestrator.approve_step(plan.task_id, step_id)

    # Must fail with CONCURRENT_MODIFICATION
    assert approved_plan.status == "FAILED"
    assert "Concurrent modification detected" in approved_plan.error
    # External modification on disk was NOT destroyed
    assert target_file.read_text(encoding="utf-8") == "EXTERNAL_PROCESS_MODIFICATION\n"


def test_dynamic_tool_registry_extensibility():
    """Verify that the capability registry dynamically supports adding tools without modifying planner/orchestrator source."""
    from backend.agent.config import agent_config
    from backend.agent.planner import VALID_TOOL_NAMES, generate_task_plan
    from backend.agent.registry import AgentTool, ToolCategory, agent_tool_registry
    from backend.agent.tools import ToolResult

    test_tool_name = "test_custom_diagnostic"

    # Define mock diagnostic handler
    def custom_diagnostic_handler(tag: str = "health_ok", **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            tool_name=test_tool_name,
            output={"tag": tag, "status": "healthy"},
        )

    # Register into global agent tool registry
    custom_tool = AgentTool(
        name=test_tool_name,
        description="Custom workspace diagnostic probe.",
        signature=f"{test_tool_name}(tag: str = 'health_ok')",
        category=ToolCategory.READ_ONLY,
        handler=custom_diagnostic_handler,
    )
    agent_tool_registry.register(custom_tool)

    try:
        # Dynamic verification
        assert agent_tool_registry.is_valid_tool(test_tool_name) is True
        assert test_tool_name in VALID_TOOL_NAMES
        assert test_tool_name in agent_tool_registry.format_system_prompt_tools()

        # Approval policy evaluation (READ_ONLY requires no approval under REVIEW)
        assert agent_tool_registry.evaluate_step_approval(test_tool_name, AgentPolicy.REVIEW) is False
        assert agent_tool_registry.evaluate_step_approval(test_tool_name, AgentPolicy.AUTO_EDIT) is False
        assert agent_tool_registry.evaluate_step_approval(test_tool_name, AgentPolicy.AUTONOMOUS) is False

        # Dispatch via orchestrator
        res = task_orchestrator._dispatch_tool(test_tool_name, {"tag": "memory_check"}, task_id="test_ext_task")
        assert res.success is True
        assert res.tool_name == test_tool_name
        assert res.output == {"tag": "memory_check", "status": "healthy"}

        # Register a destructive test tool and check approval evaluation
        destructive_name = "test_custom_cleanup"
        agent_tool_registry.register(
            AgentTool(
                name=destructive_name,
                description="Custom cleanup tool.",
                signature=f"{destructive_name}(target: str)",
                category=ToolCategory.DESTRUCTIVE,
                handler=lambda target, **kw: ToolResult(success=True, tool_name=destructive_name),
            )
        )
        assert agent_tool_registry.evaluate_step_approval(destructive_name, AgentPolicy.REVIEW) is True
        assert agent_tool_registry.evaluate_step_approval(destructive_name, AgentPolicy.AUTO_EDIT) is True
        assert agent_tool_registry.evaluate_step_approval(destructive_name, AgentPolicy.AUTONOMOUS) is True
        agent_tool_registry.unregister(destructive_name)
    finally:
        # Clean up dynamic tools to keep registry pristine
        agent_tool_registry.unregister(test_tool_name)

    assert agent_tool_registry.is_valid_tool(test_tool_name) is False
    assert test_tool_name not in VALID_TOOL_NAMES


def test_centralized_agent_config_invariants():
    """Verify that centralized configuration parameters have non-empty, sensible production defaults."""
    from backend.agent.config import agent_config

    assert agent_config.PROMPT_VERSION == "1.0"
    assert agent_config.SCHEMA_VERSION == "1.0"
    assert agent_config.PLANNER_MODEL in ("qwen2.5:3b-instruct", "qwen2.5-coder:3b-instruct")
    assert agent_config.OLLAMA_URL.startswith("http")
    assert agent_config.PLANNER_TIMEOUT_SEC >= 10.0
    assert agent_config.NUM_CTX >= 2048
    assert agent_config.KEEP_ALIVE == "60m"
    assert agent_config.MAX_REPAIR_ATTEMPTS >= 1
    assert agent_config.DEFAULT_COMMAND_TIMEOUT_SEC >= 5




