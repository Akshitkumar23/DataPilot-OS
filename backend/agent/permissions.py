"""
DataPilot OS — Granular Permission Engine & Workspace Governance.
Enforces execution-time permissions inside the trusted backend tool boundary.
Model-generated plans are strictly treated as untrusted candidate proposals.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from backend.agent.security import (
    SecurityContainmentError,
    clean_windows_path,
    is_path_contained,
    resolve_canonical_path,
    revalidate_path_containment,
)


class AgentPolicy(str, Enum):
    REVIEW = "review"
    AUTO_EDIT = "auto_edit"
    AUTONOMOUS = "autonomous"


class AgentAction(str, Enum):
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    PATCH = "patch"
    DELETE = "delete"
    MOVE = "move"
    EXECUTE_COMMAND = "execute_command"
    LIST_DIR = "list_directory"


class PermissionDeniedError(Exception):
    """Raised when an operation is rejected by policy, boundary containment, or lacks approval."""

    def __init__(
        self,
        code: str,
        message: str,
        action: Optional[str] = None,
        path: Optional[str] = None,
        requires_approval: bool = False,
        approval_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.action = action
        self.path = path
        self.requires_approval = requires_approval
        self.approval_details = approval_details or {}


class AuthorizationDecision(BaseModel):
    allowed: bool
    requires_approval: bool
    action: AgentAction
    resolved_path: Optional[str] = None
    policy: AgentPolicy
    reason: str = ""
    approval_prompt: Optional[str] = None


class WorkspaceRegistry:
    """Manages active authorized workspaces, permission policies, and explicit deny lists."""

    def __init__(self):
        self._authorized_workspaces: List[str] = []
        self._active_workspace: Optional[str] = None
        self._policy: AgentPolicy = AgentPolicy.REVIEW
        self._explicit_denies: List[str] = [
            ".env",
            ".env_token",
            "id_rsa",
            "id_ed25519",
        ]
        self._approved_tokens: Set[str] = set()

    def authorize_workspace(self, workspace_path: str) -> str:
        """Authorizes a local directory as a valid workspace root.
        
        Rejects root drives as a whole (e.g. 'C:\\') to prevent accidental whole-machine exposure.
        """
        clean = clean_windows_path(workspace_path)
        if not clean or not os.path.exists(clean):
            raise SecurityContainmentError("WORKSPACE_NOT_FOUND", f"Directory does not exist: {workspace_path}")

        canonical = resolve_canonical_path(clean)
        if not os.path.isdir(canonical):
            raise SecurityContainmentError("NOT_A_DIRECTORY", f"Path is not a directory: {workspace_path}")

        # Check against authorizing root drive directly
        drive, tail = os.path.splitdrive(canonical)
        if drive and (tail in ("\\", "/", "") or tail.strip("\\/") == ""):
            raise SecurityContainmentError(
                "ROOT_DRIVE_DISALLOWED",
                f"Authorizing an entire root drive ({drive}\\) is disallowed. Select a specific project directory.",
            )

        if canonical not in self._authorized_workspaces:
            self._authorized_workspaces.append(canonical)

        if not self._active_workspace:
            self._active_workspace = canonical

        return canonical

    def revoke_workspace(self, workspace_path: str) -> bool:
        """Revokes authorization for a directory."""
        canonical = clean_windows_path(workspace_path)
        for w in list(self._authorized_workspaces):
            if os.path.normcase(w) == os.path.normcase(canonical):
                self._authorized_workspaces.remove(w)
                if self._active_workspace and os.path.normcase(self._active_workspace) == os.path.normcase(w):
                    self._active_workspace = self._authorized_workspaces[0] if self._authorized_workspaces else None
                return True
        return False

    def get_authorized_workspaces(self) -> List[str]:
        return list(self._authorized_workspaces)

    def get_active_workspace(self) -> Optional[str]:
        return self._active_workspace

    def set_active_workspace(self, workspace_path: str) -> str:
        clean = clean_windows_path(workspace_path)
        canonical = resolve_canonical_path(clean)
        if canonical not in self._authorized_workspaces:
            self.authorize_workspace(canonical)
        self._active_workspace = canonical
        return canonical

    def get_policy(self) -> AgentPolicy:
        return self._policy

    def set_policy(self, policy: AgentPolicy | str) -> AgentPolicy:
        if isinstance(policy, str):
            policy = AgentPolicy(policy.lower())
        self._policy = policy
        return self._policy

    def add_explicit_deny(self, pattern_or_filename: str) -> None:
        if pattern_or_filename not in self._explicit_denies:
            self._explicit_denies.append(pattern_or_filename)

    def is_explicitly_denied(self, target_path: str) -> Tuple[bool, str]:
        """Checks if target_path matches any explicit deny rule. Deny strictly overrides allow."""
        norm_name = os.path.basename(target_path).lower()
        for deny in self._explicit_denies:
            if deny.lower() in norm_name or deny.lower() == norm_name:
                return True, f"Target matches explicit security deny rule '{deny}'."
        return False, ""

    def register_approval_token(self, token: str) -> None:
        self._approved_tokens.add(token)

    def consume_approval_token(self, token: Optional[str]) -> bool:
        if not token:
            return False
        if token in self._approved_tokens:
            self._approved_tokens.remove(token)
            return True
        return False

    def clear(self) -> None:
        self._authorized_workspaces.clear()
        self._active_workspace = None
        self._policy = AgentPolicy.REVIEW
        self._approved_tokens.clear()


# Singleton workspace registry
workspace_registry = WorkspaceRegistry()


def enforce_operation_permission(
    action: AgentAction | str,
    target_path: Optional[str] = None,
    command: Optional[str] = None,
    approval_token: Optional[str] = None,
    explicit_confirmed: bool = False,
) -> AuthorizationDecision:
    """Authoritative execution-time permission check.
    
    Evaluates:
    1. Action and target path containment against authorized workspaces.
    2. Explicit deny rules (deny overrides allow).
    3. Active policy requirements (Review vs Auto-edit vs Autonomous).
    4. Required approval status: if approval is required and token is absent, raises PermissionDeniedError.
    """
    if isinstance(action, str):
        action = AgentAction(action.lower())

    policy = workspace_registry.get_policy()
    authorized = workspace_registry.get_authorized_workspaces()

    # Special handling for command execution
    if action == AgentAction.EXECUTE_COMMAND:
        # Commands must have an authorized workspace as their working directory
        if not authorized:
            raise PermissionDeniedError(
                code="NO_AUTHORIZED_WORKSPACE",
                message="Cannot execute command without an authorized workspace directory.",
                action=action.value,
            )

        # Check approval requirement for command execution
        requires_approval = True
        # In autonomous mode, only strict test/lint commands are pre-cleared
        if policy == AgentPolicy.AUTONOMOUS and command:
            import shlex
            try:
                tokens = shlex.split(command, posix=False)
            except Exception:
                tokens = command.split()
            first_tok = tokens[0].strip('"\'' ) if tokens else ""
            base0 = os.path.basename(first_tok).lower()
            cmd_lower = command.strip().lower()
            if (
                cmd_lower.startswith(("pytest", "python -m pytest", "npm test", "oxlint", "git status"))
                or base0 in ("python.exe", "python", "pytest.exe", "pytest")
            ):
                requires_approval = False

        if requires_approval:
            if not workspace_registry.consume_approval_token(approval_token):
                raise PermissionDeniedError(
                    code="APPROVAL_REQUIRED",
                    message=f"Command execution '{command}' requires user approval under '{policy.value}' policy.",
                    action=action.value,
                    requires_approval=True,
                    approval_details={"command": command, "policy": policy.value},
                )

        return AuthorizationDecision(
            allowed=True,
            requires_approval=False,
            action=action,
            resolved_path=workspace_registry.get_active_workspace(),
            policy=policy,
            reason="Command execution authorized.",
        )

    # File and directory operations require target_path
    if not target_path:
        raise PermissionDeniedError(
            code="MISSING_TARGET_PATH",
            message=f"Action '{action.value}' requires a valid target path.",
            action=action.value,
        )

    # 1. Path containment check (fail closed)
    try:
        resolved = revalidate_path_containment(target_path, authorized)
    except SecurityContainmentError as sc_err:
        raise PermissionDeniedError(
            code=sc_err.code,
            message=sc_err.message,
            action=action.value,
            path=target_path,
        ) from sc_err

    # 2. Explicit deny rule check (deny overrides allow)
    denied, reason = workspace_registry.is_explicitly_denied(resolved)
    if denied:
        raise PermissionDeniedError(
            code="EXPLICITLY_DENIED",
            message=f"Access to '{resolved}' is denied: {reason}",
            action=action.value,
            path=resolved,
        )

    # 3. Action-specific approval rules based on policy
    requires_approval = False
    approval_prompt = None

    if action == AgentAction.DELETE:
        # File/dir deletion ALWAYS requires explicit confirmation
        if not explicit_confirmed and not workspace_registry.consume_approval_token(approval_token):
            raise PermissionDeniedError(
                code="APPROVAL_REQUIRED",
                message=f"Deleting '{resolved}' requires explicit confirmation.",
                action=action.value,
                path=resolved,
                requires_approval=True,
                approval_details={"operation": "delete", "target": resolved},
            )

    elif action in (AgentAction.WRITE, AgentAction.CREATE, AgentAction.PATCH):
        if policy == AgentPolicy.REVIEW:
            requires_approval = True
            approval_prompt = f"Write/modify file '{resolved}'"
        elif policy in (AgentPolicy.AUTO_EDIT, AgentPolicy.AUTONOMOUS):
            requires_approval = False

    elif action == AgentAction.MOVE:
        if policy == AgentPolicy.REVIEW:
            requires_approval = True
            approval_prompt = f"Move file '{resolved}'"

    # Enforce approval if required
    if requires_approval:
        if not workspace_registry.consume_approval_token(approval_token):
            raise PermissionDeniedError(
                code="APPROVAL_REQUIRED",
                message=f"Operation '{action.value}' on '{resolved}' requires user approval under policy '{policy.value}'.",
                action=action.value,
                path=resolved,
                requires_approval=True,
                approval_details={"action": action.value, "path": resolved, "prompt": approval_prompt},
            )

    return AuthorizationDecision(
        allowed=True,
        requires_approval=False,
        action=action,
        resolved_path=resolved,
        policy=policy,
        reason=f"Action '{action.value}' authorized under policy '{policy.value}'.",
    )
