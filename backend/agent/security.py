"""
DataPilot OS — Agent Security & Path Authorization Broker.
Provides hardened Windows path normalization, canonical containment validation,
symlink/junction escape guards, TOCTOU mitigation, and secret redaction.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple


class SecurityContainmentError(Exception):
    """Raised when a path fails containment or escapes an authorized workspace boundary."""

    def __init__(self, code: str, message: str, path: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path


# Standard ignored directories for recursive traversal disclosures
DEFAULT_EXCLUDED_DIRS: Set[str] = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "dist-electron",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

# Regex pattern for secret redaction
_SECRET_PATTERNS = [
    # Bearer tokens & JWTs
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_\-\.]{16,})"),
    re.compile(r"(?i)(token[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9_\-\.]{16,})([\"']?)"),
    # Generic API Keys (sk-..., gsk-..., key-...)
    re.compile(r"(?i)\b(sk-[A-Za-z0-9]{20,})\b"),
    re.compile(r"(?i)\b(ghp_[A-Za-z0-9]{20,})\b"),
    re.compile(r"(?i)\b(AIza[0-9A-Za-z-_]{35})\b"),
    # Passwords in URLs (e.g. postgresql://user:password@host)
    re.compile(r"://([^:]+):([^@]+)@"),
    # Environment assignments (KEY=value, SECRET=value, PASSWORD=value)
    re.compile(r"(?i)(password|passwd|secret|api_key|apikey|auth_token)\s*=\s*['\"]?([^'\"\s\r\n]{6,})['\"]?"),
    # AWS Access Key ID & Secret Key
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
]


def redact_secrets(text: Optional[str]) -> str:
    """Redacts known token, key, and password formats from text, logs, and outputs."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    redacted = text
    for pattern in _SECRET_PATTERNS:
        # For URL pattern
        if "://" in pattern.pattern:
            redacted = pattern.sub(r"://\1:[REDACTED_PASSWORD]@", redacted)
        # For labeled patterns with prefix
        elif "(?i)(bearer" in pattern.pattern or "(?i)(token" in pattern.pattern:
            redacted = pattern.sub(r"\1[REDACTED_TOKEN]\3" if r"\3" in pattern.pattern else r"\1[REDACTED_TOKEN]", redacted)
        elif "(?i)(password" in pattern.pattern:
            redacted = pattern.sub(r"\1=[REDACTED_SECRET]", redacted)
        else:
            redacted = pattern.sub(r"[REDACTED_SECRET]", redacted)

    return redacted


def clean_windows_path(path_str: str) -> str:
    """Strips quotes, normalizes slashes, and removes Windows extended prefix where safe."""
    if not path_str:
        return ""
    p = path_str.strip().strip('"').strip("'")
    if p.startswith("\\\\?\\UNC\\") or p.startswith("//?/UNC/"):
        p = "\\\\" + p[8:]
    elif p.startswith("\\\\?\\") or p.startswith("//?/"):
        p = p[4:]
    return p



def resolve_canonical_path(path_input: str | Path, strict: bool = False) -> str:
    """Resolves an absolute, canonical, real path on the local filesystem.
    
    Resolves symlinks, junctions, and relative navigation (`.` and `..`).
    """
    clean_str = clean_windows_path(str(path_input))
    if not clean_str:
        raise SecurityContainmentError("INVALID_PATH", "Path cannot be empty", clean_str)

    try:
        p = Path(clean_str)
        # If relative, anchor to current working directory
        if not p.is_absolute():
            p = Path.cwd() / p
        
        # Realpath resolves symlinks and junctions
        real = os.path.realpath(str(p))
        norm = os.path.normpath(real)
        return norm
    except Exception as exc:
        raise SecurityContainmentError("CANONICAL_RESOLUTION_FAILED", f"Could not resolve path '{path_input}': {exc}", clean_str) from exc


def is_path_contained(target_path: str | Path, root_path: str | Path) -> bool:
    """Strictly checks if target_path is contained within root_path.
    
    Handles Windows case-insensitivity, drive letter comparisons,
    symlinks/junctions, and non-existent leaf targets.
    Fails closed if containment cannot be established.
    """
    try:
        target_str = clean_windows_path(str(target_path))
        root_str = clean_windows_path(str(root_path))

        if not target_str or not root_str:
            return False

        # Resolve root canonical path (root must exist)
        if not os.path.exists(root_str):
            return False
        canon_root = resolve_canonical_path(root_str)

        # For target: if exists, resolve realpath directly
        if os.path.exists(target_str):
            canon_target = resolve_canonical_path(target_str)
        else:
            # If target does not exist yet (e.g. creating new file/dir),
            # resolve the closest existing ancestor to guard against symlink escapes
            target_p = Path(target_str)
            if not target_p.is_absolute():
                target_p = Path(canon_root) / target_p

            ancestor = target_p
            leaf_parts = []
            while not ancestor.exists():
                leaf_parts.append(ancestor.name)
                parent = ancestor.parent
                if parent == ancestor:
                    # Reached root without finding existing path
                    return False
                ancestor = parent

            canon_ancestor = resolve_canonical_path(str(ancestor))
            # Verify ancestor itself is contained in root
            if not _check_containment_normalized(canon_ancestor, canon_root):
                return False

            # Check that leaf parts do not contain traversal
            leaf_parts.reverse()
            for part in leaf_parts:
                if part in ("..", ".", "") or "/" in part or "\\" in part:
                    return False

            canon_target = os.path.normpath(os.path.join(canon_ancestor, *leaf_parts))

        return _check_containment_normalized(canon_target, canon_root)
    except Exception:
        # Fail closed on any resolution or permission error
        return False


def _check_containment_normalized(target: str, root: str) -> bool:
    """Internal helper comparing normalized case paths."""
    # Compare drive letters or UNC share names on Windows
    target_drive, _ = os.path.splitdrive(target)
    root_drive, _ = os.path.splitdrive(root)
    if target_drive and root_drive and target_drive.upper() != root_drive.upper():
        return False

    norm_target = os.path.normcase(os.path.normpath(target))
    norm_root = os.path.normcase(os.path.normpath(root))

    if norm_target == norm_root:
        return True

    # Trailing separator check ensures subpaths match boundaries exactly and handles
    # drive roots (e.g. C:\) and UNC roots (e.g. \\server\share\) without sibling collision
    root_with_sep = norm_root if norm_root.endswith(os.sep) else norm_root + os.sep
    if norm_target.startswith(root_with_sep):
        return True

    try:
        common = os.path.commonpath([norm_target, norm_root])
        return common == norm_root
    except ValueError:
        return False



def revalidate_path_containment(target_path: str | Path, authorized_roots: Sequence[str | Path]) -> str:
    """Revalidates path containment immediately before filesystem execution.
    
    Fails closed if the path is not contained within at least one authorized root.
    Returns the resolved, normalized canonical path.
    """
    if not authorized_roots:
        raise SecurityContainmentError(
            "NO_AUTHORIZED_WORKSPACE",
            "No workspace directory has been authorized. Select an authorized workspace folder first.",
            str(target_path)
        )

    clean_target = clean_windows_path(str(target_path))
    if not clean_target:
        raise SecurityContainmentError("INVALID_PATH", "Target path cannot be empty.", clean_target)

    for root in authorized_roots:
        if is_path_contained(clean_target, root):
            # Target is valid within this root
            if os.path.exists(clean_target):
                return resolve_canonical_path(clean_target)
            else:
                # Return normalized combined path
                target_p = Path(clean_target)
                if not target_p.is_absolute():
                    target_p = Path(resolve_canonical_path(str(root))) / target_p
                return os.path.normpath(str(target_p))

    raise SecurityContainmentError(
        "PATH_NOT_AUTHORIZED",
        f"Path '{target_path}' is outside authorized workspace boundaries.",
        clean_target
    )


def is_excluded_directory(dir_name: str) -> bool:
    """Checks if a directory name is in the standard excluded/ignored list."""
    norm = dir_name.strip().lower()
    return norm in {d.lower() for d in DEFAULT_EXCLUDED_DIRS}
