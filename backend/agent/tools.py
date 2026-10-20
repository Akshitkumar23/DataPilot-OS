"""
DataPilot OS — Reusable Filesystem & Process Tool Layer.
Provides typed, production-grade tools with fail-closed authorization,
atomic writes, concurrent modification guards, tree-kill process execution, and secret redaction.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from backend.agent.audit import agent_audit
from backend.agent.permissions import (
    AgentAction,
    PermissionDeniedError,
    enforce_operation_permission,
    workspace_registry,
)
from backend.agent.security import (
    DEFAULT_EXCLUDED_DIRS,
    SecurityContainmentError,
    clean_windows_path,
    is_excluded_directory,
    redact_secrets,
    revalidate_path_containment,
)


class ToolResult(BaseModel):
    success: bool
    tool_name: str
    path: Optional[str] = None
    output: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def compute_sha256(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


def is_binary_file(file_path: str, sample_size: int = 8192) -> bool:
    """Detects whether a file is binary by inspecting the first sample_size bytes for null bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
            if b"\x00" in chunk:
                return True
            # Check high proportion of non-text characters
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})
            non_text = chunk.translate(None, text_chars)
            return len(non_text) / len(chunk) > 0.3 if chunk else False
    except Exception:
        return False


def detect_encoding_and_read(file_path: str, max_bytes: int = 1048576) -> Tuple[str, str, str]:
    """Reads file text attempting UTF-8, then UTF-16, then Latin-1.
    Returns (content_text, detected_encoding, sha256_hash).
    """
    with open(file_path, "rb") as f:
        raw_bytes = f.read(max_bytes)
        file_hash = compute_sha256(raw_bytes)

    encodings = ["utf-8", "utf-8-sig", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            return text, enc, file_hash
        except UnicodeDecodeError:
            continue

    # Fallback with replacement
    return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replace", file_hash


# ─────────────────────────────────────────────────────────────────────────────
# 1. Directory Listing
# ─────────────────────────────────────────────────────────────────────────────

def list_directory(
    path: str,
    recursive: bool = False,
    max_depth: int = 2,
    include_hidden: bool = False,
    limit: int = 500,
) -> ToolResult:
    """Lists files and folders within an authorized path with bounded traversal and disclosures."""
    try:
        decision = enforce_operation_permission(AgentAction.LIST_DIR, target_path=path)
        resolved_path = decision.resolved_path

        if not os.path.exists(resolved_path):
            return ToolResult(
                success=False,
                tool_name="list_directory",
                path=path,
                error=f"Directory does not exist: {path}",
                error_code="PATH_NOT_FOUND",
            )

        if not os.path.isdir(resolved_path):
            return ToolResult(
                success=False,
                tool_name="list_directory",
                path=path,
                error=f"Path is not a directory: {path}",
                error_code="NOT_A_DIRECTORY",
            )

        entries = []
        disclosed_exclusions = set()
        count = 0

        if not recursive:
            with os.scandir(resolved_path) as it:
                for entry in it:
                    if count >= limit:
                        break
                    is_hidden = entry.name.startswith(".")
                    if is_hidden and not include_hidden:
                        continue
                    if entry.is_dir() and is_excluded_directory(entry.name):
                        disclosed_exclusions.add(entry.name)
                        continue

                    try:
                        stat = entry.stat()
                        entries.append({
                            "name": entry.name,
                            "path": clean_windows_path(entry.path),
                            "is_dir": entry.is_dir(),
                            "size": stat.st_size if entry.is_file() else None,
                            "mtime": stat.st_mtime,
                        })
                        count += 1
                    except OSError:
                        continue
        else:
            base_depth = len(Path(resolved_path).parts)
            for root, dirs, files in os.walk(resolved_path):
                current_depth = len(Path(root).parts) - base_depth
                if current_depth >= max_depth:
                    dirs.clear()
                    continue

                # Filter excluded dirs in-place to prevent traversing them
                excluded_in_root = [d for d in dirs if is_excluded_directory(d) or (d.startswith(".") and not include_hidden)]
                for d in excluded_in_root:
                    disclosed_exclusions.add(d)
                    dirs.remove(d)

                for name in dirs + files:
                    if count >= limit:
                        break
                    full_p = os.path.join(root, name)
                    try:
                        stat = os.stat(full_p)
                        is_dir = os.path.isdir(full_p)
                        entries.append({
                            "name": name,
                            "path": clean_windows_path(full_p),
                            "is_dir": is_dir,
                            "size": stat.st_size if not is_dir else None,
                            "mtime": stat.st_mtime,
                            "depth": current_depth + 1,
                        })
                        count += 1
                    except OSError:
                        continue

        # Sort entries: directories first, then alphabetical
        entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        return ToolResult(
            success=True,
            tool_name="list_directory",
            path=resolved_path,
            output=entries,
            metadata={
                "total_count": len(entries),
                "is_truncated": count >= limit,
                "disclosed_exclusions": sorted(list(disclosed_exclusions)),
            },
        )
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="list_directory", path=path, error=pde.message, error_code=pde.code)
    except Exception as exc:
        return ToolResult(success=False, tool_name="list_directory", path=path, error=str(exc), error_code="LIST_DIR_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 2. File Metadata Inspection
# ─────────────────────────────────────────────────────────────────────────────

def get_file_metadata(path: str) -> ToolResult:
    """Inspects detailed filesystem metadata for an authorized file."""
    try:
        decision = enforce_operation_permission(AgentAction.READ, target_path=path)
        resolved_path = decision.resolved_path

        if not os.path.exists(resolved_path):
            return ToolResult(
                success=False,
                tool_name="get_file_metadata",
                path=path,
                error=f"File does not exist: {path}",
                error_code="FILE_NOT_FOUND",
            )

        stat = os.stat(resolved_path)
        is_dir = os.path.isdir(resolved_path)
        is_binary = False
        line_count = None
        sha256 = None

        if not is_dir:
            is_binary = is_binary_file(resolved_path)
            if not is_binary and stat.st_size <= 2 * 1024 * 1024:
                try:
                    text, _, h = detect_encoding_and_read(resolved_path)
                    line_count = len(text.splitlines())
                    sha256 = h
                except Exception:
                    pass
            elif not is_binary:
                # Compute hash only
                with open(resolved_path, "rb") as f:
                    sha256 = hashlib.sha256(f.read()).hexdigest()

        meta = {
            "name": os.path.basename(resolved_path),
            "path": clean_windows_path(resolved_path),
            "is_dir": is_dir,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "ctime": stat.st_ctime,
            "extension": os.path.splitext(resolved_path)[1].lower(),
            "is_binary": is_binary,
            "line_count": line_count,
            "sha256": sha256,
        }

        return ToolResult(success=True, tool_name="get_file_metadata", path=resolved_path, output=meta)
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="get_file_metadata", path=path, error=pde.message, error_code=pde.code)
    except Exception as exc:
        return ToolResult(success=False, tool_name="get_file_metadata", path=path, error=str(exc), error_code="METADATA_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Read File
# ─────────────────────────────────────────────────────────────────────────────

def read_file(
    path: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    max_bytes: int = 1048576,
) -> ToolResult:
    """Reads a text file with binary detection, line range slicing, and encoding detection."""
    try:
        decision = enforce_operation_permission(AgentAction.READ, target_path=path)
        resolved_path = decision.resolved_path

        if not os.path.exists(resolved_path):
            return ToolResult(
                success=False,
                tool_name="read_file",
                path=path,
                error=f"File not found: {path}",
                error_code="FILE_NOT_FOUND",
            )

        if os.path.isdir(resolved_path):
            return ToolResult(
                success=False,
                tool_name="read_file",
                path=path,
                error=f"Path is a directory, not a file: {path}",
                error_code="IS_A_DIRECTORY",
            )

        if is_binary_file(resolved_path):
            file_size = os.path.getsize(resolved_path)
            return ToolResult(
                success=True,
                tool_name="read_file",
                path=resolved_path,
                output=f"[Binary file detected: {os.path.basename(resolved_path)} ({file_size} bytes). Raw string display suppressed.]",
                metadata={"is_binary": True, "size_bytes": file_size},
            )

        full_text, encoding, sha256 = detect_encoding_and_read(resolved_path, max_bytes=max_bytes)
        lines = full_text.splitlines(keepends=True)
        total_lines = len(lines)

        start = max(1, line_start) if line_start is not None else 1
        end = min(total_lines, line_end) if line_end is not None else total_lines

        if start > total_lines:
            sliced_text = ""
        else:
            sliced_lines = lines[start - 1 : end]
            sliced_text = "".join(sliced_lines)

        return ToolResult(
            success=True,
            tool_name="read_file",
            path=resolved_path,
            output=sliced_text,
            metadata={
                "encoding": encoding,
                "sha256": sha256,
                "total_lines": total_lines,
                "line_start": start,
                "line_end": end,
                "is_binary": False,
            },
        )
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="read_file", path=path, error=pde.message, error_code=pde.code)
    except Exception as exc:
        return ToolResult(success=False, tool_name="read_file", path=path, error=str(exc), error_code="READ_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Search Files & Contents
# ─────────────────────────────────────────────────────────────────────────────

def search_files(workspace_path: str, pattern: str, max_results: int = 100) -> ToolResult:
    """Searches for files matching a filename wildcard or substring within an authorized workspace."""
    try:
        decision = enforce_operation_permission(AgentAction.READ, target_path=workspace_path)
        resolved_root = decision.resolved_path

        matches = []
        pattern_norm = pattern.strip().lower()
        has_wildcard = any(c in pattern_norm for c in ("*", "?", "["))

        for root, dirs, files in os.walk(resolved_root):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if not is_excluded_directory(d) and not d.startswith(".")]

            for fname in files:
                fname_lower = fname.lower()
                is_match = fnmatch.fnmatch(fname_lower, pattern_norm) if has_wildcard else (pattern_norm in fname_lower)
                if is_match:
                    full_p = os.path.join(root, fname)
                    matches.append({
                        "name": fname,
                        "path": clean_windows_path(full_p),
                        "relative_path": clean_windows_path(os.path.relpath(full_p, resolved_root)),
                    })
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        return ToolResult(
            success=True,
            tool_name="search_files",
            path=resolved_root,
            output=matches,
            metadata={"match_count": len(matches), "pattern": pattern},
        )
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="search_files", path=workspace_path, error=pde.message, error_code=pde.code)
    except Exception as exc:
        return ToolResult(success=False, tool_name="search_files", path=workspace_path, error=str(exc), error_code="SEARCH_ERROR")


def search_file_contents(
    workspace_path: str,
    query: str,
    max_results: int = 100,
    max_file_size: int = 524288,
) -> ToolResult:
    """Searches text contents across workspace files with line numbers and matched snippets."""
    try:
        decision = enforce_operation_permission(AgentAction.READ, target_path=workspace_path)
        resolved_root = decision.resolved_path

        matches = []
        query_norm = query.strip()
        query_lower = query_norm.lower()

        for root, dirs, files in os.walk(resolved_root):
            dirs[:] = [d for d in dirs if not is_excluded_directory(d) and not d.startswith(".")]

            for fname in files:
                full_p = os.path.join(root, fname)
                try:
                    if os.path.getsize(full_p) > max_file_size or is_binary_file(full_p):
                        continue
                    text, _, _ = detect_encoding_and_read(full_p, max_bytes=max_file_size)
                    for idx, line in enumerate(text.splitlines(), start=1):
                        if query_lower in line.lower():
                            matches.append({
                                "file": clean_windows_path(full_p),
                                "relative_path": clean_windows_path(os.path.relpath(full_p, resolved_root)),
                                "line_number": idx,
                                "matched_snippet": line.strip()[:200],
                            })
                            if len(matches) >= max_results:
                                break
                except Exception:
                    continue

                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        return ToolResult(
            success=True,
            tool_name="search_file_contents",
            path=resolved_root,
            output=matches,
            metadata={"match_count": len(matches), "query": query},
        )
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="search_file_contents", path=workspace_path, error=pde.message, error_code=pde.code)
    except Exception as exc:
        return ToolResult(success=False, tool_name="search_file_contents", path=workspace_path, error=str(exc), error_code="SEARCH_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Directory & File Creation
# ─────────────────────────────────────────────────────────────────────────────

def create_directory(path: str, approval_token: Optional[str] = None) -> ToolResult:
    """Creates a directory at an exact path within an authorized workspace."""
    try:
        decision = enforce_operation_permission(
            AgentAction.CREATE, target_path=path, approval_token=approval_token
        )
        resolved_path = decision.resolved_path

        # TOCTOU pre-operation revalidation
        revalidate_path_containment(resolved_path, workspace_registry.get_authorized_workspaces())

        os.makedirs(resolved_path, exist_ok=True)

        agent_audit.log(
            task_id="tool_exec",
            operation="create_directory",
            path=resolved_path,
            status="SUCCESS",
        )

        return ToolResult(
            success=True,
            tool_name="create_directory",
            path=resolved_path,
            output=f"Directory created: {resolved_path}",
        )
    except PermissionDeniedError as pde:
        return ToolResult(
            success=False,
            tool_name="create_directory",
            path=path,
            error=pde.message,
            error_code=pde.code,
            metadata={"requires_approval": pde.requires_approval},
        )
    except Exception as exc:
        return ToolResult(success=False, tool_name="create_directory", path=path, error=str(exc), error_code="MKDIR_ERROR")


def create_file(
    path: str,
    content: str,
    overwrite: bool = False,
    approval_token: Optional[str] = None,
) -> ToolResult:
    """Safely creates a new file at an exact path with atomic replacement and read-back verification."""
    try:
        decision = enforce_operation_permission(
            AgentAction.CREATE if not os.path.exists(path) else AgentAction.WRITE,
            target_path=path,
            approval_token=approval_token,
        )
        resolved_path = decision.resolved_path

        # TOCTOU revalidation
        revalidate_path_containment(resolved_path, workspace_registry.get_authorized_workspaces())

        parent_dir = os.path.dirname(resolved_path)
        os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(resolved_path) and not overwrite:
            return ToolResult(
                success=False,
                tool_name="create_file",
                path=resolved_path,
                error=f"File already exists: '{resolved_path}'. Specify overwrite=True to replace.",
                error_code="FILE_ALREADY_EXISTS",
            )

        # Atomic write: write to temporary file in the EXACT same parent directory
        tmp_name = f".tmp_{uuid.uuid4().hex}_{os.path.basename(resolved_path)}"
        tmp_path = os.path.join(parent_dir, tmp_name)

        content_bytes = content.encode("utf-8")
        with open(tmp_path, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replacement
        os.replace(tmp_path, resolved_path)

        # Read-back verification
        verified_text, _, saved_hash = detect_encoding_and_read(resolved_path)
        expected_hash = compute_sha256(content_bytes)

        if saved_hash != expected_hash:
            return ToolResult(
                success=False,
                tool_name="create_file",
                path=resolved_path,
                error="Read-back checksum verification failed after write.",
                error_code="VERIFICATION_FAILED",
            )

        agent_audit.log(
            task_id="tool_exec",
            operation="create_file",
            path=resolved_path,
            status="SUCCESS",
            details={"sha256": saved_hash, "bytes": len(content_bytes)},
        )

        return ToolResult(
            success=True,
            tool_name="create_file",
            path=resolved_path,
            output=f"File created successfully: {resolved_path}",
            metadata={"sha256": saved_hash, "bytes": len(content_bytes)},
        )
    except PermissionDeniedError as pde:
        return ToolResult(
            success=False,
            tool_name="create_file",
            path=path,
            error=pde.message,
            error_code=pde.code,
            metadata={"requires_approval": pde.requires_approval},
        )
    except Exception as exc:
        return ToolResult(success=False, tool_name="create_file", path=path, error=str(exc), error_code="WRITE_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Apply File Patch / Edit
# ─────────────────────────────────────────────────────────────────────────────

def get_diff(path: str, new_content: str) -> ToolResult:
    """Generates unified diff between existing file and proposed content without writing to disk."""
    try:
        decision = enforce_operation_permission(AgentAction.READ, target_path=path)
        resolved_path = decision.resolved_path

        orig_lines = []
        if os.path.exists(resolved_path):
            orig_text, _, _ = detect_encoding_and_read(resolved_path)
            orig_lines = orig_text.splitlines(keepends=True)

        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(resolved_path)}",
            tofile=f"b/{os.path.basename(resolved_path)}",
            lineterm="",
        ))

        return ToolResult(
            success=True,
            tool_name="get_diff",
            path=resolved_path,
            output="\n".join(diff),
            metadata={"has_changes": len(diff) > 0},
        )
    except Exception as exc:
        return ToolResult(success=False, tool_name="get_diff", path=path, error=str(exc), error_code="DIFF_ERROR")


def apply_file_patch(
    path: str,
    new_content: str,
    expected_hash: Optional[str] = None,
    approval_token: Optional[str] = None,
) -> ToolResult:
    """Applies an update to an existing file with concurrent modification check, atomic replacement, and read-back verification."""
    try:
        decision = enforce_operation_permission(
            AgentAction.PATCH, target_path=path, approval_token=approval_token
        )
        resolved_path = decision.resolved_path

        # TOCTOU revalidation
        revalidate_path_containment(resolved_path, workspace_registry.get_authorized_workspaces())

        if not os.path.exists(resolved_path):
            return ToolResult(
                success=False,
                tool_name="apply_file_patch",
                path=path,
                error=f"Target file does not exist to patch: {path}",
                error_code="FILE_NOT_FOUND",
            )

        orig_text, orig_encoding, current_hash = detect_encoding_and_read(resolved_path)

        # Concurrent modification guard
        if expected_hash is not None:
            if not isinstance(expected_hash, str):
                return ToolResult(
                    success=False,
                    tool_name="apply_file_patch",
                    path=resolved_path,
                    error=f"Invalid expected_hash: expected string, got {type(expected_hash).__name__}.",
                    error_code="INVALID_HASH_FORMAT",
                    metadata={"expected_hash": str(expected_hash)},
                )
            clean_expected = expected_hash.strip().lower()
            import re
            if not re.fullmatch(r"^[0-9a-f]{64}$", clean_expected):
                return ToolResult(
                    success=False,
                    tool_name="apply_file_patch",
                    path=resolved_path,
                    error=(
                        f"Malformed expected_hash '{expected_hash}'. "
                        "Expected a 64-character SHA-256 hexadecimal digest string."
                    ),
                    error_code="INVALID_HASH_FORMAT",
                    metadata={"expected_hash": expected_hash},
                )
            if current_hash.lower() != clean_expected:
                return ToolResult(
                    success=False,
                    tool_name="apply_file_patch",
                    path=resolved_path,
                    error=(
                        f"Concurrent modification detected on '{resolved_path}'. "
                        f"Expected hash '{clean_expected[:8]}...' but found '{current_hash[:8]}...'. "
                        "File was changed externally. Reconcile before saving."
                    ),
                    error_code="CONCURRENT_MODIFICATION",
                    metadata={"current_hash": current_hash, "expected_hash": clean_expected},
                )

        # Detect original line ending convention
        crlf = "\r\n" in orig_text
        target_linesep = "\r\n" if crlf else "\n"
        normalized_content = "\n".join(new_content.splitlines())
        if crlf:
            normalized_content = normalized_content.replace("\n", "\r\n")
        if (new_content.endswith("\n") or new_content.endswith("\r\n")) and not normalized_content.endswith(target_linesep):
            normalized_content += target_linesep

        # Generate unified diff
        diff_res = get_diff(resolved_path, normalized_content)
        diff_text = diff_res.output if diff_res.success else ""

        # Atomic write
        parent_dir = os.path.dirname(resolved_path)
        tmp_name = f".tmp_{uuid.uuid4().hex}_{os.path.basename(resolved_path)}"
        tmp_path = os.path.join(parent_dir, tmp_name)

        content_bytes = normalized_content.encode(orig_encoding if "replace" not in orig_encoding else "utf-8")
        with open(tmp_path, "wb") as f:
            f.write(content_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, resolved_path)

        # Read-back verification
        verified_text, _, new_hash = detect_encoding_and_read(resolved_path)
        expected_saved_hash = compute_sha256(content_bytes)

        if new_hash != expected_saved_hash:
            return ToolResult(
                success=False,
                tool_name="apply_file_patch",
                path=resolved_path,
                error="Read-back hash verification failed after patching.",
                error_code="VERIFICATION_FAILED",
            )

        agent_audit.log(
            task_id="tool_exec",
            operation="apply_file_patch",
            path=resolved_path,
            status="SUCCESS",
            details={"old_hash": current_hash, "new_hash": new_hash},
        )

        return ToolResult(
            success=True,
            tool_name="apply_file_patch",
            path=resolved_path,
            output=diff_text,
            metadata={"new_hash": new_hash, "old_hash": current_hash},
        )
    except PermissionDeniedError as pde:
        return ToolResult(
            success=False,
            tool_name="apply_file_patch",
            path=path,
            error=pde.message,
            error_code=pde.code,
            metadata={"requires_approval": pde.requires_approval},
        )
    except Exception as exc:
        return ToolResult(success=False, tool_name="apply_file_patch", path=path, error=str(exc), error_code="PATCH_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Move & Delete File
# ─────────────────────────────────────────────────────────────────────────────

def move_file(
    source_path: str,
    target_path: str,
    overwrite: bool = False,
    approval_token: Optional[str] = None,
) -> ToolResult:
    """Moves or renames a file/directory within authorized workspace roots."""
    try:
        # Enforce read on source and write on target
        decision_src = enforce_operation_permission(AgentAction.READ, target_path=source_path)
        decision_dst = enforce_operation_permission(AgentAction.WRITE, target_path=target_path, approval_token=approval_token)

        src_resolved = decision_src.resolved_path
        dst_resolved = decision_dst.resolved_path

        if not os.path.exists(src_resolved):
            return ToolResult(success=False, tool_name="move_file", path=source_path, error=f"Source does not exist: {source_path}", error_code="FILE_NOT_FOUND")

        if os.path.exists(dst_resolved) and not overwrite:
            return ToolResult(success=False, tool_name="move_file", path=target_path, error=f"Destination already exists: {target_path}", error_code="DESTINATION_EXISTS")

        os.makedirs(os.path.dirname(dst_resolved), exist_ok=True)
        shutil.move(src_resolved, dst_resolved)

        agent_audit.log(task_id="tool_exec", operation="move_file", path=dst_resolved, status="SUCCESS", details={"from": src_resolved})

        return ToolResult(
            success=True,
            tool_name="move_file",
            path=dst_resolved,
            output=f"Moved '{src_resolved}' to '{dst_resolved}'",
        )
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="move_file", path=source_path, error=pde.message, error_code=pde.code, metadata={"requires_approval": pde.requires_approval})
    except Exception as exc:
        return ToolResult(success=False, tool_name="move_file", path=source_path, error=str(exc), error_code="MOVE_ERROR")


def delete_file(
    path: str,
    explicit_confirmed: bool = False,
    approval_token: Optional[str] = None,
) -> ToolResult:
    """Deletes a file or directory strictly upon explicit confirmation and approval."""
    try:
        decision = enforce_operation_permission(
            AgentAction.DELETE,
            target_path=path,
            approval_token=approval_token,
            explicit_confirmed=explicit_confirmed,
        )
        resolved_path = decision.resolved_path

        # TOCTOU revalidation
        revalidate_path_containment(resolved_path, workspace_registry.get_authorized_workspaces())

        if not os.path.exists(resolved_path):
            return ToolResult(success=False, tool_name="delete_file", path=path, error=f"Target does not exist: {path}", error_code="PATH_NOT_FOUND")

        if os.path.isdir(resolved_path):
            shutil.rmtree(resolved_path)
        else:
            os.remove(resolved_path)

        agent_audit.log(task_id="tool_exec", operation="delete_file", path=resolved_path, status="SUCCESS")

        return ToolResult(success=True, tool_name="delete_file", path=resolved_path, output=f"Deleted: {resolved_path}")
    except PermissionDeniedError as pde:
        return ToolResult(success=False, tool_name="delete_file", path=path, error=pde.message, error_code=pde.code, metadata={"requires_approval": pde.requires_approval})
    except Exception as exc:
        return ToolResult(success=False, tool_name="delete_file", path=path, error=str(exc), error_code="DELETE_ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Controlled Command Execution
# ─────────────────────────────────────────────────────────────────────────────

# Global registry of running process PIDs for cancellation
_ACTIVE_PROCESSES: Dict[str, subprocess.Popen] = {}
_CANCELLED_TASKS: Set[str] = set()

# Disallowed shell interpreters and wrappers
_SHELL_INTERPRETERS: Set[str] = {
    "cmd.exe", "cmd", "powershell.exe", "powershell", "pwsh.exe", "pwsh",
    "bash.exe", "bash", "sh.exe", "sh", "wscript.exe", "wscript", "cscript.exe", "cscript",
}

# Diagnostic builtins implemented natively in Python without spawning any shell
_DIAGNOSTIC_BUILTINS: Set[str] = {
    "echo", "dir", "type", "ver", "vol", "cls",
}

# Destructive shell builtins strictly disallowed in execute_command.
# Directs callers to dedicated, audited, and permission-governed workspace tools.
_DISALLOWED_SHELL_BUILTINS: Set[str] = {
    "del", "erase", "rd", "rmdir", "copy", "move", "ren", "rename", "md", "mkdir",
}

# Dangerous shell metacharacters prohibited in shell builtin or command invocations.
# Blocks command chaining (&, &&, |, ||), redirection (>, >>, <), escaping (^),
# variable expansion (%, !), grouping ((), []), delimiters (;, ,, =), quoting breakouts (", ', `),
# and newline/null command separation (\n, \r, \0).
_DANGEROUS_SHELL_OPERATORS: Set[str] = {
    "&", "|", ">", "<", "^", "%", "!", "(", ")", ";", ",", "=", '"', "'", "`", "\n", "\r", "\0",
}


def _handle_native_echo(args: List[str]) -> Tuple[int, str, str]:
    """Natively handles echo without invoking cmd.exe."""
    text = " ".join(args) + "\n" if args else "\n"
    return 0, text, ""


def _handle_native_dir(args: List[str], cwd: str, authorized_workspaces: List[str]) -> Tuple[int, str, str]:
    """Natively handles directory listing without invoking cmd.exe."""
    target_dir = cwd
    bare_format = False
    for arg in args:
        if arg.lower() in ("/b", "-b"):
            bare_format = True
        elif not arg.startswith(("/", "-")):
            candidate = arg
            if not os.path.isabs(candidate):
                candidate = os.path.join(cwd, candidate)
            target_dir = revalidate_path_containment(candidate, authorized_workspaces)

    if not os.path.exists(target_dir):
        return 1, "", "File Not Found\n"

    if os.path.isfile(target_dir):
        stat = os.stat(target_dir)
        if bare_format:
            return 0, f"{os.path.basename(target_dir)}\n", ""
        mtime_str = time.strftime("%m/%d/%Y  %I:%M %p", time.localtime(stat.st_mtime))
        out = (
            f" Volume in drive C has no label.\n"
            f" Directory of {os.path.dirname(target_dir)}\n\n"
            f"{mtime_str}    {stat.st_size:>14,d} {os.path.basename(target_dir)}\n"
            f"               1 File(s) {stat.st_size:>14,d} bytes\n"
        )
        return 0, out, ""

    entries = []
    try:
        with os.scandir(target_dir) as it:
            for entry in it:
                try:
                    stat = entry.stat(follow_symlinks=False)
                    entries.append((entry.name, entry.is_dir(follow_symlinks=False), stat.st_size, stat.st_mtime))
                except OSError:
                    continue
    except Exception as exc:
        return 1, "", f"{exc}\n"

    entries.sort(key=lambda x: (not x[1], x[0].lower()))

    if bare_format:
        out = "\n".join(e[0] for e in entries) + ("\n" if entries else "")
        return 0, out, ""

    lines = [
        " Volume in drive C has no label.",
        f" Directory of {target_dir}\n",
    ]
    file_count = 0
    file_bytes = 0
    dir_count = 0
    for name, is_d, size, mtime in entries:
        mtime_str = time.strftime("%m/%d/%Y  %I:%M %p", time.localtime(mtime))
        if is_d:
            lines.append(f"{mtime_str}    <DIR>          {name}")
            dir_count += 1
        else:
            lines.append(f"{mtime_str}    {size:>14,d} {name}")
            file_count += 1
            file_bytes += size

    lines.append(f"              {file_count} File(s) {file_bytes:>14,d} bytes")
    lines.append(f"              {dir_count} Dir(s)")
    return 0, "\n".join(lines) + "\n", ""


def _handle_native_type(args: List[str], cwd: str, authorized_workspaces: List[str]) -> Tuple[int, str, str]:
    """Natively handles file display without invoking cmd.exe."""
    if not args:
        return 1, "", "The syntax of the command is incorrect.\n"
    target_arg = args[0]
    target = target_arg if os.path.isabs(target_arg) else os.path.join(cwd, target_arg)
    canon_target = revalidate_path_containment(target, authorized_workspaces)
    if not os.path.exists(canon_target):
        return 1, "", f"The system cannot find the file specified: {target_arg}\n"
    if os.path.isdir(canon_target):
        return 1, "", f"Access is denied: {target_arg}\n"
    with open(canon_target, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return 0, content, ""


def _handle_native_ver() -> Tuple[int, str, str]:
    """Natively returns OS version without invoking cmd.exe."""
    if sys.platform == "win32":
        v = sys.getwindowsversion()
        return 0, f"Microsoft Windows [Version {v.major}.{v.minor}.{v.build}]\n", ""
    return 0, f"{sys.platform} {sys.version.split()[0]}\n", ""


def _handle_native_vol(cwd: str) -> Tuple[int, str, str]:
    """Natively returns volume information without invoking cmd.exe."""
    drive = os.path.splitdrive(cwd)[0] or "C:"
    return 0, f" Volume in drive {drive[0].upper()} has no label.\n Volume Serial Number is 0000-0000\n", ""


def _handle_native_cls() -> Tuple[int, str, str]:
    """Natively handles cls without invoking cmd.exe."""
    return 0, "", ""


def resolve_trusted_executable(
    cmd_name: str,
    cwd: str,
    authorized_workspaces: List[str],
) -> Tuple[Optional[str], str, Optional[str]]:
    """Resolves an executable strictly through trusted locations, preventing CWD binary shadowing.
    
    Returns (resolved_path, status, error_message).
    """
    clean_cmd = cmd_name.strip('"\'')
    if not clean_cmd:
        return None, "EMPTY_COMMAND", "Executable name cannot be empty."

    base_name = os.path.basename(clean_cmd).lower()

    # 1. Block shell interpreters explicitly
    if base_name in _SHELL_INTERPRETERS or clean_cmd.lower() in _SHELL_INTERPRETERS:
        return None, "UNTRUSTED_SHELL_OPERATOR", f"Direct invocation of shell interpreter '{base_name}' is disallowed."

    # 2. Block batch scripts explicitly from workspace/untrusted execution
    if base_name.endswith((".bat", ".cmd")):
        return None, "UNTRUSTED_BATCH_FILE", f"Executing batch script '{clean_cmd}' is disallowed due to Windows shell interpretation risks."

    # 3. Explicit path specified (contains slashes)
    if "/" in clean_cmd or "\\" in clean_cmd:
        target_path = clean_cmd if os.path.isabs(clean_cmd) else os.path.join(cwd, clean_cmd)
        target_norm = os.path.normpath(target_path)
        if not os.path.exists(target_norm):
            return None, "EXECUTABLE_NOT_FOUND", f"Executable not found: {clean_cmd}"
        if target_norm.lower().endswith((".bat", ".cmd")):
            return None, "UNTRUSTED_BATCH_FILE", f"Executing workspace batch script '{clean_cmd}' is disallowed."
        return target_norm, "OK", None

    # 4. Bare command lookup: Prevent CWD binary shadowing!
    # Build trusted PATH containing venv and system PATH, strictly excluding cwd and workspaces
    trusted_dirs: List[str] = []
    venv_scripts = os.path.join(sys.prefix, "Scripts")
    if os.path.isdir(venv_scripts):
        trusted_dirs.append(venv_scripts)
    if os.path.isdir(sys.prefix):
        trusted_dirs.append(sys.prefix)

    system_path = os.environ.get("PATH", "").split(os.pathsep)
    norm_cwd = os.path.normcase(os.path.realpath(cwd))
    for p in system_path:
        if not p or not os.path.isdir(p):
            continue
        try:
            real_p = os.path.normcase(os.path.realpath(p))
            # Exclude current working directory from PATH search
            if real_p == norm_cwd or real_p.startswith(norm_cwd + os.sep):
                continue
            # Exclude any authorized workspace directory from PATH search
            if any(real_p == os.path.normcase(os.path.realpath(ws)) or real_p.startswith(os.path.normcase(os.path.realpath(ws)) + os.sep) for ws in authorized_workspaces):
                continue
            trusted_dirs.append(p)
        except Exception:
            continue

    trusted_path_str = os.pathsep.join(trusted_dirs)
    resolved = shutil.which(clean_cmd, path=trusted_path_str)

    if not resolved:
        # Fallback for Python invocation if python is requested
        if clean_cmd.lower() in ("python", "python3", "python.exe"):
            resolved = sys.executable
        else:
            return None, "EXECUTABLE_NOT_FOUND", f"Executable '{clean_cmd}' not found on trusted system PATH."

    if resolved.lower().endswith((".bat", ".cmd")):
        # System batch file from trusted system directory (e.g. npm.cmd)
        return resolved, "TRUSTED_BATCH", None

    return resolved, "OK", None


def execute_command(
    command: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout_seconds: int = 30,
    max_output_chars: int = 50000,
    task_id: str = "default_cmd",
    approval_token: Optional[str] = None,
) -> ToolResult:
    """Executes an approved terminal command using structured arguments, timeout, tree termination, and output limits."""
    try:
        decision = enforce_operation_permission(
            AgentAction.EXECUTE_COMMAND,
            command=command,
            approval_token=approval_token,
        )

        authorized_workspaces = workspace_registry.get_authorized_workspaces()
        target_cwd = cwd or workspace_registry.get_active_workspace()

        if not target_cwd:
            return ToolResult(
                success=False,
                tool_name="execute_command",
                error="No workspace directory set as working directory.",
                error_code="NO_CWD",
            )

        # Enforce cwd containment
        canon_cwd = revalidate_path_containment(target_cwd, authorized_workspaces)

        # Prepare invocation list
        if args and isinstance(args, list):
            cmd_list = [a for a in args]
        else:
            try:
                cmd_list = shlex.split(command, posix=False if sys.platform == "win32" else True)
            except Exception:
                cmd_list = command.split()

        if cmd_list:
            cmd_list = [
                t[1:-1] if ((t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'"))) and len(t) >= 2 else t
                for t in cmd_list
            ]

        if not cmd_list:
            return ToolResult(success=False, tool_name="execute_command", error="Empty command string.", error_code="EMPTY_COMMAND")

        # Check for shell builtins or cmd.exe wrappers
        first_token = cmd_list[0] if cmd_list else ""
        cleaned_first = first_token.split(";")[0].split(",")[0].split("=")[0].strip("() \t'\"")
        first_lower = cleaned_first.lower()
        first_base = os.path.basename(first_lower)

        # 1. Fail closed on destructive builtins (del, erase, rmdir, rd, copy, move, etc.)
        if first_lower in _DISALLOWED_SHELL_BUILTINS or any(t.lower().split(";")[0].split(",")[0].strip("() \t'\"") in _DISALLOWED_SHELL_BUILTINS for t in cmd_list):
            disallowed = first_lower if first_lower in _DISALLOWED_SHELL_BUILTINS else [t for t in cmd_list if t.lower().split(";")[0].split(",")[0].strip("() \t'\"") in _DISALLOWED_SHELL_BUILTINS][0]
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error=f"Shell builtin '{disallowed}' is disallowed in execute_command. Use dedicated workspace tools (delete_file, move_file, create_file) instead.",
                error_code="BUILTIN_DISALLOWED",
            )

        # 2. Block direct invocation of shell interpreters (cmd.exe, cmd, powershell, etc.)
        if first_base in _SHELL_INTERPRETERS or first_lower in _SHELL_INTERPRETERS:
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error=f"Direct invocation of shell interpreter '{first_base}' is disallowed.",
                error_code="UNTRUSTED_SHELL_OPERATOR",
            )

        # 3. Native execution of read-only diagnostic builtins (echo, dir, type, ver, vol, cls)
        # ZERO subprocess or cmd.exe spawned!
        if first_lower in _DIAGNOSTIC_BUILTINS or first_token.lower() in _DIAGNOSTIC_BUILTINS:
            # Check for shell metacharacters in raw command and tokens
            if any(op in command for op in _DANGEROUS_SHELL_OPERATORS) or any(any(op in t for op in _DANGEROUS_SHELL_OPERATORS) for t in cmd_list):
                return ToolResult(
                    success=False,
                    tool_name="execute_command",
                    path=canon_cwd,
                    error="Command contains untrusted shell operators or quoting metacharacters not permitted in shell builtins.",
                    error_code="UNTRUSTED_SHELL_OPERATOR",
                )

            start_time = time.time()
            try:
                if first_lower == "echo":
                    exit_code, stdout_data, stderr_data = _handle_native_echo(cmd_list[1:])
                elif first_lower == "dir":
                    exit_code, stdout_data, stderr_data = _handle_native_dir(cmd_list[1:], canon_cwd, authorized_workspaces)
                elif first_lower == "type":
                    exit_code, stdout_data, stderr_data = _handle_native_type(cmd_list[1:], canon_cwd, authorized_workspaces)
                elif first_lower == "ver":
                    exit_code, stdout_data, stderr_data = _handle_native_ver()
                elif first_lower == "vol":
                    exit_code, stdout_data, stderr_data = _handle_native_vol(canon_cwd)
                elif first_lower == "cls":
                    exit_code, stdout_data, stderr_data = _handle_native_cls()
                else:
                    exit_code, stdout_data, stderr_data = 0, "", ""
            except SecurityContainmentError as sce:
                return ToolResult(
                    success=False,
                    tool_name="execute_command",
                    path=canon_cwd,
                    error=f"Target path is outside authorized workspace boundaries: {sce.message}",
                    error_code="PATH_NOT_AUTHORIZED",
                )

            duration_ms = int((time.time() - start_time) * 1000)
            clean_stdout = redact_secrets(stdout_data or "")
            clean_stderr = redact_secrets(stderr_data or "")
            is_truncated = False
            if len(clean_stdout) > max_output_chars:
                clean_stdout = clean_stdout[:max_output_chars] + f"\n\n[OUTPUT TRUNCATED: Exceeded {max_output_chars} character limit]"
                is_truncated = True

            agent_audit.log(
                task_id=task_id,
                operation="execute_command",
                command=command,
                exit_code=exit_code,
                status="SUCCESS" if exit_code == 0 else "FAILED",
                details={"native_diagnostic": True, "duration_ms": duration_ms},
            )

            return ToolResult(
                success=(exit_code == 0),
                tool_name="execute_command",
                path=canon_cwd,
                output={
                    "command": redact_secrets(command),
                    "exit_code": exit_code,
                    "stdout": clean_stdout,
                    "stderr": clean_stderr,
                    "duration_ms": duration_ms,
                    "timed_out": False,
                    "cancelled": False,
                    "is_truncated": is_truncated,
                },
                error=clean_stderr if exit_code != 0 else None,
                error_code="COMMAND_FAILED" if exit_code != 0 else None,
                metadata={"cwd": canon_cwd, "duration_ms": duration_ms, "native": True},
            )

        # 4. Standard Executable Runner
        # Block dangerous chaining and redirection operators in raw command
        if any(op in command for op in ("&", "|", ">", "<", "^", "\n", "\r", "\0")):
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error="Command contains untrusted shell operators ('&', '|', '>', '<', '^') not permitted in terminal commands.",
                error_code="UNTRUSTED_SHELL_OPERATOR",
            )

        # Resolve trusted executable and protect against CWD binary shadowing
        resolved_exe, exe_status, exe_err = resolve_trusted_executable(cmd_list[0], canon_cwd, authorized_workspaces)
        if exe_status == "UNTRUSTED_BATCH_FILE":
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error=exe_err or "Executing untrusted batch scripts (.bat, .cmd) is disallowed.",
                error_code="UNTRUSTED_BATCH_FILE",
            )
        if exe_status == "UNTRUSTED_SHELL_OPERATOR":
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error=exe_err or "Shell interpreter invocation disallowed.",
                error_code="UNTRUSTED_SHELL_OPERATOR",
            )
        if exe_status == "EXECUTABLE_NOT_FOUND" or not resolved_exe:
            return ToolResult(
                success=False,
                tool_name="execute_command",
                path=canon_cwd,
                error=exe_err or f"Executable '{cmd_list[0]}' not found.",
                error_code="EXECUTABLE_NOT_FOUND",
            )

        if exe_status == "TRUSTED_BATCH":
            # For trusted system batch scripts (e.g. npm.cmd), ensure arguments do not contain batch escape characters
            for token in cmd_list[1:]:
                if any(op in token for op in ("%", "!", "&", "|", ">", "<", "^", '"', "\n", "\r")):
                    return ToolResult(
                        success=False,
                        tool_name="execute_command",
                        path=canon_cwd,
                        error="Arguments to batch scripts cannot contain shell expansion or redirection characters ('%', '!', '&', '|', '>', '<').",
                        error_code="UNTRUSTED_SHELL_OPERATOR",
                    )

        cmd_list[0] = resolved_exe



        start_time = time.time()
        timed_out = False
        cancelled = False

        # Set creationflags on Windows to avoid opening extra consoles
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

        proc = subprocess.Popen(
            cmd_list,
            cwd=canon_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=creationflags,
        )

        _ACTIVE_PROCESSES[task_id] = proc

        stdout_data, stderr_data = "", ""
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc.pid)
            try:
                stdout_data, stderr_data = proc.communicate(timeout=2)
            except Exception:
                pass
        finally:
            _ACTIVE_PROCESSES.pop(task_id, None)

        if task_id in _CANCELLED_TASKS:
            cancelled = True
            _CANCELLED_TASKS.discard(task_id)

        duration_ms = int((time.time() - start_time) * 1000)
        exit_code = proc.returncode if (not timed_out and not cancelled) else -1

        # Redact secrets from output and command
        clean_stdout = redact_secrets(stdout_data or "")
        clean_stderr = redact_secrets(stderr_data or "")

        # Truncate if exceeds limit
        is_truncated = False
        if len(clean_stdout) > max_output_chars:
            clean_stdout = clean_stdout[:max_output_chars] + f"\n\n[OUTPUT TRUNCATED: Exceeded {max_output_chars} character limit]"
            is_truncated = True

        if len(clean_stderr) > max_output_chars:
            clean_stderr = clean_stderr[:max_output_chars] + f"\n\n[STDERR TRUNCATED: Exceeded {max_output_chars} character limit]"
            is_truncated = True

        status_str = "SUCCESS" if (exit_code == 0 and not timed_out and not cancelled) else ("CANCELLED" if cancelled else "FAILED")
        agent_audit.log(
            task_id=task_id,
            operation="execute_command",
            command=command,
            exit_code=exit_code,
            status=status_str,
            details={"duration_ms": duration_ms, "timed_out": timed_out, "cancelled": cancelled},
        )

        success = (exit_code == 0) and not timed_out and not cancelled

        if timed_out:
            err_code = "TIMEOUT"
        elif cancelled:
            err_code = "CANCELLED"
        elif not success:
            err_code = "COMMAND_FAILED"
        else:
            err_code = None

        return ToolResult(
            success=success,
            tool_name="execute_command",
            path=canon_cwd,
            output={
                "command": redact_secrets(command),
                "exit_code": exit_code,
                "stdout": clean_stdout,
                "stderr": clean_stderr,
                "duration_ms": duration_ms,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "is_truncated": is_truncated,
            },
            error=clean_stderr if not success else None,
            error_code=err_code,
            metadata={"cwd": canon_cwd, "duration_ms": duration_ms},
        )
    except PermissionDeniedError as pde:
        return ToolResult(
            success=False,
            tool_name="execute_command",
            error=pde.message,
            error_code=pde.code,
            metadata={"requires_approval": pde.requires_approval, "details": pde.approval_details},
        )
    except Exception as exc:
        return ToolResult(success=False, tool_name="execute_command", error=str(exc), error_code="EXECUTION_ERROR")


def cancel_running_command(task_id: str) -> bool:
    """Cancels an ongoing command by terminating its process tree."""
    _CANCELLED_TASKS.add(task_id)
    proc = _ACTIVE_PROCESSES.get(task_id)
    if proc:
        _kill_process_tree(proc.pid)
        return True
    return False



def _kill_process_tree(pid: int) -> None:
    """Recursively kills a process tree on Windows or POSIX."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(pid), 9)
    except Exception:
        try:
            os.kill(pid, 9)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 9. Verification Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(task_id: str, checks: List[Dict[str, Any]]) -> ToolResult:
    """Runs targeted test assertions or command verifications and returns structured outcome."""
    results = []
    all_passed = True

    for check in checks:
        check_type = check.get("type", "file_exists")
        desc = check.get("description", check_type)

        if check_type == "file_exists":
            p = check.get("path", "")
            if not p or not str(p).strip():
                p = workspace_registry.get_active_workspace() or "."
            exists = os.path.exists(p)
            results.append({"description": desc, "passed": exists, "details": f"Path '{p}' exists: {exists}"})
            if not exists:
                all_passed = False

        elif check_type == "file_contains":
            p = check.get("path", "")
            expected = check.get("expected", "")
            passed = False
            if os.path.exists(p):
                text, _, _ = detect_encoding_and_read(p)
                passed = expected in text
            results.append({"description": desc, "passed": passed, "details": f"Expected snippet in '{p}': {passed}"})
            if not passed:
                all_passed = False

        elif check_type == "command_exit_zero":
            cmd = check.get("command", "")
            res = execute_command(command=cmd, task_id=f"verify_{task_id}")
            passed = res.success and (res.output or {}).get("exit_code") == 0
            results.append({"description": desc, "passed": passed, "details": res.output})
            if not passed:
                all_passed = False

    agent_audit.log(
        task_id=task_id,
        operation="run_verification",
        status="SUCCESS" if all_passed else "FAILED",
        verification_status="PASSED" if all_passed else "FAILED",
        details={"checks_total": len(checks), "all_passed": all_passed},
    )

    return ToolResult(
        success=all_passed,
        tool_name="run_verification",
        output={"all_passed": all_passed, "checks": results},
    )
