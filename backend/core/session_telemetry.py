"""
DataPilot OS — Unified Session Telemetry & Observability Engine.

Writes comprehensive, human- and AI-readable end-to-end chronological traces
of every user click, dataset change, query lifecycle stage, execution step,
and output into a single durable session log (session.log).

Designed for complete transparency so logs can be shared directly with any
developer or AI assistant for instant debugging and evaluation.
"""

from __future__ import annotations

import os
import sys
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _get_session_log_path() -> str:
    env_path = os.getenv("DATAPILOT_SESSION_LOG", "").strip()
    if env_path:
        return env_path

    if sys.platform == "win32":
        base = os.getenv("APPDATA") or os.path.expanduser("~")
        log_dir = os.path.join(base, "DataPilot OS", "logs")
    else:
        log_dir = os.path.expanduser("~/.datapilot/logs")

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass
    return os.path.join(log_dir, "session.log")


_SESSION_LOG_PATH = _get_session_log_path()
_LOG_LOCK = threading.Lock()


def write_session_line(line: str) -> None:
    """Thread-safe append of a single formatted block/line to session.log."""
    try:
        os.makedirs(os.path.dirname(_SESSION_LOG_PATH), exist_ok=True)
        with _LOG_LOCK:
            with open(_SESSION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
    except Exception:
        pass


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_ui_event(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log an interaction from the UI (click, tab change, file opened, etc.)."""
    detail_str = ""
    if details:
        try:
            detail_str = " | " + ", ".join(f"{k}={v}" for k, v in details.items() if v is not None)
        except Exception:
            detail_str = f" | {details}"

    line = f"[{_now()}] [UI_EVENT] {action}{detail_str}"
    write_session_line(line)


def log_query_received(
    request_id: str,
    query: str,
    dataset_name: Optional[str] = None,
    row_count: Optional[int] = None,
    col_count: Optional[int] = None,
    dataset_hash: Optional[str] = None,
) -> None:
    """Log the reception of a new analytical query with initial context."""
    ds_info = f"Dataset: {dataset_name or 'Unspecified'}"
    if row_count is not None and col_count is not None:
        ds_info += f" ({row_count:,} rows x {col_count} cols)"
    if dataset_hash:
        ds_info += f" | Hash: {dataset_hash[:8]}"

    banner = (
        f"\n+==============================================================================\n"
        f"| [USER_QUERY] [ID: {request_id}] at {_now()}\n"
        f"| Query: \"{query}\"\n"
        f"| {ds_info}\n"
        f"+------------------------------------------------------------------------------"
    )
    write_session_line(banner)


def log_grounding_completed(
    request_id: str,
    temporal_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    dataset_hash: Optional[str] = None,
) -> None:
    """Log dataset profile and column semantic role detection."""
    lines = [
        f"| [STAGE 1: GROUNDING & PROFILING]",
        f"|   * Profile hash: {dataset_hash[:8] if dataset_hash else 'N/A'}",
        f"|   * Temporal columns: {temporal_cols or 'None'}",
        f"|   * Numeric columns ({len(numeric_cols or [])}): {numeric_cols[:6] if numeric_cols else 'None'}{'...' if len(numeric_cols or []) > 6 else ''}",
        f"|   * Categorical columns ({len(categorical_cols or [])}): {categorical_cols[:6] if categorical_cols else 'None'}{'...' if len(categorical_cols or []) > 6 else ''}",
    ]
    write_session_line("\n".join(lines))


def log_planning_completed(
    request_id: str,
    tool_name: str,
    task_family: str,
    arguments: Dict[str, Any],
    inferred_bindings: Optional[List[Any]] = None,
    attempt: int = 1,
    model: str = "ollama",
) -> None:
    """Log Ollama / Neural Planner intent resolution and slot bindings."""
    lines = [
        f"| [STAGE 2: INTENT PLANNING & SLOT RESOLUTION]",
        f"|   * Planner: {model} (Attempt #{attempt})",
        f"|   * Selected Capability: '{tool_name}' (TaskFamily: {task_family})",
        f"|   * Arguments:",
    ]
    for k, v in arguments.items():
        lines.append(f"|       - {k}: {v}")

    if inferred_bindings:
        lines.append(f"|   * Policy & Profile Provenance:")
        for b in inferred_bindings:
            col = getattr(b, "column", None) or (b.get("column", "") if isinstance(b, dict) else "")
            src = getattr(b, "source", None) or (b.get("source", "") if isinstance(b, dict) else "")
            ev = getattr(b, "evidence", None) or (b.get("evidence", "") if isinstance(b, dict) else "")
            lines.append(f"|       - {col} <- source: {src} ({ev})")

    write_session_line("\n".join(lines))


def log_feasibility_gate(
    request_id: str,
    decision: str,
    status_code: int,
    checks_passed: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> None:
    """Log scientific feasibility gate validation."""
    lines = [
        f"| [STAGE 3: SCIENTIFIC FEASIBILITY & GUARDRAILS]",
        f"|   * Decision: {decision.upper()} (HTTP {status_code})",
    ]
    if checks_passed:
        for c in checks_passed:
            lines.append(f"|   * Check passed: {c}")
    if warnings:
        for w in warnings:
            lines.append(f"|   * Warning: {w}")

    write_session_line("\n".join(lines))


def log_operator_execution(
    request_id: str,
    operator: str,
    rows_in: int,
    rows_out: int,
    duration_sec: float,
    chart_type: str = "none",
    chart_generated: bool = False,
    manifest_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Log deterministic operator execution metrics."""
    lines = [
        f"| [STAGE 4: DETERMINISTIC OPERATOR EXECUTION]",
        f"|   * Operator: '{operator}' (Execution time: {duration_sec:.3f}s)",
        f"|   * Rows: {rows_in:,} in -> {rows_out:,} out",
        f"|   * Chart requested: '{chart_type}' | Materialized: {'YES' if chart_generated else 'NO'}",
    ]
    if manifest_summary:
        lines.append(f"|   * Manifest:")
        for k, v in manifest_summary.items():
            lines.append(f"|       - {k}: {v}")

    write_session_line("\n".join(lines))


def log_result_verification(
    request_id: str,
    is_valid: bool,
    issues: Optional[List[str]] = None,
) -> None:
    """Log independent post-execution validation."""
    lines = [
        f"| [STAGE 5: INDEPENDENT RESULT VERIFICATION]",
        f"|   * Oracle Parity: {'PASSED' if is_valid else 'FAILED'}",
    ]
    if issues:
        for issue in issues:
            lines.append(f"|   * Validation Issue: {issue}")
    write_session_line("\n".join(lines))


def log_query_success(
    request_id: str,
    duration_sec: float,
    summary_text: str,
    output_shape: Optional[tuple] = None,
) -> None:
    """Log the successful conclusion of query execution."""
    shape_str = f" {output_shape[0]:,} rows x {output_shape[1]} cols" if output_shape else ""
    summary_preview = summary_text.replace("\n", " ")[:140] if summary_text else "Execution completed."

    footer = (
        f"+------------------------------------------------------------------------------\n"
        f"| [RESULT] SUCCEEDED in {duration_sec:.2f}s{shape_str}\n"
        f"| Summary: {summary_preview}\n"
        f"+==============================================================================\n"
    )
    write_session_line(footer)


def log_query_clarification(
    request_id: str,
    reason: str,
    candidate_columns: Optional[List[str]] = None,
    suggested_options: Optional[List[str]] = None,
) -> None:
    """Log when a query halts for clarification."""
    cand_str = f"Candidates: {candidate_columns}" if candidate_columns else ""
    opt_str = f"Options: {suggested_options}" if suggested_options else ""

    block = (
        f"+------------------------------------------------------------------------------\n"
        f"| [RESULT] NEEDS CLARIFICATION (HTTP 422)\n"
        f"| Reason: {reason}\n"
        f"| {cand_str} {opt_str}\n"
        f"+==============================================================================\n"
    )
    write_session_line(block)


def log_query_halted(
    request_id: str,
    code: str,
    message: str,
    stage: str,
    required_action: Optional[str] = None,
) -> None:
    """Log when a query halts due to unsupported, unsafe, or error condition."""
    block = (
        f"+------------------------------------------------------------------------------\n"
        f"| [RESULT] HALTED / {code} at Stage '{stage}'\n"
        f"| Message: {message}\n"
        f"| Required Action: {required_action or 'Modify prompt or adjust dataset.'}\n"
        f"+==============================================================================\n"
    )
    write_session_line(block)
