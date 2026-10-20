"""
DataPilot OS — Single Authoritative Chat & Query API Router.
Handles POST /api/chat and POST /api/query requests.
Returns canonical query response contracts with appropriate HTTP status codes:
- HTTP 503: Ollama / Intent Planner Unavailable
- HTTP 422: Plan Invalid / Column Resolution Error / Ambiguous Intent / Code Validation Error
- HTTP 500: Sandbox Execution Error / Result Validation Error
- HTTP 200: Query Succeeded
"""

import logging
import os
import uuid
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.core.request_context import request_context_scope

try:
    from backend.core.dataset_registry import dataset_registry
except ImportError:
    dataset_registry = None

from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse, QueryErrorDetail

try:
    from backend.persistence.workflow_store import workflow_store
except ImportError:
    workflow_store = None

logger = logging.getLogger(__name__)

_COLUMN_KEYS = {"column", "columns", "target_column", "time_column", "feature_columns", "dimensions", "metrics", "group_by"}


def _collect_required_columns(value: Any, key: Optional[str] = None) -> List[str]:
    """Extract explicit column references from the persisted semantic plan."""
    columns: List[str] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            columns.extend(_collect_required_columns(child_value, child_key))
    elif isinstance(value, list):
        for item in value:
            columns.extend(_collect_required_columns(item, key))
    elif key in _COLUMN_KEYS and isinstance(value, str) and value.strip():
        columns.append(value.strip())
    return list(dict.fromkeys(columns))


def _persist_workflow(req: "ChatRequest", response: CanonicalQueryResponse, request_id: str) -> None:
    if not workflow_store:
        return
    try:
        workflow_id = str((req.context or {}).get("workflow_id") or f"wf_{request_id}")
        result = response.result or {}
        plan = response.plan or {}
        dataset_name = ""
        if req.dataset_id and dataset_registry:
            entry = dataset_registry.get(req.dataset_id)
            dataset_name = getattr(entry, "name", "") if entry else ""
        workflow_store.save({
            "workflow_id": workflow_id,
            "session_id": request_id,
            "user_query": req.query,
            "dataset_ids": [req.dataset_id] if req.dataset_id else [],
            "dataset_name": dataset_name,
            "intent": plan,
            "required_columns": _collect_required_columns(plan),
            "code": result.get("code") if isinstance(result, dict) else None,
            "insight_summary": (result.get("response") or result.get("answer") or "")[:500] if isinstance(result, dict) else "",
            "status": response.status,
        })
        response.workflow_id = workflow_id
    except Exception as exc:
        logger.warning("WORKFLOW_PERSIST_FAILED | request_id=%s | %s", request_id, exc)

router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    query: str = Field(..., description="User natural language or structured analytical prompt")
    dataset_id: Optional[str] = Field(None, description="Optional active dataset ID")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    request_id: Optional[str] = Field(None, description="Optional request tracking ID")

@router.post("/chat", response_model=CanonicalQueryResponse)
@router.post("/query", response_model=CanonicalQueryResponse)
def chat_copilot(req: ChatRequest):
    """
    Primary Natural Language Query Endpoint for DataPilot OS.
    Returns canonical CanonicalQueryResponse object.
    Raises HTTPException with exact status code on failure.
    """
    request_id = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    query_raw = req.query.strip() if req.query else ""
    
    if not query_raw:
        err_payload = CanonicalQueryResponse(
            request_id=request_id,
            status="FAILED",
            phase="PLANNING",
            plan=None,
            result=None,
            error=QueryErrorDetail(
                code="INVALID_QUERY",
                message="Query prompt cannot be empty.",
                phase="PLANNING",
                request_id=request_id
            )
        )
        raise HTTPException(status_code=400, detail=err_payload.model_dump())

    # Bind this request to one dataset; never substitute another dataset on failure.
    df = None
    context = dict(req.context or {})
    context_dataset_id = context.get("dataset_id")

    def dataset_error(code: str, message: str, http_status: int):
        payload = CanonicalQueryResponse(
            request_id=request_id, status="FAILED", phase="PLAN_VALIDATION",
            error=QueryErrorDetail(code=code, message=message,
                                   phase="PLAN_VALIDATION", request_id=request_id),
        )
        raise HTTPException(status_code=http_status, detail=payload.model_dump())

    if req.dataset_id and context_dataset_id and req.dataset_id != context_dataset_id:
        dataset_error("DATASET_CONTEXT_MISMATCH", "Selected dataset and query context disagree. Refresh the dataset selection.", 409)
    resolved_id = req.dataset_id or context_dataset_id
    try:
        if not resolved_id and dataset_registry:
            active_dataset = dataset_registry.get_active()
            if active_dataset is not None:
                resolved_id = getattr(active_dataset, "dataset_id", None)
        if resolved_id:
            if dataset_registry is None:
                dataset_error("DATASET_REGISTRY_UNAVAILABLE", "Dataset registry is unavailable.", 503)
            df = dataset_registry.get_dataframe(resolved_id)
            if df is None:
                dataset_error("DATASET_NOT_AVAILABLE", "The selected dataset could not be loaded. Reconnect it before retrying.", 404)
            # Operators receive a request-local frame, not the shared registry cache.
            df = df.copy(deep=True)
            context["dataset_id"] = resolved_id
            entry = dataset_registry.get(resolved_id) if hasattr(dataset_registry, "get") else None
            if entry and getattr(entry, "name", None):
                context["dataset_name"] = entry.name
                df.attrs["name"] = entry.name
    except HTTPException:
        raise
    except Exception:
        logger.exception("[request_id=%s] DATASET_LOAD_FAILED", request_id)
        dataset_error("DATASET_LOAD_FAILED", "The selected dataset could not be read. Check the dataset connection.", 503)
    req = req.model_copy(update={"dataset_id": resolved_id, "context": context})

    # 2. Process via Canonical QueryOrchestrator
    try:
        from backend.core.error_contract import map_status_to_http_code, sanitize_traceback
        from backend.ds_engine.dataset_profiler import DatasetProfiler

        dataset_hash = None
        if df is not None:
            try:
                dataset_hash = DatasetProfiler.compute_hash(df)
            except Exception:
                pass
        if dataset_hash:
            context["dataset_hash"] = dataset_hash
            req = req.model_copy(update={"context": context})

        model_version = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

        output_frames = []
        with request_context_scope(
            request_id=request_id,
            dataset_id=resolved_id,
            dataset_hash=dataset_hash,
            model_version=model_version,
        ) as req_ctx:
            response: CanonicalQueryResponse = QueryOrchestrator.process_query(
                query_raw=query_raw,
                df=df,
                conversation_history=req.conversation_history,
                context=req.context,
                request_id=request_id,
                output_frames=output_frames,
            )

            _persist_workflow(req, response, request_id)

            # 3. HTTP Status Code Mapping
            status_upper = (response.status or "").upper().strip()

            if status_upper == "SUCCEEDED" and output_frames and resolved_id and dataset_registry:
                manifest_op = (response.manifest or {}).get("operator", "")
                plan_family = (response.plan or {}).get("primary_intent", "")
                if manifest_op in ("data_cleaning", "feature_transformation") or plan_family in ("DATA_CLEANING", "FEATURE_ENGINEERING"):
                    dataset_registry.update_dataframe(resolved_id, output_frames[-1])
                    try:
                        from backend.persistence.provenance_store import provenance_store
                        plan = response.plan or {}
                        manifest = response.manifest or {}
                        cols_used = manifest.get("columns_used", []) or plan.get("required_columns", [])
                        method = (plan.get("arguments") or {}).get("method", manifest_op or plan_family)
                        dataset_name = context.get("dataset_name", "")
                        if not dataset_name and resolved_id:
                            ent = dataset_registry.get(resolved_id)
                            dataset_name = getattr(ent, "name", "") if ent else ""
                        provenance_store.record_event(
                            dataset_id=resolved_id,
                            dataset_name=dataset_name,
                            operation=str(method),
                            capability=manifest_op or plan_family,
                            affected_columns=cols_used,
                            parameters=(plan.get("arguments") or {}),
                            df_before=df,
                            df_after=output_frames[-1],
                            status="SUCCESS",
                            source={
                                "type": "NATURAL_LANGUAGE",
                                "query_id": request_id,
                                "query_raw": query_raw,
                                "plan_id": plan.get("plan_id"),
                                "intent_type": plan_family
                            },
                            finding=(response.result or {}).get("response") or (response.result or {}).get("answer") or "",
                            code=(response.result or {}).get("code") or "",
                        )
                    except Exception as prov_err:
                        logger.warning(f"Failed to record natural language provenance: {prov_err}")

            elif status_upper == "SUCCEEDED" and df is not None and resolved_id:
                manifest_op = (response.manifest or {}).get("operator", "")
                plan_family = (response.plan or {}).get("primary_intent", "")
                eda_ops = ("general_eda", "distribution_comparison", "correlation_analysis", "statistical_test")
                eda_families = ("GENERAL_EDA", "EDA", "DISTRIBUTION_COMPARISON", "CORRELATION_ANALYSIS", "STATISTICAL_TEST")
                if manifest_op in eda_ops or plan_family in eda_families:
                    try:
                        from backend.persistence.provenance_store import provenance_store
                        plan = response.plan or {}
                        manifest = response.manifest or {}
                        cols_used = manifest.get("columns_used", []) or plan.get("required_columns", [])
                        method = (plan.get("arguments") or {}).get("method", manifest_op or plan_family)
                        dataset_name = context.get("dataset_name", "")
                        if not dataset_name and resolved_id and dataset_registry:
                            ent = dataset_registry.get(resolved_id)
                            dataset_name = getattr(ent, "name", "") if ent else ""
                        provenance_store.record_event(
                            dataset_id=resolved_id,
                            dataset_name=dataset_name,
                            operation=str(method),
                            capability="EDA",
                            affected_columns=cols_used,
                            parameters=(plan.get("arguments") or {}),
                            df_before=df,
                            df_after=df,
                            status="SUCCESS",
                            source={
                                "type": "NATURAL_LANGUAGE",
                                "query_id": request_id,
                                "query_raw": query_raw,
                                "plan_id": plan.get("plan_id"),
                                "intent_type": plan_family
                            },
                            result_summary={
                                "operation": str(method),
                                "row_count": len(df),
                                "columns_analyzed": len(cols_used)
                            },
                            finding=(response.result or {}).get("response") or (response.result or {}).get("answer") or "",
                            code=(response.result or {}).get("code") or "",
                        )
                    except Exception as prov_err:
                        logger.warning(f"Failed to record natural language EDA provenance: {prov_err}")

            if status_upper in ("FAILED", "NEEDS_CLARIFICATION"):
                manifest_op = (response.manifest or {}).get("operator", "")
                plan_family = (response.plan or {}).get("primary_intent", "")
                eda_ops = ("general_eda", "distribution_comparison", "correlation_analysis", "statistical_test")
                eda_families = ("GENERAL_EDA", "EDA", "DISTRIBUTION_COMPARISON", "CORRELATION_ANALYSIS", "STATISTICAL_TEST")
                is_clean_trans = manifest_op in ("data_cleaning", "feature_transformation") or plan_family in ("DATA_CLEANING", "FEATURE_ENGINEERING")
                is_eda = manifest_op in eda_ops or plan_family in eda_families
                if df is not None and resolved_id and (is_clean_trans or is_eda):
                    try:
                        from backend.persistence.provenance_store import provenance_store
                        plan = response.plan or {}
                        err_code = getattr(response.error, "code", "") if not isinstance(response.error, dict) else (response.error or {}).get("code", "")
                        err_msg = getattr(response.error, "message", "") if not isinstance(response.error, dict) else (response.error or {}).get("message", "")
                        method = (plan.get("arguments") or {}).get("method", manifest_op or plan_family)
                        dataset_name = context.get("dataset_name", "")
                        if not dataset_name and resolved_id and dataset_registry:
                            ent = dataset_registry.get(resolved_id)
                            dataset_name = getattr(ent, "name", "") if ent else ""
                        provenance_store.record_event(
                            dataset_id=resolved_id,
                            dataset_name=dataset_name,
                            operation=str(method),
                            capability="EDA" if is_eda else (manifest_op or plan_family),
                            affected_columns=plan.get("required_columns", []),
                            parameters=(plan.get("arguments") or {}),
                            df_before=df,
                            df_after=None,
                            status="FAILED" if status_upper == "FAILED" else "NEEDS_CLARIFICATION",
                            source={
                                "type": "NATURAL_LANGUAGE",
                                "query_id": request_id,
                                "query_raw": query_raw,
                                "plan_id": plan.get("plan_id"),
                                "intent_type": plan_family
                            },
                            error={"error_code": err_code, "message": err_msg},
                        )
                    except Exception as prov_err:
                        logger.warning(f"Failed to record failed NL provenance: {prov_err}")

            if status_upper == "NEEDS_CLARIFICATION":
                logger.info(f"[request_id={request_id}] CHAT_ENDPOINT_CLARIFY | HTTP 422")
                raise HTTPException(status_code=422, detail=response.model_dump())

            if status_upper in ["UNSUPPORTED", "UNSAFE"]:
                logger.info(f"[request_id={request_id}] CHAT_ENDPOINT_{status_upper} | HTTP 400")
                raise HTTPException(status_code=400, detail=response.model_dump())

            if status_upper == "FAILED":
                err_code = ""
                if response.error:
                    err_code = getattr(response.error, "code", "") if not isinstance(response.error, dict) else response.error.get("code", "")

                status_code = map_status_to_http_code(status_upper, err_code)
                logger.warning(f"[request_id={request_id}] CHAT_ENDPOINT_FAILED | HTTP {status_code} | code={err_code}")
                raise HTTPException(status_code=status_code, detail=response.model_dump())

            return response

    except HTTPException:
        raise
    except Exception as err:
        from backend.core.error_contract import sanitize_traceback
        logger.error(f"[request_id={request_id}] CHAT_ENDPOINT_UNHANDLED_ERROR: {err}", exc_info=True)
        safe_msg = sanitize_traceback(str(err))
        err_payload = CanonicalQueryResponse(
            request_id=request_id,
            status="FAILED",
            decision="failed",
            phase="EXECUTING",
            plan=None,
            analysis_spec=None,
            execution_graph=None,
            result=None,
            error=QueryErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                error_code="INTERNAL_SERVER_ERROR",
                message=f"Analytical engine error: {safe_msg}",
                required_action="Review query parameters or check backend logs.",
                stage="EXECUTING",
                phase="EXECUTING",
                request_id=request_id
            )
        )
        raise HTTPException(status_code=500, detail=err_payload.model_dump())
