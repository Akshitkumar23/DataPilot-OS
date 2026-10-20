"""
DataPilot OS — Phase 7 Governed SQL API Router.
Endpoints:
- POST /api/sql/validate: Validate SQL without execution.
- POST /api/sql/execute: Execute DuckDB query against active dataset DataFrame.
- GET /api/sql/history: Retrieve local query execution history.
- GET /api/sql/schema: Retrieve active dataset schema with DuckDB data types.
- POST /api/sql/export: Export executed SQL or query results as CSV.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field
import pandas as pd
import duckdb

from backend.core.dataset_registry import dataset_registry
from backend.sql_engine.sql_models import SQLValidationResult, SQLQueryResult, SQLQueryHistoryEntry
from backend.sql_engine.sql_validator import SQLValidator
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.sql_engine.query_history import query_history_manager
from backend.sql_engine.relation_catalog import relation_catalog
from backend.sql_engine.relationship_resolver import RelationshipResolver, RelationshipEvidence
from backend.sql_engine.join_planner import JoinPlanner, JoinPlan, JoinStep, JoinExecutionResult

router = APIRouter(prefix="/api/sql", tags=["sql"])


class SQLValidateRequest(BaseModel):
    sql: str
    dataset_id: str


class SQLExecuteRequest(BaseModel):
    sql: str
    dataset_id: str
    page: int = 1
    page_size: int = 50


class SQLExportRequest(BaseModel):
    sql: str
    dataset_id: str
    format: str = "csv"  # "csv" | "sql"


def _get_dataset_and_df(dataset_id: str):
    if not dataset_registry:
        raise HTTPException(status_code=500, detail="Dataset registry unavailable")
    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found in registry")
    df = dataset_registry.get_dataframe(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"DataFrame for dataset '{dataset_id}' not found in memory")
    return entry, df


@router.post("/validate", response_model=SQLValidationResult)
def validate_sql(req: SQLValidateRequest):
    """Validate SQL query against active dataset schema without execution."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    return SQLValidator.validate(req.sql, df, entry.name)


@router.post("/execute", response_model=SQLQueryResult)
def execute_sql(req: SQLExecuteRequest):
    """Execute governed read-only SQL query against active dataset DataFrame."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    res = GovernedDuckDBManager.execute_query(
        sql=req.sql,
        df=df,
        dataset_id=req.dataset_id,
        dataset_name=entry.name,
        page=req.page,
        page_size=req.page_size,
        record_provenance=True,
        source_file_path=getattr(entry, "file_path", None)
    )
    if res.status == "FAILED" and res.error_code in ("UNKNOWN_COLUMN", "UNSUPPORTED_SQL_OPERATION", "INVALID_SQL"):
        # Return 422 for client validation/contract errors so UI error handlers recognize structured failure
        raise HTTPException(status_code=422, detail={
            "status": "FAILED",
            "error_code": res.error_code,
            "message": res.error_message,
            "sql": req.sql
        })
    return res


@router.get("/history")
def get_query_history(dataset_id: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=200)):
    """Retrieve persistent query history for the specified dataset or all datasets."""
    entries = query_history_manager.get_history(dataset_id=dataset_id, limit=limit)
    return {"history": [e.model_dump() if hasattr(e, "model_dump") else e for e in entries]}


@router.delete("/history")
def clear_query_history(dataset_id: Optional[str] = Query(None)):
    """Clear query history for the specified dataset or all datasets."""
    query_history_manager.clear_history(dataset_id=dataset_id)
    return {"status": "SUCCEEDED", "message": "Query history cleared"}


@router.get("/schema")
def get_dataset_sql_schema(dataset_id: str = Query(...)):
    """Get active dataset schema enriched with DuckDB native data types."""
    entry, df = _get_dataset_and_df(dataset_id)
    # Query DuckDB for exact types
    columns_info = []
    con = None
    try:
        con = duckdb.connect(database=":memory:")
        con.register("active_dataset", df)
        rel = con.query("SELECT * FROM active_dataset LIMIT 1")
        col_names = rel.columns
        col_types = [str(t) for t in rel.types]

        for name, dtype in zip(col_names, col_types):
            null_count = int(df[name].isna().sum()) if name in df.columns else 0
            sample_val = None
            if name in df.columns and len(df[name].dropna()) > 0:
                sample_val = str(df[name].dropna().iloc[0])
            columns_info.append({
                "name": str(name),
                "type": dtype,
                "nullable": null_count > 0,
                "null_count": null_count,
                "sample": sample_val
            })
    finally:
        if con:
            con.close()

    return {
        "dataset_id": dataset_id,
        "name": entry.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info
    }


@router.get("/tables")
def get_catalog_tables():
    """Retrieve all relations currently registered in the Relation Catalog for multi-table queries."""
    relations = relation_catalog.list_relations()
    return {
        "tables": [r.model_dump() for r in relations],
        "active_table": relation_catalog._active_table_name
    }


@router.get("/relationships")
def get_candidate_relationships():
    """Discover candidate foreign key relationships across all registered tables in catalog."""
    tables_map = relation_catalog.get_all_dataframes()
    candidates = RelationshipResolver.discover_candidate_relationships(tables_map)
    return {
        "candidates": [c.model_dump() for c in candidates]
    }


class GovernedJoinRequest(BaseModel):
    left_table: str
    right_table: str
    join_type: str = "left"
    left_keys: List[str]
    right_keys: List[str]
    allow_fan_out: bool = False
    new_dataset_name: Optional[str] = None


@router.post("/join")
def execute_governed_join(req: GovernedJoinRequest):
    """Executes a governed cross-table join and creates a new enriched dataset."""
    tables_map = relation_catalog.get_all_dataframes()
    if req.left_table not in tables_map:
        raise HTTPException(status_code=404, detail=f"Left table '{req.left_table}' not found in catalog")
    if req.right_table not in tables_map:
        raise HTTPException(status_code=404, detail=f"Right table '{req.right_table}' not found in catalog")

    plan = JoinPlan(
        left_table=req.left_table,
        steps=[
            JoinStep(
                right_table=req.right_table,
                join_type=req.join_type.lower(),
                left_keys=req.left_keys,
                right_keys=req.right_keys
            )
        ],
        allow_fan_out=req.allow_fan_out
    )

    res = JoinPlanner.execute_join_plan(plan, tables_map)
    if res.status != "SUCCEEDED":
        raise HTTPException(status_code=422, detail={
            "status": res.status,
            "error_code": res.error_code,
            "message": res.error_message,
            "warnings": res.warnings,
            "impact": res.impact.model_dump() if res.impact else None
        })

    # Register output as a new dataset in dataset_registry
    import uuid
    new_ds_id = f"ds_join_{uuid.uuid4().hex[:6]}"
    target_name = req.new_dataset_name or f"{req.left_table}_joined_{req.right_table}"
    from backend.core.dataset_registry import DatasetEntry
    res_df = res.result_df
    res_fingerprint = DatasetProfiler.compute_hash(res_df)

    entry = DatasetEntry(
        dataset_id=new_ds_id,
        name=f"{target_name}.csv",
        file_path=f"memory://{target_name}",
        format="csv",
        row_count=len(res_df),
        column_count=len(res_df.columns),
        knowledge_model={"columns": list(res_df.columns), "row_count": len(res_df), "column_count": len(res_df.columns)},
        loaded_at=datetime.now(timezone.utc).isoformat(),
        file_hash=res_fingerprint
    )
    dataset_registry.register(new_ds_id, entry)
    dataset_registry._dataframes_cache[new_ds_id] = res_df
    dataset_registry.set_active(new_ds_id)

    # Record multi-source provenance
    from backend.persistence.provenance_store import provenance_store
    left_rel = relation_catalog.get_relation(req.left_table)
    right_rel = relation_catalog.get_relation(req.right_table)
    left_ds_id = left_rel.dataset_id if left_rel else req.left_table
    right_ds_id = right_rel.dataset_id if right_rel else req.right_table

    provenance_store.record_multi_source_event(
        operation="CROSS_TABLE_JOIN",
        primary_dataset_id=left_ds_id,
        primary_dataset_name=req.left_table,
        joined_dataset_ids=[right_ds_id],
        source_fingerprints={
            left_ds_id: left_rel.fingerprint if left_rel else "",
            right_ds_id: right_rel.fingerprint if right_rel else ""
        },
        output_dataset_id=new_ds_id,
        output_fingerprint=res_fingerprint,
        join_plan=plan.model_dump(),
        metrics={
            "left_rows": res.left_rows,
            "right_rows": res.right_rows,
            "result_rows": res.result_rows,
            "expansion_ratio": res.expansion_ratio,
            "dropped_rows": res.dropped_rows,
            "duplicated_rows": res.duplicated_rows,
            "left_null_keys": res.left_null_keys,
            "right_null_keys": res.right_null_keys,
            "unmatched_left": res.unmatched_left,
            "unmatched_right": res.unmatched_right
        },
        sql=res.compiled_sql,
        status="SUCCESS",
        execution_time_ms=0.0
    )

    return {
        "status": "SUCCEEDED",
        "new_dataset_id": new_ds_id,
        "new_dataset_name": f"{target_name}.csv",
        "row_count": len(res_df),
        "column_count": len(res_df.columns),
        "compiled_sql": res.compiled_sql,
        "expansion_ratio": res.expansion_ratio,
        "warnings": res.warnings,
        "sample_rows": res_df.head(10).to_dict(orient="records")
    }


@router.post("/export")
def export_sql_result(req: SQLExportRequest):
    """Export the query result as CSV or raw SQL script."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    if req.format.lower() == "sql":
        return Response(content=req.sql, media_type="text/plain", headers={
            "Content-Disposition": f"attachment; filename={entry.name}_query.sql"
        })

    # Execute full query for CSV export without pagination clamp
    res = GovernedDuckDBManager.execute_query(
        sql=req.sql,
        df=df,
        dataset_id=req.dataset_id,
        dataset_name=entry.name,
        page=1,
        page_size=100000,
        record_provenance=False
    )
    if res.status != "SUCCEEDED":
        raise HTTPException(status_code=422, detail=res.error_message or "Query execution failed")

    res_df = pd.DataFrame(res.rows)
    csv_str = res_df.to_csv(index=False)
    return Response(content=csv_str, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename={entry.name}_query_result.csv"
    })


class SQLNLIntentRequest(BaseModel):
    dataset_id: str
    intent: Dict[str, Any]


@router.post("/nl-intent")
def compile_nl_intent(req: SQLNLIntentRequest):
    """Compile structured SQLIntent into DuckDB SQL via deterministic schema grounding."""
    entry, df = _get_dataset_and_df(req.dataset_id)
    try:
        from backend.sql_engine.sql_models import SQLIntent
        from backend.sql_engine.sql_builder import DeterministicSQLBuilder
        sql_intent = SQLIntent(**req.intent)
        compiled_sql = DeterministicSQLBuilder.build(sql_intent, df, table_name="active_dataset")
        validation = SQLValidator.validate(compiled_sql, df, entry.name)
        return {
            "status": "SUCCEEDED",
            "sql": compiled_sql,
            "validation": validation.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


class RelationalIntentAPIRequest(BaseModel):
    query: str


@router.post("/relational-intent")
def parse_relational_intent_api(req: RelationalIntentAPIRequest):
    """
    Interprets natural language relational requests via mandatory Ollama semantic layer.
    Zero local regex/token fallback. Explicit structured error states on failure.
    """
    from backend.intelligence.relational_intent_parser import RelationalIntentParser
    result = RelationalIntentParser.parse_intent_result(req.query)
    if not result.success:
        raise HTTPException(status_code=422, detail={
            "status": "FAILED",
            "error_code": result.error_code,
            "message": result.error_message,
            "details": result.details
        })
    return {
        "status": "SUCCEEDED",
        "intent": result.intent.model_dump() if result.intent else None,
        "source": result.source
    }


