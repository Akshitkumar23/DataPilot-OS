"""
DataPilot OS — Datasets API Router.
All dataset operations: connect, upload, profile, relationships.
Never silently joins datasets. Relationship proposals require user approval.
"""
import json
import logging
import os
import shutil
from typing import List, Optional, Any, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Service imports — graceful if modules not yet fully available
try:
    from backend.dataset_engine.loader import dataset_loader
    from backend.dataset_engine.profiler import DatasetProfiler
    from backend.core.dataset_registry import dataset_registry
    DATASET_ENGINE_AVAILABLE = True
    _profiler = DatasetProfiler()
except ImportError as e:
    logger.error(f"DatasetEngine import failed: {e}")
    DATASET_ENGINE_AVAILABLE = False

try:
    from backend.dataset_engine.relationships import relationship_detector
    RELATIONSHIP_ENGINE_AVAILABLE = True
except ImportError:
    RELATIONSHIP_ENGINE_AVAILABLE = False

UPLOAD_DIR = os.path.join(
    os.getenv("DATAPILOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..")),
    "uploads"
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ConnectRequest(BaseModel):
    source_type: str
    file_path: str
    sheet_name: Optional[str] = None
    table_name: Optional[str] = None


class SelectSheetRequest(BaseModel):
    sheet_name: str


class SaveAsDatasetRequest(BaseModel):
    name: str
    columns: List[str]
    rows: List[Any]
    parent_dataset_id: Optional[str] = None
    source_request_id: Optional[str] = None
    description: Optional[str] = None


class RelationshipApproveRequest(BaseModel):
    proposal_id: str


def compute_profile_summary(df: pd.DataFrame, dataset_id: str, name: str) -> Dict[str, Any]:
    """Calculate mathematically trustworthy dataset profile metrics on the active DataFrame."""
    row_count = int(len(df))
    col_count = int(len(df.columns))
    total_cells = int(row_count * col_count)

    missing_cells = int(df.isna().sum().sum())
    missing_percentage = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        try:
            duplicate_rows = int(df.astype(str).duplicated().sum())
        except Exception:
            duplicate_rows = 0
    duplicate_percentage = round((duplicate_rows / row_count * 100), 2) if row_count > 0 else 0.0

    columns_info = []
    mixed_type_columns = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        null_percentage = round((null_count / max(row_count, 1) * 100), 2) if row_count > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        non_null_count = row_count - null_count
        unique_percentage = round((unique_count / max(non_null_count, 1) * 100), 2) if non_null_count > 0 else 0.0

        dtype_str = str(series.dtype)
        norm_type = "STR"

        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_bool_dtype(series):
                norm_type = "BOOL"
            else:
                norm_type = "NUM"
        elif pd.api.types.is_datetime64_any_dtype(series):
            norm_type = "DATE"
        elif pd.api.types.is_bool_dtype(series):
            norm_type = "BOOL"
        else:
            valid_sample = series.dropna().head(50)
            if not valid_sample.empty:
                types_in_sample = set(type(x) for x in valid_sample)
                if not types_in_sample.issubset({int, float}) and len(types_in_sample) > 1:
                    mixed_type_columns.append(str(col))

                if all(isinstance(v, bool) for v in valid_sample):
                    norm_type = "BOOL"
                elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in valid_sample):
                    norm_type = "NUM"
                else:
                    try:
                        parsed = pd.to_datetime(valid_sample, errors="coerce", format="mixed")
                        if parsed.notnull().mean() >= 0.95:
                            norm_type = "DATE"
                    except Exception:
                        pass

        columns_info.append({
            "name": str(col),
            "dtype": dtype_str,
            "normalized_type": norm_type,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
        })

    if mixed_type_columns:
        schema_status = {
            "status": "Warning",
            "message": f"Mixed types detected in: {', '.join(mixed_type_columns[:3])}",
            "is_valid": False,
            "mixed_columns": mixed_type_columns,
        }
    else:
        schema_status = {
            "status": "Valid",
            "message": "All columns type-consistent",
            "is_valid": True,
            "mixed_columns": [],
        }

    return {
        "dataset_id": dataset_id,
        "name": name,
        "row_count": row_count,
        "column_count": col_count,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "schema_status": schema_status,
        "columns": columns_info,
    }


@router.post("/connect")
def connect_dataset(request: ConnectRequest):
    """Connect a dataset by file path. Runs full profiling. No silent data modification."""
    if not DATASET_ENGINE_AVAILABLE:
        return JSONResponse(status_code=503, content={
            "error": True, "error_code": "DATASET_ENGINE_UNAVAILABLE",
            "message": "Dataset engine not loaded.",
            "required_action": "Check backend/dataset_engine/ modules for import errors."
        })

    target_path = request.file_path.strip('"').strip("'").strip()
    if not os.path.exists(target_path):
        candidates = [
            os.path.join(os.getcwd(), target_path),
            os.path.abspath(target_path)
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                found = True
                break
        if not found:
            return JSONResponse(status_code=404, content={
                "error": True, "error_code": "DATASET_INVALID",
                "message": f"File not found: {request.file_path}",
                "required_action": "Provide a valid file path to an existing dataset file."
            })

    try:
        df, load_result = dataset_loader.load(
            target_path,
            sheet_name=request.sheet_name,
            table_name=request.table_name
        )
    except Exception as load_err:
        return JSONResponse(status_code=400, content={
            "error": True, "error_code": "DATASET_INVALID",
            "message": f"Failed to load dataset: {str(load_err)}",
            "required_action": "Check the file format and path."
        })

    if not load_result.success or df is None:
        return JSONResponse(status_code=400, content={
            "error": True, "error_code": "DATASET_INVALID",
            "message": load_result.error or "Failed to load dataset.",
            "required_action": "Check the file format and try again."
        })

    import hashlib, uuid
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    name = os.path.basename(target_path or request.file_path)

    try:
        knowledge = _profiler.profile(
            df=df,
            dataset_id=dataset_id,
            name=name,
            file_path=target_path,
            format=load_result.format.value,
            file_hash=load_result.file_hash
        )
    except Exception as prof_err:
        logger.error(f"Dataset profiling failed: {prof_err}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": True, "error_code": "DATASET_PROFILE_ERROR",
            "message": f"Profiling failed: {str(prof_err)}",
            "required_action": "Inspect dataset for unusual datatypes or corruption."
        })

    from backend.core.dataset_registry import DatasetEntry
    is_native = not os.path.abspath(target_path).startswith(os.path.abspath(UPLOAD_DIR))
    storage_type = "source_native" if is_native else "managed_upload"

    entry = DatasetEntry(
        dataset_id=dataset_id,
        name=name,
        file_path=target_path,
        format=load_result.format.value,
        row_count=load_result.row_count,
        column_count=load_result.column_count,
        knowledge_model=knowledge.model_dump(),
        loaded_at=knowledge.created_at,
        file_hash=load_result.file_hash,
        initial_df_hash=DatasetProfiler.compute_hash(df),
        is_mutated=False,
        source_file_mtime=os.path.getmtime(target_path) if os.path.isfile(target_path) else None,
        source_file_size=os.path.getsize(target_path) if os.path.isfile(target_path) else None,
        sheet_name=load_result.sheet_name,
        available_sheets=load_result.available_sheets,
        is_source_native=is_native,
        source_storage=storage_type
    )
    dataset_registry.register(dataset_id, entry)
    dataset_registry._dataframes_cache[dataset_id] = df
    dataset_registry.set_active(dataset_id)

    res = knowledge.model_dump()
    prof_sum = compute_profile_summary(df, dataset_id, name)
    prof_sum["is_source_native"] = is_native
    prof_sum["source_storage"] = storage_type
    prof_sum["sheet_name"] = load_result.sheet_name
    prof_sum["available_sheets"] = load_result.available_sheets
    prof_sum["safe_display_path"] = entry.get_safe_path()

    res["profile_summary"] = prof_sum
    res["is_source_native"] = is_native
    res["source_storage"] = storage_type
    res["sheet_name"] = load_result.sheet_name
    res["available_sheets"] = load_result.available_sheets
    res["safe_display_path"] = entry.get_safe_path()
    res["source_descriptor"] = entry.get_source_descriptor()
    return res


@router.get("/excel-sheets")
def get_excel_sheets(file_path: str = Query(...)):
    """Discovers all worksheet names in an Excel workbook (.xlsx or .xls) without full dataset loading."""
    target_path = file_path.strip('"').strip("'").strip()
    if not os.path.exists(target_path):
        candidates = [
            os.path.join(os.getcwd(), target_path),
            os.path.abspath(target_path)
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    sheets = dataset_loader.get_excel_sheets(target_path)
    return {
        "file_path": target_path,
        "sheets": sheets,
        "sheet_count": len(sheets)
    }


@router.post("/{dataset_id}/select-sheet")
def select_excel_sheet(dataset_id: str, req: SelectSheetRequest):
    """Switches active worksheet for an existing registered Excel dataset.
    Loads from original source file without copying to uploads.
    """
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")

    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    ext = os.path.splitext(entry.file_path)[1].lower()
    if ext not in [".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail="Worksheet selection is only supported for Excel workbooks (.xlsx, .xls)")

    df, load_result = dataset_loader.load(entry.file_path, sheet_name=req.sheet_name)
    if not load_result.success or df is None:
        raise HTTPException(status_code=400, detail=load_result.error or f"Failed to load sheet '{req.sheet_name}'")

    knowledge = _profiler.profile(
        df=df,
        dataset_id=dataset_id,
        name=entry.name,
        file_path=entry.file_path,
        format=entry.format,
        file_hash=entry.file_hash
    )

    entry.sheet_name = load_result.sheet_name
    entry.available_sheets = load_result.available_sheets
    entry.row_count = load_result.row_count
    entry.column_count = load_result.column_count
    entry.knowledge_model = knowledge.model_dump()
    entry.initial_df_hash = DatasetProfiler.compute_hash(df)
    entry.is_mutated = False

    dataset_registry.register(dataset_id, entry)
    dataset_registry.update_dataframe(dataset_id, df)
    entry.is_mutated = False

    prof_sum = compute_profile_summary(df, dataset_id, entry.name)
    prof_sum["is_source_native"] = entry.is_source_native
    prof_sum["source_storage"] = entry.source_storage
    prof_sum["sheet_name"] = entry.sheet_name
    prof_sum["available_sheets"] = entry.available_sheets
    prof_sum["safe_display_path"] = entry.get_safe_path()

    res = knowledge.model_dump()
    res["profile_summary"] = prof_sum
    res["source_descriptor"] = entry.get_source_descriptor()
    res["is_source_native"] = entry.is_source_native
    res["source_storage"] = entry.source_storage
    res["sheet_name"] = entry.sheet_name
    res["available_sheets"] = entry.available_sheets
    res["safe_display_path"] = entry.get_safe_path()
    return res


@router.post("/{dataset_id}/refresh")
def refresh_dataset(dataset_id: str):
    """Refreshes the dataset from its original source file on disk.
    Resets drift status and recomputes profile and catalog metadata.
    """
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")

    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    if not os.path.exists(entry.file_path):
        raise HTTPException(status_code=404, detail=f"Original source file not found on disk: {entry.file_path}")

    df, load_result = dataset_loader.load(entry.file_path, sheet_name=entry.sheet_name)
    if not load_result.success or df is None:
        raise HTTPException(status_code=400, detail=load_result.error or "Failed to refresh dataset from disk")

    knowledge = _profiler.profile(
        df=df,
        dataset_id=dataset_id,
        name=entry.name,
        file_path=entry.file_path,
        format=entry.format,
        file_hash=load_result.file_hash
    )

    entry.row_count = load_result.row_count
    entry.column_count = load_result.column_count
    entry.file_hash = load_result.file_hash
    entry.source_file_mtime = os.path.getmtime(entry.file_path)
    entry.source_file_size = os.path.getsize(entry.file_path)
    entry.knowledge_model = knowledge.model_dump()
    entry.initial_df_hash = DatasetProfiler.compute_hash(df)
    entry.is_mutated = False
    entry.drift_detected = False
    entry.drift_reason = None
    if load_result.available_sheets:
        entry.available_sheets = load_result.available_sheets

    dataset_registry.register(dataset_id, entry)
    dataset_registry.update_dataframe(dataset_id, df)
    entry.is_mutated = False

    prof_sum = compute_profile_summary(df, dataset_id, entry.name)
    prof_sum["is_source_native"] = entry.is_source_native
    prof_sum["source_storage"] = entry.source_storage
    prof_sum["sheet_name"] = entry.sheet_name
    prof_sum["available_sheets"] = entry.available_sheets
    prof_sum["safe_display_path"] = entry.get_safe_path()

    res = knowledge.model_dump()
    res["profile_summary"] = prof_sum
    res["source_descriptor"] = entry.get_source_descriptor()
    res["is_source_native"] = entry.is_source_native
    res["source_storage"] = entry.source_storage
    res["sheet_name"] = entry.sheet_name
    res["available_sheets"] = entry.available_sheets
    res["safe_display_path"] = entry.get_safe_path()
    return res


@router.get("/{dataset_id}/drift-status")
def get_drift_status(dataset_id: str):
    """Inspects current source file state on disk and returns drift metadata."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")

    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    entry.check_drift()
    curr_mtime = os.path.getmtime(entry.file_path) if os.path.exists(entry.file_path) else None
    curr_size = os.path.getsize(entry.file_path) if os.path.exists(entry.file_path) else None

    return {
        "dataset_id": dataset_id,
        "is_source_native": entry.is_source_native,
        "source_storage": entry.source_storage,
        "source_file_path": entry.file_path,
        "safe_display_path": entry.get_safe_path(),
        "drift_detected": entry.drift_detected,
        "drift_reason": entry.drift_reason,
        "is_mutated": entry.is_mutated,
        "recorded_file_mtime": entry.source_file_mtime,
        "current_file_mtime": curr_mtime,
        "recorded_file_size": entry.source_file_size,
        "current_file_size": curr_size,
        "recorded_file_hash": entry.file_hash,
        "sheet_name": entry.sheet_name,
        "available_sheets": entry.available_sheets,
        "source_descriptor": entry.get_source_descriptor()
    }


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file then connect and profile it."""
    import uuid
    safe_filename = os.path.basename(file.filename or f"dataset_{uuid.uuid4().hex[:6]}.csv")
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": True, "error_code": "DATASET_INVALID",
            "message": f"File upload failed: {e}",
            "required_action": "Try uploading again."
        })

    ext = os.path.splitext(safe_filename)[1].lower().lstrip(".") or "csv"
    req = ConnectRequest(source_type=ext, file_path=dest_path)
    return connect_dataset(req)


@router.get("")
@router.get("/")
def list_datasets():
    """List all currently connected datasets."""
    if not DATASET_ENGINE_AVAILABLE:
        return []
    entries = dataset_registry.list_all()
    active_id = dataset_registry.get_active().dataset_id if dataset_registry.get_active() else None
    return [
        {
            "dataset_id": e.dataset_id,
            "name": e.name,
            "format": e.format,
            "row_count": e.row_count,
            "column_count": e.column_count,
            "loaded_at": e.loaded_at,
            "is_active": (e.dataset_id == active_id),
            "is_source_native": e.is_source_native,
            "source_storage": e.source_storage,
            "sheet_name": e.sheet_name,
            "available_sheets": e.available_sheets,
            "drift_detected": e.drift_detected,
            "safe_display_path": e.get_safe_path(),
            "source_descriptor": e.get_source_descriptor()
        }
        for e in entries
    ]


@router.get("/data")
def get_active_dataset_data(
    limit: int = Query(default=100, ge=1, le=10_000),
    dataset_id: Optional[str] = None,
):
    """Return a bounded, JSON-safe preview of the selected or active dataset.

    This is the canonical replacement for the retired ``/api/dataset/data``
    endpoint.  The endpoint never guesses a dataset: it uses an explicitly
    supplied id, or the registry's explicitly active dataset.
    """
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")

    entry = dataset_registry.get(dataset_id) if dataset_id else dataset_registry.get_active()
    if entry is None:
        raise HTTPException(status_code=404, detail="No active dataset. Connect and activate a dataset first.")

    df = dataset_registry.get_dataframe(entry.dataset_id)
    if df is None:
        raise HTTPException(status_code=500, detail=f"Dataset '{entry.dataset_id}' could not be loaded")

    # Pandas' JSON encoder gives the renderer JSON-safe nulls, timestamps and
    # numeric scalars without leaking NumPy objects across the API boundary.
    rows = json.loads(df.head(limit).to_json(orient="records", date_format="iso"))
    profile_summary = compute_profile_summary(df, entry.dataset_id, entry.name)
    profile_summary["is_source_native"] = entry.is_source_native
    profile_summary["source_storage"] = entry.source_storage
    profile_summary["sheet_name"] = entry.sheet_name
    profile_summary["available_sheets"] = entry.available_sheets
    profile_summary["safe_display_path"] = entry.get_safe_path()

    return {
        "dataset_id": entry.dataset_id,
        "rows": rows,
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "returned_rows": len(rows),
        "profile": profile_summary,
        "source_descriptor": entry.get_source_descriptor(),
        "is_source_native": entry.is_source_native,
        "source_storage": entry.source_storage,
        "sheet_name": entry.sheet_name,
        "available_sheets": entry.available_sheets
    }


@router.get("/profile")
def get_active_dataset_profile(dataset_id: Optional[str] = None):
    """Return the real-time, mathematically computed data profile for the selected or active dataset."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")

    entry = dataset_registry.get(dataset_id) if dataset_id else dataset_registry.get_active()
    if entry is None:
        raise HTTPException(status_code=404, detail="No active dataset. Connect and activate a dataset first.")

    df = dataset_registry.get_dataframe(entry.dataset_id)
    if df is None:
        raise HTTPException(status_code=500, detail=f"Dataset '{entry.dataset_id}' could not be loaded")

    return compute_profile_summary(df, entry.dataset_id, entry.name)


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get full DatasetKnowledgeModel for a specific dataset."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")
    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return entry.knowledge_model


@router.get("/{dataset_id}/profile")
async def get_dataset_profile(dataset_id: str):
    """Get the calculated data profile for a specific dataset."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")
    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    df = dataset_registry.get_dataframe(dataset_id)
    if df is not None:
        return compute_profile_summary(df, entry.dataset_id, entry.name)
    return entry.knowledge_model


@router.get("/{dataset_id}/relationships")
async def get_relationships(dataset_id: str):
    """Detect relationships between connected datasets. Returns proposals only — never auto-joins."""
    if not RELATIONSHIP_ENGINE_AVAILABLE:
        return []
    all_entries = dataset_registry.list_all()
    if len(all_entries) < 2:
        return []

    dfs = {}
    profiles = {}
    for entry in all_entries:
        df = dataset_registry.get_dataframe(entry.dataset_id)
        if df is not None:
            dfs[entry.dataset_id] = df
            profiles[entry.dataset_id] = entry.knowledge_model

    proposals = relationship_detector.detect(dfs, profiles)
    return [p.model_dump() for p in proposals]


@router.post("/{dataset_id}/relationships/{proposal_id}/approve")
async def approve_relationship(dataset_id: str, proposal_id: str):
    """User-initiated approval of a dataset relationship proposal."""
    return {"status": "approved", "proposal_id": proposal_id, "requires_user_confirmation": True}


@router.post("/{dataset_id}/activate")
async def activate_dataset(dataset_id: str):
    """Set a dataset as the active dataset for new queries."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")
    success = dataset_registry.set_active(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return {"status": "activated", "dataset_id": dataset_id}


@router.delete("/{dataset_id}")
async def disconnect_dataset(dataset_id: str):
    """Disconnect and remove a dataset from the registry."""
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")
    dataset_registry.remove(dataset_id)
    return {"status": "disconnected", "dataset_id": dataset_id}


@router.post("/save-result")
def save_result_as_dataset(req: SaveAsDatasetRequest):
    """Save an analytical result table as a first-class registered dataset.
    
    Generates new dataset ID, calculates fingerprint, and records provenance parent.
    Never mutates or overwrites the source dataset.
    """
    if not DATASET_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset engine not available")
    
    if not req.columns or not req.rows:
        raise HTTPException(status_code=400, detail="Result table must contain non-empty columns and rows")
    
    import uuid
    # Construct DataFrame safely from records dicts or lists
    if len(req.rows) > 0 and isinstance(req.rows[0], dict):
        df = pd.DataFrame(req.rows)
        # Re-index to enforce column order
        cols_present = [c for c in req.columns if c in df.columns]
        df = df[cols_present]
    else:
        df = pd.DataFrame(req.rows, columns=req.columns)

    safe_name = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in req.name).strip()
    if not safe_name.endswith(".csv"):
        safe_name = f"{safe_name}.csv"
    
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    dest_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{safe_name}")
    df.to_csv(dest_path, index=False)
    
    file_size = os.path.getsize(dest_path)
    import hashlib
    with open(dest_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        knowledge = _profiler.profile(
            df=df,
            dataset_id=dataset_id,
            name=safe_name,
            file_path=dest_path,
            format="csv",
            file_hash=file_hash
        )
    except Exception as prof_err:
        logger.error(f"Dataset profiling failed for saved result: {prof_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(prof_err)}")

    from backend.core.dataset_registry import DatasetEntry
    entry = DatasetEntry(
        dataset_id=dataset_id,
        name=safe_name,
        file_path=dest_path,
        format="csv",
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model=knowledge.model_dump(),
        loaded_at=knowledge.created_at,
        file_hash=file_hash,
        is_source_native=False,
        source_storage="managed_upload"
    )
    dataset_registry.register(dataset_id, entry)
    dataset_registry.register(dataset_id, df)

    # Record provenance parent event if governance engine is active
    try:
        from backend.persistence.provenance_store import provenance_store
        provenance_store.record_event(
            dataset_id=dataset_id,
            dataset_name=safe_name,
            operation="save_result_as_dataset",
            capability="datasets",
            affected_columns=list(df.columns),
            parameters={
                "parent_dataset_id": req.parent_dataset_id,
                "source_request_id": req.source_request_id,
                "rows": len(df),
                "columns": list(df.columns),
                "file_hash": file_hash,
            },
            df_before=None,
            df_after=df,
            status="SUCCESS",
            code="",
            finding=f"Promoted query result to persistent dataset '{safe_name}' ({len(df):,} rows).",
            result_summary={"rows": len(df), "columns": list(df.columns), "file_hash": file_hash}
        )
    except Exception as prov_err:
        logger.warning(f"Failed to record result dataset provenance: {prov_err}")

    prof_sum = compute_profile_summary(df, dataset_id, safe_name)
    prof_sum["is_source_native"] = False
    prof_sum["source_storage"] = "promoted_result"
    return {
        "status": "SUCCEEDED",
        "dataset_id": dataset_id,
        "name": safe_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "file_hash": file_hash,
        "parent_dataset_id": req.parent_dataset_id,
        "is_source_native": False,
        "source_storage": "promoted_result",
        "profile_summary": prof_sum
    }

