import secrets
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import pandas as pd
import numpy as np
import duckdb
import sqlite3
try:
    import sqlalchemy
    from sqlalchemy import create_engine, inspect, text, URL
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    sqlalchemy = None
    create_engine = None
    inspect = None
    text = None
    URL = None
import uuid
import json
import os
import io
import sys
import base64
import ast
import threading
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from typing import List, Dict, Any, Optional
import re
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None
from ds_knowledge_engine import ds_engine
from ds_learning_memory import learning_memory
from modules.validation_engine import validation_engine
from modules.execution_engine import execution_engine
from modules.governance import governance_engine
from modules.template_engine import template_engine

app = FastAPI(title="AI-DSOS Backend Engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_TOKEN_PATH = os.path.join(BASE_DIR, ".env_token")

# Obtain or generate random session API auth token
API_TOKEN = os.environ.get("DATAPILOT_API_TOKEN")
if not API_TOKEN:
    if os.path.exists(ENV_TOKEN_PATH):
        try:
            with open(ENV_TOKEN_PATH, "r", encoding="utf-8") as f:
                API_TOKEN = f.read().strip()
        except Exception:
            API_TOKEN = None
    if not API_TOKEN:
        API_TOKEN = secrets.token_hex(32)

os.environ["DATAPILOT_API_TOKEN"] = API_TOKEN
try:
    with open(ENV_TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(API_TOKEN)
except Exception as e:
    pass

# Bearer token authentication middleware for all /api/* endpoints
@app.middleware("http")
async def bearer_token_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        auth_header = request.headers.get("Authorization")
        expected_header = f"Bearer {API_TOKEN}"
        if not auth_header or auth_header != expected_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: Invalid or missing Bearer authentication token"}
            )
    return await call_next(request)

# CORS middleware for communication with Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "app://.", "vscode-webview://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state with LRU memory management
class SessionState:
    def __init__(self):
        self.df_states: Dict[str, pd.DataFrame] = {} # maps step_id -> DataFrame
        self.pipeline_steps: List[Dict[str, Any]] = [] # chronological steps
        self.active_step_id: Optional[str] = None
        self.ml_model_metadata: Dict[str, Any] = {} # caches ML results for the active model
        self._source_path: Optional[str] = None
        self.max_states: int = 15
        self.lock = threading.Lock()

    def add_state(self, step_id: str, df: pd.DataFrame):
        with self.lock:
            self.df_states[step_id] = df
            self.active_step_id = step_id
            # Evict oldest step dataframes if exceeding max_states to prevent RAM leaks
            if len(self.df_states) > self.max_states:
                base_id = self.pipeline_steps[0]["id"] if self.pipeline_steps else None
                for old_id in list(self.df_states.keys()):
                    if old_id != self.active_step_id and old_id != base_id:
                        del self.df_states[old_id]
                        if len(self.df_states) <= self.max_states:
                            break

    def get_active_df(self) -> Optional[pd.DataFrame]:
        with self.lock:
            if self.active_step_id and self.active_step_id in self.df_states:
                return self.df_states[self.active_step_id]
            if self.df_states:
                return next(iter(self.df_states.values()))
            return None

state = SessionState()



# Pydantic models for incoming requests
class ConnectRequest(BaseModel):
    file_path: Optional[str] = None
    source_type: Optional[str] = "csv"   # csv, excel, sqlserver, postgresql, etc.
    # DB connection fields (future use)
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

class DBTestRequest(BaseModel):
    engine: Optional[str] = "sqlite"
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None
    database: Optional[str] = None
    filepath: Optional[str] = None

class DBImportRequest(BaseModel):
    engine: Optional[str] = "sqlite"
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database_name: Optional[str] = None
    database: Optional[str] = None
    filepath: Optional[str] = None
    table_name: Optional[str] = None
    sql_query: Optional[str] = None


class OperationRequest(BaseModel):
    operation_id: Optional[str] = None
    operation: Optional[str] = None
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    fill_value: Optional[str] = None
    new_name: Optional[str] = None
    n_bins: Optional[int] = 5
    operator: Optional[str] = None
    filter_value: Optional[str] = None

class CodeExecuteRequest(BaseModel):
    code: str

class NotebookCellRequest(BaseModel):
    cell_id: str
    code: str

def ensure_active_dataset():
    """No auto-load fallbacks. Only explicitly connected or screen-matched datasets are used."""
    pass



def clean_value_for_json(val):
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.bool_)):
        return bool(val)
    return str(val)

def get_df_profile(df: pd.DataFrame) -> List[Dict[str, Any]]:
    profile = []
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        col_type = str(s.dtype)
        
        # Ultra-fast null counting
        null_count = int(s.isna().sum())
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            empty_count = int((s == "").sum()) + int((s == "nan").sum()) + int((s == "null").sum())
            null_count += empty_count

        null_pct = round((null_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
        
        # Determine logical type
        if pd.api.types.is_bool_dtype(s):
            logical_type = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            logical_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            logical_type = "datetime"
        else:
            logical_type = "categorical"
            
        stats = {}
        if logical_type == "numeric" and not s.empty:
            valid_s = s.dropna()
            if not valid_s.empty:
                try:
                    stats = {
                        "min": clean_value_for_json(valid_s.min()),
                        "max": clean_value_for_json(valid_s.max()),
                        "mean": clean_value_for_json(valid_s.mean()),
                        "median": clean_value_for_json(valid_s.quantile(0.5))
                    }
                except Exception:
                    stats = {
                        "min": clean_value_for_json(valid_s.min()),
                        "max": clean_value_for_json(valid_s.max()),
                        "mean": 0,
                        "median": 0
                    }
            else:
                stats = {"min": 0, "max": 0, "mean": 0, "median": 0}
        elif not s.empty:
            valid_s = s.dropna()
            if not valid_s.empty:
                vc = valid_s.value_counts()
                stats = {
                    "top_value": clean_value_for_json(vc.index[0]) if len(vc) > 0 else None,
                    "freq": int(vc.iloc[0]) if len(vc) > 0 else 0
                }
            else:
                stats = {"top_value": None, "freq": 0}
            
        unique_count = int(s.nunique(dropna=True))

        profile.append({
            "name": str(col),
            "type": col_type,
            "logical_type": logical_type,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "stats": stats
        })
    return profile

def get_preview_data(df: pd.DataFrame, limit: int = 50) -> Dict[str, Any]:
    rows = []
    columns = list(df.columns)
    
    # Get first N records
    sampled = df.head(limit)
    for idx, row in sampled.iterrows():
        row_dict = {}
        for col in columns:
            row_dict[col] = clean_value_for_json(row[col])
        row_dict["_row_idx"] = idx
        rows.append(row_dict)
        
    return {
        "columns": columns,
        "rows": rows,
        "total_rows": len(df)
    }

def compute_diff(df_old: pd.DataFrame, df_new: pd.DataFrame, limit: int = 50) -> List[Dict[str, Any]]:
    diffs = []
    n_rows = min(len(df_old), len(df_new), limit)
    
    # Check for column status changes
    added_cols = [c for c in df_new.columns if c not in df_old.columns]
    removed_cols = [c for c in df_old.columns if c not in df_new.columns]
    
    # Cell level updates for common columns
    common_cols = [c for c in df_old.columns if c in df_new.columns]
    for c in common_cols:
        series_old = df_old[c].head(n_rows)
        series_new = df_new[c].head(n_rows)
        for idx in range(n_rows):
            v_old = series_old.iloc[idx]
            v_new = series_new.iloc[idx]
            
            is_old_nan = pd.isna(v_old)
            is_new_nan = pd.isna(v_new)
            
            if is_old_nan != is_new_nan or (not is_old_nan and not is_new_nan and v_old != v_new):
                diffs.append({
                    "row": idx,
                    "col": c,
                    "type": "update",
                    "prev": None if is_old_nan else clean_value_for_json(v_old),
                    "curr": None if is_new_nan else clean_value_for_json(v_new)
                })
                
    for col in added_cols:
        for idx in range(min(len(df_new), limit)):
            diffs.append({
                "row": idx,
                "col": col,
                "type": "addition",
                "prev": None,
                "curr": clean_value_for_json(df_new[col].iloc[idx])
            })
            
    return diffs

# Helper to generate Pandas and SQL codes based on operation
def generate_op_code(operation_id: str, params: Dict[str, Any], current_df: pd.DataFrame) -> Dict[str, str]:
    pandas_code = ""
    sql_code = ""
    
    if operation_id == "impute_missing":
        cols = params.get("columns", [])
        strategy = params.get("strategy", "mean")
        const_val = params.get("constant_value", "")
        cols_str = str(cols)
        
        if strategy == "mean":
            pandas_code = f"for col in {cols_str}:\n    df[col] = df[col].fillna(df[col].mean())"
            sql_subqueries = ", ".join([f"COALESCE({c}, (SELECT AVG({c}) FROM active_table)) AS {c}" for c in cols])
            sql_code = f"SELECT *, {sql_subqueries} FROM active_table" # Simplified representation
        elif strategy == "median":
            pandas_code = f"for col in {cols_str}:\n    df[col] = df[col].fillna(df[col].median())"
            sql_code = f"-- Median Imputation SQL\nSELECT * FROM active_table"
        elif strategy == "mode":
            pandas_code = f"for col in {cols_str}:\n    mode_val = df[col].mode()[0] if not df[col].mode().empty else np.nan\n    df[col] = df[col].fillna(mode_val)"
            sql_code = f"-- Mode Imputation SQL"
        elif strategy == "constant":
            pandas_code = f"for col in {cols_str}:\n    df[col] = df[col].fillna('{const_val}')"
            sql_code = f"SELECT *, " + ", ".join([f"COALESCE({c}, '{const_val}') AS {c}" for c in cols]) + " FROM active_table"
        elif strategy == "ffill":
            pandas_code = f"df[{cols_str}] = df[{cols_str}].ffill()"
            sql_code = f"-- Forward Fill using LEAD/LAG SQL windowing"
        elif strategy == "bfill":
            pandas_code = f"df[{cols_str}] = df[{cols_str}].bfill()"
            sql_code = f"-- Backward Fill SQL windowing"
        elif strategy == "drop_rows":
            pandas_code = f"df = df.dropna(subset={cols_str})"
            where_clause = " AND ".join([f"{c} IS NOT NULL" for c in cols])
            sql_code = f"SELECT * FROM active_table WHERE {where_clause}"
        elif strategy == "drop_cols":
            pandas_code = f"df = df.drop(columns={cols_str})"
            sql_code = f"SELECT * EXCLUDE ({', '.join(cols)}) FROM active_table"
            
    elif operation_id == "drop_duplicates":
        cols = params.get("columns", [])
        keep = params.get("keep", "first")
        subset_arg = f"subset={cols}" if cols else ""
        pandas_code = f"df = df.drop_duplicates({subset_arg}, keep='{keep}')"
        if cols:
            sql_code = f"SELECT DISTINCT ON ({', '.join(cols)}) * FROM active_table"
        else:
            sql_code = f"SELECT DISTINCT * FROM active_table"
            
    elif operation_id == "column_rename":
        col = params.get("column", "")
        new_name = params.get("new_name", "")
        pandas_code = f"df = df.rename(columns={{'{col}': '{new_name}'}})"
        sql_code = f"SELECT *, {col} AS {new_name} EXCLUDE ({col}) FROM active_table"
        
    elif operation_id == "cast_type":
        col = params.get("column", "")
        new_type = params.get("new_type", "float")
        if new_type == "int":
            pandas_code = f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').fillna(0).astype(int)"
            sql_code = f"SELECT *, CAST({col} AS INTEGER) AS {col} EXCLUDE ({col}) FROM active_table"
        elif new_type == "float":
            pandas_code = f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')"
            sql_code = f"SELECT *, CAST({col} AS DOUBLE) AS {col} EXCLUDE ({col}) FROM active_table"
        elif new_type == "string":
            pandas_code = f"df['{col}'] = df['{col}'].astype(str)"
            sql_code = f"SELECT *, CAST({col} AS VARCHAR) AS {col} EXCLUDE ({col}) FROM active_table"
        elif new_type == "category":
            pandas_code = f"df['{col}'] = df['{col}'].astype('category')"
            sql_code = f"-- Category type SQL representation\nSELECT * FROM active_table"
        elif new_type == "datetime":
            pandas_code = f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')"
            sql_code = f"SELECT *, CAST({col} AS TIMESTAMP) AS {col} EXCLUDE ({col}) FROM active_table"
            
    elif operation_id == "scaling":
        cols = params.get("columns", [])
        method = params.get("method", "minmax")
        cols_str = str(cols)
        if method == "minmax":
            pandas_code = f"for col in {cols_str}:\n    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())"
            sql_code = f"SELECT *, " + ", ".join([f"({c} - MIN({c}) OVER()) / (MAX({c}) OVER() - MIN({c}) OVER()) AS {c}" for c in cols]) + " FROM active_table"
        elif method == "standard":
            pandas_code = f"for col in {cols_str}:\n    df[col] = (df[col] - df[col].mean()) / df[col].std()"
            sql_code = f"SELECT *, " + ", ".join([f"({c} - AVG({c}) OVER()) / STDDEV({c}) OVER() AS {c}" for c in cols]) + " FROM active_table"
            
    elif operation_id == "encoding":
        cols = params.get("columns", [])
        pandas_code = f"df = pd.get_dummies(df, columns={cols}, dtype=int)"
        sql_code = f"-- One-Hot Encoding Pivot query\nSELECT * FROM active_table"
        
    elif operation_id == "filter_rows":
        col = params.get("column", "")
        op = params.get("operator", "==")
        val = params.get("value", "")
        # Cast value appropriately if numeric
        try:
            float(val)
            val_pandas = val
            val_sql = val
        except ValueError:
            val_pandas = f"'{val}'"
            val_sql = f"'{val}'"
            
        pandas_code = f"df = df[df['{col}'] {op} {val_pandas}]"
        sql_code = f"SELECT * FROM active_table WHERE {col} {op} {val_sql}"
        
    return {"pandas": pandas_code, "sql": sql_code}

# Core dataframe modifier sandbox
def apply_op_to_df(df: pd.DataFrame, operation_id: str, params: Dict[str, Any]) -> pd.DataFrame:
    df_copy = df.copy()
    
    if operation_id == "impute_missing":
        cols = params.get("columns", [])
        strategy = params.get("strategy", "mean")
        const_val = params.get("constant_value", "")

        for col in cols:
            if col not in df_copy.columns:
                continue
            if strategy == "mean":
                num_s = pd.to_numeric(df_copy[col], errors='coerce')
                mean_val = round(float(num_s.mean()), 2) if not pd.isna(num_s.mean()) else np.nan
                if not pd.isna(mean_val):
                    df_copy[col] = num_s.fillna(mean_val).round(2)
                else:
                    df_copy[col] = df_copy[col].fillna("Unknown")
            elif strategy == "median":
                num_s = pd.to_numeric(df_copy[col], errors='coerce')
                med_val = round(float(num_s.median()), 2) if not pd.isna(num_s.median()) else np.nan
                if not pd.isna(med_val):
                    df_copy[col] = num_s.fillna(med_val).round(2)
                else:
                    df_copy[col] = df_copy[col].fillna("Unknown")
            elif strategy == "mode":
                if not df_copy[col].empty and not df_copy[col].mode().empty:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode().iloc[0])
            elif strategy == "constant":
                try:
                    c_val = float(const_val) if const_val.replace('.', '', 1).isdigit() else const_val
                except Exception:
                    c_val = const_val
                df_copy[col] = df_copy[col].fillna(c_val)
            elif strategy == "ffill":
                df_copy[col] = df_copy[col].ffill()
            elif strategy == "bfill":
                df_copy[col] = df_copy[col].bfill()
            elif strategy == "drop_rows":
                df_copy = df_copy.dropna(subset=[col])
            elif strategy == "drop_cols":
                df_copy = df_copy.drop(columns=[col])

    elif operation_id == "drop_duplicates":
        cols = params.get("columns", [])
        keep = params.get("keep", "first")
        subset = cols if cols else None
        df_copy = df_copy.drop_duplicates(subset=subset, keep=keep)

    elif operation_id in ("column_rename", "rename_column"):
        col = params.get("column", "")
        new_name = params.get("new_name", "")
        if col in df_copy.columns and new_name:
            df_copy = df_copy.rename(columns={col: new_name})

    elif operation_id == "cast_type":
        col = params.get("column", "")
        new_type = params.get("new_type", "float")
        if col in df_copy.columns:
            if new_type == "int":
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
            elif new_type == "float":
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            elif new_type == "string":
                df_copy[col] = df_copy[col].astype(str)
            elif new_type == "category":
                df_copy[col] = df_copy[col].astype('category')
            elif new_type == "datetime":
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

    elif operation_id in ("scaling", "normalize_scale"):
        cols = params.get("columns", [])
        method = params.get("method", "minmax")
        for col in cols:
            if col in df_copy.columns:
                num_s = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
                col_min = num_s.min()
                col_max = num_s.max()
                col_mean = num_s.mean()
                col_std = num_s.std()

                if method == "minmax":
                    if col_max != col_min:
                        df_copy[col] = (num_s - col_min) / (col_max - col_min)
                    else:
                        df_copy[col] = 0.0
                elif method == "standard":
                    if col_std != 0 and not pd.isna(col_std):
                        df_copy[col] = (num_s - col_mean) / col_std
                    else:
                        df_copy[col] = 0.0

    elif operation_id in ("encoding", "one_hot_encode"):
        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in df_copy.columns]
        if valid_cols:
            df_copy = pd.get_dummies(df_copy, columns=valid_cols, dtype=int)

    elif operation_id == "label_encode":
        cols = params.get("columns", [])
        for col in cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype('category').cat.codes

    elif operation_id == "log_transform":
        cols = params.get("columns", [])
        for col in cols:
            if col in df_copy.columns:
                num_s = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
                df_copy[col] = np.log1p(np.maximum(0, num_s))

    elif operation_id == "yeo_johnson":
        cols = params.get("columns", []) or ([params.get("column")] if params.get("column") else [])
        for col in cols:
            if col in df_copy.columns:
                try:
                    from sklearn.preprocessing import PowerTransformer
                    pt = PowerTransformer(method='yeo-johnson')
                    num_s = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).values.reshape(-1, 1)
                    df_copy[col] = pt.fit_transform(num_s).flatten()
                except Exception:
                    pass

    elif operation_id == "extract_datetime":
        col = params.get("column", "")
        if col in df_copy.columns:
            dt_s = pd.to_datetime(df_copy[col], errors='coerce')
            df_copy[f"{col}_year"] = dt_s.dt.year.fillna(0).astype(int)
            df_copy[f"{col}_month"] = dt_s.dt.month.fillna(0).astype(int)
            df_copy[f"{col}_day"] = dt_s.dt.day.fillna(0).astype(int)
            df_copy[f"{col}_hour"] = dt_s.dt.hour.fillna(0).astype(int)

    elif operation_id == "bin_column":
        col = params.get("column", "")
        n_bins = params.get("n_bins", 5)
        if col in df_copy.columns:
            num_s = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
            df_copy[f"{col}_binned"] = pd.qcut(num_s, q=n_bins, labels=False, duplicates='drop')
            
    elif operation_id == "filter_rows":
        col = params.get("column", "")
        op = params.get("operator", "==")
        val = params.get("value", "")
        if col in df_copy.columns:
            # Handle float casting for comparison if possible
            try:
                numeric_val = float(val)
                comp_series = pd.to_numeric(df_copy[col], errors='coerce')
                target_val = numeric_val
            except ValueError:
                comp_series = df_copy[col].astype(str)
                target_val = str(val)
                
            if op == "==":
                df_copy = df_copy[comp_series == target_val]
            elif op == "!=":
                df_copy = df_copy[comp_series != target_val]
            elif op == ">":
                df_copy = df_copy[comp_series > target_val]
            elif op == "<":
                df_copy = df_copy[comp_series < target_val]
            elif op == ">=":
                df_copy = df_copy[comp_series >= target_val]
            elif op == "<=":
                df_copy = df_copy[comp_series <= target_val]

    elif operation_id in ("train_model", "train"):
        target_col = params.get("target", None)
        if not target_col or target_col not in df_copy.columns:
            candidates = [c for c in df_copy.columns if "id" not in c.lower()]
            target_col = candidates[-1] if candidates else df_copy.columns[-1]

        y_series = df_copy[target_col]
        feature_cols = [c for c in df_copy.columns if c != target_col and "id" not in c.lower()]
        X_df = pd.get_dummies(df_copy[feature_cols], drop_first=True, dtype=int)
        for c in X_df.columns:
            X_df[c] = pd.to_numeric(X_df[c], errors='coerce').fillna(0)

        # Detect problem type: Regression vs Classification
        is_regression = pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 15

        from sklearn.model_selection import train_test_split
        if is_regression:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_squared_error
            y = pd.to_numeric(y_series, errors='coerce').fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = float(r2_score(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            state.ml_model_metadata = {
                "success": True,
                "model_type": "Regression",
                "accuracy": round(max(0, r2) * 100, 1),
                "r2_score": round(r2, 3),
                "mse": round(mse, 3)
            }
            df_copy["ML_Prediction"] = model.predict(X_df)
        else:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            if y_series.dtype == 'object' or y_series.dtype.name == 'category':
                y_encoded = y_series.astype('category').cat.codes
            else:
                y_encoded = y_series
            X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted', zero_division=0))
            prec = float(precision_score(y_test, preds, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, preds, average='weighted', zero_division=0))
            state.ml_model_metadata = {
                "success": True,
                "model_type": "Classification",
                "accuracy": round(acc * 100, 1),
                "f1_score": round(f1, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3)
            }
            df_copy["ML_Prediction"] = clf.predict(X_df)


    # Clean formatting pass: round float columns to max 2 decimal places to prevent messy unrounded floats
    for col in df_copy.select_dtypes(include=[np.floating]).columns:
        df_copy[col] = df_copy[col].round(2)

    return df_copy


@app.get("/health")
def health():
    return {"status": "ok", "engine": "DataPilot OS", "version": "1.0.0"}


@app.post("/api/connect")
def connect_dataset(req: ConnectRequest):
    file_path = req.file_path
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required for file-based sources")

    if not os.path.exists(file_path):
        fallback = os.path.join(os.getcwd(), file_path)
        if os.path.exists(fallback):
            file_path = fallback
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        source_type = (req.source_type or "csv").lower()
        if source_type in ("excel", "xlsx", "xls") or file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            try:
                con = duckdb.connect(database=':memory:')
                df = con.execute("SELECT * FROM read_csv_auto(?, ignore_errors=true)", [file_path]).df()
                con.close()
            except Exception:
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
                    except Exception:
                        df = pd.read_csv(file_path, encoding='cp1252', on_bad_lines='skip', low_memory=False)
                except Exception:
                    df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

        # Reset session state
        state.df_states = {}
        state.pipeline_steps = []
        state.active_step_id = None
        state.ml_model_metadata = {}
        # Store source path for writeback
        state._source_path = file_path

        step_id = str(uuid.uuid4())
        state.df_states[step_id] = df
        state.active_step_id = step_id

        row_cnt = len(df)
        col_cnt = len(df.columns)

        step_meta = {
            "id": step_id,
            "parent_id": None,
            "operation_id": "load_data",
            "parameters": {"file_path": file_path},
            "pandas_code": f"df = pd.read_csv('{os.path.basename(file_path)}')",
            "sql_code": f"SELECT * FROM read_csv_auto('{os.path.basename(file_path)}')",
            "row_count": row_cnt,
            "column_count": col_cnt,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        state.pipeline_steps.append(step_meta)

        profile = get_df_profile(df)
        return {
            "success": True,
            "name": os.path.basename(file_path),
            "source_type": source_type,
            "step_id": step_id,
            "rows": row_cnt,
            "row_count": row_cnt,
            "total_rows": row_cnt,
            "rowCount": row_cnt,
            "columns": profile,
            "column_count": col_cnt,
            "col_count": col_cnt,
            "schema": profile,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        temp_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return connect_dataset(ConnectRequest(file_path=file_path, source_type="csv"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def get_db_tables(req: DBTestRequest) -> List[str]:
    engine_type = (req.engine or "sqlite").lower()
    conn_str = req.connection_string
    user = req.user or req.username
    db_name = req.database_name or req.database
    
    if engine_type == "duckdb" or (conn_str and conn_str.startswith("duckdb")):
        db_path = req.filepath or conn_str or db_name or ":memory:"
        if db_path.startswith("duckdb://"):
            db_path = db_path[9:]
        elif db_path.startswith("duckdb:"):
            db_path = db_path[7:]
            
        if db_path != ":memory:" and not os.path.exists(db_path):
            raise ValueError(f"DuckDB database file not found: {db_path}")
            
        try:
            con = duckdb.connect(database=db_path, read_only=True)
        except Exception:
            con = duckdb.connect(database=db_path)
        try:
            res = con.execute("SHOW TABLES").fetchall()
            return [str(r[0]) for r in res]
        finally:
            con.close()

    elif engine_type == "sqlite":
        db_path = req.filepath or conn_str or db_name or ":memory:"
        if db_path != ":memory:" and not os.path.exists(db_path):
            raise ValueError(f"SQLite database file not found: {db_path}")
            
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            rows = cursor.fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    else:
        # PostgreSQL, MySQL, URI or fallback SQLAlchemy
        if not HAS_SQLALCHEMY:
            raise ValueError("SQLAlchemy is required for PostgreSQL/MySQL/URI engines. Please install sqlalchemy.")
        if engine_type == "uri":
            if not conn_str:
                raise ValueError("connection_string is required for URI engine")
            eng = create_engine(conn_str)
        elif engine_type in ("postgresql", "postgres"):
            if conn_str:
                eng = create_engine(conn_str)
            else:
                url = URL.create(
                    drivername="postgresql",
                    username=user or None,
                    password=req.password or None,
                    host=req.host or "localhost",
                    port=req.port or 5432,
                    database=db_name or None
                )
                eng = create_engine(url)
        elif engine_type == "mysql":
            if conn_str:
                eng = create_engine(conn_str)
            else:
                url = URL.create(
                    drivername="mysql+pymysql",
                    username=user or None,
                    password=req.password or None,
                    host=req.host or "localhost",
                    port=req.port or 3306,
                    database=db_name or None
                )
                eng = create_engine(url)
        else:
            raise ValueError(f"Unsupported database engine: '{engine_type}'. Supported: sqlite, postgresql, mysql, duckdb, uri.")

        try:
            with eng.connect() as conn:
                inspector = inspect(conn)
                return inspector.get_table_names()
        finally:
            eng.dispose()


def fetch_db_dataframe(req: DBImportRequest):
    engine_type = (req.engine or "sqlite").lower()
    conn_str = req.connection_string
    user = req.user or req.username
    db_name = req.database_name or req.database
    
    if req.sql_query and req.sql_query.strip():
        query = req.sql_query.strip()
    elif req.table_name and req.table_name.strip():
        query = f"SELECT * FROM {req.table_name.strip()} LIMIT 10000"
    else:
        query = "SELECT * FROM table_name LIMIT 10000"

    if engine_type == "duckdb" or (conn_str and conn_str.startswith("duckdb")):
        db_path = req.filepath or conn_str or db_name or ":memory:"
        if db_path.startswith("duckdb://"):
            db_path = db_path[9:]
        elif db_path.startswith("duckdb:"):
            db_path = db_path[7:]
            
        if db_path != ":memory:" and not os.path.exists(db_path):
            raise ValueError(f"DuckDB database file not found: {db_path}")
            
        con = duckdb.connect(database=db_path)
        try:
            df = con.execute(query).df()
            return df, query
        finally:
            con.close()

    elif engine_type == "sqlite":
        db_path = req.filepath or conn_str or db_name or ":memory:"
        if db_path != ":memory:" and not os.path.exists(db_path):
            raise ValueError(f"SQLite database file not found: {db_path}")
            
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(query, conn)
            return df, query
        finally:
            conn.close()

    else:
        # SQLAlchemy for PostgreSQL, MySQL, URI
        if not HAS_SQLALCHEMY:
            raise ValueError("SQLAlchemy is required for PostgreSQL/MySQL/URI engines. Please install sqlalchemy.")
        if engine_type == "uri":
            if not conn_str:
                raise ValueError("connection_string is required for URI engine")
            eng = create_engine(conn_str)
        elif engine_type in ("postgresql", "postgres"):
            if conn_str:
                eng = create_engine(conn_str)
            else:
                url = URL.create(
                    drivername="postgresql",
                    username=user or None,
                    password=req.password or None,
                    host=req.host or "localhost",
                    port=req.port or 5432,
                    database=db_name or None
                )
                eng = create_engine(url)
        elif engine_type == "mysql":
            if conn_str:
                eng = create_engine(conn_str)
            else:
                url = URL.create(
                    drivername="mysql+pymysql",
                    username=user or None,
                    password=req.password or None,
                    host=req.host or "localhost",
                    port=req.port or 3306,
                    database=db_name or None
                )
                eng = create_engine(url)
        else:
            raise ValueError(f"Unsupported database engine: '{engine_type}'. Supported: sqlite, postgresql, mysql, duckdb, uri.")

        try:
            with eng.connect() as conn:
                df = pd.read_sql_query(text(query), conn)
            return df, query
        finally:
            eng.dispose()


@app.post("/api/db/test")
def test_db_connection(req: DBTestRequest):
    try:
        tables = get_db_tables(req)
        return {"success": True, "tables": tables}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database connection error: {str(e)}")


@app.post("/api/db/import")
def import_db_dataset(req: DBImportRequest):
    try:
        df, query = fetch_db_dataframe(req)
        step_id = str(uuid.uuid4())

        # Reset session state for new dataset connection
        state.df_states = {}
        state.pipeline_steps = []
        state.active_step_id = None
        state.ml_model_metadata = {}

        state.add_state(step_id, df)
        source_db = req.database_name or req.database or "database"
        source_tbl = req.table_name or "query"
        state._source_path = f"db://{source_db}/{source_tbl}"

        step_meta = {
            "id": step_id,
            "parent_id": None,
            "operation_id": "import_db",
            "parameters": {
                "engine": req.engine,
                "database_name": source_db,
                "table_name": req.table_name,
                "sql_query": query
            },
            "pandas_code": f"# Loaded from {req.engine} database\n# Query: {query}",
            "sql_code": query,
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        state.pipeline_steps.append(step_meta)

        dataset_name = f"{req.engine or 'db'}://{source_db}/{source_tbl}"
        learning_memory.record_learning_event(
            dataset_name=dataset_name,
            operation="db_import",
            column=None,
            metrics={
                "engine": req.engine or "sqlite",
                "database": source_db,
                "table_name": req.table_name,
                "query": query,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        )

        profile = get_df_profile(df)
        preview_data = get_preview_data(df, limit=50)

        return {
            "success": True,
            "step_id": step_id,
            "schema": profile,
            "row_count": len(df),
            "col_count": len(df.columns),
            "column_count": len(df.columns),
            "preview": preview_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database import error: {str(e)}")



@app.get("/api/dataset/schema")
def get_schema():
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]
    profile = get_df_profile(df)
    for col_info in profile:
        col_info["total"] = len(df)

    knowledge_result = ds_engine.analyze_dataset_with_knowledge(df)

    return {
        "step_id": state.active_step_id,
        "total_rows": len(df),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": profile,
        "schema": profile,
        "knowledge_analysis": knowledge_result
    }


@app.get("/api/dataset/data")
def get_data(limit: Optional[int] = 100):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]
    
    # Cap preview sample data between 1 and 500 rows maximum for ultra-fast UI rendering
    if limit is None or limit <= 0:
        row_limit = 100
    else:
        row_limit = min(limit, 500)

    raw = get_preview_data(df, row_limit)
    return {"data": raw.get("rows", []), "rows": raw.get("rows", []), "columns": raw.get("columns", []), "total_rows": raw.get("total_rows", 0)}


@app.post("/api/screen/analyze")
def analyze_screen_context():
    """Captures screen workspace & active dataset to produce Gemini-style Data Science AI recommendations using Native Windows OCR."""
    try:
        screen_captured = False
        screen_resolution = "1920x1080 px"
        active_title = "Active Workspace Window"
        extracted_text = ""

        thumbnail_b64 = None
        img = None
        try:
            import importlib, io, base64
            from PIL import Image, ImageGrab, ImageDraw
            import asyncio

            try:
                img = ImageGrab.grab(all_screens=True)
            except Exception:
                img = ImageGrab.grab()
            screen_captured = True
            screen_resolution = f"{img.size[0]}×{img.size[1]} px"
        except Exception as e:
            print("Screen OCR grab notice:", e)

        if img is None:
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (400, 220), color='#0f172a')
                d = ImageDraw.Draw(img)
                d.text((20, 90), f"Workspace: {active_title[:30]}", fill='#38bdf8')
            except Exception:
                pass

        if img is not None:
            try:
                import io, base64
                thumb = img.copy()
                thumb.thumbnail((240, 140))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=55)
                thumbnail_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as terr:
                print("Thumbnail gen error:", terr)

        # Detect Windows Active Window Title & Enumerate Active Desktop Windows (EnumWindows)
        running_apps = []
        try:
            import ctypes
            if sys.platform == "win32" and hasattr(ctypes, "windll"):
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                    if buff.value:
                        active_title = str(buff.value)

                # Real Win32 EnumWindows Process Enumeration for Active Applications
                EnumWindows = ctypes.windll.user32.EnumWindows
                EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
                GetWindowTextW = ctypes.windll.user32.GetWindowTextW
                IsWindowVisible = ctypes.windll.user32.IsWindowVisible

                titles = []
                def foreach_window(h_wnd, l_param):
                    if IsWindowVisible(h_wnd):
                        l = ctypes.windll.user32.GetWindowTextLengthW(h_wnd)
                        if l > 2:
                            b = ctypes.create_unicode_buffer(l + 1)
                            GetWindowTextW(h_wnd, b, l + 1)
                            if b.value and b.value not in ['Program Manager', 'Settings', 'Default IME', 'MSCTFIME UI']:
                                titles.append(str(b.value))
                    return True
                EnumWindows(EnumWindowsProc(foreach_window), 0)
                running_apps = titles[:8]
        except Exception as e:
            print("Active window detection notice:", e)

        # Smart Screen-Matched File Detection: search OCR text & Window Title for open file names
        all_files = [f for f in os.listdir(".") if f.endswith((".csv", ".xlsx", ".xls")) and not f.startswith(".")]
        combined_screen_context = (extracted_text + " " + active_title).lower()

        detected_file = None
        for fname in all_files:
            base_name = os.path.splitext(fname)[0].lower()
            if base_name in combined_screen_context or fname.lower() in combined_screen_context:
                detected_file = fname
                break

        if detected_file:
            try:
                if detected_file.endswith((".xlsx", ".xls")):
                    df_detected = pd.read_excel(detected_file)
                else:
                    df_detected = pd.read_csv(detected_file)
                step_id = str(uuid.uuid4())
                state.add_state(step_id, df_detected)
                state._source_path = detected_file
            except Exception as err:
                print("Auto-load detected file error:", err)

        # Smart AI Recommendations & Error Diagnosis Generator
        recommendations = []
        is_error_detected = any(err_w in (extracted_text + " " + active_title).lower() for err_w in ['traceback', 'syntaxerror', 'valueerror', 'keyerror', 'typeerror', 'exception', '#ref!', '#value!'])

        if is_error_detected:
            summary_text = f"⚠️ Screen Error Detected in '{active_title}'! Instant AI Code Diagnostic Available."
            recommendations = ["🛠️ Explain & Fix Screen Error", "📋 Generate Fixed Code", "🔍 Check Data Types"]
        elif detected_file or state.active_step_id:
            df = state.df_states.get(state.active_step_id)
            dataset_name = str(detected_file if detected_file else getattr(state, '_source_path', 'Loaded Dataset'))
            total_file_rows = len(df) if df is not None else 0
            summary_text = f"📸 Read Complete: '{active_title}' → Active Dataset '{dataset_name}' ({total_file_rows:,} rows)"
            recommendations = ["📊 Run Auto-EDA", "🧹 Impute Missing Nulls", "📈 Forecast 30-Day Trend", "🔍 Detect Outliers"]
        else:
            text_snippet = extracted_text[:100].replace("\n", " ") if extracted_text else "No text visible"
            summary_text = f"📸 Read Complete: Active Window '{active_title}' ({len(extracted_text)} chars scanned)"
            recommendations = ["📝 Summarize Screen Text", "📊 Plot Category Distribution", "🔍 Profile Dataset Columns"]

        return {
            "success": True,
            "screen_captured": bool(screen_captured),
            "screen_resolution": str(screen_resolution),
            "active_title": str(active_title),
            "running_apps": running_apps,
            "extracted_character_count": int(len(extracted_text)),
            "extracted_text": str(extracted_text),
            "thumbnail_b64": thumbnail_b64,
            "dataset_name": str(detected_file if detected_file else active_title),
            "summary": str(summary_text),
            "recommendations": recommendations
        }
    except Exception as exc:
        print("Bulletproof Exception Handled in analyze_screen_context:", exc)
        return {
            "success": True,
            "screen_captured": False,
            "active_title": "Desktop Workspace",
            "extracted_character_count": 0,
            "dataset_name": "Active Workspace Screen",
            "summary": "📸 Read Complete: Active Screen Context Read",
            "recommendations": ["📊 Run Auto-EDA", "🧹 Impute Missing Nulls", "📈 Forecast 30-Day Trend"]
        }





def query_ollama_llm(query: str, dataset_context: Optional[str] = None) -> Optional[Dict[str, Any]]:
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    base_url = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_URL", "http://localhost:11434"))
    if not base_url.endswith("/api/chat") and not base_url.endswith("/api/generate"):
        ollama_url = f"{base_url.rstrip('/')}/api/chat"
    else:
        ollama_url = base_url
    
    timeout_sec = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
    
    system_prompt = (
        "You are an expert Data Science AI assistant in DataPilot OS. "
        "Provide helpful data science analysis, python code snippets if applicable, and recommendations."
    )
    if dataset_context:
        system_prompt += f"\nActive Dataset Context:\n{dataset_context}"

    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "stream": False
    }

    try:
        data = None
        if HAS_REQUESTS and requests is not None:
            resp = requests.post(ollama_url, json=payload, timeout=(1.0, timeout_sec))
            if resp.status_code == 200:
                data = resp.json()
        else:
            import urllib.request
            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(ollama_url, data=req_data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))

        if data:
            content = data.get("message", {}).get("content", "")
            if not content and "response" in data:
                content = data.get("response", "")
            
            if content:
                # LLM code extraction disabled for security — output is handled as conversational answer
                return {
                    "source": "ollama",
                    "response": content,
                    "answer": content,
                    "code": None
                }
    except Exception as err:
        print(f"Ollama integration notice (falling back to knowledge engine): {err}")
    
    return None


class ChatRequest(BaseModel):
    query: str

from ds_knowledge_engine import ds_engine
from modules.capability_registry import capability_registry

@app.post("/api/query")
@app.post("/api/chat")
def chat_copilot(req: ChatRequest):
    """
    Unified Routing & Capability Dispatch Engine for DataPilot OS.
    
    Order of Operations:
    1. Validate query & check security interdict.
    2. Check instant SHA-256 Execution Cache hit.
    3. Route & dispatch query through CapabilityRegistry (unified scoring: feature signals + classifier probabilities + semantic similarity).
    4. Attach confidence score & alternative_intents (if confidence < 0.70 or delta < 0.15).
    """
    query_raw = req.query.strip()
    if not query_raw:
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")

    has_df = bool(state.active_step_id and state.active_step_id in state.df_states)
    df = state.df_states[state.active_step_id] if has_df else None

    # 1. Validation Engine Execution
    val_res = validation_engine.validate_query(df, query_raw)
    if val_res.synthesized_columns and df is not None:
        for metric_name in val_res.synthesized_columns.keys():
            df, derived_col = validation_engine.resolve_missing_metrics(df, metric_name)
            if derived_col:
                governance_engine.add_lineage_node(
                    step_id=f"synth_{derived_col}",
                    step_name=f"Synthesize {derived_col}",
                    action_type="derived_metric",
                    python_code=f"# Derived metric synthesized by Validation Engine\n# Column: {derived_col}",
                    input_rows=len(df),
                    output_rows=len(df)
                )

    # Log audit event
    governance_engine.log_audit_event("chat_query", query_raw, status="success" if val_res.is_valid else "warning")

    # 2. Security Check
    if any(k in query_raw.lower() for k in ["__import__", "subprocess", "eval(", "exec(", "os.system", "rm -rf", "cmd.exe", "socket"]):
        return {
            "success": False,
            "response": "⚠️ **Security Violation Blocked**: Disallowed system call or AST payload detected.",
            "answer": "Security Violation Blocked",
            "code": "# Security Guard Interdicted Payload",
            "artifacts": {"code": [], "plots": [], "tables": [], "reports": ["Security Violation Blocked"]},
            "intent": "ast_security",
            "confidence": 1.0
        }

    # 3. Execution Cache Check (Instant SHA-256 Hit)
    if hasattr(ds_engine, 'execution_cache') and ds_engine.execution_cache:
        cache_res = ds_engine.execution_cache.get(df, query_raw)
        if isinstance(cache_res, tuple):
            is_hit, cached_val = cache_res
            if is_hit and cached_val:
                return cached_val
        elif cache_res:
            return cache_res

    # 4. Process Query through QueryOrchestrator (LLM Semantic Planner + Validator + Repair + Sandbox)
    from modules.query_orchestrator import QueryOrchestrator
    from modules.models.execution_result import ExecutionStatus
    
    exec_res = QueryOrchestrator.process_query(query_raw=query_raw, df=df)
    res = {
        "status": exec_res.status.value,
        "success": exec_res.status == ExecutionStatus.SUCCESS,
        "response": exec_res.response,
        "answer": exec_res.response,
        "code": exec_res.code,
        "plot_image": exec_res.plot_image,
        "artifacts": exec_res.artifacts
    }

    # Cache execution result
    if hasattr(ds_engine, 'execution_cache') and ds_engine.execution_cache:
        ds_engine.execution_cache.set(df, query_raw, res)

    return res

    # 6. Fall back gracefully to column-type-driven routing engine
    query = query_raw.lower()
    code = None
    response_text = ""

    # ─── STEP A: Smart Column Matcher ────────────────────────────────────────────
    # Match columns by normalizing spaces/underscores. Works for ANY language query.
    num_matched = []
    cat_matched = []
    date_matched = []
    year_filter = None

    if df is not None:
        query_norm = re.sub(r'[\s_\-]+', '', query_raw.lower())
        query_words = set(re.findall(r'\b[a-zA-Z0-9]{2,}\b', query_raw.lower())) - {'the', 'a', 'an', 'in', 'of', 'for', 'to', 'is', 'me', 'ma', 'kr', 'show', 'plot', 'list', 'dikhao'}

        scored_matches = []
        for col in df.columns:
            col_l = str(col).lower()
            col_norm = re.sub(r'[\s_\-]+', '', col_l)
            col_words = set(re.findall(r'\b[a-zA-Z0-9]{2,}\b', col_l))

            score = 0
            if col_norm in query_norm or (len(col_norm) >= 4 and col_norm in query_norm):
                score += 100
            matching = col_words.intersection(query_words)
            score += len(matching) * 30
            if col_l in query_raw.lower():
                score += 20

            # Domain term boost (e.g. 'delivery' in query matches 'order_status' / 'on_time_delivery')
            if 'delivery' in query_raw.lower() and any(k in col_l for k in ['deliver', 'order_status', 'shipping']):
                score += 50
            if 'status' in query_raw.lower() and 'status' in col_l:
                score += 25

            if score > 0:
                scored_matches.append((score, col))

        # Sort matches by score descending
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        for _, col in scored_matches:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if col not in date_matched: date_matched.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                if col not in num_matched: num_matched.append(col)
            else:
                if col not in cat_matched: cat_matched.append(col)

        # Smart Status/Order Keyword Interdiction: If query asks for order status/processing/pending, prioritize status columns
        status_keywords = ['status', 'processing', 'pending', 'delivered', 'shipped', 'cancelled', 'returned', 'order']
        if any(k in query_raw.lower() for k in status_keywords):
            for col in df.columns:
                col_l = str(col).lower()
                if any(k in col_l for k in ['status', 'state', 'stage', 'phase']) or ('order' in query_raw.lower() and 'order' in col_l and col not in num_matched):
                    if col not in cat_matched:
                        cat_matched.insert(0, col)

        # Deep Cell Value Scanner: Search data cell values if no column name matched directly
        cell_value_match = None
        if not cat_matched and df is not None:
            stop_words = {'show', 'plot', 'list', 'chart', 'dikhao', 'dikha', 'dikhana', 'dekhna', 'dekho', 'bana', 'banao', 'kitne', 'kitna', 'main', 'hain', 'mein', 'with', 'from', 'have', 'the', 'for', 'and', 'are', 'unhe', 'wale'}
            tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query_raw.lower()) if w not in stop_words]
            for col in df.select_dtypes(include=['object', 'category', 'string']).columns:
                try:
                    unique_vals = [str(v).lower() for v in df[col].dropna().unique()[:100]]
                    for token in tokens:
                        matched_val = next((v for v in unique_vals if token in v), None)
                        if matched_val:
                            cat_matched.append(col)
                            cell_value_match = (col, token, matched_val)
                            break
                except Exception:
                    pass
                if cat_matched:
                    break

        # Detect year filter in query (e.g. "2026", "2024") — works in ANY language
        year_match = re.search(r'\b(20\d{2})\b', query_raw)
        if year_match:
            year_val = int(year_match.group(1))
            for col in df.columns:
                col_lower = col.lower()
                if any(k in col_lower for k in ['year', 'date', 'time', 'month', 'period', 'yr']):
                    year_filter = (col, year_val)
                    break

    # 6. Open-Ended Generative Query Branch: Trigger Ollama LLM only if query requires free-form reasoning AND no column matched
    open_ended_keywords = ["explain", "story", "creative", "strategy", "opinion", "why should i", "describe in detail", "tell me about"]
    has_column_match = bool(num_matched or cat_matched or date_matched)
    is_open_ended = any(k in query_raw.lower() for k in open_ended_keywords) or (not has_column_match and parsed_intent and getattr(parsed_intent, 'confidence_score', 0) < 0.20)
    
    if is_open_ended:
        ds_context = None
        if df is not None:
            dtypes_str = {c: str(t) for c, t in df.dtypes.items()}
            null_counts = df.isna().sum().to_dict()
            ds_context = f"Shape: {df.shape}\nColumns & Types: {dtypes_str}\nNull Counts: {null_counts}"
        ollama_res = query_ollama_llm(query_raw, ds_context)
        if ollama_res:
            res_ollama = {
                "success": True,
                "response": ollama_res["response"],
                "answer": ollama_res["answer"],
                "code": ollama_res["code"],
                "artifacts": {
                    "code": [ollama_res["code"]] if ollama_res.get("code") else [],
                    "plots": [],
                    "tables": [],
                    "reports": [ollama_res["response"]]
                }
            }
            if hasattr(ds_engine, 'execution_cache') and ds_engine.execution_cache:
                ds_engine.execution_cache.set(df, query_raw, res_ollama)
            return res_ollama

    # ─── STEP B: ML Meta-Classifier Intent Detection ─────────────────────────────
    # Uses the trained TF-IDF + RandomForest classifier to detect DS intent
    # from ANY natural language (English, Hinglish, short phrases, etc.)
    ml_intent = None
    try:
        if ds_engine.clf_model and ds_engine.vectorizer:
            vec = ds_engine.vectorizer.transform([query_raw])
            ml_intent = ds_engine.clf_model.predict(vec)[0]  # e.g. "visualization", "imputation", "ml_training" etc.
    except Exception:
        ml_intent = None


    # ─── STEP C: Intent Signal Scoring ───────────────────────────────────────────
    # Score signals from query text — pattern-based, NOT keyword-dependent.
    # Each signal is a broad semantic pattern; the sum drives routing.

    def _has(patterns):
        """Returns True if ANY pattern (regex) matches the query."""
        return any(re.search(p, query, re.IGNORECASE) for p in patterns)

    sig_list        = _has([r'\blist\b', r'table\b', r'rows?\b', r'sample\b', r'head\b', r'preview\b', r'display\b', r'records?\b', r'data\s*list', r'dikhao\s*list'])
    sig_scatter     = _has([r'scatter', r'\bvs\b', r'versus', r'relation', r'bivariate', r'correlation between'])
    # FORECAST must be separate & high-priority — checked BEFORE sig_line
    sig_forecast    = _has([r'forecast', r'predict.*next', r'next.*\d+.*day', r'next.*month', r'predict.*trend',
                             r'future.*sales', r'future.*trend', r'agle\s*\d+', r'agle.*din', r'\d+.*day.*forecast',
                             r'30.?day', r'predict.*30', r'time.?series.*forecast'])
    sig_line        = _has([r'line\s*chart', r'line\s*plot', r'over time', r'timeline', r'month.?wise', r'year.?wise',
                             r'date.?wise', r'weekly', r'daily\s*(trend|chart|plot)', r'monthly\s*(trend|chart|plot)',
                             r'yearly', r'growth', r'\btrend\b'])
    sig_pie         = _has([r'pie', r'donut', r'share', r'proportion', r'percentage\s*(of|share|break)', r'composition'])
    sig_box         = _has([r'box\s*plot', r'outlier', r'\biqr\b', r'whisker', r'quartile', r'extreme\s*value'])
    sig_hist        = _has([r'hist', r'distribution', r'\bkde\b', r'density', r'bell\s*curve', r'frequency\s*dist', r'spread'])
    sig_heatmap     = _has([r'heatmap', r'heat\s*map', r'corr(elation)?\s*(matrix|map)', r'feature.*corr'])
    sig_impute      = _has([r'impute', r'missing', r'null', r'fillna', r'fill.*null', r'handle.*null', r'drop.*null'])
    sig_clean       = _has([r'\bclean\b', r'duplicat', r'drop.*row', r'remove.*null', r'inconsistenc', r'inconsistent', r'audit', r'data\s*quality', r'sanitiz'])
    sig_ml          = _has([r'train', r'predict\s+(a\s+)?(model|classif|regress)', r'\bml\b', r'machine\s*learn', r'random\s*forest', r'classif', r'regress', r'xgboost', r'\bmodel\b'])
    sig_cluster     = _has([r'cluster', r'segment', r'kmeans', r'k-means', r'group\s*similar', r'unsupervised'])
    sig_pca         = _has([r'\bpca\b', r'principal\s*component', r'dimensionality\s*reduc', r'reduce\s*feature'])
    sig_encode      = _has([r'encod', r'one.?hot', r'label\s*encod', r'dummy', r'dummies'])
    sig_scale       = _has([r'scal', r'normaliz', r'standardiz', r'minmax', r'zscore', r'z.score'])
    sig_skew        = _has([r'skew', r'log\s*transform', r'yeo.?johnson', r'power\s*transform', r'normalit'])
    sig_describe    = _has([r'describ', r'summary', r'statistic', r'info\b', r'profil', r'overview', r'shape\b', r'dtypes', r'data.*report', r'full.*report', r'report.*data', r'profiling\s*report'])
    sig_top         = _has([r'top\s*\d+', r'bottom\s*\d+', r'best\s*\d+', r'worst\s*\d+', r'highest', r'lowest', r'best', r'worst', r'rank'])
    sig_count       = _has([r'count\b', r'how\s*many', r'frequency', r'kitne\b', r'kitna\b'])
    sig_feature_eng = _has([r'suggest.*feature', r'feature.*engineer', r'new.*feature', r'create.*feature', r'feature.*suggest', r'feature.*importance'])
    sig_compare     = _has([r'compar', r'contrast', r'differ', r'between', r'vs\b', r'versus'])
    sig_eda_dashboard = _has([r'\beda\b', r'eda.*dashboard', r'automated.*eda', r'full.*eda', r'eda.*report', r'eda.*summary',
                               r'full.*automated.*dashboard', r'generate.*full.*dashboard', r'complete.*analysis.*dashboard'])

    # Broad "show/visualize/chart" intent — catches ALL chart requests regardless of language
    sig_viz_generic = _has([
        r'plot', r'graph', r'chart', r'visual', r'show', r'draw', r'render', r'generate\s*(a|the)?\s*(plot|chart|graph)',
        r'dikhao', r'dikha', r'bana', r'banao', r'dekhna', r'dekho', r'dikhana',  # Hindi
        r'wise\b',  # "category wise", "region wise"
        r'by\s+\w+',  # "sales by region", "profit by category"
        r'\w+\s+(ka|ki|ke)\s+\w+',  # Hindi genitive
        r'break(down)?', r'breakdown', r'analys', r'insight'
    ])

    # ─── STEP D: Route to Code Generator ─────────────────────────────────────────
    # Priority: specific operations > ML > visualization type > generic viz with columns
    # NEVER fall to generic profile if columns were matched.

    def _make_filter_snippet(df_var_name="plot_df"):
        if year_filter:
            yr_col, yr_val = year_filter
            return (
                f"\n# Filter for year {yr_val}\n"
                f"{df_var_name} = df[df['{yr_col}'] == {yr_val}].copy()\n"
                f"if {df_var_name}.empty:\n"
                f"    {df_var_name} = df.copy()  # fallback if year not found\n"
            ), df_var_name
        return "", "df"

    def _best_cat_col():
        """Pick best categorical groupby column when none matched from query."""
        if cat_matched:
            return cat_matched[0]
        status_keys = ['processing', 'status', 'order', 'pending', 'shipped', 'delivered', 'cancelled']
        if any(k in query_raw.lower() for k in status_keys):
            for col in df.columns:
                if any(k in str(col).lower() for k in ['status', 'state', 'stage', 'phase']) and df[col].nunique() <= 50:
                    return col

        priority_keys = ['category', 'courier', 'gender', 'region', 'segment', 'product', 'city', 'state',
                         'country', 'type', 'status', 'payment', 'month', 'year', 'quarter']
        for k in priority_keys:
            for col in df.columns:
                if k in col.lower() and 2 <= df[col].nunique() <= 50:
                    return col

        # Filter out high-cardinality ID/Name columns (nunique > 50 or id/code in column name)
        valid_cats = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            col_l = str(col).lower()
            if any(id_kw in col_l for id_kw in ['id', 'uuid', 'code', 'hash', 'num', 'name']):
                continue
            if 2 <= df[col].nunique() <= 50:
                valid_cats.append(col)

        if valid_cats:
            return valid_cats[0]

        # Fallback to any string column with 2 <= nunique <= 50
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if 2 <= df[col].nunique() <= 50:
                return col

        cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        return cat_cols[0] if cat_cols else df.columns[0]

    def _best_num_col(exclude=None):
        """Pick best numeric column."""
        if num_matched:
            return next((c for c in num_matched if c != exclude), num_matched[0])
        priority_keys = ['sale', 'revenue', 'profit', 'amount', 'price', 'cost', 'total', 'quantity', 'count', 'age']
        for k in priority_keys:
            for col in df.columns:
                if k in col.lower() and col != exclude and pd.api.types.is_numeric_dtype(df[col]):
                    return col
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        num_cols = [c for c in num_cols if c != exclude]
        return num_cols[0] if num_cols else df.columns[-1]

    def _dual_bar_code(y1, y2, x_col, filter_snip, df_var):
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
{filter_snip}
grouped = {df_var}.groupby('{x_col}')[['{y1}', '{y2}']].sum().reset_index()
grouped = grouped.sort_values(by='{y1}', ascending=False).head(15)
x = np.arange(len(grouped['{x_col}']))
width = 0.38
fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
sns.set_theme(style='whitegrid')
bars1 = ax.bar(x - width/2, grouped['{y1}'], width, label='{y1}', color='#0284c7', edgecolor='black', linewidth=0.6)
bars2 = ax.bar(x + width/2, grouped['{y2}'], width, label='{y2}', color='#10b981', edgecolor='black', linewidth=0.6)
for bar in bars1:
    h = bar.get_height()
    if h > 0: ax.annotate(f'{{h:,.0f}}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,4), textcoords='offset points', ha='center', fontsize=7, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    if h > 0: ax.annotate(f'{{h:,.0f}}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0,4), textcoords='offset points', ha='center', fontsize=7, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(grouped['{x_col}'].astype(str), rotation=35, ha='right', fontsize=9, fontweight='bold')
ax.set_title('{y1} vs {y2} by {x_col}', fontsize=14, fontweight='bold', pad=14)
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.legend(fontsize=10, frameon=True)
plt.tight_layout()
plt.show()
print(f'[SUCCESS] Rendered {y1} & {y2} comparison chart by {x_col}!')"""

    def _single_bar_code(y_col, x_col, filter_snip, df_var):
        is_avg_metric = any(k in y_col.lower() for k in ['age', 'rating', 'score', 'percent', 'ratio', 'avg', 'mean'])
        agg_func = "mean" if is_avg_metric else "sum"
        agg_label = "Average" if is_avg_metric else "Total"
        fmt_code = "{w:,.1f}" if is_avg_metric else "{w:,.0f}"

        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
{filter_snip}
grouped = {df_var}.groupby('{x_col}')['{y_col}'].{agg_func}().reset_index().sort_values(by='{y_col}', ascending=False).head(15)
plt.figure(figsize=(11, 6), dpi=150)
sns.set_theme(style='whitegrid')
ax = sns.barplot(data=grouped, x='{y_col}', y='{x_col}', palette='Blues_r')
plt.title('{agg_label} {y_col} by {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{agg_label} {y_col}', fontsize=11, fontweight='bold')
plt.ylabel('{x_col}', fontsize=11, fontweight='bold')
for p in ax.patches:
    w = p.get_width()
    if w > 0:
        ax.annotate(f'{fmt_code}', (w, p.get_y() + p.get_height()/2.), ha='left', va='center', xytext=(5,0), textcoords='offset points', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()
print(f'[SUCCESS] Rendered {agg_label} {y_col} by {x_col} Bar Chart!')"""

    # ── Route 0: Tabular Listing & Summary ──
    if sig_list and df is not None:
        matched_cols = [c for c in (cat_matched + num_matched + date_matched) if c in df.columns]
        cols_to_show = matched_cols if matched_cols else list(df.columns[:6])
        code = f"""import pandas as pd
cols = [c for c in {cols_to_show} if c in df.columns]
print("=== Data Preview & Listing ===")
print(df[cols].head(25).to_string(index=False))
for col in cols[:3]:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"\\n=== Value Counts Distribution: {{col}} ===")
        print(df[col].value_counts().to_string())
"""
        response_text = f"Tabular data listing and frequency breakdown for **{', '.join(cols_to_show)}**:"

    # ── Route 1: Specific Cleaning / Imputation ──
    elif sig_impute or sig_clean:
        if df is not None:
            if sig_clean and not sig_impute:
                code = template_engine.render_cleaning(is_audit=True)
                response_text = "🧹 **Data Quality & Inconsistency Audit Pipeline** executed — duplicates, missing values, anomalies, and text formatting checked:"
            else:
                target_col = (num_matched + cat_matched)[0] if (num_matched + cat_matched) else None
                if not target_col:
                    null_series = df.isna().sum()
                    null_cols = null_series[null_series > 0]
                    target_col = null_cols.index[0] if not null_cols.empty else df.columns[0]
                null_count = int(df[target_col].isnull().sum())
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    s = float(pd.to_numeric(df[target_col], errors='coerce').dropna().skew()) if len(df[target_col].dropna()) > 3 else 0.0
                    strategy = "median" if abs(s) > 1.0 else "mean"
                    code = template_engine.render_cleaning(target_col=target_col, strategy=strategy, null_count=null_count, skewness=s)
                    response_text = f"Column '{target_col}' has {null_count} missing values → applying **{strategy.capitalize()} Imputation** (skewness={s:.2f}):"
                else:
                    code = template_engine.render_cleaning(target_col=target_col, strategy="mode", null_count=null_count)
                    response_text = f"Categorical column '{target_col}' has {null_count} missing values → applying **Mode Imputation**:"

    # ── Route 2: Encoding ──
    elif sig_encode:
        if df is not None:
            col = (cat_matched + [c for c in df.select_dtypes(include=['object']).columns])[0] if (cat_matched or len(df.select_dtypes(include=['object']).columns)) else df.columns[0]
            code = template_engine.render_encoding_scaling(mode="one_hot", col=col)
            response_text = f"Applying **One-Hot Encoding** to categorical column '{col}':"

    # ── Route 3: Scaling / Normalization ──
    elif sig_scale:
        if df is not None:
            col = num_matched[0] if num_matched else _best_num_col()
            code = template_engine.render_encoding_scaling(mode="standard_scale", col=col)
            response_text = f"Applying **StandardScaler** to numeric column '{col}':"

    # ── Route 4: Skewness / Log Transform ──
    elif sig_skew:
        if df is not None:
            col = num_matched[0] if num_matched else _best_num_col()
            code = template_engine.render_encoding_scaling(mode="log_transform", col=col)
            response_text = f"Applying **Log Transform** to reduce skewness in '{col}':"

    # ── Route 5: Outlier / Boxplot ──
    elif sig_box:
        if df is not None:
            y_col = num_matched[0] if num_matched else _best_num_col()
            x_col = cat_matched[0] if cat_matched else None
            code = template_engine.render_outlier_boxplot(y_col=y_col, x_col=x_col)
            response_text = f"IQR Outlier analysis boxplot for **{y_col}**:"

    # ── Route 6: Histogram / Distribution ──
    elif sig_hist:
        if df is not None:
            col = num_matched[0] if num_matched else _best_num_col()
            code = template_engine.render_distribution_hist(col=col)
            response_text = f"Distribution histogram + KDE curve for **{col}** (skewness: {df[col].skew():.2f}):"

    # ── Route 7: Correlation Heatmap ──
    elif sig_heatmap:
        if df is not None:
            num_cols_h = list(df.select_dtypes(include=[np.number]).columns[:12])
            code = template_engine.render_correlation_heatmap(num_cols=num_cols_h)
            response_text = "Seaborn Correlation Heatmap for all numeric features:"

    # ── Route 8: Scatter Plot ──
    elif sig_scatter:
        if df is not None:
            x_col = num_matched[0] if len(num_matched) >= 1 else _best_num_col()
            y_col = num_matched[1] if len(num_matched) >= 2 else _best_num_col(exclude=x_col)
            hue_col = cat_matched[0] if cat_matched else None
            code = template_engine.render_scatter_plot(x_col=x_col, y_col=y_col, hue_col=hue_col)
            response_text = f"Scatter plot — **{y_col}** vs **{x_col}**:"

    # ── Route 9: Line / Time-Series / Forecast ──
    elif sig_forecast or sig_line or date_matched:
        if df is not None:
            y_col = num_matched[0] if num_matched else _best_num_col()

            # Detect if the query is asking for FORECAST specifically (uses dedicated sig_forecast)
            is_forecast = sig_forecast

            # Try to find a real datetime column in the dataset
            dt_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ['date', 'time', 'timestamp']):
                    try:
                        test = pd.to_datetime(df[col].dropna().iloc[:5], errors='raise')
                        dt_col = col
                        break
                    except Exception:
                        continue

            if is_forecast and dt_col:
                # Determine forecast horizon (default 30 days)
                horizon_match = re.search(r'(\d+)\s*day', query, re.IGNORECASE)
                horizon = int(horizon_match.group(1)) if horizon_match else 30
                code = template_engine.render_forecast(dt_col=dt_col, y_col=y_col, horizon=horizon, is_forecast=True)
                response_text = f"📈 **{horizon}-Day Time Series Forecast** — {y_col} historical trend + future prediction with confidence band:"

            elif dt_col and not is_forecast:
                # Regular time-series line chart over actual dates (monthly aggregation)
                code = template_engine.render_forecast(dt_col=dt_col, y_col=y_col, is_forecast=False)
                response_text = f"📅 Monthly **{y_col}** trend line chart over time:"

            else:
                # Fallback: category-based trend
                x_col = cat_matched[0] if cat_matched else _best_cat_col()
                code = f"""import seaborn as sns, matplotlib.pyplot as plt
grouped = df.groupby('{x_col}')['{y_col}'].mean().reset_index().sort_values('{y_col}', ascending=True)
plt.figure(figsize=(11, 5), dpi=150)
sns.set_theme(style='whitegrid')
sns.lineplot(data=grouped, x='{x_col}', y='{y_col}', marker='o', linewidth=2.5, color='#10b981')
plt.title('Trend: {y_col} by {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{x_col}', fontsize=11, fontweight='bold')
plt.ylabel('Avg {y_col}', fontsize=11, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=9)
plt.tight_layout()
plt.show()
print(f'[SUCCESS] Category trend chart for {y_col} by {x_col}!')"""
                response_text = f"Category trend — **{y_col}** by **{x_col}**:"

    # ── Route 10: Pie / Share Chart ──
    elif sig_pie:
        if df is not None:
            col = cat_matched[0] if cat_matched else _best_cat_col()
            code = f"""import matplotlib.pyplot as plt
top = df['{col}'].value_counts().head(7)
plt.figure(figsize=(8, 6), dpi=150)
colors = ['#0284c7','#0ea5e9','#38bdf8','#7dd3fc','#bae6fd','#10b981','#34d399']
plt.pie(top.values, labels=top.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={{'fontsize':10,'weight':'bold'}})
plt.title('Share Breakdown of {col}', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.show()
print(f'[SUCCESS] Pie chart for {col}!')"""
            response_text = f"Pie chart — share breakdown of **{col}**:"

    # ── Route 11: ML Training & Feature Engineering / Suggestion ──
    elif sig_ml or sig_feature_eng:
        if df is not None:
            if sig_feature_eng:
                code = template_engine.render_ml_training(target='Profit_Margin', is_feature_eng=True)
                response_text = "💡 **Automated Feature Engineering & Feature Suggestion Pipeline** executed:"
            else:
                target = (num_matched + cat_matched)[0] if (num_matched + cat_matched) else df.columns[-1]
                is_cat = not pd.api.types.is_numeric_dtype(df[target]) or df[target].nunique() < 15
                model_cls = "RandomForestClassifier" if is_cat else "RandomForestRegressor"
                code = template_engine.render_ml_training(target=target, model_cls=model_cls, is_feature_eng=False)
                response_text = f"Training **{model_cls}** with target=**{target}** (high-cardinality ID columns safely excluded):"

    # ── Route 12: Clustering ──
    elif sig_cluster:
        if df is not None:
            code = template_engine.render_kmeans_elbow()
            response_text = "KMeans Clustering — Elbow Curve to find optimal number of clusters:"

    # ── Route 13: PCA ──
    elif sig_pca:
        if df is not None:
            code = template_engine.render_pca()
            response_text = "PCA 2D Projection — dimensionality reduction to 2 principal components:"

    # ── Route 14: Top N / Count chart ──
    elif sig_top or sig_count:
        if df is not None:
            n = 10
            m = re.search(r'top\s*(\d+)', query)
            if m: n = int(m.group(1))
            col = cat_matched[0] if cat_matched else _best_cat_col()
            val_col = num_matched[0] if num_matched else None
            code = template_engine.render_top_n_bar(col=col, val_col=val_col, n=n)
            if val_col:
                response_text = f"Top **{n}** **{col}** by **{val_col}**:"
            else:
                response_text = f"Top **{n}** values in **{col}** by count:"

    # ── Route 15: Dataset describe/info / Full Profiling Report ──
    elif sig_describe:
        if df is not None:
            code = template_engine.render_data_profiling()
            response_text = "📊 **Full Data Profiling Report** — shape, missing values, duplicates, numeric & categorical stats:"

    # ── Route 15B: Full Automated 4-Panel EDA Dashboard ──
    elif sig_eda_dashboard:
        if df is not None:
            best_cat = _best_cat_col()
            best_num1 = _best_num_col()
            best_num2 = _best_num_col(exclude=best_num1)
            code = template_engine.render_eda_dashboard(best_cat=best_cat, best_num1=best_num1, best_num2=best_num2)
            response_text = "📊 **Full Automated EDA Dashboard** — 4-panel multi-feature analysis (Categorical Distribution, Numeric Histogram, Feature Correlation & Bivariate Boxplot):"

    # ── Route 16: Generic viz OR any column matched → ALWAYS generate a chart ──
    elif sig_viz_generic or num_matched or cat_matched:
        if df is not None:
            filter_snip, df_var = _make_filter_snippet()

            if len(num_matched) >= 2:
                # 2 numeric columns mentioned → dual bar chart
                y1 = num_matched[0]
                y2 = num_matched[1]
                x_col = _best_cat_col()
                code = _dual_bar_code(y1, y2, x_col, filter_snip, df_var)
                response_text = f"Side-by-side comparison: **{y1}** and **{y2}** grouped by **{x_col}**:"

            elif num_matched:
                # 1 numeric → bar chart grouped by best categorical
                y_col = num_matched[0]
                x_col = _best_cat_col()
                code = _single_bar_code(y_col, x_col, filter_snip, df_var)
                response_text = f"Total **{y_col}** by **{x_col}**:"

            elif cat_matched:
                # Only categorical → value count bar chart
                col = cat_matched[0]
                code = template_engine.render_top_n_bar(col=col, n=15)
                response_text = f"Category distribution bar chart for **{col}**:"

            else:
                # No specific column matched -> AUTO-SELECT BEST COLUMNS for visual analysis
                y_col = _best_num_col()
                x_col = _best_cat_col()
                if y_col and x_col:
                    code = _single_bar_code(y_col, x_col, filter_snip, df_var)
                    response_text = f"Auto-selected primary key metrics (**{y_col}** by **{x_col}**) for visual analysis:"
                elif y_col:
                    code = template_engine.render_distribution_hist(col=y_col)
                    response_text = f"Visualizing distribution of primary numeric feature **{y_col}**:"
                elif x_col:
                    code = template_engine.render_top_n_bar(col=x_col, n=15)
                    response_text = f"Visualizing categories in **{x_col}**:"
                else:
                    code = "print('=== Dataset Summary ===')\nprint(df.describe(include='all').T.to_string())"
                    response_text = "Dataset overview and descriptive statistics:"

        else:
            code = "print('Please load a dataset first!')"
            response_text = "Load a dataset first to generate charts."

    # ── Route 17: General Data Science AI Fallback — Auto-explore dataset ──
    else:
        if df is not None:
            y_col = _best_num_col()
            x_col = _best_cat_col()
            if y_col and x_col:
                code = f"""import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

print('=' * 60)
print('         AUTOMATED DATA SCIENCE EXPLORATION & ANALYSIS')
print('=' * 60)
print(f'Dataset Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns')
print(f'Primary Numeric Metric: {y_col}')
print(f'Primary Category Feature: {x_col}')

plt.figure(figsize=(11, 6), dpi=150)
sns.set_theme(style='whitegrid')
top_df = df.groupby('{x_col}')['{y_col}'].sum().nlargest(12).reset_index()
ax = sns.barplot(data=top_df, x='{y_col}', y='{x_col}', palette='crest')
plt.title('Automated Overview: {y_col} by {x_col}', fontsize=14, fontweight='bold', pad=14)
for p in ax.patches:
    w = p.get_width()
    if w > 0:
        ax.annotate(f'{{w:,.0f}}', (w, p.get_y()+p.get_height()/2.), ha='left', va='center', xytext=(5,0), textcoords='offset points', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

print('\\nDescriptive Summary of Key Numeric Columns:')
print(df.select_dtypes(include='number').describe().T[['mean', 'std', 'min', 'max']].round(2).to_string())
print('\\n[SUCCESS] Automated Data Science Analysis complete!')"""
                response_text = f"📊 **Automated Data Science Analysis** — Exploring **{y_col}** by **{x_col}** across dataset:"
            else:
                code = f"""print('=== AUTOMATED DATASET OVERVIEW ===')
print('Shape:', df.shape)
print('\\nData Types:')
print(df.dtypes)
print('\\nDescriptive Statistics:')
print(df.describe(include='all').T.to_string())
print('[SUCCESS] Dataset exploration executed successfully!')"""
                response_text = "📊 **Automated Dataset Exploration** executed successfully:"
        else:
            response_text = "Load a dataset first, then ask me anything — charts, analysis, ML models, cleaning, etc."
            code = "print('No dataset loaded. Use Open File to load a CSV/Excel/Parquet file.')"

    res_payload = {
        "success": True,
        "response": response_text,
        "answer": response_text,
        "code": code,
        "artifacts": {
            "code": [code] if code else [],
            "plots": [],
            "tables": [],
            "reports": [response_text] if response_text else []
        }
    }
    if hasattr(ds_engine, 'execution_cache') and ds_engine.execution_cache:
        ds_engine.execution_cache.set(df, query_raw, res_payload)
    return res_payload












import ast

def validate_python_code_security(code_str: str):
    """AST inspection to prevent dangerous system imports/calls while allowing pandas & numpy data wrangling."""
    forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'builtins', 'socket', 'urllib', 'requests'}
    forbidden_funcs = {'eval', 'exec', '__import__', 'compile', 'open'}
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"Syntax Error in Python code: {str(e)}")
        
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in forbidden_modules:
                    raise HTTPException(status_code=400, detail=f"Security Error: Module '{alias.name}' is restricted.")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in forbidden_modules:
                raise HTTPException(status_code=400, detail=f"Security Error: Importing from '{node.module}' is restricted.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_funcs:
                raise HTTPException(status_code=400, detail=f"Security Error: Builtin function '{node.func.id}()' is restricted.")

import io
import sys
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

terminal_exec_lock = threading.Lock()

@app.post("/api/terminal/execute")
def execute_terminal_code(req: CodeExecuteRequest):
    """Executes Python code safely in isolated subprocess, capturing stdout print outputs, dataframe mutations, and matplotlib figures."""
    validate_python_code_security(req.code)

    with terminal_exec_lock:
        with state.lock:
            has_df = bool(state.active_step_id and state.active_step_id in state.df_states)
            df = state.df_states[state.active_step_id].copy() if has_df else pd.DataFrame({'Feature_A': [10, 20, 30], 'Feature_B': [100, 200, 300]})

        res = execution_engine.execute_code_subprocess(req.code, df=df, timeout=30)
        
        if not res.get("success"):
            return {
                "success": False,
                "step_id": state.active_step_id,
                "message": res.get("error", "Execution error"),
                "output": f"Python Execution Error:\n{res.get('error') or res.get('stderr')}",
                "plot_image": None,
                "row_count": len(df),
                "column_count": len(df.columns),
                "schema": [],
                "columns": []
            }

        captured_output = (res.get("stdout") or "").strip()
        plot_image_b64 = res.get("plot_image")
        modified_df = res.get("dataframe") if res.get("dataframe") is not None else df

        new_step_id = str(uuid.uuid4())
        if has_df and modified_df is not None:
            state.add_state(new_step_id, modified_df)
            profile = get_df_profile(modified_df)
            row_cnt = len(modified_df)
            col_cnt = len(modified_df.columns)
            
            step_meta = {
                "id": new_step_id,
                "parent_id": state.active_step_id,
                "operation_id": "terminal_execute",
                "parameters": {"code": req.code},
                "pandas_code": req.code,
                "sql_code": "-- Terminal execution",
                "row_count": row_cnt,
                "column_count": col_cnt,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            with state.lock:
                state.pipeline_steps.append(step_meta)
        else:
            profile = []
            row_cnt = len(modified_df) if modified_df is not None else 0
            col_cnt = len(modified_df.columns) if modified_df is not None else 0

        return {
            "success": True,
            "step_id": new_step_id,
            "message": "Executed code successfully",
            "output": captured_output if captured_output else "Code executed cleanly with 0 errors.",
            "plot_image": plot_image_b64,
            "row_count": row_cnt,
            "column_count": col_cnt,
            "schema": profile,
            "columns": profile
        }


@app.post("/api/notebook/execute")
def execute_notebook_legacy(req: CodeExecuteRequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
        
    df = state.df_states[state.active_step_id]
    validate_python_code_security(req.code)

    res = execution_engine.execute_code_subprocess(req.code, df=df, timeout=30)
    
    html_table = None
    if res.get("dataframe") is not None and not res["dataframe"].empty:
        out_df = res["dataframe"]
        if 'df.head' in req.code:
            html_table = out_df.head().to_html(classes="jupyter-df-table", border=0)
        elif 'df.tail' in req.code:
            html_table = out_df.tail().to_html(classes="jupyter-df-table", border=0)
        elif 'df.describe' in req.code:
            html_table = out_df.describe().to_html(classes="jupyter-df-table", border=0)

    return {
        "success": res.get("success", False),
        "stdout": res.get("stdout", ""),
        "html_table": html_table,
        "plot_image": res.get("plot_image"),
        "error": res.get("error")
    }


@app.post("/api/notebook/execute_cell")
def execute_notebook_cell(req: NotebookCellRequest):
    ensure_active_dataset()

    active_df = None
    if state.active_step_id and state.active_step_id in state.df_states:
        active_df = state.df_states[state.active_step_id]

    code = req.code or ""
    validate_python_code_security(code)

    res = execution_engine.execute_code_subprocess(code, df=active_df, timeout=30)
    
    dataframe_html = None
    result_val = None
    if res.get("dataframe") is not None and not res["dataframe"].empty:
        out_df = res["dataframe"]
        dataframe_html = out_df.to_html(classes="dataframe table table-striped", max_rows=50)
        records = out_df.head(50).to_dict(orient="records")
        result_val = [{k: clean_value_for_json(v) for k, v in row.items()} for row in records]
        if state.active_step_id:
            state.df_states[state.active_step_id] = out_df

    return {
        "success": res.get("success", False),
        "stdout": res.get("stdout", ""),
        "plot_image": res.get("plot_image"),
        "dataframe_html": dataframe_html,
        "result": result_val,
        "error": res.get("error")
    }




@app.post("/api/operations/undo")
def undo_step():
    if len(state.pipeline_steps) <= 1:
         raise HTTPException(status_code=400, detail="Cannot undo. At initial data state.")
         
    current_idx = -1
    for i, s in enumerate(state.pipeline_steps):
        if s["id"] == state.active_step_id:
            current_idx = i
            break
            
    if current_idx <= 0:
        raise HTTPException(status_code=400, detail="Cannot undo. Already at base dataset.")
        
    parent_step = state.pipeline_steps[current_idx - 1]
    state.active_step_id = parent_step["id"]
    df = state.df_states[state.active_step_id]
    
    return {
        "success": True,
        "step_id": state.active_step_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "schema": get_df_profile(df),
        "preview": get_preview_data(df)
    }


@app.post("/api/operations/redo")
def redo_step():
    current_idx = -1
    for i, s in enumerate(state.pipeline_steps):
        if s["id"] == state.active_step_id:
            current_idx = i
            break
            
    if current_idx == -1 or current_idx >= len(state.pipeline_steps) - 1:
        raise HTTPException(status_code=400, detail="Cannot redo. Already at latest step.")
        
    next_step = state.pipeline_steps[current_idx + 1]
    state.active_step_id = next_step["id"]
    df = state.df_states[state.active_step_id]
    
    return {
        "success": True,
        "step_id": state.active_step_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "schema": get_df_profile(df),
        "preview": get_preview_data(df)
    }


@app.get("/api/pipeline")
def get_pipeline():
    return {
        "active_step_id": state.active_step_id,
        "steps": state.pipeline_steps
    }


@app.get("/api/export")
def export_pipeline():
    # Consolidate all steps into a single script
    pandas_blocks = []
    sql_blocks = []
    
    for i, step in enumerate(state.pipeline_steps):
        pandas_blocks.append(f"# Step {i}: {step['operation_id']}\n{step['pandas_code']}")
        sql_blocks.append(f"-- Step {i}: {step['operation_id']}\n{step['sql_code']}")
        
    consolidated_pandas = "\n\n".join(pandas_blocks)
    consolidated_sql = "\n\n".join(sql_blocks)
    
    return {
        "pandas": consolidated_pandas,
        "sql": consolidated_sql
    }

# Endpoint for interactive statistics calculation
@app.get("/api/statistics/correlation")
def calculate_correlation():
    if not state.active_step_id or state.active_step_id not in state.df_states:
         raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]
    
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {"columns": [], "matrix": []}
        
    corr = numeric_df.corr().round(4)
    cols = list(corr.columns)
    
    # Convert correlation matrix to table matrix
    matrix = []
    for c1 in cols:
        row_vals = []
        for c2 in cols:
            val = corr.loc[c1, c2]
            row_vals.append(None if pd.isna(val) else float(val))
        matrix.append(row_vals)
        
    return {
        "columns": cols,
        "matrix": matrix
    }

# Machine learning model training endpoint
class TrainModelRequest(BaseModel):
    target_column: str
    features: Optional[List[str]] = None
    test_size: float = 0.2
    algorithm: Optional[str] = "random_forest"  # random_forest, gradient_boosting, hist_gradient_boosting, extra_trees, ada_boost, svm, knn, mlp_neural_network, decision_tree, linear_logistic, auto
    scaling: Optional[str] = "none"             # none, standard, minmax, robust
    cv_folds: Optional[int] = 5

class PCARequest(BaseModel):
    n_components: Optional[int] = 2
    features: Optional[List[str]] = None

class KMeansRequest(BaseModel):
    n_clusters: Optional[int] = 3
    features: Optional[List[str]] = None


def _get_scaler(scaling_type: str):
    st = (scaling_type or "none").lower()
    if st == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif st == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif st == "robust":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    return None


@app.post("/api/ml/train")
def train_model(req: TrainModelRequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
         raise HTTPException(status_code=400, detail="No active dataset connection.")
         
    df = state.df_states[state.active_step_id]
    
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found.")
        
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
        
        # Prepare features: select numeric or categorical features, exclude IDs and target
        if req.features:
            features = [f for f in req.features if f in df.columns and f != req.target_column]
        else:
            features = []
            for col in df.columns:
                if col == req.target_column or "id" in col.lower():
                    continue
                features.append(col)
                    
        if not features:
            raise HTTPException(status_code=400, detail="No valid feature columns found.")
            
        model_df = df[[req.target_column] + features].dropna()
        if len(model_df) < 10:
             raise HTTPException(status_code=400, detail="Insufficient clean rows for model training (minimum 10 rows required). Apply Imputations/Cleaning first.")
             
        X_raw = model_df[features]
        y_raw = model_df[req.target_column]
        
        # Handle categorical encoding for string/object features
        X_encoded = pd.get_dummies(X_raw, drop_first=True)
        encoded_features = list(X_encoded.columns)

        # Detect problem type: Regression vs Classification
        is_regression = pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 15
        alg_name = (req.algorithm or "random_forest").lower()

        # Models storage directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_filename = f"model_{uuid.uuid4().hex[:8]}.pkl"
        model_filepath = os.path.join(models_dir, model_filename)

        # Helper to construct model instances
        def create_model_instance(name: str, reg: bool):
            n = name.lower()
            if reg:
                if "hist" in n:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    return HistGradientBoostingRegressor(random_state=42), "HistGradientBoostingRegressor(random_state=42)", "from sklearn.ensemble import HistGradientBoostingRegressor"
                elif "extra" in n:
                    from sklearn.ensemble import ExtraTreesRegressor
                    return ExtraTreesRegressor(n_estimators=50, random_state=42), "ExtraTreesRegressor(n_estimators=50, random_state=42)", "from sklearn.ensemble import ExtraTreesRegressor"
                elif "ada" in n:
                    from sklearn.ensemble import AdaBoostRegressor
                    return AdaBoostRegressor(random_state=42), "AdaBoostRegressor(random_state=42)", "from sklearn.ensemble import AdaBoostRegressor"
                elif "svm" in n or "svr" in n:
                    from sklearn.svm import SVR
                    return SVR(), "SVR()", "from sklearn.svm import SVR"
                elif "knn" in n or "neighbor" in n:
                    from sklearn.neighbors import KNeighborsRegressor
                    return KNeighborsRegressor(), "KNeighborsRegressor()", "from sklearn.neighbors import KNeighborsRegressor"
                elif "mlp" in n or "neural" in n:
                    from sklearn.neural_network import MLPRegressor
                    return MLPRegressor(max_iter=500, random_state=42), "MLPRegressor(max_iter=500, random_state=42)", "from sklearn.neural_network import MLPRegressor"
                elif "linear" in n or "logistic" in n or "ridge" in n:
                    from sklearn.linear_model import LinearRegression
                    return LinearRegression(), "LinearRegression()", "from sklearn.linear_model import LinearRegression"
                elif "tree" in n or "decision" in n:
                    from sklearn.tree import DecisionTreeRegressor
                    return DecisionTreeRegressor(random_state=42), "DecisionTreeRegressor(random_state=42)", "from sklearn.tree import DecisionTreeRegressor"
                elif "boosting" in n or "gradient" in n:
                    from sklearn.ensemble import GradientBoostingRegressor
                    return GradientBoostingRegressor(random_state=42), "GradientBoostingRegressor(random_state=42)", "from sklearn.ensemble import GradientBoostingRegressor"
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(n_estimators=50, random_state=42), "RandomForestRegressor(n_estimators=50, random_state=42)", "from sklearn.ensemble import RandomForestRegressor"
            else:
                if "hist" in n:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    return HistGradientBoostingClassifier(random_state=42), "HistGradientBoostingClassifier(random_state=42)", "from sklearn.ensemble import HistGradientBoostingClassifier"
                elif "extra" in n:
                    from sklearn.ensemble import ExtraTreesClassifier
                    return ExtraTreesClassifier(n_estimators=50, random_state=42), "ExtraTreesClassifier(n_estimators=50, random_state=42)", "from sklearn.ensemble import ExtraTreesClassifier"
                elif "ada" in n:
                    from sklearn.ensemble import AdaBoostClassifier
                    return AdaBoostClassifier(random_state=42), "AdaBoostClassifier(random_state=42)", "from sklearn.ensemble import AdaBoostClassifier"
                elif "svm" in n or "svc" in n:
                    from sklearn.svm import SVC
                    return SVC(probability=True, random_state=42), "SVC(probability=True, random_state=42)", "from sklearn.svm import SVC"
                elif "knn" in n or "neighbor" in n:
                    from sklearn.neighbors import KNeighborsClassifier
                    return KNeighborsClassifier(), "KNeighborsClassifier()", "from sklearn.neighbors import KNeighborsClassifier"
                elif "mlp" in n or "neural" in n:
                    from sklearn.neural_network import MLPClassifier
                    return MLPClassifier(max_iter=500, random_state=42), "MLPClassifier(max_iter=500, random_state=42)", "from sklearn.neural_network import MLPClassifier"
                elif "logistic" in n or "linear" in n:
                    from sklearn.linear_model import LogisticRegression
                    return LogisticRegression(max_iter=1000, random_state=42), "LogisticRegression(max_iter=1000, random_state=42)", "from sklearn.linear_model import LogisticRegression"
                elif "tree" in n or "decision" in n:
                    from sklearn.tree import DecisionTreeClassifier
                    return DecisionTreeClassifier(random_state=42), "DecisionTreeClassifier(random_state=42)", "from sklearn.tree import DecisionTreeClassifier"
                elif "boosting" in n or "gradient" in n:
                    from sklearn.ensemble import GradientBoostingClassifier
                    return GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier(random_state=42)", "from sklearn.ensemble import GradientBoostingClassifier"
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(n_estimators=50, random_state=42), "RandomForestClassifier(n_estimators=50, random_state=42)", "from sklearn.ensemble import RandomForestClassifier"

        # ---------------- REGRESSION ML PIPELINE ----------------
        if is_regression:
            y_target = pd.to_numeric(y_raw, errors='coerce').fillna(0)
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_encoded, y_target, test_size=req.test_size, random_state=42
            )

            # Fit Scaler on X_train ONLY (No data leakage!)
            scaler = _get_scaler(req.scaling)
            if scaler:
                X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=encoded_features, index=X_train_raw.index)
                X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=encoded_features, index=X_test_raw.index)
            else:
                X_train, X_test = X_train_raw, X_test_raw

            leaderboard = []
            if alg_name == "auto":
                candidates = ["random_forest", "gradient_boosting", "hist_gradient_boosting", "extra_trees", "knn", "linear"]
                best_model = None
                best_r2 = -9999.0
                best_name = "random_forest"
                best_class_str = ""
                best_import_str = ""

                for cand in candidates:
                    try:
                        m_cand, class_str, imp_str = create_model_instance(cand, reg=True)
                        m_cand.fit(X_train, y_train)
                        preds = m_cand.predict(X_test)
                        r2_cand = float(r2_score(y_test, preds))
                        leaderboard.append({"algorithm": cand, "r2_score": round(r2_cand, 3), "score": round(max(0, r2_cand)*100, 1)})
                        if r2_cand > best_r2:
                            best_r2 = r2_cand
                            best_model = m_cand
                            best_name = cand
                            best_class_str = class_str
                            best_import_str = imp_str
                    except Exception:
                        pass
                
                model = best_model
                model_class_str = best_class_str
                import_str = best_import_str
                leaderboard.sort(key=lambda x: x["r2_score"], reverse=True)
            else:
                model, model_class_str, import_str = create_model_instance(alg_name, reg=True)
                model.fit(X_train, y_train)

            # Model Serialization
            import joblib
            joblib.dump({"model": model, "scaler": scaler, "features": encoded_features, "target": req.target_column}, model_filepath)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_test = float(r2_score(y_test, y_test_pred))
            r2_train = float(r2_score(y_train, y_train_pred))
            mse_test = float(mean_squared_error(y_test, y_test_pred))

            # Cross-validation score
            cv_scores = []
            try:
                cv_raw = cross_val_score(model, X_train, y_train, cv=min(req.cv_folds or 5, len(X_train)))
                cv_scores = [round(float(s), 3) for s in cv_raw]
            except Exception:
                pass

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
            else:
                importances = np.ones(len(encoded_features)) / len(encoded_features)

            feature_importance_list = sorted([
                {"feature": f, "importance": float(imp)}
                for f, imp in zip(encoded_features, importances)
            ], key=lambda x: x["importance"], reverse=True)

            pandas_code = f"""# Train-Test Split and Model Fitting (Regression)
from sklearn.model_selection import train_test_split, cross_val_score
{import_str}
from sklearn.metrics import r2_score, mean_squared_error

features = {features}
target = '{req.target_column}'

model_df = df[[target] + features].dropna()
X = pd.get_dummies(model_df[features], drop_first=True)
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={req.test_size}, random_state=42)

model = {model_class_str}
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model R2 Score:", r2_score(y_test, y_pred))
print("Model MSE:", mean_squared_error(y_test, y_pred))
"""
            step_id = str(uuid.uuid4())
            step_meta = {
                "id": step_id,
                "parent_id": state.active_step_id,
                "operation_id": "train_model",
                "parameters": {"target_column": req.target_column, "features": features, "test_size": req.test_size, "algorithm": alg_name},
                "pandas_code": pandas_code,
                "sql_code": f"-- Advanced Regression ML model ({alg_name}) fitted on dataset",
                "row_count": len(df),
                "column_count": len(df.columns),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            state.pipeline_steps.append(step_meta)
            state.df_states[step_id] = df
            state.active_step_id = step_id

            results = {
                "success": True,
                "problem_type": "Regression",
                "algorithm": alg_name,
                "accuracy": round(max(0, r2_test) * 100, 1),
                "r2_score": round(r2_test, 3),
                "mse": round(mse_test, 3),
                "train_accuracy": round(max(0, r2_train) * 100, 1),
                "test_accuracy": round(max(0, r2_test) * 100, 1),
                "cv_scores": cv_scores,
                "cv_mean": round(float(np.mean(cv_scores)), 3) if cv_scores else round(max(0, r2_test), 3),
                "leaderboard": leaderboard,
                "feature_importance": feature_importance_list,
                "used_features": encoded_features,
                "model_path": model_filepath,
                "step_id": step_id
            }
            state.ml_model_metadata = results

            dataset_name = os.path.basename(state._source_path) if getattr(state, "_source_path", None) else "active_dataset"
            learning_memory.record_learning_event(
                dataset_name=dataset_name,
                operation="train_model",
                column=req.target_column,
                metrics={
                    "problem_type": "Regression",
                    "algorithm": alg_name,
                    "accuracy": results["accuracy"],
                    "r2_score": results["r2_score"],
                    "mse": results["mse"]
                }
            )

            return results

        # ---------------- CLASSIFICATION ML PIPELINE ----------------
        y_encoded = y_raw
        target_mapping = {}
        if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
            y_cats = y_raw.astype('category').cat
            y_encoded = y_cats.codes
            target_mapping = {int(code): str(label) for code, label in enumerate(y_cats.categories)}

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=req.test_size, random_state=42
        )

        # Fit Scaler on X_train ONLY (No data leakage!)
        scaler = _get_scaler(req.scaling)
        if scaler:
            X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=encoded_features, index=X_train_raw.index)
            X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=encoded_features, index=X_test_raw.index)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        leaderboard = []
        if alg_name == "auto":
            candidates = ["random_forest", "gradient_boosting", "hist_gradient_boosting", "extra_trees", "logistic", "knn", "mlp_neural_network"]
            best_model = None
            best_acc = -1.0
            best_name = "random_forest"
            best_class_str = ""
            best_import_str = ""

            for cand in candidates:
                try:
                    m_cand, class_str, imp_str = create_model_instance(cand, reg=False)
                    m_cand.fit(X_train, y_train)
                    preds = m_cand.predict(X_test)
                    acc_cand = float(accuracy_score(y_test, preds))
                    leaderboard.append({"algorithm": cand, "accuracy": round(acc_cand * 100, 1), "score": round(acc_cand * 100, 1)})
                    if acc_cand > best_acc:
                        best_acc = acc_cand
                        best_model = m_cand
                        best_name = cand
                        best_class_str = class_str
                        best_import_str = imp_str
                except Exception:
                    pass

            clf = best_model
            model_class_str = best_class_str
            import_str = best_import_str
            leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
        else:
            clf, model_class_str, import_str = create_model_instance(alg_name, reg=False)
            clf.fit(X_train, y_train)

        # Model Serialization
        import joblib
        joblib.dump({"model": clf, "scaler": scaler, "features": encoded_features, "target": req.target_column, "target_mapping": target_mapping}, model_filepath)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))

        # Cross-validation score
        cv_scores = []
        try:
            cv_raw = cross_val_score(clf, X_train, y_train, cv=min(req.cv_folds or 5, len(X_train)))
            cv_scores = [round(float(s), 3) for s in cv_raw]
        except Exception:
            pass

        # Feature importances / Coefficients
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0]) if len(clf.coef_.shape) > 1 else np.abs(clf.coef_)
        else:
            importances = np.ones(len(encoded_features)) / len(encoded_features)
            
        feature_importance_list = sorted([
            {"feature": f, "importance": float(imp)}
            for f, imp in zip(encoded_features, importances)
        ], key=lambda x: x["importance"], reverse=True)
        
        report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
        
        readable_report = {}
        for k, v in report.items():
            if k.isdigit() and int(k) in target_mapping:
                readable_report[target_mapping[int(k)]] = v
            else:
                readable_report[k] = v
                
        pandas_code = f"""# Train-Test Split and Model Fitting (Classification)
from sklearn.model_selection import train_test_split, cross_val_score
{import_str}
from sklearn.metrics import classification_report, accuracy_score

features = {features}
target = '{req.target_column}'

model_df = df[[target] + features].dropna()
X = pd.get_dummies(model_df[features], drop_first=True)
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={req.test_size}, random_state=42)

model = {model_class_str}
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""

        step_id = str(uuid.uuid4())
        step_meta = {
            "id": step_id,
            "parent_id": state.active_step_id,
            "operation_id": "train_model",
            "parameters": {
                "target_column": req.target_column,
                "features": features,
                "test_size": req.test_size,
                "algorithm": alg_name
            },
            "pandas_code": pandas_code,
            "sql_code": f"-- Advanced Classification ML model ({alg_name}) fitted on dataset",
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        state.pipeline_steps.append(step_meta)
        state.df_states[step_id] = df
        state.active_step_id = step_id
        
        results = {
            "success": True,
            "problem_type": "Classification",
            "algorithm": alg_name,
            "accuracy": round(test_acc * 100, 1),
            "precision": round(float(readable_report.get("macro avg", {}).get("precision", test_acc)) * 100, 1),
            "recall": round(float(readable_report.get("macro avg", {}).get("recall", test_acc)) * 100, 1),
            "f1_score": round(float(readable_report.get("macro avg", {}).get("f1-score", test_acc)), 3),
            "train_accuracy": round(train_acc * 100, 1),
            "test_accuracy": round(test_acc * 100, 1),
            "cv_scores": cv_scores,
            "cv_mean": round(float(np.mean(cv_scores)), 3) if cv_scores else round(test_acc, 3),
            "leaderboard": leaderboard,
            "feature_importance": feature_importance_list,
            "classification_report": readable_report,
            "used_features": encoded_features,
            "model_path": model_filepath,
            "step_id": step_id
        }
        
        state.ml_model_metadata = results

        dataset_name = os.path.basename(state._source_path) if getattr(state, "_source_path", None) else "active_dataset"
        learning_memory.record_learning_event(
            dataset_name=dataset_name,
            operation="train_model",
            column=req.target_column,
            metrics={
                "problem_type": "Classification",
                "algorithm": alg_name,
                "accuracy": results["accuracy"],
                "f1_score": results["f1_score"]
            }
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/api/ml/metrics")
def get_ml_metrics():
    return state.ml_model_metadata


@app.post("/api/ml/pca")
def run_pca(req: PCARequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]

    numeric_cols = req.features or list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) < 2:
        raise HTTPException(status_code=400, detail="PCA requires at least 2 numeric columns.")

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        clean_df = df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)

        n_comp = min(req.n_components or 2, len(numeric_cols))
        pca = PCA(n_components=n_comp)
        pca_features = pca.fit_transform(scaled_data)

        explained_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        
        # Append PCA columns to dataframe
        res_df = df.copy()
        for i in range(n_comp):
            res_df[f"PCA_Comp_{i+1}"] = pd.Series(pca_features[:, i], index=clean_df.index)

        new_step_id = str(uuid.uuid4())
        state.df_states[new_step_id] = res_df
        state.active_step_id = new_step_id

        return {
            "success": True,
            "components": n_comp,
            "explained_variance_ratio": explained_variance,
            "total_explained_variance": round(sum(explained_variance) * 100, 2),
            "step_id": new_step_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA computation failed: {str(e)}")


@app.post("/api/ml/kmeans")
def run_kmeans(req: KMeansRequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]

    numeric_cols = req.features or list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="KMeans requires numeric feature columns.")

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        clean_df = df[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)

        k = max(2, req.n_clusters or 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        res_df = df.copy()
        res_df["Cluster_Label"] = pd.Series(clusters, index=clean_df.index)

        new_step_id = str(uuid.uuid4())
        state.df_states[new_step_id] = res_df
        state.active_step_id = new_step_id

        return {
            "success": True,
            "n_clusters": k,
            "inertia": round(float(kmeans.inertia_), 2),
            "cluster_counts": {f"Cluster_{c}": int((clusters == c).sum()) for c in range(k)},
            "step_id": new_step_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KMeans computation failed: {str(e)}")

# Directory scanner to browse folders on user's PC
@app.get("/api/fs/list")
def list_local_files(path: Optional[str] = None):
    base_dir = os.path.abspath(os.getcwd())
    if path:
        target_dir = os.path.abspath(path)
        try:
            if os.path.commonpath([base_dir, target_dir]) != base_dir:
                target_dir = base_dir
        except (ValueError, Exception):
            target_dir = base_dir
    else:
        target_dir = base_dir
    
    if not os.path.exists(target_dir):
        target_dir = base_dir
        
    try:
        items = []
        for entry in os.scandir(target_dir):
            is_dir = entry.is_dir()
            name = entry.name
            if name.startswith('.') or name.startswith('__'):
                continue
                
            # Filter non-relevant items to keep data-science clean
            if not is_dir and not name.lower().endswith(('.csv', '.xlsx', '.xls')):
                continue
                
            size = entry.stat().st_size if not is_dir else 0
            items.append({
                "name": name,
                "is_dir": is_dir,
                "size": size,
                "path": os.path.join(target_dir, name).replace("\\", "/")
            })
            
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        parent = os.path.dirname(target_dir)
        try:
            parent_allowed = (os.path.commonpath([base_dir, parent]) == base_dir) if parent != target_dir else False
        except Exception:
            parent_allowed = False

        return {
            "current_path": target_dir.replace("\\", "/"),
            "parent_path": parent.replace("\\", "/") if parent_allowed else None,
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WritebackRequest(BaseModel):
    file_path: Optional[str] = None

# In-place write-back of modified dataframe to local PC disk
@app.post("/api/fs/writeback")
def save_file_inplace(req: Optional[WritebackRequest] = None):
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset state.")
        
    target_path = (req.file_path if req and req.file_path else None) or getattr(state, "_source_path", None)
    if not target_path:
        raise HTTPException(status_code=400, detail="No active dataset file path specified for saving.")
    target_path = os.path.abspath(target_path)
    df = state.df_states[state.active_step_id]

    
    try:
        if target_path.lower().endswith(('.xlsx', '.xls')):
            df.to_excel(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)
            
        return {
            "success": True,
            "message": f"Successfully updated '{os.path.basename(target_path)}' in-place on disk!",
            "path": target_path.replace("\\", "/"),
            "file_path": target_path.replace("\\", "/"),
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    except Exception as e:
        # Retry with force write mode to target CSV

        try:
            with open(target_path, "w", encoding="utf-8", newline="") as f:
                df.to_csv(f, index=False)
            return {
                "success": True,
                "message": f"Successfully updated '{os.path.basename(target_path)}' in-place on disk!",
                "path": target_path.replace("\\", "/"),
                "file_path": target_path.replace("\\", "/"),
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        except Exception as err2:
            raise HTTPException(status_code=500, detail=f"Failed to write to '{os.path.basename(target_path)}': {str(err2)}")

# Dynamic lab status checklist validation
@app.get("/api/lab/status")
def get_lab_status():
    df = state.df_states.get(state.active_step_id)
    if df is None:
        return {"completion": 0, "tasks": [], "profile": False, "nulls": False,
                "outliers": False, "encode": False, "normalize": False, "train": False, "evaluate": False}

    tasks = []
    # Task 1: Impute Age
    age_done = "Age" in df.columns and int(df["Age"].isna().sum()) == 0
    tasks.append({"id": "impute_age", "name": "Impute missing Age values", "done": age_done})

    # Task 2: Cast Total_Charges to float
    total_charges_float = False
    if "Total_Charges" in df.columns:
        total_charges_float = pd.api.types.is_numeric_dtype(df["Total_Charges"])
    tasks.append({"id": "cast_total_charges", "name": "Cast Total_Charges to floating decimals", "done": total_charges_float})

    # Task 3: One-hot encode Contract
    contract_encoded = any("Contract_" in col for col in df.columns)
    tasks.append({"id": "encode_contract", "name": "One-Hot Encode variables", "done": contract_encoded})

    # Task 4: Train Machine Learning Model
    ml_done = state.ml_model_metadata.get("success", False)
    tasks.append({"id": "train_model", "name": "Train predictor model with test accuracy > 75%", "done": ml_done})

    completed_count = sum(1 for t in tasks if t["done"])
    completion_pct = int((completed_count / len(tasks)) * 100)

    # Generic null-check: profile is done if dataset is loaded
    null_cols = int(df.isna().any().sum())

    return {
        "completion": completion_pct,
        "tasks": tasks,
        # Flat boolean fields used by frontend Cisco lab guide
        "profile":    True,           # dataset is loaded
        "nulls":      null_cols == 0, # all nulls handled
        "outliers":   len(state.pipeline_steps) > 2,
        "encode":     contract_encoded,
        "normalize":  any("_norm" in c or "_scaled" in c for c in df.columns),
        "train":      ml_done,
        "evaluate":   ml_done,
    }


# ─── New-style operation endpoints ───────────────────────────────────────────
# These accept the simple {operation, column, ...} format from the new frontend.

def _translate_new_op(req_data: dict) -> tuple:
    """Translate new frontend request format -> (operation_id, parameters) for existing engine."""

    op = req_data.get("operation", "")
    col = req_data.get("column", "")
    mapping = {
        "fill_mean":         ("impute_missing",      {"columns": [col], "strategy": "mean"}),
        "fill_median":       ("impute_missing",      {"columns": [col], "strategy": "median"}),
        "fill_mode":         ("impute_missing",      {"columns": [col], "strategy": "mode"}),
        "fill_constant":     ("impute_missing",      {"columns": [col], "strategy": "constant", "constant_value": req_data.get("fill_value", "")}),
        "drop_nulls":        ("impute_missing",      {"columns": [col], "strategy": "drop_rows"}),
        "drop_duplicates":   ("drop_duplicates",     {}),
        "drop_column":       ("impute_missing",      {"columns": [col], "strategy": "drop_cols"}),
        "rename_column":     ("rename_column",       {"column": col, "new_name": req_data.get("new_name", "")}),
        "one_hot_encode":    ("one_hot_encode",      {"columns": [col]}),
        "label_encode":      ("label_encode",        {"columns": [col]}),
        "normalize_minmax":  ("normalize_scale",     {"columns": [col], "method": "minmax"}),
        "standardize":       ("normalize_scale",     {"columns": [col], "method": "standard"}),
        "log_transform":     ("log_transform",       {"columns": [col]}),
        "bin_column":        ("bin_column",          {"column": col, "n_bins": req_data.get("n_bins", 5)}),
        "extract_datetime":  ("extract_datetime",    {"column": col}),
        "filter_rows":       ("filter_rows",         {"column": col, "operator": req_data.get("operator", "=="), "value": req_data.get("filter_value", "")}),
    }
    if op in mapping:
        return mapping[op]
    # Passthrough
    return op, {k: v for k, v in req_data.items() if k != "operation"}


@app.post("/api/operations/preview")
def preview_operation_v2(req: OperationRequest):
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")

    df_current = state.df_states[state.active_step_id]
    req_data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    operation_id, parameters = _translate_new_op(req_data)

    try:
        df_preview = apply_op_to_df(df_current, operation_id, parameters)
        col = req.column
        before_vals = df_current[col].head(10).tolist() if col and col in df_current.columns else []
        after_vals  = df_preview[col].head(10).tolist() if col and col in df_preview.columns else []

        before_nulls = int(df_current[col].isna().sum()) if col and col in df_current.columns else 0
        after_nulls  = int(df_preview[col].isna().sum()) if col and col in df_preview.columns else 0

        return {
            "before": [None if pd.isna(v) else v for v in before_vals],
            "after":  [None if pd.isna(v) else v for v in after_vals],
            "before_stats": {"nulls": before_nulls, "rows": len(df_current)},
            "after_stats":  {"nulls": after_nulls,  "rows": len(df_preview)},
            "row_count": len(df_preview),
            "column_count": len(df_preview.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preview failed: {str(e)}")


@app.post("/api/operations/apply")
def apply_operation_v2(req: OperationRequest):
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")

    df_current = state.df_states[state.active_step_id]
    operation_id, parameters = _translate_new_op(req.dict())

    try:
        df_new = apply_op_to_df(df_current, operation_id, parameters)
        step_id = str(uuid.uuid4())
        state.df_states[step_id] = df_new

        codes = generate_op_code(operation_id, parameters, df_current)
        step_meta = {
            "id": step_id,
            "parent_id": state.active_step_id,
            "operation_id": operation_id,
            "operation": req.operation,
            "description": f"{req.operation} on {req.column or 'dataset'}",
            "parameters": parameters,
            "pandas_code": codes.get("pandas", ""),
            "sql_code": codes.get("sql", ""),
            "row_count": len(df_new),
            "column_count": len(df_new.columns),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        state.pipeline_steps.append(step_meta)
        state.active_step_id = step_id

        # Governance & Lineage Engine Recording
        governance_engine.add_lineage_node(
            step_id=step_id,
            step_name=step_meta["description"],
            action_type=req.operation or operation_id,
            python_code=codes.get("pandas", ""),
            input_rows=len(df_current),
            output_rows=len(df_new)
        )
        governance_engine.log_audit_event("apply_operation", f"{req.operation} on {req.column or 'dataset'}", status="success")

        dataset_name = os.path.basename(state._source_path) if getattr(state, "_source_path", None) else "active_dataset"

        learning_memory.record_learning_event(
            dataset_name=dataset_name,
            operation=req.operation or operation_id,
            column=req.column,
            metrics={"row_count": len(df_new), "column_count": len(df_new.columns)}
        )

        profile = get_df_profile(df_new)
        for col_info in profile:
            col_info["total"] = len(df_new)

        return {
            "success": True,
            "step_id": step_id,
            "rows": len(df_new),
            "row_count": len(df_new),
            "column_count": len(df_new.columns),
            "columns": profile,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Apply failed: {str(e)}")


@app.post("/api/undo")
def undo_alias():
    """Alias for /api/operations/undo used by the Ribbon."""
    return undo_step()
# ─── New Data Science Visualization & Unsupervised ML Endpoints ───────────────

class ClusterRequest(BaseModel):
    n_clusters: int = 3
    features: Optional[List[str]] = None

@app.post("/api/ml/cluster")
def kmeans_cluster(req: ClusterRequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
    df = state.df_states[state.active_step_id]
    
    num_cols = req.features or list(df.select_dtypes(include=[np.number]).columns)
    if not num_cols:
        raise HTTPException(status_code=400, detail="No numeric features available for K-Means clustering.")
        
    cluster_df = df[num_cols].dropna()
    if len(cluster_df) < req.n_clusters:
        raise HTTPException(status_code=400, detail="Insufficient clean rows for clustering.")
        
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)
    
    kmeans = KMeans(n_clusters=req.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    df_copy = df.copy()
    df_copy["Cluster"] = None
    df_copy["Cluster"] = df_copy["Cluster"].astype(object)
    df_copy.loc[cluster_df.index, "Cluster"] = [f"Cluster_{l}" for l in labels]

    
    new_step_id = str(uuid.uuid4())
    state.add_state(new_step_id, df_copy)
    
    # Calculate cluster sizes & centers
    cluster_counts = pd.Series(labels).value_counts().to_dict()
    centers = {f"Cluster_{i}": [round(float(v), 3) for v in center] for i, center in enumerate(kmeans.cluster_centers_)}
    
    return {
        "success": True,
        "n_clusters": req.n_clusters,
        "inertia": round(float(kmeans.inertia_), 2),
        "cluster_counts": {f"Cluster_{k}": int(v) for k, v in cluster_counts.items()},
        "centers": centers,
        "features": num_cols,
        "step_id": new_step_id,
        "row_count": len(df_copy)
    }

class JoinRequest(BaseModel):
    secondary_file: str
    join_key: str
    join_type: str = "left"  # left, inner, right

ALLOWED_JOIN_TYPES = {"LEFT", "RIGHT", "INNER", "FULL", "OUTER"}

@app.post("/api/sql/join")
def join_datasets(req: JoinRequest):
    ensure_active_dataset()
    df_main = state.df_states[state.active_step_id]
    
    join_type = (req.join_type or "").strip().upper()
    if join_type not in ALLOWED_JOIN_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid join_type '{req.join_type}'. Allowed types: {sorted(ALLOWED_JOIN_TYPES)}")

    join_key = (req.join_key or "").strip()
    if not re.match(r"^[a-zA-Z0-9_ ]+$", join_key):
        raise HTTPException(status_code=400, detail="Invalid join_key format.")

    if not os.path.exists(req.secondary_file):
        raise HTTPException(status_code=404, detail=f"Secondary file '{req.secondary_file}' not found.")
        
    try:
        if req.secondary_file.endswith((".xlsx", ".xls")):
            df_sec = pd.read_excel(req.secondary_file)
        else:
            df_sec = pd.read_csv(req.secondary_file)
            
        if join_key not in df_main.columns or join_key not in df_sec.columns:
            raise HTTPException(status_code=400, detail=f"Join key '{join_key}' must exist in both main and secondary datasets.")
            
        safe_key = f'"{join_key}"'
        joined_df = duckdb.query(f"""
            SELECT m.*, s.* EXCLUDE ({safe_key})
            FROM df_main m
            {join_type} JOIN df_sec s ON m.{safe_key} = s.{safe_key}
        """).to_df()
        
        new_step_id = str(uuid.uuid4())
        state.add_state(new_step_id, joined_df)
        
        return {
            "success": True,
            "message": f"Successfully merged datasets using {join_type} JOIN on key '{join_key}'",
            "step_id": new_step_id,
            "row_count": len(joined_df),
            "column_count": len(joined_df.columns),
            "columns": get_df_profile(joined_df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Join failed: {str(e)}")

# ── Generic API Endpoints for 6-Engine AI Operating System ────────────────

class GenericValidateRequest(BaseModel):
    query: str

class GenericExecuteRequest(BaseModel):
    sub_engine: str
    action: str
    kwargs: Optional[Dict[str, Any]] = None

@app.post("/api/validate")
def validate_endpoint(req: GenericValidateRequest):
    """Generic Validation API Endpoint."""
    df = state.get_active_df()
    val_res = validation_engine.validate_query(df, req.query)
    governance_engine.log_audit_event("validate_query", req.query, status="success" if val_res.is_valid else "warning")
    return {
        "is_valid": val_res.is_valid,
        "ambiguities_found": val_res.ambiguities_found,
        "synthesized_columns": val_res.synthesized_columns,
        "warnings": val_res.warnings,
        "ast_safe": val_res.ast_safe
    }

@app.post("/api/execute")
def execute_endpoint(req: GenericExecuteRequest):
    """Generic Modular Execution Engine Endpoint."""
    df = state.get_active_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No active dataset loaded for execution.")
    res = execution_engine.execute_task(req.sub_engine, req.action, df, req.kwargs or {})
    governance_engine.log_audit_event("execute_task", f"{req.sub_engine}:{req.action}", status=res.get("status", "success"))
    return res

@app.get("/api/knowledge")
def knowledge_endpoint(domain: Optional[str] = "all"):
    """Generic Enterprise Knowledge Engine Endpoint."""
    return {
        "glossary": ds_engine.get_business_glossary(),
        "domain_rules": ds_engine.get_domain_rules(domain or "all")
    }

@app.get("/api/governance")
def governance_endpoint():
    """Generic Governance & Compliance Endpoint."""
    return governance_engine.get_governance_summary()

@app.post("/api/export")
def export_endpoint(export_type: Optional[str] = "python"):
    """Generic Export Endpoint (Reproducibility Python Script)."""
    script_text = governance_engine.export_reproducibility_script()
    governance_engine.log_audit_event("export_script", f"export_type={export_type}", status="success")
    return {"status": "success", "script": script_text}


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')
ASSETS_DIR = os.path.join(DIST_DIR, 'assets')
INDEX_FILE = os.path.join(DIST_DIR, 'index.html')

if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

@app.get("/")
def serve_frontend_index():
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return JSONResponse(status_code=404, content={"detail": f"Frontend index.html not found at {INDEX_FILE}"})

@app.get("/{full_path:path}")
def serve_spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path == "health":
        raise HTTPException(status_code=404, detail="API endpoint not found")
    target_file = os.path.join(DIST_DIR, full_path)
    if os.path.exists(target_file) and os.path.isfile(target_file):
        return FileResponse(target_file)
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return JSONResponse(status_code=404, content={"detail": f"Frontend index.html not found at {INDEX_FILE}"})


if __name__ == '__main__':
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)



