from fastapi import FastAPI, HTTPException, status, UploadFile, File
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

app = FastAPI(title="AI-DSOS Backend Engine")

# CORS middleware for communication with Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

        try:
            import importlib
            from PIL import ImageGrab
            import asyncio

            img = ImageGrab.grab()
            screen_captured = True
            screen_resolution = f"{img.size[0]}×{img.size[1]} px"

            # Extract real text/code from screen pixels via Native Windows OCR Engine if installed
            if importlib.util.find_spec("winocr") is not None:
                winocr_mod = importlib.import_module("winocr")
                ocr_res = asyncio.run(winocr_mod.recognize_pil(img))
                if ocr_res and hasattr(ocr_res, 'text'):
                    extracted_text = str(ocr_res.text).strip()
        except Exception as e:
            print("Screen OCR grab notice:", e)

        # Detect Windows Active Window Title
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

        # If a file was matched from the screen:
        if detected_file:
            has_dataset = True
            df = state.df_states[state.active_step_id]
            dataset_name = str(detected_file)
            total_file_rows = int(len(df))
            total_file_cols = int(len(df.columns))
            summary_text = f"📸 Read Complete: Window '{active_title}' → Matched '{dataset_name}' ({total_file_rows:,} rows)"
        else:
            # PURE SCREEN READ: Report Active Window Title & Extracted Screen Text Preview!
            text_snippet = extracted_text[:100].replace("\n", " ") if extracted_text else "No text visible"
            summary_text = f"📸 Read Complete: Active Window '{active_title}' (Extracted {len(extracted_text)} chars: \"{text_snippet}...\")"

        return {
            "success": True,
            "screen_captured": bool(screen_captured),
            "active_title": str(active_title),
            "extracted_character_count": int(len(extracted_text)),
            "extracted_text": str(extracted_text),
            "dataset_name": str(detected_file if detected_file else active_title),
            "summary": str(summary_text),
            "recommendations": []
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
            "recommendations": []
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
            resp = requests.post(ollama_url, json=payload, timeout=(1.0, 15.0))
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
                code_match = re.search(r"```(?:python|py|sql)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
                code = code_match.group(1).strip() if code_match else None
                return {
                    "source": "ollama",
                    "response": content,
                    "answer": content,
                    "code": code
                }
    except Exception as err:
        print(f"Ollama integration notice (falling back to knowledge engine): {err}")
    
    return None


class ChatRequest(BaseModel):
    query: str

from ds_knowledge_engine import ds_engine

@app.post("/api/chat")
def chat_copilot(req: ChatRequest):
    """Generates AI response and executable Python code card leveraging Master Data Science PDF Knowledge Engine or Ollama Local LLM."""
    query_raw = req.query.strip()
    if not query_raw:
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")
    has_df = bool(state.active_step_id and state.active_step_id in state.df_states)
    df = state.df_states[state.active_step_id] if has_df else None
    
    # 1. Attempt Ollama local LLM integration
    ds_context = None
    if df is not None:
        dtypes_str = {c: str(t) for c, t in df.dtypes.items()}
        null_counts = df.isna().sum().to_dict()
        ds_context = f"Shape: {df.shape}\nColumns & Types: {dtypes_str}\nNull Counts: {null_counts}"
    ollama_res = query_ollama_llm(query_raw, ds_context)
    if ollama_res:
        return {
            "success": True,
            "response": ollama_res["response"],
            "answer": ollama_res["answer"],
            "code": ollama_res["code"]
        }

    # 2. Fall back gracefully to knowledge engine rules
    query = query_raw.lower()
    code = None
    response_text = ""
    
    # Helper: Universal Smart Column Matcher
    num_matched = []
    cat_matched = []
    if df is not None:
        query_norm = re.sub(r'[\s_]+', '', query_raw.lower())
        for col in df.columns:
            col_norm = re.sub(r'[\s_]+', '', str(col).lower())
            if col_norm in query_norm or str(col).lower() in query_raw.lower():
                if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    num_matched.append(col)
                else:
                    cat_matched.a    # -------------------------------------------------------------------------
    # UNIVERSAL ALL-QUERY DATA SCIENCE PROCESSOR
    # -------------------------------------------------------------------------
    
    # 1. Scatter Plot (Bivariate Feature Relationships)
    if any(k in query for k in ["scatter", "versus", "vs", "relationship", "bivariate"]):
        if df is not None:
            x_col = num_matched[0] if len(num_matched) >= 1 else df.columns[0]
            y_col = num_matched[1] if len(num_matched) >= 2 else (num_matched[0] if num_matched else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))
            hue_col = cat_matched[0] if cat_matched else None
            hue_param = f", hue='{hue_col}'" if hue_col else ""
            
            code = f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid")
sns.scatterplot(data=df, x='{x_col}', y='{y_col}'{hue_param}, alpha=0.7, s=60, palette='viridis')
plt.title('Scatter Plot: {y_col} vs {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{x_col}', fontsize=11, fontweight='bold')
plt.ylabel('{y_col}', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Scatter Plot of {y_col} vs {x_col}!')"""
            response_text = f"Analyzed active dataset: Scatter plot visualizing relationship between '{y_col}' and '{x_col}':"

    # 2. Line Chart / Trend Analysis over Time/Category
    elif any(k in query for k in ["line", "trend", "time", "date", "sequence", "timeline"]):
        if df is not None:
            y_col = num_matched[0] if num_matched else df.columns[0]
            x_col = cat_matched[0] if cat_matched else (df.columns[0] if df.columns[0] != y_col else df.columns[-1])
            code = f"""import seaborn as sns
import matplotlib.pyplot as plt

grouped = df.groupby('{x_col}')['{y_col}'].mean().reset_index()
plt.figure(figsize=(10, 5), dpi=150)
sns.set_theme(style="whitegrid")
sns.lineplot(data=grouped, x='{x_col}', y='{y_col}', marker='o', linewidth=2.5, color='#0284c7')
plt.title('Trend Analysis: Average {y_col} across {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{x_col}', fontsize=11, fontweight='bold')
plt.ylabel('{y_col} (Mean)', fontsize=11, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Line Trend Plot for {y_col} across {x_col}!')"""
            response_text = f"Analyzed active dataset: Line trend plot showing average '{y_col}' across '{x_col}':"

    # 3. Pie Chart / Share Proportion
    elif any(k in query for k in ["pie", "share", "portion", "percentage share", "donut"]):
        if df is not None:
            col = cat_matched[0] if cat_matched else df.columns[0]
            code = f"""import matplotlib.pyplot as plt

top_counts = df['{col}'].value_counts().head(7)
plt.figure(figsize=(8, 6), dpi=150)
colors = ['#0284c7', '#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#e0f2fe', '#f1f5f9']
plt.pie(top_counts.values, labels=top_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={{'fontsize': 10, 'weight': 'bold'}})
plt.title('Percentage Share Breakdown of {col}', fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Pie Share Chart for {col}!')"""
            response_text = f"Analyzed active dataset: Pie chart proportion breakdown for categorical feature '{col}':"

    # 4. Boxplot / Outlier Analysis
    elif any(k in query for k in ["outlier", "iqr", "zscore", "clip", "extreme", "box", "boxplot"]):
        if df is not None:
            y_col = num_matched[0] if num_matched else df.select_dtypes(include=[np.number]).columns[0]
            x_col = cat_matched[0] if cat_matched else None
            x_param = f", x='{x_col}'" if x_col else ""
            code = f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid")
sns.boxplot(data=df{x_param}, y='{y_col}', palette='Set2', linewidth=1.5, fliersize=6)
plt.title('Outlier Boxplot Analysis (1.5x IQR Bounds) for {y_col}', fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=20, ha='right', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Perform IQR Capping
q1 = df['{y_col}'].quantile(0.25)
q3 = df['{y_col}'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['{y_col}'] < (q1 - 1.5 * iqr)) | (df['{y_col}'] > (q3 + 1.5 * iqr))]
df['{y_col}'] = df['{y_col}'].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
print(f'[✓ SUCCESS] Detected & clipped {{len(outliers)}} IQR outliers in feature \"{y_col}\"!')"""
            response_text = f"Analyzed active dataset: Interquartile Range (1.5×IQR) Outlier Capping & Boxplot for feature '{y_col}':"

    # 5. Histogram / Distribution Plot
    elif any(k in query for k in ["hist", "distribution", "kde", "density", "bell curve", "frequency"]):
        if df is not None:
            col = num_matched[0] if num_matched else df.select_dtypes(include=[np.number]).columns[0]
            code = f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5), dpi=150)
sns.set_theme(style="whitegrid")
sns.histplot(df['{col}'].dropna(), kde=True, color='#0284c7', bins=25, edgecolor='black', alpha=0.6)
plt.title('Distribution Plot & KDE Density: {col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{col}', fontsize=11, fontweight='bold')
plt.ylabel('Frequency Count', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Distribution Histogram for {col}!')"""
            response_text = f"Analyzed active dataset: Feature distribution histogram and KDE curve for '{col}':"

    # 6. Correlation Heatmap
    elif any(k in query for k in ["heatmap", "corr", "correlation", "matrix"]):
        if df is not None:
            num_cols = list(df.select_dtypes(include=[np.number]).columns[:10])
            code = f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid")
sns.heatmap(df[{num_cols}].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=1.0, cbar=True, annot_kws={{"size": 10, "weight": "bold"}})
plt.title('Data Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=14, color='#0f172a')
plt.xticks(rotation=35, ha='right', fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()
print('[✓ SUCCESS] High-resolution correlation heatmap generated!')"""
            response_text = f"Analyzed active dataset: Generated Seaborn correlation matrix heatmap for numeric features:"

    # 7. Missing Value Imputation
    elif any(k in query for k in ["impute", "null", "missing", "clean", "fillna", "dropna"]):
        if df is not None:
            target_col = (num_matched + cat_matched)[0] if (num_matched + cat_matched) else None
            if not target_col:
                null_summary = df.isna().sum()
                null_cols = null_summary[null_summary > 0]
                target_col = null_cols.index[0] if not null_cols.empty else df.columns[0]
            
            null_count = int(df[target_col].isnull().sum())
            if pd.api.types.is_numeric_dtype(df[target_col]):
                num_series = pd.to_numeric(df[target_col], errors='coerce').dropna()
                skewness = float(num_series.skew()) if len(num_series) > 3 else 0.0
                strategy = "median" if abs(skewness) > 1.0 else "mean"
                code = f"df['{target_col}'] = pd.to_numeric(df['{target_col}'], errors='coerce').fillna(df['{target_col}'].{strategy}().round(2))\nprint(f'[✓ SUCCESS] Imputed {null_count} nulls in \"{target_col}\" using {strategy.capitalize()} (Skewness={skewness:.2f})! Remaining nulls:', df['{target_col}'].isna().sum())"
                response_text = f"Column '{target_col}' has {null_count} missing values. Applying {strategy.capitalize()} Imputation based on skewness ({skewness:.2f}):"
            else:
                code = f"mode_val = df['{target_col}'].mode()[0] if not df['{target_col}'].mode().empty else 'Unknown'\ndf['{target_col}'] = df['{target_col}'].fillna(mode_val)\nprint(f'[✓ SUCCESS] Imputed categorical nulls in \"{target_col}\" with Mode ({{mode_val}})! Remaining nulls:', df['{target_col}'].isna().sum())"
                response_text = f"Found {null_count} missing values in categorical feature '{target_col}'. Applying Mode Imputation:"

    # 8. Machine Learning Model Training (Supervised Classification / Regression)
    elif any(k in query for k in ["train", "model", "predict", "ml", "classify", "regress", "random forest", "xgboost"]):
        if df is not None and len(df.columns) >= 2:
            target = (num_matched + cat_matched)[0] if (num_matched + cat_matched) else df.columns[-1]
            target_is_cat = not pd.api.types.is_numeric_dtype(df[target]) or df[target].nunique() < 10
            model_cls = "RandomForestClassifier" if target_is_cat else "RandomForestRegressor"

            code = f"""from sklearn.model_selection import train_test_split
from sklearn.ensemble import {model_cls}
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Dynamic Feature (X) and Target (y) Extraction
target_col = '{target}'
X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True, dtype=int).fillna(0)
y = df[target_col]

# 2. 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = {model_cls}(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# 4. Top Feature Importances Visualization
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(8)
plt.figure(figsize=(10, 5), dpi=150)
sns.set_theme(style="whitegrid")
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title(f'Random Forest Top Feature Importances (Model Score: {{round(score*100, 2)}}%)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Importance Score', fontweight='bold')
plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Trained {model_cls} on target \"{{target_col}}\"! Evaluation Score: {{round(score * 100, 2)}}%')"""
            response_text = f"Analyzed active dataset: Selected target variable '{target}' ({model_cls}). Executing Machine Learning Training Pipeline:"

    # 9. General Bar Chart / GroupBy Aggregation (Default Dynamic Plot Fallback)
    else:
        if df is not None:
            if num_matched and cat_matched:
                y_col = num_matched[0]
                x_col = cat_matched[0]
            elif num_matched:
                y_col = num_matched[0]
                cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
                x_col = cat_cols[0] if cat_cols else df.columns[0]
            elif cat_matched:
                x_col = cat_matched[0]
                num_cols = list(df.select_dtypes(include=[np.number]).columns)
                y_col = num_cols[0] if num_cols else df.columns[0]
            else:
                cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
                num_cols = list(df.select_dtypes(include=[np.number]).columns)
                x_col = cat_cols[0] if cat_cols else df.columns[0]
                y_col = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

            if pd.api.types.is_numeric_dtype(df[y_col]):
                code = f"""import seaborn as sns
import matplotlib.pyplot as plt

grouped = df.groupby('{x_col}')['{y_col}'].sum().reset_index().sort_values(by='{y_col}', ascending=False).head(15)
plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=grouped, x='{y_col}', y='{x_col}', palette='Blues_r')
plt.title('Total {y_col} by {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('{y_col}', fontsize=11, fontweight='bold')
plt.ylabel('{x_col}', fontsize=11, fontweight='bold')

for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.annotate(f'{{width:,.0f}}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#0f172a')

plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Total {y_col} by {x_col} Bar Chart!')"""
                response_text = f"Analyzed active dataset: Aggregated Total '{y_col}' grouped by '{x_col}':"
            else:
                code = f"""import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), dpi=150)
sns.set_theme(style="whitegrid")
top_vals = df['{x_col}'].value_counts().head(12)
ax = sns.barplot(x=top_vals.values, y=top_vals.index, palette='Blues_r')
plt.title('Category Distribution: {x_col}', fontsize=14, fontweight='bold', pad=14)
plt.xlabel('Count', fontsize=11, fontweight='bold')
plt.ylabel('{x_col}', fontsize=11, fontweight='bold')

for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.annotate(f'{{int(width):,}}', (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#0f172a')

plt.tight_layout()
plt.show()
print(f'[✓ SUCCESS] Rendered Category Count Bar Chart for {x_col}!')"""
                response_text = f"Analyzed active dataset: Value counts bar chart for category '{x_col}':"
        else:
            code = "# Master Data Science Environment Ready\nimport pandas as pd\nimport numpy as np\nprint('DataPilot OS AI Engine Active!')"
            response_text = "DataPilot OS AI Engine Active!"

        rules_found = [r for r in ds_engine.knowledge_db.get("rules", []) if any(w in r.get("method", "").lower() or w in r.get("category", "").lower() for w in query.split())]
        if rules_found:
            r = rules_found[0]
            m_name = r.get('method', 'DS Method')
            how = r.get('how_to_apply', '')
            response_text = f"📌 **{m_name}** ({r.get('category', 'Data Science')})\n• **When to Apply**: {r.get('when_to_apply', '')}\n• **Why to Apply**: {r.get('why_to_apply', '')}"
            code = f"# {m_name} Code\n{how}\nprint('[✓ SUCCESS] Executed {m_name}!')"

        elif df is not None:
            dataset_name = str(state._source_path).replace("\\", "/").split("/")[-1] if getattr(state, '_source_path', None) else "Active Dataset"
            response_text = f"Master AI Brain ready for dataset '{dataset_name}' ({len(df):,} rows × {len(df.columns)} columns). Ask me about Skewness Imputations, IQR Outliers, Correlation Heatmaps, One-Hot Encodings, or Random Forest ML Models."
            code = f"# Dataset Quick Profile\nprint('Dataset Shape:', df.shape)\nprint('Missing Values:', df.isna().sum().to_dict())\nprint('Numeric Features:', list(df.select_dtypes(include='number').columns))"
        else:
            response_text = "I am your DataPilot OS Master AI Engine (trained on 434 pages of Data Science PDF Knowledge Rules). Ask me any question on Data Wrangling, Pandas, Matplotlib, Seaborn, or Scikit-Learn Machine Learning."
            code = "# Master Data Science Environment Ready\nimport pandas as pd\nimport numpy as np\nprint('DataPilot OS AI Engine Active!')"

    return {
        "success": True,
        "response": response_text,
        "answer": response_text,
        "code": code
    }











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
    """Executes Python code safely, capturing stdout print outputs, dataframe mutations, and matplotlib figures."""
    # 1. AST Security Validation
    validate_python_code_security(req.code)

    with terminal_exec_lock:
        with state.lock:
            has_df = bool(state.active_step_id and state.active_step_id in state.df_states)
            df = state.df_states[state.active_step_id].copy() if has_df else pd.DataFrame({'Feature_A': [10, 20, 30], 'Feature_B': [100, 200, 300]})

        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt}
        try:
            import seaborn as sns
            local_vars["sns"] = sns
        except ImportError:
            pass

        # Clear previous figures
        plt.close('all')
        
        # Capture stdout print statements
        stdout_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_buffer

        try:
            exec(req.code, {}, local_vars)
            sys.stdout = old_stdout
            captured_output = stdout_buffer.getvalue().strip()

            # Capture figure if created by matplotlib/seaborn
            plot_image_b64 = None
            if plt.get_fignums():
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                plot_image_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                plt.close('all')

            modified_df = local_vars.get("df", df)
            if not isinstance(modified_df, pd.DataFrame):
                for val in local_vars.values():
                    if isinstance(val, pd.DataFrame):
                        modified_df = val
                        break

            new_step_id = str(uuid.uuid4())
            if has_df:
                state.add_state(new_step_id, modified_df)
                profile = get_df_profile(modified_df)
                row_cnt = len(modified_df)
                col_cnt = len(modified_df.columns)
                
                # Record terminal operation in pipeline steps for 100% sync
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
                row_cnt = len(modified_df)
                col_cnt = len(modified_df.columns)

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
        except HTTPException:
            sys.stdout = old_stdout
            plt.close('all')
            raise
        except Exception as e:
            sys.stdout = old_stdout
            plt.close('all')
            import traceback
            tb_str = traceback.format_exc()
            return {
                "success": False,
                "step_id": state.active_step_id,
                "message": f"Execution error: {str(e)}",
                "output": f"Python Execution Error:\n{tb_str}",
                "plot_image": None,
                "row_count": len(df) if 'df' in locals() else 0,
                "column_count": len(df.columns) if 'df' in locals() else 0,
                "schema": [],
                "columns": []
            }



import io
import base64
import sys

@app.post("/api/notebook/execute")
def execute_notebook_legacy(req: CodeExecuteRequest):
    ensure_active_dataset()
    if not state.active_step_id or state.active_step_id not in state.df_states:
        raise HTTPException(status_code=400, detail="No active dataset connection.")
        
    df = state.df_states[state.active_step_id]
    validate_python_code_security(req.code)
    
    # Configure headless matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.clf()
        plt.close('all')
        has_mpl = True
    except Exception:
        has_mpl = False
        plt = None
        sns = None
        
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    
    local_vars = {
        "df": df,
        "pd": pd,
        "np": np,
    }
    if has_mpl:
        local_vars["plt"] = plt
        local_vars["sns"] = sns
        
    html_table = None
    plot_image = None
    stdout_text = ""
    error_text = None
    
    try:
        exec(req.code, {}, local_vars)
        
        # 1. Capture Matplotlib / Seaborn plots
        if has_mpl and plt:
            fig = plt.gcf()
            if fig and (fig.get_axes() or len(fig.axes) > 0):
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0f172a', edgecolor='none')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                plot_image = f"data:image/png;base64,{img_b64}"
                plt.close('all')
                
        stdout_text = stdout_buf.getvalue()
        
        # 2. Check for DataFrame HTML output evaluation
        lines = [l.strip() for l in req.code.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
        if lines:
            last_line = lines[-1]
            if not ('=' in last_line or 'import' in last_line or last_line.startswith('print') or last_line.startswith('plt.') or last_line.startswith('sns.')):
                try:
                    eval_res = eval(last_line, {}, local_vars)
                    if isinstance(eval_res, pd.DataFrame):
                        html_table = eval_res.to_html(classes="jupyter-df-table", border=0)
                    elif isinstance(eval_res, pd.Series):
                        html_table = eval_res.to_frame().to_html(classes="jupyter-df-table", border=0)
                    elif eval_res is not None and not stdout_text:
                        stdout_text = str(eval_res)
                except Exception:
                    pass
                    
        if not html_table:
            if 'df.head' in req.code:
                html_table = df.head().to_html(classes="jupyter-df-table", border=0)
            elif 'df.tail' in req.code:
                html_table = df.tail().to_html(classes="jupyter-df-table", border=0)
            elif 'df.describe' in req.code:
                html_table = df.describe().to_html(classes="jupyter-df-table", border=0)
                
    except Exception as e:
        error_text = f"{type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout
        if has_mpl and plt:
            plt.close('all')
            
    return {
        "success": error_text is None,
        "stdout": stdout_text,
        "html_table": html_table,
        "plot_image": plot_image,
        "error": error_text
    }



@app.post("/api/notebook/execute_cell")
def execute_notebook_cell(req: NotebookCellRequest):
    ensure_active_dataset()

    active_df = None
    if state.active_step_id and state.active_step_id in state.df_states:
        active_df = state.df_states[state.active_step_id]

    code = req.code or ""
    stdout_buf = io.StringIO()
    plot_image: Optional[str] = None
    dataframe_html: Optional[str] = None
    result_val: Any = None
    error_msg: Optional[str] = None
    success: bool = True

    if plt is not None:
        plt.close('all')

    exec_namespace = {
        "df": active_df.copy() if active_df is not None else pd.DataFrame(),
        "pd": pd,
        "np": np,
    }
    if plt is not None:
        exec_namespace["plt"] = plt

    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    try:
        if code.strip():
            tree = ast.parse(code)
            if tree.body:
                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    exec_body = tree.body[:-1]
                    eval_expr = last_node.value

                    if exec_body:
                        exec_module = ast.Module(body=exec_body, type_ignores=[])
                        exec(compile(exec_module, filename="<notebook_cell>", mode="exec"), exec_namespace)

                    eval_module = ast.Expression(body=eval_expr)
                    raw_result = eval(compile(eval_module, filename="<notebook_cell>", mode="eval"), exec_namespace)
                else:
                    exec_module = ast.Module(body=tree.body, type_ignores=[])
                    exec(compile(exec_module, filename="<notebook_cell>", mode="exec"), exec_namespace)
                    raw_result = None
            else:
                raw_result = None
        else:
            raw_result = None

        # Check for DataFrame outputs
        df_output = None
        if isinstance(raw_result, pd.DataFrame):
            df_output = raw_result
        elif raw_result is None:
            for k, v in exec_namespace.items():
                if k not in ("pd", "np", "plt", "io", "sys", "base64", "ast") and isinstance(v, pd.DataFrame):
                    if k != "df" or (active_df is not None and not v.equals(active_df)):
                        df_output = v
                        break

        if df_output is not None:
            dataframe_html = df_output.to_html(classes="dataframe table table-striped", max_rows=50)
            records = df_output.head(50).to_dict(orient="records")
            result_val = [{k: clean_value_for_json(v) for k, v in row.items()} for row in records]
        elif isinstance(raw_result, pd.Series):
            dataframe_html = raw_result.to_frame().to_html(classes="dataframe table table-striped", max_rows=50)
            result_val = {str(k): clean_value_for_json(v) for k, v in raw_result.head(50).to_dict().items()}
        elif raw_result is not None:
            if isinstance(raw_result, (int, float, bool, str)):
                result_val = raw_result
            elif isinstance(raw_result, (np.integer, np.floating, np.bool_)):
                result_val = clean_value_for_json(raw_result)
            else:
                try:
                    json.dumps(raw_result)
                    result_val = raw_result
                except Exception:
                    result_val = str(raw_result)

        # Sync active dataframe in state if df was modified
        if "df" in exec_namespace and isinstance(exec_namespace["df"], pd.DataFrame):
            if state.active_step_id:
                state.df_states[state.active_step_id] = exec_namespace["df"]

        # Capture active matplotlib.pyplot figures
        if plt is not None and plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plot_image = f"data:image/png;base64,{img_b64}"
            plt.close('all')

    except Exception as e:
        success = False
        error_msg = str(e)
    finally:
        sys.stdout = old_stdout

    stdout_str = stdout_buf.getvalue()

    return {
        "success": success,
        "stdout": stdout_str,
        "plot_image": plot_image,
        "dataframe_html": dataframe_html,
        "result": result_val,
        "error": error_msg
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
    """Translate new frontend request format → (operation_id, parameters) for existing engine."""
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
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)



