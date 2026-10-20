"""
DataPilot OS — Multi-Format Dataset Loader.
Loads CSV, XLSX, XLS, JSON, Parquet, SQLite, DuckDB formats.
Returns DataLoadResult with dataframe + metadata.
Never silently modifies data — always reports what was done.
"""

import os
import hashlib
import pandas as pd
import sqlite3
import duckdb
from enum import Enum
from typing import Optional, Tuple, List, Any
from pydantic import BaseModel, Field

class SupportedFormat(str, Enum):
    CSV = "CSV"
    XLSX = "XLSX"
    XLS = "XLS"
    JSON = "JSON"
    PARQUET = "PARQUET"
    SQLITE = "SQLITE"
    DUCKDB = "DUCKDB"

class DataLoadResult(BaseModel):
    success: bool
    format: SupportedFormat
    row_count: int = 0
    column_count: int = 0
    file_path: str
    file_size_bytes: int = 0
    file_hash: str = ""
    encoding: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    sheet_name: Optional[str] = None
    available_sheets: List[str] = Field(default_factory=list)

class DatasetLoader:
    def detect_format(self, file_path: str) -> SupportedFormat:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return SupportedFormat.CSV
        elif ext == '.xlsx':
            return SupportedFormat.XLSX
        elif ext == '.xls':
            return SupportedFormat.XLS
        elif ext == '.json':
            return SupportedFormat.JSON
        elif ext in ['.parquet', '.pq']:
            return SupportedFormat.PARQUET
        elif ext in ['.sqlite', '.db', '.sqlite3']:
            return SupportedFormat.SQLITE
        elif ext in ['.duckdb', '.ddb']:
            return SupportedFormat.DUCKDB
        else:
            # Default to CSV if unknown
            return SupportedFormat.CSV

    def get_excel_sheets(self, file_path: str) -> List[str]:
        """Discovers all worksheet names in an Excel workbook (.xlsx or .xls) without full dataset loading."""
        if not os.path.exists(file_path):
            return []
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.xlsx':
                import openpyxl
                wb = openpyxl.load_workbook(file_path, read_only=True)
                sheets = list(wb.sheetnames)
                wb.close()
                return sheets
            elif ext == '.xls':
                import xlrd
                wb = xlrd.open_workbook(file_path, on_demand=True)
                return list(wb.sheet_names())
        except Exception:
            return []
        return []

    def compute_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""

    def load(self, file_path: str, sheet_name: Optional[str] = None, table_name: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], DataLoadResult]:
        result = DataLoadResult(success=False, format=SupportedFormat.CSV, file_path=file_path)
        try:
            if not os.path.exists(file_path):
                result.error = f"File not found: {file_path}"
                return None, result

            result.file_size_bytes = os.path.getsize(file_path)
            result.file_hash = self.compute_file_hash(file_path)
            result.format = self.detect_format(file_path)

            df = None
            if result.format == SupportedFormat.CSV:
                df, enc, warns, err = self._load_csv(file_path)
                result.encoding = enc
                result.warnings.extend(warns)
                if err: result.error = err
            elif result.format in [SupportedFormat.XLSX, SupportedFormat.XLS]:
                df, warns, err, loaded_sheet, sheets = self._load_excel(file_path, sheet_name)
                result.warnings.extend(warns)
                result.sheet_name = loaded_sheet
                result.available_sheets = sheets
                if err: result.error = err
            elif result.format == SupportedFormat.JSON:
                df, warns, err = self._load_json(file_path)
                result.warnings.extend(warns)
                if err: result.error = err
            elif result.format == SupportedFormat.PARQUET:
                df, warns, err = self._load_parquet(file_path)
                result.warnings.extend(warns)
                if err: result.error = err
            elif result.format == SupportedFormat.SQLITE:
                df, warns, err = self._load_sqlite(file_path, table_name)
                result.warnings.extend(warns)
                if err: result.error = err
            elif result.format == SupportedFormat.DUCKDB:
                df, warns, err = self._load_duckdb(file_path, table_name)
                result.warnings.extend(warns)
                if err: result.error = err

            if df is not None:
                result.success = True
                result.row_count = len(df)
                result.column_count = len(df.columns)
            else:
                if not result.error:
                    result.error = "Failed to load data, dataframe is None."
            
            return df, result
        except Exception as e:
            result.error = str(e)
            return None, result

    def _load_csv(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str], Optional[str]]:
        encodings = ['utf-8', 'latin-1', 'cp1252']
        warnings = []
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                if len(df.columns) == 1 and len(df) > 0:
                    first_col = str(df.columns[0])
                    if ';' in first_col or '\t' in first_col or '|' in first_col:
                        try:
                            df_sniff = pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
                            if len(df_sniff.columns) > 1:
                                warnings.append("Auto-detected non-comma CSV delimiter.")
                                return df_sniff, enc, warnings, None
                        except Exception:
                            pass
                return df, enc, warnings, None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
                    warnings.append(f"Loaded with python CSV parser engine due to: {e}")
                    return df, enc, warnings, None
                except Exception:
                    pass
                return None, None, warnings, str(e)
        return None, None, warnings, "Failed to decode CSV with common encodings."

    def _load_excel(self, file_path: str, sheet_name: Optional[str]) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str], Optional[str], List[str]]:
        warnings = []
        sheets = self.get_excel_sheets(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'

        target_sheet = sheet_name
        if not target_sheet and sheets:
            target_sheet = sheets[0]

        if sheets and target_sheet and target_sheet not in sheets:
            return None, warnings, f"Sheet '{target_sheet}' not found in workbook. Available sheets: {sheets}", None, sheets

        try:
            if target_sheet:
                df = pd.read_excel(file_path, sheet_name=target_sheet, engine=engine)
            else:
                df = pd.read_excel(file_path, engine=engine)
                if hasattr(df, 'name'):
                    target_sheet = df.name

            if df is not None and df.empty:
                warnings.append(f"Sheet '{target_sheet}' is empty.")

            return df, warnings, None, target_sheet, sheets
        except Exception as e:
            return None, warnings, str(e), target_sheet, sheets

    def _load_json(self, file_path: str) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
        warnings = []
        try:
            df = pd.read_json(file_path)
            return df, warnings, None
        except Exception:
            try:
                df = pd.read_json(file_path, lines=True)
                return df, warnings, None
            except Exception as e:
                return None, warnings, str(e)

    def _load_parquet(self, file_path: str) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
        warnings = []
        try:
            df = pd.read_parquet(file_path)
            return df, warnings, None
        except Exception as e:
            return None, warnings, str(e)

    def _load_sqlite(self, file_path: str, table_name: Optional[str]) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
        warnings = []
        try:
            conn = sqlite3.connect(file_path)
            if not table_name:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    return None, warnings, "No tables found in SQLite database."
                table_name = tables[0][0]
                warnings.append(f"No table_name provided, loaded first table: {table_name}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df, warnings, None
        except Exception as e:
            return None, warnings, str(e)

    def _load_duckdb(self, file_path: str, table_name: Optional[str]) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
        warnings = []
        try:
            conn = duckdb.connect(file_path)
            if not table_name:
                tables = conn.execute("SHOW TABLES").fetchall()
                if not tables:
                    return None, warnings, "No tables found in DuckDB database."
                table_name = tables[0][0]
                warnings.append(f"No table_name provided, loaded first table: {table_name}")
            df = conn.execute(f"SELECT * FROM {table_name}").df()
            conn.close()
            return df, warnings, None
        except Exception as e:
            return None, warnings, str(e)

dataset_loader = DatasetLoader()
