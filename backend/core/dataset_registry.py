"""
DataPilot OS — Multi-Dataset Registry.
Singleton that holds all connected datasets in memory by ID.
No dataset is silently switched or auto-joined.
"""
import threading
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import os
import logging

class DatasetEntry(BaseModel):
    dataset_id: str
    name: str
    file_path: str
    format: str
    row_count: int
    column_count: int
    knowledge_model: Dict[str, Any]
    loaded_at: str
    file_hash: str
    initial_df_hash: Optional[str] = None
    is_mutated: bool = False
    source_file_mtime: Optional[float] = None
    source_file_size: Optional[int] = None
    sheet_name: Optional[str] = None
    available_sheets: List[str] = Field(default_factory=list)
    is_source_native: bool = True
    source_storage: str = "source_native"
    drift_detected: bool = False
    drift_reason: Optional[str] = None

    def get_safe_path(self) -> str:
        """Returns safe, user-facing display path without disclosing private filesystem hierarchies."""
        if not self.file_path:
            return ""
        return os.path.basename(self.file_path)

    def get_source_descriptor(self) -> Dict[str, Any]:
        """Provides authoritative source metadata contract for UI badges and execution planner."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "file_path": self.file_path,
            "safe_display_path": self.get_safe_path(),
            "format": self.format,
            "sheet_name": self.sheet_name,
            "available_sheets": self.available_sheets,
            "is_source_native": self.is_source_native,
            "source_storage": self.source_storage,
            "is_mutated": self.is_mutated,
            "drift_detected": self.drift_detected,
            "drift_reason": self.drift_reason,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_hash": self.file_hash,
            "source_file_mtime": self.source_file_mtime,
            "source_file_size": self.source_file_size,
        }

    def check_drift(self) -> bool:
        """Inspects disk file state against recorded mtime, size, and hash without mutating dataset state."""
        if not self.file_path or not self.is_source_native:
            return False
        if not os.path.exists(self.file_path):
            self.drift_detected = True
            self.drift_reason = f"Source file missing or moved: {self.get_safe_path()}"
            return True
        try:
            curr_size = os.path.getsize(self.file_path)
            curr_mtime = os.path.getmtime(self.file_path)
            if self.source_file_size is not None and curr_size != self.source_file_size:
                self.drift_detected = True
                self.drift_reason = f"Source file size changed ({self.source_file_size} -> {curr_size} bytes)"
                return True
            if self.source_file_mtime is not None and abs(curr_mtime - self.source_file_mtime) > 1e-4:
                import hashlib
                sha = hashlib.sha256()
                with open(self.file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha.update(chunk)
                new_hash = sha.hexdigest()
                if self.file_hash and new_hash != self.file_hash:
                    self.drift_detected = True
                    self.drift_reason = "Source file content modified externally on disk"
                    return True
        except Exception as e:
            self.drift_detected = True
            self.drift_reason = f"Error checking source file drift: {e}"
            return True
        self.drift_detected = False
        self.drift_reason = None
        return False

    def get_context(self) -> Dict[str, Any]:
        cols = []
        if isinstance(self.knowledge_model, dict):
            if "columns" in self.knowledge_model:
                c_val = self.knowledge_model["columns"]
                if isinstance(c_val, dict):
                    cols = list(c_val.keys())
                elif isinstance(c_val, list):
                    cols = c_val
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "columns": cols,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "knowledge_model": self.knowledge_model,
            "sheet_name": self.sheet_name,
            "is_source_native": self.is_source_native,
            "source_storage": self.source_storage,
            "safe_display_path": self.get_safe_path()
        }

class DatasetRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self._datasets: Dict[str, DatasetEntry] = {}
        self._active_dataset_id: Optional[str] = None
        self._registry_lock = threading.RLock()
        self._dataframes_cache: Dict[str, pd.DataFrame] = {}

    def register(self, dataset_id: str, entry: Any) -> None:
        with self._registry_lock:
            if isinstance(entry, pd.DataFrame):
                self._dataframes_cache[dataset_id] = entry
                # Sync into relation_catalog if entry exists
                d_entry = self._datasets.get(dataset_id)
                tbl_name = d_entry.name if d_entry else dataset_id
                from backend.sql_engine.relation_catalog import relation_catalog
                relation_catalog.register_dataset(dataset_id=dataset_id, name=tbl_name, df=entry, is_active=(self._active_dataset_id == dataset_id))
            else:
                self._datasets[dataset_id] = entry
                # If dataframe is already cached, sync to relation_catalog
                if dataset_id in self._dataframes_cache:
                    from backend.sql_engine.relation_catalog import relation_catalog
                    relation_catalog.register_dataset(dataset_id=dataset_id, name=entry.name, df=self._dataframes_cache[dataset_id], is_active=True)
            self._active_dataset_id = dataset_id

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        with self._registry_lock:
            return self._datasets.get(dataset_id)

    def get_active(self) -> Optional[DatasetEntry]:
        with self._registry_lock:
            if self._active_dataset_id and self._active_dataset_id in self._datasets:
                return self._datasets[self._active_dataset_id]
            return None

    def set_active(self, dataset_id: str) -> bool:
        with self._registry_lock:
            if dataset_id in self._datasets:
                self._active_dataset_id = dataset_id
                return True
            return False

    def list_all(self) -> List[DatasetEntry]:
        with self._registry_lock:
            return list(self._datasets.values())

    def remove(self, dataset_id: str) -> None:
        with self._registry_lock:
            if dataset_id in self._datasets:
                del self._datasets[dataset_id]
                if self._active_dataset_id == dataset_id:
                    self._active_dataset_id = None
            if dataset_id in self._dataframes_cache:
                del self._dataframes_cache[dataset_id]
            from backend.sql_engine.relation_catalog import relation_catalog
            relation_catalog.unregister_dataset(dataset_id)

    def get_dataframe(self, dataset_id: str) -> Optional[pd.DataFrame]:
        with self._registry_lock:
            if dataset_id in self._dataframes_cache:
                return self._dataframes_cache[dataset_id]
            entry = self.get(dataset_id)
            if not entry:
                return None
                
            try:
                from backend.dataset_engine.loader import dataset_loader
                if entry.format.lower() in ["csv"]:
                    df, _ = dataset_loader.load(entry.file_path)
                    if df is None:
                        df = pd.read_csv(entry.file_path)
                elif entry.format.lower() in ["json"]:
                    df = pd.read_json(entry.file_path)
                elif entry.format.lower() in ["parquet"]:
                    df = pd.read_parquet(entry.file_path)
                elif entry.format.lower() in ["xlsx", "xls"]:
                    df, _ = dataset_loader.load(entry.file_path, sheet_name=entry.sheet_name)
                    if df is None:
                        df = pd.read_excel(entry.file_path, sheet_name=entry.sheet_name)
                else:
                    return None
                    
                self._dataframes_cache[dataset_id] = df
                from backend.sql_engine.relation_catalog import relation_catalog
                relation_catalog.register_dataset(dataset_id=dataset_id, name=entry.name, df=df, is_active=(self._active_dataset_id == dataset_id))
                return df
            except Exception as e:
                logging.error(f"Failed to load dataset {dataset_id}: {str(e)}")
                return None

    def update_dataframe(self, dataset_id: str, new_df: pd.DataFrame) -> bool:
        with self._registry_lock:
            self._dataframes_cache[dataset_id] = new_df
            entry = self._datasets.get(dataset_id)
            if entry:
                entry.is_mutated = True
                entry.row_count = int(len(new_df))
                entry.column_count = int(len(new_df.columns))
                if isinstance(entry.knowledge_model, dict):
                    entry.knowledge_model["columns"] = list(new_df.columns)
                    entry.knowledge_model["row_count"] = int(len(new_df))
                    entry.knowledge_model["column_count"] = int(len(new_df.columns))
                from backend.sql_engine.relation_catalog import relation_catalog
                relation_catalog.register_dataset(dataset_id=dataset_id, name=entry.name, df=new_df, is_active=(self._active_dataset_id == dataset_id))
            return True

    def clear(self) -> None:
        with self._registry_lock:
            self._datasets.clear()
            self._active_dataset_id = None
            self._dataframes_cache.clear()
            from backend.sql_engine.relation_catalog import relation_catalog
            relation_catalog.clear()

dataset_registry = DatasetRegistry()
