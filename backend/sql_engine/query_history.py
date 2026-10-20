"""
DataPilot OS — Phase 7 Governed SQL Query History.
Maintains persistent, dataset-partitioned query history for the SQL Studio.
Zero hardcoding: dynamically tracks queries by dataset ID and dataset fingerprint.
"""
import os
import json
from datetime import datetime, timezone
from typing import List, Optional
import threading

from backend.sql_engine.sql_models import SQLQueryHistoryEntry


class SQLQueryHistoryManager:
    """Thread-safe persistent manager for SQL query history."""
    HISTORY_FILE = "backend/data/sql_history.json"

    def __init__(self, data_path: Optional[str] = None):
        target_path = data_path or getattr(self.__class__, "HISTORY_FILE", "backend/data/sql_history.json")
        runtime_data_dir = os.getenv("DATAPILOT_DATA_DIR")
        self.data_path = (
            os.path.join(runtime_data_dir, "sql_history.json")
            if runtime_data_dir else (target_path if os.path.isabs(str(target_path)) else os.path.join(os.getcwd(), str(target_path)))
        )
        self._entries: List[SQLQueryHistoryEntry] = []
        self._lock = threading.RLock()
        self._load_from_disk()

    def _load_from_disk(self):
        with self._lock:
            if os.path.exists(self.data_path):
                try:
                    with open(self.data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self._entries = [SQLQueryHistoryEntry(**item) for item in data]
                except Exception:
                    self._entries = []

    def _persist_to_disk(self):
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                with open(self.data_path, "w", encoding="utf-8") as f:
                    json.dump([e.model_dump() for e in self._entries], f, indent=2)
            except Exception:
                pass

    def add_entry(
        self,
        query_id: str,
        dataset_id: str,
        dataset_fingerprint: str,
        sql: str,
        execution_time_ms: float,
        status: str,
        row_count: int,
        error: Optional[str] = None
    ) -> SQLQueryHistoryEntry:
        """Record an executed query in history."""
        entry = SQLQueryHistoryEntry(
            query_id=query_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            sql=sql,
            execution_time_ms=round(execution_time_ms, 2),
            status=status.upper(),
            row_count=row_count,
            error=error
        )
        with self._lock:
            # Maintain last 500 entries
            self._entries.insert(0, entry)
            if len(self._entries) > 500:
                self._entries = self._entries[:500]
            self._persist_to_disk()
        return entry

    @classmethod
    def get_history(cls, limit: int = 50, dataset_id: Optional[str] = None) -> List[Any]:
        """Retrieve recent query history, optionally filtered by dataset."""
        target_file = getattr(cls, "HISTORY_FILE", None)
        if target_file and str(target_file) != str(query_history_manager.data_path):
            query_history_manager.data_path = str(target_file)
            query_history_manager._load_from_disk()
        entries = query_history_manager._get_entries(dataset_id=dataset_id, limit=limit)
        return [e.model_dump() for e in entries]

    def _get_entries(self, dataset_id: Optional[str] = None, limit: int = 50) -> List[SQLQueryHistoryEntry]:
        with self._lock:
            if not dataset_id:
                return self._entries[:limit]
            filtered = [e for e in self._entries if e.dataset_id == dataset_id]
            return filtered[:limit]

    def clear_history(self, dataset_id: Optional[str] = None) -> None:
        """Clear history for a specific dataset or entirely."""
        with self._lock:
            if dataset_id:
                self._entries = [e for e in self._entries if e.dataset_id != dataset_id]
            else:
                self._entries = []
            self._persist_to_disk()

    # Alias for convenience
    get_history_entries = get_history
    get_entries = get_history
    get_entries = get_history


# Singleton instance
query_history_manager = SQLQueryHistoryManager()
