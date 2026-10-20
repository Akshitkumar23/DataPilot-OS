"""
DataPilot OS — Phase 7 / FW-04 Governed In-Memory Relation Catalog.
Provides thread-safe multi-relation catalog management for DuckDB and DataPilot OS.
Preserves dataset identities, computes cryptographic fingerprints, enforces table-name
canonicalization without silent overwrites, and maintains 'active_dataset' as an alias.
Zero domain or dataset hardcoding.
"""

import threading
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import duckdb

from backend.ds_engine.dataset_profiler import DatasetProfiler


class RelationCatalogEntry(BaseModel):
    """Immutable metadata descriptor for a registered relation in the Relation Catalog."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_name: str
    dataset_id: str
    row_count: int
    column_count: int
    columns: List[str]
    column_types: Dict[str, str]
    nullable_columns: List[str]
    primary_key_candidates: List[str]
    fingerprint: str
    loaded_at: str


class RelationCatalog:
    """
    Authoritative In-Memory Relation Catalog for DataPilot OS.
    Thread-safe registry mapping canonical SQL table names to DataFrames and metadata.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self._relations: Dict[str, RelationCatalogEntry] = {}  # table_name -> entry
        self._dataframes: Dict[str, pd.DataFrame] = {}         # table_name -> df
        self._dataset_id_to_table: Dict[str, str] = {}         # dataset_id -> table_name
        self._active_table_name: Optional[str] = None
        self._catalog_lock = threading.RLock()

    @staticmethod
    def canonicalize_table_name(name: str) -> str:
        """
        Converts a filename or raw name into a valid, deterministic SQL identifier.
        e.g. 'Customer Orders-2026.csv' -> 'customer_orders_2026'
        """
        base = str(name).strip()
        if "." in base:
            base = base.rsplit(".", 1)[0]
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_").lower()
        cleaned = re.sub(r"_+", "_", cleaned)
        if not cleaned:
            cleaned = "tbl"
        if cleaned[0].isdigit():
            cleaned = f"t_{cleaned}"
        return cleaned

    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        df: pd.DataFrame,
        is_active: bool = False
    ) -> RelationCatalogEntry:
        """
        Registers or updates a dataset DataFrame in the relation catalog.
        Prevents silent overwrites: if a different dataset has the same canonical name,
        appends a deterministic numeric suffix.
        """
        with self._catalog_lock:
            canonical_base = self.canonicalize_table_name(name)
            target_name = canonical_base

            # If this dataset_id was already registered under a table_name, we might update it
            existing_table = self._dataset_id_to_table.get(dataset_id)
            if existing_table and existing_table in self._relations:
                target_name = existing_table
            else:
                # Disambiguate table name collision with different dataset_id
                suffix = 2
                while target_name in self._relations and self._relations[target_name].dataset_id != dataset_id:
                    target_name = f"{canonical_base}_{suffix}"
                    suffix += 1

            # Introspect types and primary key candidates using DuckDB and Pandas
            col_types = {}
            con = None
            try:
                con = duckdb.connect(database=":memory:")
                con.register("tmp_reg", df)
                rel = con.query("SELECT * FROM tmp_reg LIMIT 1")
                for c_name, c_type in zip(rel.columns, rel.types):
                    col_types[str(c_name)] = str(c_type)
            except Exception:
                for c in df.columns:
                    col_types[str(c)] = str(df[c].dtype)
            finally:
                if con:
                    try:
                        con.close()
                    except Exception:
                        pass

            # Detect nullability and PK candidates
            nullable = []
            pk_candidates = []
            n_rows = len(df)
            for c in df.columns:
                null_count = int(df[c].isna().sum())
                if null_count > 0:
                    nullable.append(str(c))
                elif n_rows > 0 and int(df[c].nunique()) == n_rows:
                    pk_candidates.append(str(c))

            fingerprint = DatasetProfiler.compute_hash(df)
            now_iso = datetime.now(timezone.utc).isoformat()

            entry = RelationCatalogEntry(
                table_name=target_name,
                dataset_id=dataset_id,
                row_count=n_rows,
                column_count=len(df.columns),
                columns=[str(c) for c in df.columns],
                column_types=col_types,
                nullable_columns=nullable,
                primary_key_candidates=pk_candidates,
                fingerprint=fingerprint,
                loaded_at=now_iso
            )

            self._relations[target_name] = entry
            self._dataframes[target_name] = df
            self._dataset_id_to_table[dataset_id] = target_name

            if is_active or self._active_table_name is None:
                self._active_table_name = target_name

            return entry

    def unregister_dataset(self, dataset_id: str) -> bool:
        """Removes a dataset from the relation catalog."""
        with self._catalog_lock:
            table_name = self._dataset_id_to_table.get(dataset_id)
            if not table_name:
                return False
            del self._dataset_id_to_table[dataset_id]
            if table_name in self._relations:
                del self._relations[table_name]
            if table_name in self._dataframes:
                del self._dataframes[table_name]
            if self._active_table_name == table_name:
                self._active_table_name = next(iter(self._relations.keys())) if self._relations else None
            return True

    def set_active(self, identifier: str) -> bool:
        """Sets active table by dataset_id or table_name."""
        with self._catalog_lock:
            if identifier in self._relations:
                self._active_table_name = identifier
                return True
            if identifier in self._dataset_id_to_table:
                self._active_table_name = self._dataset_id_to_table[identifier]
                return True
            return False

    def get_active(self) -> Optional[RelationCatalogEntry]:
        """Returns the active relation entry."""
        with self._catalog_lock:
            if self._active_table_name and self._active_table_name in self._relations:
                return self._relations[self._active_table_name]
            return None

    def get_active_dataframe(self) -> Optional[pd.DataFrame]:
        """Returns the active relation DataFrame."""
        with self._catalog_lock:
            if self._active_table_name and self._active_table_name in self._dataframes:
                return self._dataframes[self._active_table_name]
            return None

    def get_relation(self, table_name_or_id: str) -> Optional[RelationCatalogEntry]:
        """Resolves relation by table_name or dataset_id."""
        with self._catalog_lock:
            if table_name_or_id in self._relations:
                return self._relations[table_name_or_id]
            tbl = self._dataset_id_to_table.get(table_name_or_id)
            if tbl and tbl in self._relations:
                return self._relations[tbl]
            return None

    def get_dataframe(self, table_name_or_id: str) -> Optional[pd.DataFrame]:
        """Returns the DataFrame for a given table_name or dataset_id."""
        with self._catalog_lock:
            if table_name_or_id in self._dataframes:
                return self._dataframes[table_name_or_id]
            tbl = self._dataset_id_to_table.get(table_name_or_id)
            if tbl and tbl in self._dataframes:
                return self._dataframes[tbl]
            return None

    def list_relations(self) -> List[RelationCatalogEntry]:
        """Returns a snapshot list of all registered relation entries."""
        with self._catalog_lock:
            return list(self._relations.values())

    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Returns a mapping of table_name -> DataFrame for all relations."""
        with self._catalog_lock:
            return dict(self._dataframes)

    def clear(self) -> None:
        """Clears all catalog entries and dataframes."""
        with self._catalog_lock:
            self._relations.clear()
            self._dataframes.clear()
            self._dataset_id_to_table.clear()
            self._active_table_name = None


relation_catalog = RelationCatalog()
