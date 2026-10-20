"""
DataPilot OS — Phase 7 Governed SQL Models.
Typed Pydantic contracts for DuckDB validation, execution, intent, and history.
Strictly schema-driven: zero dataset-specific hardcoding, zero silent fallbacks.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SQLValidationResult(BaseModel):
    """Structured response from the SQL validator."""
    model_config = ConfigDict(extra="forbid")
    status: str = "VALID"  # "VALID" | "INVALID"
    query: str
    referenced_columns: List[str] = Field(default_factory=list)
    referenced_tables: List[str] = Field(default_factory=list)
    estimated_operation: str = "SELECTION"  # "SELECTION" | "AGGREGATION" | "FILTER" | "WINDOW" | "CTE" | "JOIN"
    error_code: Optional[str] = None
    message: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == "VALID"


class SQLQueryResult(BaseModel):
    """Structured response from a governed DuckDB query execution."""
    model_config = ConfigDict(extra="forbid")
    query_id: str
    dataset_id: str
    dataset_fingerprint: str
    sql: str
    status: str = "SUCCEEDED"  # "SUCCEEDED" | "FAILED"
    columns: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    row_count: int = 0
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    page: int = 1
    page_size: int = 50
    total_pages: int = 1
    execution_time_ms: float = 0.0
    explain_plan: Optional[str] = None
    execution_mode: str = "in_memory_dataframe"  # "direct_file_scan" | "in_memory_dataframe"
    pushdown_supported: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class SQLIntent(BaseModel):
    """Typed semantic intent contract for Natural-Language-to-SQL planning."""
    model_config = ConfigDict(extra="forbid")
    operation: str = "select"  # select | aggregate | filter | window | order
    selected_columns: List[str] = Field(default_factory=list)
    aggregations: List[Dict[str, str]] = Field(default_factory=list)  # e.g. [{"column": "x", "func": "avg", "alias": "avg_x"}]
    group_by: List[str] = Field(default_factory=list)
    filters: List[Dict[str, Any]] = Field(default_factory=list)  # e.g. [{"column": "x", "operator": ">", "value": 10}]
    order_by: List[Dict[str, str]] = Field(default_factory=list)  # e.g. [{"column": "x", "direction": "desc"}]
    limit: Optional[int] = None
    time_grain: Optional[str] = None  # e.g. "year", "month", "day"
    requested_output: Optional[str] = None


class SQLQueryHistoryEntry(BaseModel):
    """Persistent local query history entry for governed SQL operations."""
    model_config = ConfigDict(extra="forbid")
    query_id: str
    timestamp: str
    dataset_id: str
    dataset_fingerprint: str
    sql: str
    execution_time_ms: float
    status: str
    row_count: int
    error: Optional[str] = None
