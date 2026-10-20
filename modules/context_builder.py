"""
Context Builder module for DataPilot OS.
Assembles rich query context including dataset schema, column types, conversation memory,
and capabilities dynamically fetched from capability_registry to pass to Ollama Semantic Brain.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from modules.capability_registry import capability_registry

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Constructs a comprehensive context payload for Ollama semantic planning.
    Source of truth for available capabilities is capability_registry.
    """

    @classmethod
    def build_context(
        cls,
        query_raw: str,
        df: Optional[pd.DataFrame] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        available_capabilities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        schema_info: Dict[str, Any] = {}

        if df is not None and not df.empty:
            columns_meta = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                sample_vals = [str(val) for val in df[col].dropna().head(3).tolist()]
                col_meta = {
                    "column_name": str(col),
                    "dtype": col_type,
                    "sample_values": sample_vals,
                    "null_count": int(df[col].isnull().sum()),
                    "unique_count": int(df[col].nunique())
                }
                columns_meta.append(col_meta)

            schema_info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": columns_meta
            }

        # Dynamically fetch available capabilities from CapabilityRegistry
        if available_capabilities is None:
            try:
                available_capabilities = capability_registry.list_capabilities()
            except Exception:
                available_capabilities = ["eda", "forecast", "ml", "clean", "sql", "group_aggregate"]

        context = {
            "query": query_raw,
            "schema": schema_info,
            "conversation_memory": conversation_history or [],
            "available_capabilities": available_capabilities
        }
        return context
