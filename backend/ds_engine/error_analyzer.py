"""
DataPilot OS — Execution Error Analyzer.
Classifies Python runtime errors, identifies root causes,
and prepares structured repair proposals for the Repair Engine.

This is NOT a fallback. It is an evidence-driven repair system.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
import re

class ErrorCategory(str, Enum):
    KEY_ERROR = "KEY_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    VALUE_ERROR = "VALUE_ERROR"
    ATTRIBUTE_ERROR = "ATTRIBUTE_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    TIMEOUT = "TIMEOUT"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    UNKNOWN = "UNKNOWN"

class ErrorAnalysis(BaseModel):
    category: ErrorCategory
    root_cause: str
    affected_line: Optional[int] = None
    affected_symbol: Optional[str] = None
    suggested_fix: str
    available_alternatives: List[str]
    is_repairable: bool
    repair_type: str

class ErrorAnalyzer:
    def analyze(self, error_output: str, code: str, dataset_context: dict) -> ErrorAnalysis:
        columns = dataset_context.get("columns", [])
        line_num = self._extract_error_line(error_output)
        
        if "KeyError" in error_output:
            match = re.search(r"KeyError: ['\"](.*?)['\"]", error_output)
            key = match.group(1) if match else "Unknown"
            sim_cols = self._find_similar_columns(key, columns)
            return ErrorAnalysis(
                category=ErrorCategory.KEY_ERROR,
                root_cause=f"Column '{key}' not found.",
                affected_line=line_num,
                affected_symbol=key,
                suggested_fix=f"Replace '{key}' with similar column.",
                available_alternatives=sim_cols,
                is_repairable=True,
                repair_type="column_name"
            )
            
        elif "ImportError" in error_output or "ModuleNotFoundError" in error_output:
            return ErrorAnalysis(
                category=ErrorCategory.IMPORT_ERROR,
                root_cause="Missing package import.",
                affected_line=line_num,
                affected_symbol=None,
                suggested_fix="Check ALLOWED_IMPORTS.",
                available_alternatives=[],
                is_repairable=False,
                repair_type="import"
            )
            
        elif "TypeError" in error_output:
            return ErrorAnalysis(
                category=ErrorCategory.TYPE_ERROR,
                root_cause="Type mismatch.",
                affected_line=line_num,
                affected_symbol=None,
                suggested_fix="Cast type appropriately.",
                available_alternatives=[],
                is_repairable=True,
                repair_type="type_cast"
            )
            
        return ErrorAnalysis(
            category=ErrorCategory.UNKNOWN,
            root_cause="Unknown error occurred.",
            affected_line=line_num,
            affected_symbol=None,
            suggested_fix="Needs Ollama analysis.",
            available_alternatives=[],
            is_repairable=True,
            repair_type="unknown"
        )

    def _find_similar_columns(self, column_name: str, available_columns: List[str]) -> List[str]:
        # Simple case-insensitive matching for demonstration
        similar = []
        for col in available_columns:
            if column_name.lower() in col.lower() or col.lower() in column_name.lower():
                similar.append(col)
        return similar[:3]

    def _extract_error_line(self, error_output: str) -> Optional[int]:
        matches = re.findall(r"line (\d+)", error_output)
        if matches:
            return int(matches[-1])
        return None

error_analyzer = ErrorAnalyzer()
