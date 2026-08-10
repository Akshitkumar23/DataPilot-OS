"""
Validation Engine & Ambiguity Guard for DataPilot OS.

This module acts as an execution guard to prevent AI hallucinations, resolve schema
ambiguities (e.g., auto-deriving missing metrics like Profit = Revenue - Cost), and enforce
AST code security before execution.
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class ValidationResult:
    """Dataclass holding validation evaluation details."""
    is_valid: bool
    ambiguities_found: List[str] = field(default_factory=list)
    synthesized_columns: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    ast_safe: bool = True
    suggested_fix: Optional[str] = None
    query_plan: Dict[str, Any] = field(default_factory=dict)


class ValidationEngine:
    """
    Validation Engine & Schema Ambiguity Guard.
    Validates queries, resolves missing metrics, checks AST security,
    and guards against execution errors or hallucinations.
    """

    def __init__(self) -> None:
        logger.info("Initializing ValidationEngine")
        # Formula mappings for common business metrics synthesis
        self._metric_formulas = {
            "profit": {
                "inputs": [["revenue", "sales", "gross_revenue"], ["cost", "expense", "expenses", "cogs"]],
                "formula": lambda df, r, c: df[r] - df[c],
                "expr_str": "{revenue_col} - {cost_col}"
            },
            "margin": {
                "inputs": [["revenue", "sales"], ["cost", "expense", "expenses"]],
                "formula": lambda df, r, c: (df[r] - df[c]) / df[r].replace(0, pd.NA),
                "expr_str": "({revenue_col} - {cost_col}) / {revenue_col}"
            },
            "conversion_rate": {
                "inputs": [["conversions", "orders", "purchases"], ["visits", "clicks", "sessions"]],
                "formula": lambda df, conv, vis: df[conv] / df[vis].replace(0, pd.NA),
                "expr_str": "{conv_col} / {vis_col}"
            }
        }

    def validate_query(self, df: Optional[pd.DataFrame], raw_query: str, parsed_intent: Optional[Any] = None) -> ValidationResult:
        """
        Validates a query against the current dataset schema.

        Args:
            df (pd.DataFrame): Active dataset dataframe.
            raw_query (str): Raw string query from user.
            parsed_intent (Any): Parsed intent object or dict.

        Returns:
            ValidationResult: Detailed validation result with ambiguity fixes.
        """
        result = ValidationResult(is_valid=True)

        if df is None or df.empty:
            result.is_valid = False
            result.warnings.append("No active dataset loaded. Please ingest data first.")
            return result

        columns_lower = {col.lower(): col for col in df.columns}
        query_lower = raw_query.lower()

        # Check for missing metric synthesis (e.g. Profit, Margin)
        for metric_name, details in self._metric_formulas.items():
            if metric_name in query_lower and metric_name not in columns_lower:
                # Try to synthesize
                rev_match = next((columns_lower[c] for c in details["inputs"][0] if c in columns_lower), None)
                cost_match = next((columns_lower[c] for c in details["inputs"][1] if c in columns_lower), None)

                if rev_match and cost_match:
                    expr = details["expr_str"].format(revenue_col=rev_match, cost_col=cost_match, conv_col=rev_match, vis_col=cost_match)
                    result.synthesized_columns[metric_name] = f"{metric_name.capitalize()} derived as ({expr})"
                    result.ambiguities_found.append(f"Column '{metric_name}' missing. Resolved by calculating: {expr}")
                else:
                    result.warnings.append(f"Metric '{metric_name}' requested but required source columns were missing.")

        # AST Safety Guard check
        ast_safe, ast_err = self.validate_ast_safety(raw_query)
        if not ast_safe:
            result.is_valid = False
            result.ast_safe = False
            result.warnings.append(f"AST Security Interception: {ast_err}")

        return result

    def resolve_missing_metrics(self, df: pd.DataFrame, requested_metric: str) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Derives a missing metric column on the DataFrame if possible.

        Returns:
            Tuple[pd.DataFrame, Optional[str]]: (updated_df, new_column_name_or_none)
        """
        if df is None or df.empty:
            return df, None

        columns_lower = {col.lower(): col for col in df.columns}
        metric_lower = requested_metric.lower()

        if metric_lower in columns_lower:
            return df, columns_lower[metric_lower]

        if metric_lower in self._metric_formulas:
            details = self._metric_formulas[metric_lower]
            rev_match = next((columns_lower[c] for c in details["inputs"][0] if c in columns_lower), None)
            cost_match = next((columns_lower[c] for c in details["inputs"][1] if c in columns_lower), None)

            if rev_match and cost_match:
                new_col = metric_lower.capitalize()
                df[new_col] = details["formula"](df, rev_match, cost_match)
                logger.info(f"Derived new column '{new_col}' using formula ({rev_match} - {cost_match})")
                return df, new_col

        return df, None

    def validate_ast_safety(self, code_snippet: str) -> Tuple[bool, str]:
        """
        Validates Python code snippets against security AST constraints.
        Prevents execution of malicious system commands, file destruction, etc.
        """
        forbidden_calls = {"exec", "eval", "__import__", "compile", "open"}
        forbidden_modules = {"os", "subprocess", "sys", "shutil", "socket", "pickle", "pty", "importlib"}

        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_modules:
                            return False, f"Forbidden module import: '{alias.name}'"
                elif isinstance(node, ast.ImportFrom):
                    if node.module in forbidden_modules:
                        return False, f"Forbidden module import: '{node.module}'"
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                        return False, f"Forbidden function call: '{node.func.id}'"
            return True, "AST Security Validated"
        except SyntaxError:
            # Query is not valid python code (likely natural language), which is safe to pass to LLM/Parsers
            return True, "Natural Language Query"
        except Exception as e:
            return False, f"AST Parse Error: {str(e)}"


# Singleton instance
validation_engine = ValidationEngine()
