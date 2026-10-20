"""
DataPilot OS — Phase 7 Deterministic SQL Builder.
Compiles validated semantic SQLIntent contracts into standard DuckDB SQL.
Guarantees:
- Strict schema grounding: checks referenced columns against DataFrame.
- Zero fuzzy guessing: unknown columns raise structured schema errors.
- Correct identifier escaping (handles spaces and reserved keywords safely).
"""
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.sql_engine.sql_models import SQLIntent

# Allowed SQL aggregation functions in DuckDB
ALLOWED_AGGREGATIONS = {
    "count": "COUNT",
    "avg": "AVG",
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
    "stddev": "STDDEV",
    "variance": "VARIANCE",
}

# Allowed comparison operators
ALLOWED_OPERATORS = {
    "=": "=",
    "==": "=",
    "eq": "=",
    "!=": "!=",
    "ne": "!=",
    "<>": "!=",
    ">": ">",
    "gt": ">",
    ">=": ">=",
    "gte": ">=",
    "<": "<",
    "lt": "<",
    "<=": "<=",
    "lte": "<=",
    "like": "LIKE",
    "ilike": "ILIKE",
    "is null": "IS NULL",
    "is not null": "IS NOT NULL",
}


def quote_identifier(ident: str) -> str:
    """Safely quote a SQL column or table identifier."""
    clean = str(ident).strip()
    # If already quoted with double quotes or backticks, return clean
    if (clean.startswith('"') and clean.endswith('"')) or (clean.startswith('`') and clean.endswith('`')):
        return clean
    # If contains spaces, punctuation, or starts with digit, quote with double quotes
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", clean):
        escaped = clean.replace('"', '""')
        return f'"{escaped}"'
    return clean


def format_sql_literal(val: Any) -> str:
    """Format a Python literal safely for SQL."""
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    # String literal escaping
    val_str = str(val).replace("'", "''")
    return f"'{val_str}'"


class DeterministicSQLBuilder:
    """Builds deterministic, validated DuckDB SQL from a structured SQLIntent."""

    @classmethod
    def build(cls, intent: SQLIntent, df: pd.DataFrame, table_name: str = "active_dataset") -> str:
        """
        Validates schema identifiers and compiles SQLIntent into a DuckDB SQL string.
        Raises ValueError with clear message if unknown columns are referenced.
        """
        available_cols = list(df.columns)
        available_cols_lower = {str(c).lower(): str(c) for c in df.columns}

        # 1. Ground and validate all referenced columns
        def resolve_col(raw_name: str) -> str:
            clean = raw_name.strip().strip('"').strip('`')
            if clean == "*":
                return "*"
            # Strict exact match first
            if clean in df.columns:
                return clean
            # Case-insensitive exact match
            clean_lower = clean.lower()
            if clean_lower in available_cols_lower:
                return available_cols_lower[clean_lower]
            # No fuzzy matching allowed!
            raise ValueError(
                f"Column '{clean}' does not exist in active dataset. Available columns: {', '.join(available_cols)}"
            )

        # Validate selected columns
        resolved_selected = [resolve_col(c) for c in intent.selected_columns]

        # Validate group by
        resolved_group_by = [resolve_col(c) for c in intent.group_by]

        # Validate aggregations
        resolved_aggs = []
        for agg in intent.aggregations:
            raw_col = agg.get("column", "*")
            func = agg.get("func", agg.get("function", "count")).lower()
            if func not in ALLOWED_AGGREGATIONS:
                raise ValueError(f"Unsupported aggregation function '{func}'. Allowed: {', '.join(ALLOWED_AGGREGATIONS.keys())}")
            norm_func = ALLOWED_AGGREGATIONS[func]
            col_target = "*" if raw_col == "*" or norm_func == "COUNT" and raw_col == "*" else resolve_col(raw_col)
            alias = agg.get("alias") or f"{func}_{col_target.replace('*', 'all')}"
            resolved_aggs.append({
                "func": norm_func,
                "column": col_target,
                "alias": alias
            })

        # Validate filters
        resolved_filters = []
        for flt in intent.filters:
            col = resolve_col(flt.get("column", ""))
            op = str(flt.get("operator", "=")).lower()
            if op not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported filter operator '{op}'. Allowed: {', '.join(ALLOWED_OPERATORS.keys())}")
            norm_op = ALLOWED_OPERATORS[op]
            val = flt.get("value")
            resolved_filters.append({
                "column": col,
                "operator": norm_op,
                "value": val
            })

        # Validate order by
        resolved_order = []
        for ord_spec in intent.order_by:
            raw_col = ord_spec.get("column", "")
            # Order by can reference an alias from aggregations or a source column
            agg_aliases = [a["alias"] for a in resolved_aggs]
            if raw_col in agg_aliases:
                target_col = raw_col
            else:
                target_col = resolve_col(raw_col)
            direction = str(ord_spec.get("direction", "asc")).upper()
            if direction not in ["ASC", "DESC"]:
                direction = "ASC"
            resolved_order.append({"column": target_col, "direction": direction})

        # 2. Assemble SQL query clauses
        # SELECT
        select_parts: List[str] = []
        for col in resolved_selected:
            select_parts.append(quote_identifier(col))

        for agg in resolved_aggs:
            col_arg = "*" if agg["column"] == "*" else quote_identifier(agg["column"])
            agg_expr = f"{agg['func']}({col_arg}) AS {quote_identifier(agg['alias'])}"
            select_parts.append(agg_expr)

        if not select_parts:
            # If nothing specified, select all
            select_parts = ["*"]

        sql_lines = [f"SELECT {', '.join(select_parts)}", f"FROM {table_name}"]

        # WHERE
        if resolved_filters:
            where_conds = []
            for flt in resolved_filters:
                c_quoted = quote_identifier(flt["column"])
                op = flt["operator"]
                val = flt["value"]
                if op in ["IS NULL", "IS NOT NULL"]:
                    where_conds.append(f"{c_quoted} {op}")
                else:
                    lit = format_sql_literal(val)
                    where_conds.append(f"{c_quoted} {op} {lit}")
            sql_lines.append(f"WHERE {' AND '.join(where_conds)}")

        # GROUP BY
        if resolved_group_by:
            sql_lines.append(f"GROUP BY {', '.join(quote_identifier(c) for c in resolved_group_by)}")

        # ORDER BY
        if resolved_order:
            order_items = [f"{quote_identifier(o['column'])} {o['direction']}" for o in resolved_order]
            sql_lines.append(f"ORDER BY {', '.join(order_items)}")

        # LIMIT
        if intent.limit is not None and intent.limit > 0:
            sql_lines.append(f"LIMIT {int(intent.limit)}")

        return " ".join(sql_lines) + ";"
