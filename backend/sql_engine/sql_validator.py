"""
DataPilot OS — Phase 7 Governed SQL Validator.
Validates DuckDB SQL queries against the active dataset schema.
Enforces read-only execution contract: rejects mutations, filesystem writes, and unknown identifiers.
Zero hardcoding, zero guessing, zero silent fallbacks.
"""
import re
from typing import List, Optional, Set, Tuple
import pandas as pd
import duckdb

from backend.sql_engine.sql_models import SQLValidationResult
from backend.sql_engine.relation_catalog import relation_catalog

# Prohibited statement prefixes and mutative keywords
FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "ALTER",
    "ATTACH",
    "DETACH",
    "COPY",
    "INSTALL",
    "LOAD",
    "INSERT",
    "CREATE",
    "REPLACE",
    "PRAGMA",
    "CALL",
    "CHECKPOINT",
    "FORCE_DROP",
    "EXPORT",
    "IMPORT",
]

# Regex pattern matching any forbidden statement as the primary command or within multiple statements
FORBIDDEN_PATTERN = re.compile(
    r"(?:^|;\s*|\b)(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE
)


class SQLValidator:
    """Validates SQL statements against DuckDB engine rules and dataset schema."""

    @classmethod
    def check_forbidden_operations(cls, sql: str) -> Optional[Tuple[str, str]]:
        """
        Check if the SQL contains forbidden mutative or administration statements.
        Returns (error_code, message) if forbidden, otherwise None.
        """
        clean = sql.strip()
        # Remove comments (-- and /* */)
        clean = re.sub(r"--.*$", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL).strip()

        # Check for multiple statements separated by semicolon where any contains forbidden operations
        statements = [s.strip() for s in clean.split(";") if s.strip()]
        for stmt in statements:
            tokens = re.findall(r"[A-Za-z_]+", stmt)
            if not tokens:
                continue
            first_token = tokens[0].upper()
            if first_token in FORBIDDEN_KEYWORDS:
                return (
                    "MUTATION_BLOCKED",
                    f"Forbidden SQL operation '{first_token}'. DataPilot SQL Studio is strictly analytical and read-only (SELECT, WITH, CTEs allowed; mutating commands are blocked)."
                )
            # Scan for dangerous commands anywhere in statement
            for kw in FORBIDDEN_KEYWORDS:
                if re.search(r"\b" + kw + r"\b", stmt, re.IGNORECASE):
                    # Special exception: if 'CREATE' or 'REPLACE' is inside a comment or string literal, we already cleaned comments.
                    # But if COPY is used (e.g. COPY ... TO ...), block it:
                    if kw in ["COPY", "ATTACH", "DETACH", "INSTALL", "LOAD", "DROP", "DELETE", "UPDATE", "ALTER", "INSERT"]:
                        return (
                            "MUTATION_BLOCKED",
                            f"Forbidden SQL operation '{kw}'. Mutating operations and external file writes are blocked."
                        )
        return None

    @classmethod
    def validate(
        cls,
        sql: str,
        df: pd.DataFrame,
        dataset_name: Optional[str] = None
    ) -> SQLValidationResult:
        """
        Perform complete validation:
        1. Check for empty SQL.
        2. Enforce read-only contract.
        3. Parse and plan using DuckDB in-memory connection with DataFrame view.
        4. Detect unknown columns and format structured error with available columns.
        5. Extract referenced columns and estimated analytical operation.
        """
        raw_sql = (sql or "").strip()
        if not raw_sql:
            return SQLValidationResult(
                status="INVALID",
                query=raw_sql,
                error_code="EMPTY_QUERY",
                message="SQL query cannot be empty."
            )

        # 1. Enforce read-only / safety contract
        forbidden = cls.check_forbidden_operations(raw_sql)
        if forbidden:
            err_code, err_msg = forbidden
            return SQLValidationResult(
                status="INVALID",
                query=raw_sql,
                error_code=err_code,
                message=err_msg
            )

        # 2. Setup isolated in-memory DuckDB connection for validation
        try:
            con = duckdb.connect(database=":memory:")
            # Register all catalog tables in validation connection
            catalog_dfs = relation_catalog.get_all_dataframes()
            for tbl_name, tbl_df in catalog_dfs.items():
                con.register(tbl_name, tbl_df)

            # Register active dataset relations
            con.register("active_dataset", df)
            con.register("dataset", df)
            con.register("df", df)
            if dataset_name:
                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_name.split(".")[0]).strip("_")
                if safe_name and not safe_name.isdigit() and safe_name not in catalog_dfs:
                    con.register(safe_name, df)

            # 3. Test query planning with EXPLAIN
            con.execute(f"EXPLAIN {raw_sql}")

        except duckdb.ParserException as pe:
            msg = str(pe).strip()
            # Clean up internal parser traceback noise if present
            clean_msg = re.sub(r"LINE \d+:.*", "", msg).strip() or msg
            return SQLValidationResult(
                status="INVALID",
                query=raw_sql,
                error_code="PARSE_ERROR",
                message=f"SQL syntax error: {clean_msg}"
            )
        except duckdb.BinderException as be:
            err_text = str(be).strip()
            # Extract unknown column if referenced
            col_match = re.search(r'Referenced column ["\']([^"\']+)["\'] not found', err_text, re.IGNORECASE)
            if not col_match:
                col_match = re.search(r'Column with name ([^\s]+) does not exist', err_text, re.IGNORECASE)
            
            if col_match:
                missing_col = col_match.group(1).strip()
                avail_cols = list(df.columns)
                return SQLValidationResult(
                    status="INVALID",
                    query=raw_sql,
                    error_code="UNKNOWN_COLUMN",
                    message=f"Column '{missing_col}' does not exist in active dataset. Available columns: {', '.join(avail_cols)}"
                )
            
            # Extract unknown table if referenced
            tbl_match = re.search(r'Table with name ["\']?([^"\'\s]+)["\']? does not exist', err_text, re.IGNORECASE)
            if tbl_match:
                missing_tbl = tbl_match.group(1).strip()
                avail_tbls = ["active_dataset"] + list(relation_catalog.get_all_dataframes().keys())
                return SQLValidationResult(
                    status="INVALID",
                    query=raw_sql,
                    error_code="UNKNOWN_TABLE",
                    message=f"Table '{missing_tbl}' does not exist. Available tables: {', '.join(sorted(set(avail_tbls)))}"
                )

            return SQLValidationResult(
                status="INVALID",
                query=raw_sql,
                error_code="BINDER_ERROR",
                message=f"SQL Binder Error: {err_text}"
            )
        except Exception as e:
            return SQLValidationResult(
                status="INVALID",
                query=raw_sql,
                error_code="EXECUTION_ERROR",
                message=f"SQL validation error: {str(e)}"
            )
        finally:
            try:
                con.close()
            except Exception:
                pass

        # 4. Extract referenced columns and tables
        ref_cols: List[str] = []
        for col in df.columns:
            # Check if column is mentioned in SQL as an identifier (word boundary or quoted)
            col_escaped = re.escape(str(col))
            if re.search(r'(?:"' + col_escaped + r'"|`' + col_escaped + r'`|\b' + col_escaped + r'\b)', raw_sql, re.IGNORECASE):
                ref_cols.append(str(col))

        ref_tables: List[str] = []
        possible_tables = ["active_dataset", "dataset", "df"] + list(relation_catalog.get_all_dataframes().keys())
        if dataset_name:
            possible_tables.append(dataset_name)
        for tbl in set(possible_tables):
            if re.search(r'\b' + re.escape(tbl) + r'\b', raw_sql, re.IGNORECASE):
                ref_tables.append(tbl)
        if not ref_tables:
            ref_tables.append("active_dataset")

        # 5. Estimate operation
        sql_upper = raw_sql.upper()
        if re.search(r"\bOVER\s*\(", sql_upper) or any(wf in sql_upper for wf in ["ROW_NUMBER()", "RANK()", "DENSE_RANK()", "NTILE("]):
            est_op = "WINDOW"
        elif sql_upper.startswith("WITH ") or re.search(r"\bWITH\b\s+[A-Za-z0-9_]+\s+AS\s*\(", sql_upper):
            est_op = "CTE"
        elif re.search(r"\bGROUP\s+BY\b", sql_upper) or any(agg in sql_upper for agg in ["COUNT(", "AVG(", "SUM(", "MIN(", "MAX("]):
            est_op = "AGGREGATION"
        elif re.search(r"\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\b", sql_upper):
            est_op = "JOIN"
        elif re.search(r"\bWHERE\b", sql_upper):
            est_op = "FILTER"
        else:
            est_op = "SELECTION"

        return SQLValidationResult(
            status="VALID",
            query=raw_sql,
            referenced_columns=ref_cols,
            referenced_tables=ref_tables or ["active_dataset"],
            estimated_operation=est_op
        )

    validate_query = validate

