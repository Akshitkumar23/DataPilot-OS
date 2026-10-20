"""
DataPilot OS — Phase 7 / FW-04 Deterministic Join Planner & Fan-Out Guard.
Validates join plans, detects multiplicative fan-out row explosion, compiles
deterministic DuckDB SQL statements, enforces column collision management, and
executes governed relational enrichment.
Zero domain or dataset hardcoding.
"""

import uuid
import re
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import duckdb

from backend.sql_engine.relationship_resolver import RelationshipResolver, RelationshipEvidence


class JoinStep(BaseModel):
    """Specification of a single join step in a join plan."""
    right_table: str
    join_type: Literal["inner", "left", "right", "full", "cross"] = "left"
    left_keys: List[str]
    right_keys: List[str]
    relationship_evidence: Optional[RelationshipEvidence] = None


class JoinPlan(BaseModel):
    """Structured deterministic plan for multi-table relational operations."""
    plan_id: str = Field(default_factory=lambda: f"join_{uuid.uuid4().hex[:8]}")
    left_table: str
    target_name: Optional[str] = None
    steps: List[JoinStep]
    selected_columns: Optional[List[str]] = None
    allow_fan_out: bool = False
    max_expansion_ratio: float = 1.5


class JoinPlanImpact(BaseModel):
    """Pre-execution assessment of row count impact, cardinality, and fan-out risk."""
    left_table: str
    right_table: str
    join_type: str
    cardinality: str
    left_rows: int
    right_rows: int
    estimated_rows: int
    expansion_ratio: float
    fan_out_risk: Literal["none", "low", "high_multiplicative"]
    requires_acknowledgement: bool
    warnings: List[str] = Field(default_factory=list)


class JoinExecutionResult(BaseModel):
    """Outcome of governed multi-table join execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal["SUCCEEDED", "FAILED", "REJECTED"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result_df: Optional[pd.DataFrame] = None
    compiled_sql: str = ""
    left_rows: int = 0
    right_rows: int = 0
    result_rows: int = 0
    expansion_ratio: float = 1.0
    dropped_rows: int = 0
    duplicated_rows: int = 0
    left_null_keys: int = 0
    right_null_keys: int = 0
    unmatched_left: int = 0
    unmatched_right: int = 0
    impact: Optional[JoinPlanImpact] = None
    warnings: List[str] = Field(default_factory=list)


class JoinPlanner:
    """
    Authoritative join compiler, validator, and executor for DataPilot OS.
    """

    @classmethod
    def analyze_impact(
        cls,
        left_df: pd.DataFrame,
        left_table: str,
        step: JoinStep,
        right_df: pd.DataFrame
    ) -> JoinPlanImpact:
        """
        Calculates projected row expansion and cardinality risk before execution.
        """
        l_rows = len(left_df)
        r_rows = len(right_df)
        warnings = []

        if step.join_type.lower() == "cross":
            est_rows = l_rows * r_rows
            exp_ratio = float(est_rows) / max(1, l_rows)
            return JoinPlanImpact(
                left_table=left_table,
                right_table=step.right_table,
                join_type="cross",
                cardinality="m:m",
                left_rows=l_rows,
                right_rows=r_rows,
                estimated_rows=est_rows,
                expansion_ratio=exp_ratio,
                fan_out_risk="high_multiplicative",
                requires_acknowledgement=True,
                warnings=["Cartesian CROSS JOIN creates multiplicative row explosion."]
            )

        # Evaluate relationship using single key or composite key representation
        l_key = step.left_keys[0]
        r_key = step.right_keys[0]

        l_s = left_df[l_key].dropna()
        r_s = right_df[r_key].dropna()

        l_is_unique = (l_s.nunique() == len(l_s)) and (int(left_df[l_key].isna().sum()) == 0) and len(l_s) > 0
        r_is_unique = (r_s.nunique() == len(r_s)) and (int(right_df[r_key].isna().sum()) == 0) and len(r_s) > 0

        if l_is_unique and r_is_unique:
            cardinality = "1:1"
        elif l_is_unique and not r_is_unique:
            cardinality = "1:m"
        elif not l_is_unique and r_is_unique:
            cardinality = "m:1"
        else:
            cardinality = "m:m"

        # Estimate rows based on join_type and cardinality
        jt = step.join_type.lower()
        if jt == "left":
            if r_is_unique:
                est_rows = l_rows  # Grain preserved! Each left row matches at most 1 right row
                exp_ratio = 1.0
                fan_risk = "none"
                req_ack = False
            else:
                # Right side has duplicates for join key -> Left rows multiply!
                # Calculate average occurrences of right keys that exist in left
                r_counts = right_df[r_key].dropna().value_counts()
                avg_r_mult = float(r_counts.mean()) if len(r_counts) > 0 else 1.0
                est_rows = int(round(l_rows * avg_r_mult))
                exp_ratio = est_rows / max(1, l_rows)
                fan_risk = "high_multiplicative" if exp_ratio > 1.5 else "low"
                req_ack = exp_ratio > 1.5
                if req_ack:
                    warnings.append(f"Join expands rows by {exp_ratio:.1f}x. Risk of altering primary table observational grain.")
        elif jt == "inner":
            matched_l = int(left_df[l_key].isin(set(right_df[r_key].dropna())).sum())
            if r_is_unique:
                est_rows = matched_l
                exp_ratio = est_rows / max(1, l_rows)
                fan_risk = "none"
                req_ack = False
            else:
                r_counts = right_df[r_key].dropna().value_counts()
                avg_r_mult = float(r_counts.mean()) if len(r_counts) > 0 else 1.0
                est_rows = int(round(matched_l * avg_r_mult))
                exp_ratio = est_rows / max(1, l_rows)
                fan_risk = "high_multiplicative" if exp_ratio > 1.5 else "low"
                req_ack = exp_ratio > 1.5
                if req_ack:
                    warnings.append(f"Inner join expands rows by {exp_ratio:.1f}x due to multiple matches in right table.")
        else:
            est_rows = max(l_rows, r_rows)
            exp_ratio = est_rows / max(1, l_rows)
            fan_risk = "low"
            req_ack = False

        return JoinPlanImpact(
            left_table=left_table,
            right_table=step.right_table,
            join_type=jt,
            cardinality=cardinality,
            left_rows=l_rows,
            right_rows=r_rows,
            estimated_rows=est_rows,
            expansion_ratio=exp_ratio,
            fan_out_risk=fan_risk,
            requires_acknowledgement=req_ack,
            warnings=warnings
        )

    @classmethod
    def compile_sql(
        cls,
        plan: JoinPlan,
        tables_map: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Compiles a deterministic, unambiguous DuckDB SQL query from the join plan.
        Handles column collisions with deterministic table suffixes.
        """
        l_tbl = plan.left_table
        l_df = tables_map[l_tbl]

        # Build column selections with collision avoidance
        all_cols_selected = []
        seen_col_names = set()

        # Step 1: Add left table columns
        for c in l_df.columns:
            clean_c = str(c)
            seen_col_names.add(clean_c)
            all_cols_selected.append(f'"{l_tbl}"."{clean_c}" AS "{clean_c}"')

        # Step 2: Add joined tables columns with renaming on collision
        for step in plan.steps:
            r_tbl = step.right_table
            r_df = tables_map[r_tbl]
            # Exclude duplicate join keys in right table if identical name
            for c in r_df.columns:
                clean_c = str(c)
                if clean_c in step.right_keys and clean_c in step.left_keys:
                    continue  # skip redundant key
                if clean_c in seen_col_names:
                    # Collision! Deterministic suffixing
                    alias = f"{clean_c}_{r_tbl}"
                    all_cols_selected.append(f'"{r_tbl}"."{clean_c}" AS "{alias}"')
                else:
                    seen_col_names.add(clean_c)
                    all_cols_selected.append(f'"{r_tbl}"."{clean_c}" AS "{clean_c}"')

        # Custom column selection filter if specified
        if plan.selected_columns:
            select_clause = ", ".join(plan.selected_columns)
        else:
            select_clause = ",\n    ".join(all_cols_selected)

        sql_parts = [f"SELECT\n    {select_clause}\nFROM \"{l_tbl}\""]

        for step in plan.steps:
            r_tbl = step.right_table
            jt = step.join_type.upper()
            if jt == "CROSS":
                sql_parts.append(f"CROSS JOIN \"{r_tbl}\"")
            else:
                on_conditions = [
                    f'"{l_tbl}"."{lk}" = "{r_tbl}"."{rk}"'
                    for lk, rk in zip(step.left_keys, step.right_keys)
                ]
                on_clause = " AND ".join(on_conditions)
                sql_parts.append(f"{jt} JOIN \"{r_tbl}\" ON {on_clause}")

        return "\n".join(sql_parts)

    @classmethod
    def execute_join_plan(
        cls,
        plan: JoinPlan,
        tables_map: Dict[str, pd.DataFrame]
    ) -> JoinExecutionResult:
        """
        Executes a validated JoinPlan using DuckDB with zero DataFrame mutation
        and comprehensive relational audit metrics.
        """
        l_tbl = plan.left_table
        if l_tbl not in tables_map:
            return JoinExecutionResult(
                status="FAILED",
                error_code="UNKNOWN_TABLE",
                error_message=f"Primary table '{l_tbl}' not found in Relation Catalog."
            )

        l_df = tables_map[l_tbl]

        # Pre-execution validation for all steps
        for step in plan.steps:
            r_tbl = step.right_table
            if r_tbl not in tables_map:
                return JoinExecutionResult(
                    status="FAILED",
                    error_code="UNKNOWN_TABLE",
                    error_message=f"Joined table '{r_tbl}' not found in Relation Catalog."
                )
            r_df = tables_map[r_tbl]

            for lk in step.left_keys:
                if lk not in l_df.columns:
                    return JoinExecutionResult(
                        status="FAILED",
                        error_code="UNKNOWN_COLUMN",
                        error_message=f"Left key '{lk}' not found in '{l_tbl}'."
                    )
            for rk in step.right_keys:
                if rk not in r_df.columns:
                    return JoinExecutionResult(
                        status="FAILED",
                        error_code="UNKNOWN_COLUMN",
                        error_message=f"Right key '{rk}' not found in '{r_tbl}'."
                    )

            # Analyze fan-out impact
            impact = cls.analyze_impact(l_df, l_tbl, step, r_df)
            if impact.fan_out_risk == "high_multiplicative" and not plan.allow_fan_out:
                return JoinExecutionResult(
                    status="REJECTED",
                    error_code="FAN_OUT_WARNING",
                    error_message=(
                        f"Join rejected due to projected row expansion of {impact.expansion_ratio:.1f}x. "
                        f"Set allow_fan_out=True or pre-aggregate '{r_tbl}' before joining."
                    ),
                    impact=impact,
                    warnings=impact.warnings
                )

        # Compile SQL
        compiled_sql = cls.compile_sql(plan, tables_map)

        # Execute using DuckDB
        con = None
        try:
            con = duckdb.connect(database=":memory:")
            con.register(l_tbl, l_df)
            for step in plan.steps:
                con.register(step.right_table, tables_map[step.right_table])

            res_df = con.query(compiled_sql).df()

            # Calculate relational audit statistics
            first_step = plan.steps[0]
            r_tbl = first_step.right_table
            r_df = tables_map[r_tbl]
            lk = first_step.left_keys[0]
            rk = first_step.right_keys[0]

            l_rows = len(l_df)
            r_rows = len(r_df)
            res_rows = len(res_df)
            exp_ratio = float(res_rows) / max(1, l_rows)

            l_nulls = int(l_df[lk].isna().sum())
            r_nulls = int(r_df[rk].isna().sum())

            # Unmatched counts
            l_keys_set = set(l_df[lk].dropna())
            r_keys_set = set(r_df[rk].dropna())
            unmatched_l = int((~l_df[lk].isin(r_keys_set) & l_df[lk].notna()).sum())
            unmatched_r = int((~r_df[rk].isin(l_keys_set) & r_df[rk].notna()).sum())

            dropped_rows = max(0, l_rows - res_rows)
            duplicated_rows = max(0, res_rows - l_rows)

            impact = cls.analyze_impact(l_df, l_tbl, first_step, r_df)

            return JoinExecutionResult(
                status="SUCCEEDED",
                result_df=res_df,
                compiled_sql=compiled_sql,
                left_rows=l_rows,
                right_rows=r_rows,
                result_rows=res_rows,
                expansion_ratio=exp_ratio,
                dropped_rows=dropped_rows,
                duplicated_rows=duplicated_rows,
                left_null_keys=l_nulls,
                right_null_keys=r_nulls,
                unmatched_left=unmatched_l,
                unmatched_right=unmatched_r,
                impact=impact,
                warnings=impact.warnings
            )
        except Exception as e:
            return JoinExecutionResult(
                status="FAILED",
                error_code="EXECUTION_ERROR",
                error_message=f"DuckDB execution error: {str(e)}",
                compiled_sql=compiled_sql
            )
        finally:
            if con:
                try:
                    con.close()
                except Exception:
                    pass
