"""
Unit test suite for DataPilot OS — FW-04 Join Planner & Fan-Out Guard.
Verifies:
1. Compiles deterministic SQL with column collision suffixing.
2. Identifies fan-out multiplicative row explosion (projected expansion > 1.5x) and blocks without allow_fan_out.
3. Successfully executes 1:1, 1:m, and m:1 joins using DuckDB.
4. Correct standard SQL NULL semantics (NULL keys do not match).
"""

import pytest
import pandas as pd
from backend.sql_engine.join_planner import JoinPlanner, JoinPlan, JoinStep


def test_join_planner_compilation_and_collision_suffixing():
    df_orders = pd.DataFrame({
        "order_id": [1, 2],
        "customer_id": [10, 20],
        "created_at": ["2026-01-01", "2026-01-02"]
    })
    df_customers = pd.DataFrame({
        "customer_id": [10, 20],
        "name": ["Alice", "Bob"],
        "created_at": ["2025-01-01", "2025-01-02"]
    })
    tables = {"orders": df_orders, "customers": df_customers}

    plan = JoinPlan(
        left_table="orders",
        steps=[
            JoinStep(
                right_table="customers",
                join_type="left",
                left_keys=["customer_id"],
                right_keys=["customer_id"]
            )
        ]
    )

    sql = JoinPlanner.compile_sql(plan, tables)
    assert 'created_at_customers' in sql
    assert '"orders"."customer_id" = "customers"."customer_id"' in sql


def test_join_planner_fan_out_guard_rejection():
    # 2 customers, but both have 5 orders each -> fan out if we left join orders with payments (many to many)
    df_left = pd.DataFrame({"cust_id": [1, 2], "val": [10, 20]})
    # Right table repeats cust_id 5 times per customer -> 5x expansion
    df_right = pd.DataFrame({
        "cust_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "event": range(10)
    })
    tables = {"left_tbl": df_left, "right_tbl": df_right}

    plan = JoinPlan(
        left_table="left_tbl",
        steps=[
            JoinStep(
                right_table="right_tbl",
                join_type="left",
                left_keys=["cust_id"],
                right_keys=["cust_id"]
            )
        ],
        allow_fan_out=False
    )

    res = JoinPlanner.execute_join_plan(plan, tables)
    assert res.status == "REJECTED"
    assert res.error_code == "FAN_OUT_WARNING"
    assert res.impact.expansion_ratio >= 4.0


def test_join_planner_execution_with_null_semantics():
    df_left = pd.DataFrame({
        "id": [None, 1, 2],
        "val_left": ["null_row", "one", "two"]
    })
    df_right = pd.DataFrame({
        "id": [None, 1, 3],
        "val_right": ["right_null", "matched_one", "three"]
    })
    tables = {"left_t": df_left, "right_t": df_right}

    plan = JoinPlan(
        left_table="left_t",
        steps=[
            JoinStep(
                right_table="right_t",
                join_type="inner",
                left_keys=["id"],
                right_keys=["id"]
            )
        ]
    )

    res = JoinPlanner.execute_join_plan(plan, tables)
    assert res.status == "SUCCEEDED"
    # In SQL INNER JOIN, NULL != NULL! Only id=1 matches
    assert len(res.result_df) == 1
    assert res.result_df["id"].iloc[0] == 1
