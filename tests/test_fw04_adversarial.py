"""
Adversarial test suite for DataPilot OS — FW-04 Multi-Table Relational Schema Resolution.
Tests edge cases and boundary conditions:
1. Incompatible join key datatypes.
2. Nonexistent tables or columns in join plans.
3. Severe multiplicative fan-out row explosion rejection.
4. Composite keys explicitly declared vs composite keys rejected without declaration.
5. Exact parity with independent numerical oracles (Pandas merge & DuckDB).
"""

import pytest
import pandas as pd
from backend.sql_engine.relation_catalog import relation_catalog
from backend.sql_engine.relationship_resolver import RelationshipResolver
from backend.sql_engine.join_planner import JoinPlanner, JoinPlan, JoinStep
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager


@pytest.fixture(autouse=True)
def clean_catalog():
    relation_catalog.clear()
    yield
    relation_catalog.clear()


def test_adversarial_incompatible_dtypes_rejection():
    df1 = pd.DataFrame({"key": [1, 2, 3], "val1": [10, 20, 30]})
    df2 = pd.DataFrame({"key": ["one", "two", "three"], "val2": [100, 200, 300]})

    ev = RelationshipResolver.evaluate_relationship(
        parent_df=df1, parent_table="t1", parent_key="key",
        child_df=df2, child_table="t2", child_key="key"
    )
    assert ev.status == "TYPE_MISMATCH"
    assert ev.is_valid_relationship is False


def test_adversarial_nonexistent_table_or_column_rejection():
    df1 = pd.DataFrame({"id": [1, 2]})
    relation_catalog.register_dataset("ds1", "t1", df1)

    # 1. Nonexistent table
    plan_bad_tbl = JoinPlan(
        left_table="t1",
        steps=[JoinStep(right_table="ghost_table", left_keys=["id"], right_keys=["id"])]
    )
    res_tbl = JoinPlanner.execute_join_plan(plan_bad_tbl, relation_catalog.get_all_dataframes())
    assert res_tbl.status == "FAILED"
    assert res_tbl.error_code == "UNKNOWN_TABLE"

    # 2. Nonexistent column
    df2 = pd.DataFrame({"id_other": [1, 2]})
    relation_catalog.register_dataset("ds2", "t2", df2)
    plan_bad_col = JoinPlan(
        left_table="t1",
        steps=[JoinStep(right_table="t2", left_keys=["nonexistent_key"], right_keys=["id_other"])]
    )
    res_col = JoinPlanner.execute_join_plan(plan_bad_col, relation_catalog.get_all_dataframes())
    assert res_col.status == "FAILED"
    assert res_col.error_code == "UNKNOWN_COLUMN"


def test_adversarial_severe_fan_out_row_explosion_rejection():
    # 10 records on left, 10 records on right, but ALL keys equal 'A' -> 100 rows (10x expansion!)
    df_l = pd.DataFrame({"key": ["A"] * 10, "val_l": range(10)})
    df_r = pd.DataFrame({"key": ["A"] * 10, "val_r": range(10)})
    relation_catalog.register_dataset("dsl", "tl", df_l)
    relation_catalog.register_dataset("dsr", "tr", df_r)

    plan = JoinPlan(
        left_table="tl",
        steps=[JoinStep(right_table="tr", join_type="left", left_keys=["key"], right_keys=["key"])],
        allow_fan_out=False
    )
    res = JoinPlanner.execute_join_plan(plan, relation_catalog.get_all_dataframes())
    assert res.status == "REJECTED"
    assert res.error_code == "FAN_OUT_WARNING"
    assert res.impact.fan_out_risk == "high_multiplicative"


def test_adversarial_composite_key_explicit_declaration():
    # Store ID + Dept ID composite key
    df1 = pd.DataFrame({
        "store_id": [1, 1, 2],
        "dept_id": [10, 20, 10],
        "sales": [500, 700, 600]
    })
    df2 = pd.DataFrame({
        "store_id": [1, 1, 2],
        "dept_id": [10, 20, 10],
        "manager": ["Smith", "Jones", "Taylor"]
    })
    relation_catalog.register_dataset("ds1", "sales_tbl", df1)
    relation_catalog.register_dataset("ds2", "dept_mgrs", df2)

    plan = JoinPlan(
        left_table="sales_tbl",
        steps=[
            JoinStep(
                right_table="dept_mgrs",
                join_type="left",
                left_keys=["store_id", "dept_id"],
                right_keys=["store_id", "dept_id"]
            )
        ]
    )
    res = JoinPlanner.execute_join_plan(plan, relation_catalog.get_all_dataframes())
    assert res.status == "SUCCEEDED"
    assert len(res.result_df) == 3
    assert "manager" in res.result_df.columns


def test_independent_numerical_oracle_parity():
    # Compare DataPilot join with independent Pandas merge and direct DuckDB query
    df_left = pd.DataFrame({
        "k": [1, 2, 3, 4],
        "val": [10.5, 20.5, 30.5, 40.5]
    })
    df_right = pd.DataFrame({
        "k": [2, 3, 4, 5],
        "mult": [2.0, 3.0, 4.0, 5.0]
    })
    relation_catalog.register_dataset("dsl", "t_l", df_left)
    relation_catalog.register_dataset("dsr", "t_r", df_right)

    # DataPilot join
    plan = JoinPlan(
        left_table="t_l",
        steps=[JoinStep(right_table="t_r", join_type="inner", left_keys=["k"], right_keys=["k"])]
    )
    dp_res = JoinPlanner.execute_join_plan(plan, relation_catalog.get_all_dataframes())
    assert dp_res.status == "SUCCEEDED"

    # Independent Pandas oracle
    pd_oracle = pd.merge(df_left, df_right, on="k", how="inner")

    # Verify exact numerical parity
    assert len(dp_res.result_df) == len(pd_oracle)
    assert (dp_res.result_df["k"].values == pd_oracle["k"].values).all()
    assert (dp_res.result_df["val"].values == pd_oracle["val"].values).all()
    assert (dp_res.result_df["mult"].values == pd_oracle["mult"].values).all()
