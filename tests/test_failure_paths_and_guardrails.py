"""
DataPilot OS — Failure Path, Type Guardrail, and Schema Protection Audit Suite.
Tests:
1. No Date Column for Forecast -> Verifies clean rejection (NO_TEMPORAL_COLUMN or AMBIGUOUS_INTENT) without fake date fabrication.
2. Non-numeric Correlation -> Verifies Pearson correlation on non-numeric text columns is cleanly handled/rejected without crashing or numerical invalidity.
3. Non-existent Column -> Verifies query referencing unknown column raises COLUMN_NOT_FOUND / IntentResolutionError cleanly.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator, ChartType
from backend.ds_engine.intent_resolver import IntentResolver, IntentResolutionError, AmbiguousIntentError
from backend.ds_engine.capability_drivers import CorrelationDriver, TimeSeriesForecastDriver
from backend.ds_engine.code_generator import code_generator
from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner
from modules.execution_engine import ModularExecutionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GuardrailAuditor")


def test_case_1_no_date_column_for_forecast():
    """
    Test Case 1: No Date Column for Forecast.
    Attempt time-series forecast on a dataset with NO temporal columns (only category, name, value).
    Verify it raises a clean error/failure (NO_TEMPORAL_COLUMN or AMBIGUOUS_INTENT) and NEVER fabricates fake dates.
    """
    print("\n" + "=" * 80)
    print("GUARDRAIL TEST 1: No Date Column for Forecast Protection")
    print("=" * 80)

    # Dataset with strictly non-temporal columns
    df_no_date = pd.DataFrame({
        "category": ["Electronics", "Furniture", "Clothing", "Groceries", "Automotive"],
        "product_name": ["Laptop", "Desk", "Shirt", "Apple", "Tire"],
        "unit_value": [1200.50, 350.00, 45.99, 2.50, 110.00]
    })
    print(f"Dataset columns: {list(df_no_date.columns)}")
    print(f"Dataset dtypes:\n{df_no_date.dtypes}")

    # Sub-test 1A: IntentResolver Level
    resolver = IntentResolver()
    forecast_intent = SemanticIntent(
        intent_type=IntentType.TIME_SERIES_FORECAST,
        primary_intent=IntentType.TIME_SERIES_FORECAST,
        target_column="unit_value",
        raw_query="Forecast unit_value for next 30 days"
    )

    resolver_failed_cleanly = False
    caught_error_code = None
    try:
        resolved = resolver.resolve(forecast_intent, df=df_no_date)
        # If it didn't raise, check if it fabricated a date column
        assert resolved.resolved_time_column is None, (
            f"VIOLATION: System fabricated fake date column: {resolved.resolved_time_column}"
        )
    except IntentResolutionError as err:
        resolver_failed_cleanly = True
        caught_error_code = err.code
        print(f"\n[PASS 1A.1] IntentResolver raised IntentResolutionError as expected: code='{err.code}', message='{err.message}'")
        assert err.code in ["NO_TEMPORAL_COLUMN", "NO_DATE_COLUMN"], (
            f"Expected NO_TEMPORAL_COLUMN or NO_DATE_COLUMN, got '{err.code}'"
        )
    except AmbiguousIntentError as amb_err:
        resolver_failed_cleanly = True
        caught_error_code = amb_err.code
        print(f"\n[PASS 1A.1] IntentResolver raised AmbiguousIntentError as expected: code='{amb_err.code}', message='{amb_err.message}'")
        assert amb_err.code == "AMBIGUOUS_INTENT"

    assert resolver_failed_cleanly, "VIOLATION: Expected IntentResolutionError or AmbiguousIntentError for dataset with no date columns"

    # Sub-test 1B: Full QueryOrchestrator Pipeline Level
    mock_plan = {
        "tool": "forecast_data",
        "arguments": {
            "value_column": "unit_value",
            # Supply a schema-valid but dataset-invalid temporal reference so
            # this test reaches the semantic feasibility guardrail instead of
            # failing the strict ToolSpec shape check.
            "time_column": "event_date",
            "horizon": 30,
            "horizon_evidence": "next 30 days",
            "frequency": "D",
            "method": "naive_last_value",
            "chart_type": "line",
        },
        "raw_query": "Forecast unit_value for next 30 days"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Forecast unit_value for next 30 days",
            df=df_no_date
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    print(f"\nPipeline Response Status: {res.status}")
    print(f"Pipeline Error: {res.error}")

    assert res.status == "FAILED", f"Expected query failure, but got status '{res.status}'"
    assert res.error is not None, "Expected error detail in response"
    assert res.error.phase == "PLAN_VALIDATION", f"Expected failure in PLAN_VALIDATION phase, got '{res.error.phase}'"
    assert "date" in res.error.message.lower() or "temporal" in res.error.message.lower(), (
        f"Error message should mention missing date/temporal requirement: {res.error.message}"
    )

    print("\n[PASS] Test Case 1: Successfully prevented forecasting without date column; zero fake dates fabricated.")


def test_case_2_non_numeric_correlation_handling():
    """
    Test Case 2: Non-numeric Correlation.
    Attempt Pearson correlation on non-numeric text columns ('employee_name', 'department').
    Verify clean rejection/handling without crashing or numerical invalidity.
    """
    print("\n" + "=" * 80)
    print("GUARDRAIL TEST 2: Non-Numeric Correlation Type Guardrail")
    print("=" * 80)

    # Dataset with categorical/text columns
    df_hr = pd.DataFrame({
        "employee_name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright"],
        "department": ["Engineering", "Sales", "Human Resources", "Marketing", "Engineering"],
        "job_level": ["L3", "L4", "L2", "L5", "L3"],
        "salary": [85000, 92000, 65000, 120000, 88000]
    })
    print(f"Dataset columns: {list(df_hr.columns)}")

    # Sub-test 2A: Pairwise Pearson Correlation on Text Columns
    pairwise_intent = SemanticIntent(
        intent_type=IntentType.CORRELATION,
        primary_intent=IntentType.CORRELATION,
        feature_columns=["employee_name", "department"],
        chart_type=ChartType.SCATTER,
        raw_query="Calculate Pearson correlation between employee_name and department"
    )

    resolver = IntentResolver()
    resolved = resolver.resolve(pairwise_intent, df=df_hr)
    assert resolved.is_valid, "ResolvedIntent should resolve valid existing column names"

    driver = CorrelationDriver()
    code, deps, outputs = driver.compile(resolved)
    print("\nGenerated Pairwise Correlation Driver Code:\n" + "-" * 40)
    print(code)
    print("-" * 40)

    # Execute in sandbox environment
    exec_engine = ModularExecutionEngine()
    exec_res = exec_engine.execute_code_subprocess(code, df=df_hr)

    print(f"\nExecution Success: {exec_res.get('success')}")
    print(f"Stdout: {exec_res.get('stdout')}")
    print(f"Stderr: {exec_res.get('stderr')}")

    # Verify execution did not crash with unhandled exception
    assert exec_res.get("success") is True, f"Sandbox crashed during non-numeric correlation: {exec_res.get('error')}"
    stdout = exec_res.get("stdout", "")
    assert "non-numeric" in stdout.lower() or "insufficient" in stdout.lower(), (
        f"Expected rejection message in stdout, got: {stdout}"
    )

    # Verify structured dataframe output is cleanly marked as non-numeric rather than invalid numbers
    out_df = exec_res.get("dataframe")
    if out_df is not None and isinstance(out_df, pd.DataFrame) and not out_df.empty:
        records = out_df.to_dict(orient="records")
        print(f"Structured Output Records: {records}")
        for r in records:
            # Must not contain fabricated NaN/inf numbers
            p_val = r.get("Pearson_r")
            if p_val is not None:
                assert not (pd.isna(p_val) or np.isnan(p_val)), "VIOLATION: NaN numeric correlation produced"
            # Must contain clean rejection status or message
            if "status" in r:
                assert r["status"] == "REJECTED_NON_NUMERIC"

    # Sub-test 2B: Full QueryOrchestrator Pipeline Integration
    mock_corr_plan = {
        "tool": "correlate_data",
        "arguments": {
            "columns": ["employee_name", "department"],
            "analysis_mode": "pairwise",
            "relationship_form": "linear",
            "correlation_methods": ["pearson"],
            "chart_type": "none",
            "smoother": "none",
            "expected_direction": "unspecified"
        },
        "raw_query": "Calculate Pearson correlation between employee_name and department"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_corr_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw="Calculate Pearson correlation between employee_name and department",
            df=df_hr
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    print(f"\nPipeline Correlation Response Status: {res.status}")
    assert res.status == "FAILED", f"Query pipeline expected failure: {res.error}"
    assert res.error is not None
    assert res.error.code in ("INSUFFICIENT_NUMERIC_COLUMNS", "TYPE_ERROR")

    print("\n[PASS] Test Case 2: Handled non-numeric correlation safely via modular pipeline.")


def test_case_3_non_existent_column_protection():
    """
    Test Case 3: Non-Existent Column Protection (Invalid Schema Guardrail).
    Ask for a column that does NOT exist in dataset ('xyz_unknown_col_123').
    Verify strict IntentResolutionError with COLUMN_NOT_FOUND and suggestion of available columns.
    """
    print("\n" + "=" * 80)
    print("GUARDRAIL TEST 3: Non-Existent Column Protection (Invalid Schema Guardrail)")
    print("=" * 80)

    # Valid dataset
    df_valid = pd.DataFrame({
        "sales": [100, 200, 300, 400],
        "region": ["North", "South", "East", "West"],
        "balance": [50.5, 80.2, 120.0, 95.3]
    })
    print(f"Active Dataset columns: {list(df_valid.columns)}")

    fake_col = "xyz_unknown_col_123"
    resolver = IntentResolver()

    # Sub-test 3A: IntentResolver with unknown target_column
    bad_target_intent = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        target_column=fake_col,
        dimensions=["region"],
        aggregation="mean",
        raw_query=f"Calculate average {fake_col} by region"
    )

    caught_col_not_found = False
    try:
        resolver.resolve(bad_target_intent, df=df_valid)
    except IntentResolutionError as err:
        caught_col_not_found = True
        print(f"\n[PASS 3A.1] Caught IntentResolutionError: code='{err.code}', invalid_cols={err.invalid_columns}")
        assert err.code == "COLUMN_NOT_FOUND", f"Expected COLUMN_NOT_FOUND, got '{err.code}'"
        assert fake_col in err.invalid_columns, f"Expected '{fake_col}' in invalid_columns, got {err.invalid_columns}"
        assert "sales" in err.message or "region" in err.message, "Error message should suggest available dataset columns"

    assert caught_col_not_found, f"VIOLATION: Expected IntentResolutionError for non-existent target column '{fake_col}'"

    # Sub-test 3B: IntentResolver with unknown filter column
    bad_filter_intent = SemanticIntent(
        intent_type=IntentType.FILTER_QUERY,
        primary_intent=IntentType.FILTER_QUERY,
        filters=[FilterSpec(column=fake_col, operator=FilterOperator.GT, value=50)],
        raw_query=f"Filter {fake_col} > 50"
    )

    caught_filter_col_not_found = False
    try:
        resolver.resolve(bad_filter_intent, df=df_valid)
    except IntentResolutionError as err:
        caught_filter_col_not_found = True
        print(f"\n[PASS 3A.2] Caught filter IntentResolutionError: code='{err.code}', invalid_cols={err.invalid_columns}")
        assert err.code == "COLUMN_NOT_FOUND"
        assert fake_col in err.invalid_columns

    assert caught_filter_col_not_found, f"VIOLATION: Expected IntentResolutionError for non-existent filter column '{fake_col}'"

    # Sub-test 3C: Full QueryOrchestrator Pipeline Level
    mock_bad_col_plan = {
        "tool": "group_data",
        "arguments": {
            "group_by": ["region"],
            "metric": fake_col,
            "aggregation": "mean"
        },
        "raw_query": f"Calculate average {fake_col} by region"
    }

    orig_extract = SemanticPlanner.extract_semantic_intent
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_bad_col_plan)

    try:
        res = QueryOrchestrator.process_query(
            query_raw=f"Calculate average {fake_col} by region",
            df=df_valid
        )
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract

    print(f"\nPipeline Response Status: {res.status}")
    print(f"Pipeline Error: {res.error}")

    assert res.status == "FAILED", f"Expected query failure, got '{res.status}'"
    assert res.error is not None, "Expected error detail in response"
    assert res.error.phase == "PLAN_VALIDATION", f"Expected failure in PLAN_VALIDATION phase, got '{res.error.phase}'"
    assert fake_col in res.error.message, f"Expected '{fake_col}' in error message: {res.error.message}"

    print(f"\n[PASS] Test Case 3: Successfully rejected non-existent column '{fake_col}' with COLUMN_NOT_FOUND.")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("STARTING FAILURE PATHS, TYPE GUARDRAILS & INVALID SCHEMA PROTECTION AUDIT")
    print("#" * 80)

    test_case_1_no_date_column_for_forecast()
    test_case_2_non_numeric_correlation_handling()
    test_case_3_non_existent_column_protection()

    print("\n" + "#" * 80)
    print("ALL 3 FAILURE PATH & GUARDRAIL AUDIT TESTS PASSED 100% CLEANLY [OK]")
    print("#" * 80 + "\n")
