"""
DataPilot OS — Tri-State Ordering Verification Suite.
Tests and verifies tri-state ordering semantics:
  1. order='desc' (Explicit Descending)
  2. order='asc'  (Explicit Ascending)
  3. order=None   (No ordering specified / Unsorted natural groupby)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath("."))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TriStateOrderingAuditor")

from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator
from backend.ds_engine.intent_schema import SemanticIntent


def create_test_sales_dataset() -> pd.DataFrame:
    """Creates a deterministic sales dataframe with varying sales amounts across regions."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "sales_amount": [150.0, 720.0, 310.0, 950.0, 620.0, 480.0, 840.0, 560.0],
        "region": ["North", "South", "North", "South", "East", "West", "East", "West"]
    })


def test_query_1_explicit_descending():
    """
    Query 1 (Explicit Descending):
    'Filter sales_amount > 500 and sort by sales_amount descending'
    Verify: intent.order == 'desc' and dataframe actually sorted descending.
    """
    query = "Filter sales_amount > 500 and sort by sales_amount descending"
    df = create_test_sales_dataset()

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED", f"Query execution failed: {res.error}"
    assert res.result is not None, "Query response result is None"
    
    intent = SemanticIntent(**res.plan)
    assert intent.order == "desc", f"Expected intent.order == 'desc', got {intent.order}"

    records = res.result.get("data", {}).get("records", [])
    assert len(records) > 0, "No records returned from execution"
    
    res_df = pd.DataFrame(records)
    assert (res_df["sales_amount"] > 500).all(), "Found sales_amount <= 500 in filtered results"
    
    sales_values = res_df["sales_amount"].tolist()
    expected_desc = sorted(sales_values, reverse=True)
    assert sales_values == expected_desc, f"Dataframe is NOT sorted descending! Got {sales_values}"


def test_query_2_explicit_ascending():
    """
    Query 2 (Explicit Ascending):
    'Filter sales_amount > 500 and sort by sales_amount ascending'
    Verify: intent.order == 'asc' and dataframe actually sorted ascending.
    """
    query = "Filter sales_amount > 500 and sort by sales_amount ascending"
    df = create_test_sales_dataset()

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED", f"Query execution failed: {res.error}"
    assert res.result is not None, "Query response result is None"
    
    intent = SemanticIntent(**res.plan)
    assert intent.order == "asc", f"Expected intent.order == 'asc', got {intent.order}"

    records = res.result.get("data", {}).get("records", [])
    assert len(records) > 0, "No records returned from execution"
    
    res_df = pd.DataFrame(records)
    assert (res_df["sales_amount"] > 500).all(), "Found sales_amount <= 500 in filtered results"
    
    sales_values = res_df["sales_amount"].tolist()
    expected_asc = sorted(sales_values)
    assert sales_values == expected_asc, f"Dataframe is NOT sorted ascending! Got {sales_values}"


def test_query_3_no_ordering_specified():
    """
    Query 3 (No ordering specified):
    'Calculate total sales_amount by region'
    Verify: intent.order is None and no arbitrary sorting is forced on the dataset.
    """
    query = "Calculate total sales_amount by region"
    df = create_test_sales_dataset()

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED", f"Query execution failed: {res.error}"
    assert res.result is not None, "Query response result is None"
    
    intent = SemanticIntent(**res.plan)
    assert intent.order is None, f"Expected intent.order is None, got {intent.order}"
    
    print(f"\n[Semantic Intent]")
    print(f" - Primary Intent: {intent.primary_intent}")
    print(f" - Dimensions: {intent.dimensions}")
    print(f" - Metrics: {intent.metrics}")
    print(f" - Aggregation: {intent.aggregation}")
    print(f" - Order: {intent.order}")

    assert intent.order is None, f"Expected intent.order is None, got {intent.order}"

    # 2. End-to-End Orchestration & Execution verification
    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    assert res.status == "SUCCEEDED", f"Query execution failed: {res.error}"
    assert res.result is not None, "Query response result is None"
    
    generated_code = res.result.get("code", "")
    print(f"\n[Generated Code]:\n{generated_code}")
    
    # Verify no arbitrary metric sorting was injected in code
    assert "sort_values(by='sales_amount'" not in generated_code, (
        "Arbitrary sorting by metric column was unexpectedly generated when order=None!"
    )

    records = res.result.get("data", {}).get("records", [])
    assert len(records) > 0, "No records returned from execution"
    
    res_df = pd.DataFrame(records)
    print("\n[Execution Result DataFrame]:")
    print(res_df.to_string(index=False))
    
    # Calculate ground truth groupby sums
    expected_totals = df.groupby("region")["sales_amount"].sum().reset_index()
    
    # Check that aggregated metric matches ground truth
    merged = pd.merge(res_df, expected_totals, on="region", suffixes=("_actual", "_expected"))
    assert len(merged) == len(expected_totals), "Group aggregation missing regions"
    np.testing.assert_allclose(
        merged["sales_amount_actual"].values,
        merged["sales_amount_expected"].values,
        err_msg="Calculated group totals do not match expected totals"
    )

    print(f">> [PASS] Test Case 3: intent.order is None and dataset preserves natural groupby without arbitrary sorting.")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("RUNNING TRI-STATE ORDERING AUDIT TEST SUITE (desc, asc, None)")
    print("#" * 80)
    
    test_query_1_explicit_descending()
    test_query_2_explicit_ascending()
    test_query_3_no_ordering_specified()
    
    print("\n" + "=" * 80)
    print("ALL 3 TRI-STATE ORDERING TEST CASES PASSED 100% CLEANLY [OK]")
    print("=" * 80 + "\n")
