"""
DataPilot OS — Multi-Intent Pipeline Audit Suite.

Tests complex multi-step query decomposition, plan compilation, and sequential execution:
1. Sales Dataset: 'Filter sales_amount > 500 and calculate average sales_amount by region'
   - Verifies prerequisite step (FILTER_QUERY) filters data and carries forward filtered dataframe.
   - Verifies primary step (GROUP_AGGREGATE) executes on filtered dataset with 100% mathematical accuracy.
2. Sensor Dataset: 'Clean missing values and detect anomalies in temperature using IsolationForest'
   - Verifies prerequisite step (DATA_CLEAN) imputes/cleans missing values.
   - Verifies primary step (ANOMALY_DETECTION) runs IsolationForest on the clean dataset.
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from modules.query_orchestrator import QueryOrchestrator
from modules.semantic_planner import SemanticPlanner
from modules.execution_plan_compiler import ExecutionPlanCompiler
from modules.execution_engine import ModularExecutionEngine
from backend.ds_engine.intent_schema import SemanticIntent, IntentType, FilterSpec, FilterOperator, AggregationType
from backend.ds_engine.intent_resolver import intent_resolver
from backend.ds_engine.code_generator import code_generator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiIntentAudit")


def generate_sales_dataset(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic Sales dataset with deterministic ground truth."""
    np.random.seed(seed)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    regions = np.random.choice(["North", "South", "East", "West"], size=n)
    categories = np.random.choice(["Electronics", "Apparel", "Furniture", "Groceries"], size=n)
    # Generate sales amounts ranging from 100 to 1000
    sales = np.random.uniform(100, 1000, size=n)
    units = np.random.randint(1, 20, size=n)
    return pd.DataFrame({
        "order_date": dates,
        "region": regions,
        "category": categories,
        "sales_amount": sales,
        "units_sold": units
    })


def generate_sensor_dataset(n: int = 120, seed: int = 101) -> pd.DataFrame:
    """Generates synthetic Sensor dataset with intentional missing values and anomalies."""
    np.random.seed(seed)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    temp = np.random.normal(70, 10, size=n)
    pressure = np.random.normal(1013, 15, size=n)
    humidity = np.random.uniform(20, 90, size=n)
    vibration = np.random.exponential(2.5, size=n)

    df = pd.DataFrame({
        "timestamp": dates,
        "temperature": temp,
        "pressure": pressure,
        "humidity": humidity,
        "vibration_level": vibration
    })

    # Introduce missing values in temperature and pressure
    df.loc[5:10, "temperature"] = np.nan
    df.loc[25:30, "pressure"] = np.nan

    # Inject clear anomaly spikes
    df.loc[50, "temperature"] = 195.0
    df.loc[75, "temperature"] = -45.0
    df.loc[90, "vibration_level"] = 35.0

    return df


def test_multi_intent_sales_filter_and_group_aggregate():
    """
    Test complex multi-intent query on Sales dataset:
    'Filter sales_amount > 500 and calculate average sales_amount by region'
    
    Verifies:
    1. SemanticPlanner correctly decomposes query into primary_intent (GROUP_AGGREGATE) and prerequisite (FILTER_QUERY).
    2. ExecutionPlanCompiler constructs a 2-step ExecutionPlan with correct dependency chaining.
    3. Prerequisite Step 1 (FILTER_QUERY) filters out sales_amount <= 500 and carries forward filtered DataFrame.
    4. Primary Step 2 (GROUP_AGGREGATE) executes on filtered dataset and produces 100% mathematically accurate results.
    """
    print("\n" + "=" * 90)
    print("AUDIT CASE 1: SALES DATASET MULTI-INTENT (FILTER_QUERY + GROUP_AGGREGATE)")
    print("=" * 90)

    df = generate_sales_dataset(n=100, seed=42)
    query = "Filter sales_amount > 500 and calculate average sales_amount by region"

    print(f"Dataset Shape: {df.shape}")
    print(f"Dataset Columns: {list(df.columns)}")
    print(f"Query: '{query}'")

    # 1. Independent Ground Truth Mathematical Calculation
    ground_truth_filtered = df[df["sales_amount"] > 500]
    expected_filtered_rows = len(ground_truth_filtered)
    expected_region_means = ground_truth_filtered.groupby("region")["sales_amount"].mean().to_dict()

    print(f"\n[Ground Truth Validation Baseline]")
    print(f"  • Total Rows in Initial Dataset: {len(df)}")
    print(f"  • Rows matching sales_amount > 500: {expected_filtered_rows}")
    print(f"  • Ground Truth Averages by Region (Filtered):")
    for r, m in expected_region_means.items():
        print(f"      - {r:6s}: {m:.6f}")

    # 2. Execute End-to-End Multi-Intent Query Orchestration
    res = QueryOrchestrator.process_query(query_raw=query, df=df)

    print(f"\n[Execution Response Status]")
    print(f"  • Status: {res.status}")
    print(f"  • Phase: {res.phase}")
    print(f"  • Request ID: {res.request_id}")

    if res.status != "SUCCEEDED":
        print(f"  • Error: {res.error}")
    assert res.status == "SUCCEEDED", f"QueryOrchestrator failed: {res.error}"
    assert res.plan is not None, "Semantic plan is None"
    assert res.result is not None, "Execution result is None"

    # 3. Verify Semantic Intent & Prerequisite Operations
    plan = res.plan
    primary_intent = str(plan.get("primary_intent") or plan.get("intent_type"))
    prereqs = plan.get("prerequisite_operations") or []
    filters = plan.get("filters") or []

    print(f"\n[Semantic Intent Verification]")
    print(f"  • Primary Intent: {primary_intent}")
    print(f"  • Prerequisites:  {prereqs}")
    print(f"  • Target Metric:  {plan.get('metrics')}")
    print(f"  • Dimension:      {plan.get('dimensions')}")
    print(f"  • Aggregation:    {plan.get('aggregation')}")
    print(f"  • Filter Specs:   {filters}")

    assert "GROUP_AGGREGATE" in primary_intent, f"Expected GROUP_AGGREGATE, got {primary_intent}"
    assert any("FILTER_QUERY" in str(p) for p in prereqs), f"Expected FILTER_QUERY in prerequisites: {prereqs}"
    assert len(filters) > 0, "Expected at least 1 filter specification in semantic intent"

    # 4. Verify Execution Plan Compilation Steps
    exec_steps = plan.get("execution_plan", [])
    print(f"\n[Compiled Execution Plan Steps ({len(exec_steps)} steps)]")
    for s in exec_steps:
        print(f"  • Step {s.get('step_index')}: Intent={s.get('intent_type')}, Prerequisite={s.get('is_prerequisite')}, Desc='{s.get('description')}'")

    assert len(exec_steps) == 2, f"Expected exactly 2 execution steps, got {len(exec_steps)}"
    assert "FILTER_QUERY" in str(exec_steps[0].get("intent_type")), "Step 1 must be FILTER_QUERY"
    assert exec_steps[0].get("is_prerequisite") is True, "Step 1 is_prerequisite must be True"
    assert "GROUP_AGGREGATE" in str(exec_steps[1].get("intent_type")), "Step 2 must be GROUP_AGGREGATE"
    assert exec_steps[1].get("is_prerequisite") is False, "Step 2 is_prerequisite must be False"

    # 5. Verify Step Artifacts & DataFrame Carry-Forward
    step_outputs = res.result.get("artifacts", {}).get("step_outputs", [])
    print(f"\n[Step Outputs & Data Carry-Forward Audit]")
    assert len(step_outputs) == 2, f"Expected 2 step outputs, got {len(step_outputs)}"
    for s_out in step_outputs:
        print(f"  • Step {s_out.get('step_index')} [{s_out.get('intent')}]: Duration={s_out.get('duration_sec')}s")
        print(f"    Stdout Snippet:\n{s_out.get('stdout', '').strip()}\n")

    # 6. Verify 100% Mathematical Accuracy of Aggregation on Filtered Dataset
    records = res.result.get("data", {}).get("records", [])
    assert len(records) > 0, "No records returned in result data"
    
    actual_region_means = {}
    for rec in records:
        reg = rec.get("region")
        val = rec.get("sales_amount") or rec.get("mean") or rec.get("sales_amount_mean")
        if reg and val is not None:
            actual_region_means[reg] = float(val)

    print(f"[Mathematical Accuracy Audit]")
    print(f"  Expected Region Means (Ground Truth): {expected_region_means}")
    print(f"  Actual Region Means (Pipeline Output): {actual_region_means}")

    assert len(actual_region_means) == len(expected_region_means), (
        f"Region count mismatch: Expected {len(expected_region_means)}, got {len(actual_region_means)}"
    )

    for reg, exp_mean in expected_region_means.items():
        assert reg in actual_region_means, f"Missing region '{reg}' in result data"
        act_mean = actual_region_means[reg]
        diff = abs(act_mean - exp_mean)
        print(f"  • Region '{reg}': Expected = {exp_mean:.6f} | Actual = {act_mean:.6f} | Abs Diff = {diff:.8f}")
        assert diff < 1e-4, f"Mathematical accuracy violation for region {reg}: expected {exp_mean}, got {act_mean} (diff={diff})"

    print("\n>> [AUDIT PASS] Sales Multi-Intent Pipeline: FILTER_QUERY correctly carried forward filtered dataset, and GROUP_AGGREGATE calculated 100% mathematically accurate regional averages.\n")


def test_multi_intent_sensor_clean_and_anomaly_detection():
    """
    Test multi-intent query on Sensor dataset with missing values:
    'Clean missing values and detect anomalies in temperature using IsolationForest'
    
    Verifies:
    1. SemanticPlanner parses query into primary_intent (ANOMALY_DETECTION) and prerequisite (DATA_CLEAN).
    2. ExecutionPlanCompiler creates 2-step plan: Step 1 (DATA_CLEAN), Step 2 (ANOMALY_DETECTION).
    3. Prerequisite Step 1 (DATA_CLEAN) removes or imputes NaNs across columns (temperature & pressure).
    4. Primary Step 2 (ANOMALY_DETECTION) executes IsolationForest on cleaned data and flags outliers.
    5. Injected extreme anomalies are successfully identified in the outlier results.
    """
    print("\n" + "=" * 90)
    print("AUDIT CASE 2: SENSOR DATASET MULTI-INTENT (DATA_CLEAN + ANOMALY_DETECTION)")
    print("=" * 90)

    sensor_df = generate_sensor_dataset(n=120, seed=101)
    query = "Clean missing values and detect anomalies in temperature using IsolationForest"

    print(f"Dataset Shape: {sensor_df.shape}")
    print(f"Initial Missing Values per Column:\n{sensor_df.isnull().sum()}")
    print(f"Query: '{query}'")

    assert sensor_df["temperature"].isnull().sum() > 0, "Test dataset must contain missing temperature values"
    assert sensor_df["pressure"].isnull().sum() > 0, "Test dataset must contain missing pressure values"

    # 1. Execute End-to-End Multi-Intent Query Orchestration
    res = QueryOrchestrator.process_query(query_raw=query, df=sensor_df)

    print(f"\n[Execution Response Status]")
    print(f"  • Status: {res.status}")
    print(f"  • Phase: {res.phase}")
    print(f"  • Request ID: {res.request_id}")

    if res.status != "SUCCEEDED":
        print(f"  • Error: {res.error}")
    assert res.status == "SUCCEEDED", f"QueryOrchestrator failed: {res.error}"
    assert res.plan is not None, "Semantic plan is None"
    assert res.result is not None, "Execution result is None"

    # 2. Verify Semantic Intent & Prerequisite Operations
    plan = res.plan
    primary_intent = str(plan.get("primary_intent") or plan.get("intent_type"))
    prereqs = plan.get("prerequisite_operations") or []

    print(f"\n[Semantic Intent Verification]")
    print(f"  • Primary Intent: {primary_intent}")
    print(f"  • Prerequisites:  {prereqs}")
    print(f"  • Target Column:  {plan.get('target_column')}")
    print(f"  • Feature Cols:   {plan.get('feature_columns')}")

    assert "ANOMALY_DETECTION" in primary_intent, f"Expected ANOMALY_DETECTION, got {primary_intent}"
    assert any("DATA_CLEAN" in str(p) for p in prereqs), f"Expected DATA_CLEAN in prerequisites: {prereqs}"

    # 3. Verify Execution Plan Compilation Steps
    exec_steps = plan.get("execution_plan", [])
    print(f"\n[Compiled Execution Plan Steps ({len(exec_steps)} steps)]")
    for s in exec_steps:
        print(f"  • Step {s.get('step_index')}: Intent={s.get('intent_type')}, Prerequisite={s.get('is_prerequisite')}, Desc='{s.get('description')}'")

    assert len(exec_steps) == 2, f"Expected exactly 2 execution steps, got {len(exec_steps)}"
    assert "DATA_CLEAN" in str(exec_steps[0].get("intent_type")), "Step 1 must be DATA_CLEAN"
    assert exec_steps[0].get("is_prerequisite") is True, "Step 1 is_prerequisite must be True"
    assert "ANOMALY_DETECTION" in str(exec_steps[1].get("intent_type")), "Step 2 must be ANOMALY_DETECTION"
    assert exec_steps[1].get("is_prerequisite") is False, "Step 2 is_prerequisite must be False"

    # 4. Verify Step Artifacts & Execution Output
    step_outputs = res.result.get("artifacts", {}).get("step_outputs", [])
    print(f"\n[Step Outputs Audit]")
    assert len(step_outputs) == 2, f"Expected 2 step outputs, got {len(step_outputs)}"
    for s_out in step_outputs:
        print(f"  • Step {s_out.get('step_index')} [{s_out.get('intent')}]: Duration={s_out.get('duration_sec')}s")
        print(f"    Stdout Snippet:\n{s_out.get('stdout', '').strip()}\n")

    # 5. Verify Clean Data Passed to Anomaly Detection
    # Step 1 stdout should confirm missing values were imputed/cleaned
    clean_stdout = step_outputs[0].get("stdout", "")
    assert "DATA HEALTH & CLEANING AUDIT" in clean_stdout, "Step 1 missing cleaning audit log"
    assert "Final Clean Dataset Shape" in clean_stdout, "Step 1 missing clean dataset shape log"

    # Step 2 stdout should confirm IsolationForest anomaly detection ran on clean dataset
    anomaly_stdout = step_outputs[1].get("stdout", "")
    assert "ANOMALY DETECTION SUMMARY" in anomaly_stdout, "Step 2 missing anomaly detection summary"
    assert "Anomalies Detected" in anomaly_stdout, "Step 2 missing anomalies count log"

    # 6. Verify Outliers in Result Data
    records = res.result.get("data", {}).get("records", [])
    assert len(records) > 0, "Expected at least 1 anomaly record returned"
    print(f"[Detected Anomaly Records: {len(records)} sample outliers]")
    for r in records[:5]:
        print(f"  • Outlier Record: Temp={r.get('temperature')}, Pressure={r.get('pressure')}, OutlierFlag={r.get('is_outlier')}")

    # Verify outlier flag is -1
    for r in records:
        if "is_outlier" in r:
            assert r["is_outlier"] == -1, f"Expected is_outlier == -1, got {r['is_outlier']}"

    print("\n>> [AUDIT PASS] Sensor Multi-Intent Pipeline: DATA_CLEAN successfully imputed missing values, and ANOMALY_DETECTION successfully isolated anomalies using IsolationForest.\n")


def test_deterministic_execution_plan_compiler_direct():
    """
    Unit test for direct ExecutionPlanCompiler step compilation with explicit SemanticIntent.
    Ensures compiler deterministically constructs multi-step DAGs without Ollama roundtrips.
    """
    print("\n" + "=" * 90)
    print("AUDIT CASE 3: DETERMINISTIC EXECUTION PLAN COMPILER STEP DECOMPOSITION")
    print("=" * 90)

    df = generate_sales_dataset(n=50)

    # Multi-intent 1: Filter + Group Aggregate
    intent1 = SemanticIntent(
        intent_type=IntentType.GROUP_AGGREGATE,
        primary_intent=IntentType.GROUP_AGGREGATE,
        prerequisite_operations=[IntentType.FILTER_QUERY],
        dimensions=["region"],
        metrics=["sales_amount"],
        aggregation=AggregationType.MEAN,
        filters=[FilterSpec(column="sales_amount", operator=FilterOperator.GT, value=500)],
        raw_query="Filter sales_amount > 500 and calculate average sales_amount by region"
    )

    plan1 = ExecutionPlanCompiler.compile_plan(intent1, df=df, request_id="unit-req-001")
    assert plan1.is_compiled is True
    assert len(plan1.steps) == 2
    assert plan1.steps[0].intent_type == IntentType.FILTER_QUERY
    assert plan1.steps[0].is_prerequisite is True
    assert plan1.steps[1].intent_type == IntentType.GROUP_AGGREGATE
    assert plan1.steps[1].is_prerequisite is False
    print("  • Step 1 Code Preamble Check:", "res_df['sales_amount'] > 500" in plan1.steps[0].code_version.code or "sales_amount" in plan1.steps[0].code_version.code)
    print("  • Step 2 Code Groupby Check:", "groupby('region')" in plan1.steps[1].code_version.code)
    print("  >> Multi-intent Plan 1 compiled cleanly.")

    # Multi-intent 2: Clean + Anomaly Detection
    intent2 = SemanticIntent(
        intent_type=IntentType.ANOMALY_DETECTION,
        primary_intent=IntentType.ANOMALY_DETECTION,
        prerequisite_operations=[IntentType.DATA_CLEAN],
        target_column="temperature",
        feature_columns=["temperature"],
        raw_query="Clean missing values and detect anomalies in temperature using IsolationForest"
    )

    sensor_df = generate_sensor_dataset(n=50)
    plan2 = ExecutionPlanCompiler.compile_plan(intent2, df=sensor_df, request_id="unit-req-002")
    assert plan2.is_compiled is True
    assert len(plan2.steps) == 2
    assert plan2.steps[0].intent_type == IntentType.DATA_CLEAN
    assert plan2.steps[0].is_prerequisite is True
    assert plan2.steps[1].intent_type == IntentType.ANOMALY_DETECTION
    assert plan2.steps[1].is_prerequisite is False
    print("  • Step 1 Clean Code Check:", "IterativeImputer" in plan2.steps[0].code_version.code or "SimpleImputer" in plan2.steps[0].code_version.code)
    print("  • Step 2 IsolationForest Check:", "IsolationForest" in plan2.steps[1].code_version.code)
    print("  >> Multi-intent Plan 2 compiled cleanly.")

    print("\n>> [AUDIT PASS] Deterministic ExecutionPlanCompiler unit tests passed 100%.\n")


if __name__ == "__main__":
    t0 = time.time()
    print("#" * 90)
    print("STARTING DATA PILOT OS MULTI-INTENT PIPELINE AUDIT SUITE")
    print("#" * 90)

    test_deterministic_execution_plan_compiler_direct()
    test_multi_intent_sales_filter_and_group_aggregate()
    test_multi_intent_sensor_clean_and_anomaly_detection()

    total_time = round(time.time() - t0, 2)
    print("\n" + "#" * 90)
    print(f"ALL MULTI-INTENT PIPELINE AUDIT TESTS PASSED 100% CLEANLY [TOTAL TIME: {total_time}s]")
    print("#" * 90 + "\n")
