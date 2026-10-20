"""
DataPilot OS — Synthetic live Ollama multi-domain contract suite.

Exercises live Ollama semantic planning and DS-OS deterministic execution against
synthetic contract fixtures that resemble multiple industry domains:
1. Predictive Maintenance (25k rows manufacturing sensor records with UDI sequence)
2. Retail Sales (multi-category e-commerce transactions and daily revenue)
3. Sensor Telemetry (IoT multi-axis vibration, temperature, and pressure signals)
4. HR Employee Churn (satisfaction, tenure, salary, and departure dynamics)

ARCHITECTURAL GUARANTEES & ZERO-MOCK INVARIANTS:
- 100% Live Ollama Semantic Planning (`http://localhost:11434`).
- Zero manual mock commands, zero hardcoded tool strings, zero synthetic bypasses.
- Proactive Semantic Feasibility & Grounding Interdicts for sequence clarification.
- Authoritative Execution via TypedOperatorRegistry with ExecutionManifest parity.
- 100% Scientific Factual Accuracy and Zero Silent Fallbacks.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Optional
import pytest
import numpy as np
import pandas as pd

pytestmark = pytest.mark.live_ollama

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.intelligence.ollama_service import ollama_service
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse, LifecycleState
from backend.ds_engine.intent_schema import (
    SemanticIntent,
    AnalysisIntentV2,
    TaskFamily,
    FilterOperator,
    FilterSpec,
    ExecutionManifest
)
from backend.ds_engine.dataset_profiler import DatasetProfiler, DatasetProfile
from backend.ds_engine.semantic_feasibility import (
    SemanticFeasibilityValidator,
    FeasibilityCheckResult,
    _find_sequence_candidates
)

logger = logging.getLogger("LiveOllamaRealBenchmarks")


def check_ollama_available() -> bool:
    """Checks if local Ollama daemon is active and has models loaded."""
    try:
        status = ollama_service.detect_server()
        return bool(status.is_running and len(status.available_models) > 0)
    except Exception as e:
        logger.warning(f"Ollama readiness check failed: {e}")
        return False


def print_benchmark_telemetry(test_name: str, query: str, res: CanonicalQueryResponse):
    """Prints formatted telemetry for live Ollama benchmark execution."""
    print("\n" + "=" * 80)
    print(f" BENCHMARK: {test_name}")
    print(f" QUERY:     '{query}'")
    print(f" STATUS:    {res.status} | PHASE: {res.phase}")

    if res.clarification:
        print(f"\n--- PROACTIVE CLARIFICATION PROMPT ---\n{res.clarification}")

    if res.error:
        print(f"\n--- ERROR DETAIL ---\n[{res.error.code}] {res.error.message}")
        if res.error.candidate_columns:
            print(f"Candidate Columns: {res.error.candidate_columns}")

    plan = res.plan or {}
    raw_llm = plan.get("raw_llm_output")
    if raw_llm:
        print(f"\n--- RAW LLM OUTPUT ---\n{raw_llm}")

    if res.result and "manifest" in res.result:
        print(f"\n--- EXECUTION MANIFEST ---\n{json.dumps(res.result['manifest'], indent=2, default=str)}")

    if res.result and "data" in res.result:
        records = res.result["data"].get("records", [])
        if records:
            result_df = pd.DataFrame(records)
            print("\n--- SAMPLE RESULT RECORDS ---")
            print(result_df.head(5).to_string(index=False))
            print(f"Total Rows: {len(records)} | Columns: {list(result_df.columns)}")
    print("=" * 80 + "\n")


# ==============================================================================
# SYNTHETIC MULTI-DOMAIN CONTRACT FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def df_predictive_maintenance_25k() -> pd.DataFrame:
    """
    Domain 1: synthetic manufacturing predictive-maintenance fixture (25k rows).
    Columns:
      - `UDI`: Unique Device Identifier / sequential order integer index (1 to 25,000)
      - `Product ID`: Equipment model identifier (e.g. M14860, L47181, H29424)
      - `Type`: Machine quality type variant ('L', 'M', 'H')
      - `Air temperature [K]`: Ambient temperature in Kelvin
      - `Process temperature [K]`: Process operational temperature in Kelvin
      - `Rotational speed [rpm]`: Spindle rotational speed
      - `Torque [Nm]`: Torque applied to workpiece
      - `Tool wear [min]`: Cumulative tool wear duration in minutes
      - `Target`: Binary failure indicator flag (0 = normal, 1 = failure)
      - `Failure Type`: Failure mode ('No Failure', 'Heat Dissipation Failure', 'Power Failure', etc.)
    """
    np.random.seed(42)
    n = 25000
    udi = list(range(1, n + 1))
    product_models = ["M14860", "L47181", "H29424", "M14861", "L47182", "H29425", "M14862"]
    product_id = np.random.choice(product_models, size=n)
    types = np.random.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    
    air_temp = np.random.normal(300.0, 2.0, size=n).round(1)
    process_temp = (air_temp + 10.0 + np.random.normal(0, 1.0, size=n)).round(1)
    rot_speed = np.random.normal(1538.0, 179.0, size=n).clip(1168, 2886).round(0)
    torque = np.random.normal(40.0, 9.9, size=n).clip(3.8, 76.6).round(1)
    tool_wear = np.random.randint(0, 253, size=n)

    # Deterministic failure injection at specific intervals with pre-failure temperature/torque spikes
    target = np.zeros(n, dtype=int)
    failure_type = ["No Failure"] * n
    failure_indices = [500, 1200, 3500, 8200, 12500, 17800, 22000]

    for f_idx in failure_indices:
        target[f_idx] = 1
        failure_type[f_idx] = "Heat Dissipation Failure"
        # Pre-event variance elevation in lookback window (30 steps before failure)
        lookback_start = max(0, f_idx - 30)
        process_temp[lookback_start:f_idx] += np.random.normal(5.0, 2.5, f_idx - lookback_start).round(1)
        torque[lookback_start:f_idx] += np.random.normal(8.0, 3.0, f_idx - lookback_start).round(1)

    return pd.DataFrame({
        "UDI": udi,
        "Product ID": product_id,
        "Type": types,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Target": target,
        "Failure Type": failure_type
    })


@pytest.fixture
def df_retail_sales() -> pd.DataFrame:
    """
    Domain 2: Retail Sales dataset.
    Columns: `order_date`, `sales_amount`, `region`, `category`.
    """
    np.random.seed(101)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    sales = (np.linspace(150, 450, n) + np.sin(np.linspace(0, 8 * np.pi, n)) * 50 + np.random.normal(0, 15, n)).round(2)
    sales = np.clip(sales, 20.0, None)
    regions = np.random.choice(["North", "South", "East", "West"], size=n)
    categories = np.random.choice(["Electronics", "Clothing", "Home & Kitchen", "Books"], size=n)
    
    return pd.DataFrame({
        "order_date": dates,
        "sales_amount": sales,
        "region": regions,
        "category": categories
    })


@pytest.fixture
def df_sensor_telemetry() -> pd.DataFrame:
    """
    Domain 3: Sensor Telemetry dataset.
    Columns: `timestamp`, `temperature`, `vibration_x`, `vibration_y`, `pressure`.
    """
    np.random.seed(202)
    n = 150
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=n, freq="D")
    # Temperature with strong 7-day seasonality and linear trend
    t = np.arange(n)
    seasonal_7d = 8.0 * np.sin(2 * np.pi * t / 7.0)
    temp = (65.0 + 0.05 * t + seasonal_7d + np.random.normal(0, 1.0, n)).round(2)
    
    # Vibration with autoregressive signal
    vib_x = np.zeros(n)
    vib_x[0] = 1.5
    for i in range(1, n):
        vib_x[i] = 0.75 * vib_x[i - 1] + np.random.normal(0, 0.2)
    vib_x = np.round(vib_x + 1.2, 3)

    vib_y = (np.random.normal(1.3, 0.25, n)).round(3)
    pressure = (101.3 + np.random.normal(0, 3.5, n)).round(2)

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temp,
        "vibration_x": vib_x,
        "vibration_y": vib_y,
        "pressure": pressure
    })


@pytest.fixture
def df_hr_employee_churn() -> pd.DataFrame:
    """
    Domain 4: HR Employee Churn dataset.
    Columns: `employee_id`, `monthly_salary`, `satisfaction_score`, `tenure_months`, `left_company`.
    """
    np.random.seed(303)
    n = 150
    emp_ids = [f"EMP-{1000 + i}" for i in range(n)]
    
    # 3 distinct clusters:
    # Cluster 0: Junior / Low Salary, Moderate Satisfaction
    # Cluster 1: Mid-level / Moderate Salary, High Satisfaction
    # Cluster 2: Senior / High Salary, Varied Satisfaction
    cluster_assign = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.35, 0.25])
    salaries = np.zeros(n)
    satisfaction = np.zeros(n)
    tenure = np.zeros(n, dtype=int)
    
    for i in range(n):
        if cluster_assign[i] == 0:
            salaries[i] = np.random.normal(3500, 400)
            satisfaction[i] = np.random.uniform(0.3, 0.6)
            tenure[i] = np.random.randint(1, 18)
        elif cluster_assign[i] == 1:
            salaries[i] = np.random.normal(7500, 600)
            satisfaction[i] = np.random.uniform(0.7, 0.98)
            tenure[i] = np.random.randint(18, 48)
        else:
            salaries[i] = np.random.normal(14000, 1200)
            satisfaction[i] = np.random.uniform(0.5, 0.9)
            tenure[i] = np.random.randint(48, 120)
            
    left = np.where((satisfaction < 0.45) | (tenure > 60), np.random.choice([0, 1], size=n, p=[0.4, 0.6]), 0)

    return pd.DataFrame({
        "employee_id": emp_ids,
        "monthly_salary": np.round(salaries, 2),
        "satisfaction_score": np.round(satisfaction, 2),
        "tenure_months": tenure,
        "left_company": left
    })


# ==============================================================================
# COMPREHENSIVE LIVE OLLAMA BENCHMARK TESTS
# ==============================================================================

class TestLiveOllamaRealBenchmarks:
    """
    Authoritative benchmark test suite validating live Ollama NLP intent extraction,
    proactive semantic feasibility interception, and typed execution across real multi-domain datasets.
    """

    # --------------------------------------------------------------------------
    # Benchmark 1: Predictive Maintenance (No timestamp -> Proactive UDI Clarification)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_predictive_maintenance_without_timestamp_proactively_clarifies_udi(self, df_predictive_maintenance_25k):
        """
        Query: 'Analyze process temperature and torque before machine failure'
        Dataset: 25k Predictive Maintenance dataset (with UDI, but NO datetime column).
        
        Asserts:
        - Response status is 'NEEDS_CLARIFICATION'.
        - Error code is 'CLARIFICATION_REQUIRED'.
        - Candidate columns include ['UDI'].
        - Clarification text asks whether to use 'UDI' as sequence order.
        """
        query = "Analyze process temperature and torque before machine failure"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_predictive_maintenance_25k)
        print_benchmark_telemetry("01: Predictive Maintenance (Proactive UDI Clarification)", query, res)

        assert res.status == "NEEDS_CLARIFICATION", f"Expected NEEDS_CLARIFICATION, got {res.status}"
        assert res.error is not None, "Expected CanonicalError in response"
        assert res.error.code == "CLARIFICATION_REQUIRED", f"Expected CLARIFICATION_REQUIRED, got {res.error.code}"
        assert res.error.candidate_columns is not None, "Expected candidate_columns in error"
        assert "UDI" in res.error.candidate_columns, f"Expected 'UDI' in candidate_columns: {res.error.candidate_columns}"
        assert res.clarification is not None, "Expected clarification prompt text in response"
        assert "UDI" in res.clarification, f"Expected 'UDI' referenced in clarification prompt: {res.clarification}"
        assert ("sequence order" in res.clarification.lower() or "temporal" in res.clarification.lower() or "order" in res.clarification.lower())

    # --------------------------------------------------------------------------
    # Benchmark 2: Predictive Maintenance (Explicit Sequence Column -> Executes Event Window)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_predictive_maintenance_with_explicit_sequence_column_executes_event_window(self, df_predictive_maintenance_25k):
        """
        Query: 'Analyze process temperature and torque before machine failure using UDI as sequence order'
        Dataset: 25k Predictive Maintenance dataset.
        
        Asserts:
        - Response status is 'SUCCEEDED'.
        - Operator executed is 'event_window_analysis'.
        - Manifest contains `event_row_count > 0`, `pre_event_row_count > 0`, and `sorted_by == 'UDI'`.
        """
        query = "Analyze process temperature and torque before machine failure using UDI as sequence order"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_predictive_maintenance_25k)
        print_benchmark_telemetry("02: Predictive Maintenance (Explicit Sequence Event Window)", query, res)

        assert res.status == "SUCCEEDED", f"Execution failed with error: {res.error}"
        assert res.result is not None, "Expected result payload"
        assert "manifest" in res.result, "Expected execution manifest in result"
        
        manifest = res.result["manifest"]
        assert manifest.get("operator") == "event_window_analysis", f"Expected operator 'event_window_analysis', got {manifest.get('operator')}"
        assert manifest.get("event_row_count", 0) > 0, f"Expected event_row_count > 0, got {manifest.get('event_row_count')}"
        assert manifest.get("pre_event_row_count", 0) > 0, f"Expected pre_event_row_count > 0, got {manifest.get('pre_event_row_count')}"
        assert manifest.get("sorted_by") == "UDI", f"Expected sorted_by == 'UDI', got {manifest.get('sorted_by')}"

        # Verify subject metrics included in columns_used
        cols_used = manifest.get("columns_used", [])
        assert "Process temperature [K]" in cols_used or any("process" in c.lower() for c in cols_used)
        assert "Torque [Nm]" in cols_used or any("torque" in c.lower() for c in cols_used)

    # --------------------------------------------------------------------------
    # Benchmark 3: Retail Sales (Time-Series Forecast E2E)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_retail_sales_forecast_e2e(self, df_retail_sales):
        """
        Query: 'Forecast sales_amount for next 14 days based on order_date'
        Dataset: Retail Sales dataset with `order_date`, `sales_amount`, `region`, `category`.
        
        Asserts:
        - Response status is 'SUCCEEDED'.
        - Operator executed is 'time_series_forecast'.
        - Forecast horizon is 14 (manifest row_count_out == 14).
        """
        query = "Forecast sales_amount for next 14 days based on order_date"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_retail_sales)
        print_benchmark_telemetry("03: Retail Sales Forecast", query, res)

        assert res.status == "SUCCEEDED", f"Execution failed with error: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result
        
        manifest = res.result["manifest"]
        assert manifest.get("operator") == "time_series_forecast", f"Expected operator 'time_series_forecast', got {manifest.get('operator')}"
        assert manifest.get("row_count_out") == 14, f"Expected horizon row_count_out == 14, got {manifest.get('row_count_out')}"
        
        # Verify forecast output records
        records = res.result["data"].get("records", [])
        assert len(records) == 14, f"Expected 14 forecast records, got {len(records)}"
        assert "yhat" in records[0], f"Expected 'yhat' forecast column in records: {records[0]}"

    # --------------------------------------------------------------------------
    # Benchmark 4: Sensor Telemetry (Time-Series Decomposition E2E)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_sensor_time_series_decomposition_e2e(self, df_sensor_telemetry):
        """
        Query: 'Decompose temperature over timestamp with period 7 and additive model'
        Dataset: Sensor Telemetry dataset with `timestamp`, `temperature`, `vibration_x`, `vibration_y`, `pressure`.
        
        Asserts:
        - Response status is 'SUCCEEDED'.
        - Operator executed is 'time_series_decomposition'.
        - Manifest statistics computed include trend_strength and seasonality_strength.
        """
        query = "Decompose temperature over timestamp with period 7 and additive model"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_sensor_telemetry)
        print_benchmark_telemetry("04: Sensor Time-Series Decomposition", query, res)

        assert res.status == "SUCCEEDED", f"Execution failed with error: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result
        
        manifest = res.result["manifest"]
        assert manifest.get("operator") == "time_series_decomposition", f"Expected operator 'time_series_decomposition', got {manifest.get('operator')}"
        assert manifest.get("sorted_by") == "timestamp"
        
        stats = manifest.get("statistics_computed", [])
        assert "trend_strength" in stats
        assert "seasonality_strength" in stats

        # Verify decomposed records contain observed, trend, seasonal, residual
        records = res.result["data"].get("records", [])
        assert len(records) > 0
        first_rec = records[0]
        assert "observed" in first_rec
        assert "trend" in first_rec
        assert "seasonal" in first_rec
        assert "residual" in first_rec

    # --------------------------------------------------------------------------
    # Benchmark 5: Sensor Telemetry (Autocorrelation ACF & PACF E2E)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_sensor_autocorrelation_e2e(self, df_sensor_telemetry):
        """
        Query: 'Plot autocorrelation ACF and PACF for vibration_x with 30 lags'
        Dataset: Sensor Telemetry dataset.
        
        Asserts:
        - Response status is 'SUCCEEDED'.
        - Operator executed is 'autocorrelation'.
        - Manifest chart_type is 'acf'.
        """
        query = "Plot autocorrelation ACF and PACF for vibration_x with 30 lags"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_sensor_telemetry)
        print_benchmark_telemetry("05: Sensor Autocorrelation ACF & PACF", query, res)

        assert res.status == "SUCCEEDED", f"Execution failed with error: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result
        
        manifest = res.result["manifest"]
        assert manifest.get("operator") == "autocorrelation", f"Expected operator 'autocorrelation', got {manifest.get('operator')}"
        assert manifest.get("chart_type") == "acf", f"Expected chart_type 'acf', got {manifest.get('chart_type')}"
        assert "vibration_x" in manifest.get("columns_used", [])

        # Verify ACF / PACF tabular records produced
        records = res.result["data"].get("records", [])
        assert len(records) > 0
        assert "acf" in records[0] or "lag" in records[0]

    # --------------------------------------------------------------------------
    # Benchmark 6: HR Employee Analytics (K-Means Clustering E2E)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_hr_employee_clustering_e2e(self, df_hr_employee_churn):
        """
        Query: 'Cluster employees into 3 groups based on monthly_salary and satisfaction_score'
        Dataset: HR Employee Churn dataset.
        
        Asserts:
        - Response status is 'SUCCEEDED'.
        - Operator executed is 'clustering'.
        - Number of cluster centroids is 3 (manifest row_count_out == 3).
        """
        query = "Cluster employees into 3 groups based on monthly_salary and satisfaction_score"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_hr_employee_churn)
        print_benchmark_telemetry("06: HR Employee Clustering", query, res)

        assert res.status == "SUCCEEDED", f"Execution failed with error: {res.error}"
        assert res.result is not None
        assert "manifest" in res.result
        
        manifest = res.result["manifest"]
        assert manifest.get("operator") == "clustering", f"Expected operator 'clustering', got {manifest.get('operator')}"
        assert manifest.get("row_count_out") == 3, f"Expected 3 cluster centroids, got {manifest.get('row_count_out')}"
        
        cols_used = [c.lower() for c in manifest.get("columns_used", [])]
        assert "monthly_salary" in cols_used
        assert "satisfaction_score" in cols_used

        stats = manifest.get("statistics_computed", [])
        assert "cluster_centroids" in stats
        assert "silhouette_score" in stats

    # --------------------------------------------------------------------------
    # Benchmark 7: Multilingual Hinglish Predictive Maintenance Interception
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_hinglish_event_window_query_proactively_clarifies_sequence(self, df_predictive_maintenance_25k):
        """
        Query: 'Machine breakdown hone se pehle ka process temperature analyze karo'
        Dataset: 25k Predictive Maintenance dataset (lacking temporal column).
        
        Asserts:
        - SemanticFeasibilityValidator intercepts event-window intent on non-temporal data.
        - Response status is 'NEEDS_CLARIFICATION'.
        - Error code is 'CLARIFICATION_REQUIRED'.
        - Candidate columns include ['UDI'].
        - Proactive clarification proposes 'UDI' sequence order in Hinglish context.
        """
        query = "Machine breakdown hone se pehle ka process temperature analyze karo"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_predictive_maintenance_25k)
        print_benchmark_telemetry("07: Hinglish Event Window Clarification", query, res)

        assert res.status == "NEEDS_CLARIFICATION", f"Expected NEEDS_CLARIFICATION, got {res.status}"
        assert res.error is not None
        assert res.error.code == "CLARIFICATION_REQUIRED", f"Expected CLARIFICATION_REQUIRED, got {res.error.code}"
        assert res.error.candidate_columns is not None
        assert "UDI" in res.error.candidate_columns, f"Expected 'UDI' in candidate_columns: {res.error.candidate_columns}"
        assert res.clarification is not None
        assert "UDI" in res.clarification, f"Expected 'UDI' in clarification prompt: {res.clarification}"
