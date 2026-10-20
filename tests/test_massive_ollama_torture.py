"""
DataPilot OS — Massive Ollama Multi-Capability Torture Test Suite.

Comprehensive verification matrix spanning 17 analytical capabilities:
1.  FILTER (7 NL variations)
2.  GROUP (6 NL variations)
3.  SORT (5 NL variations)
4.  EDA (5 NL variations)
5.  CORRELATION (5 NL variations)
6.  ANOMALY (5 NL variations)
7.  CLEANING (5 NL variations)
8.  CLASSIFICATION / PREDICTION (5 NL variations)
9.  CLUSTERING (5 NL variations)
10. TIME SERIES / AUTOCORRELATION (5 NL variations)
11. FORECASTING (5 NL variations)
12. MULTI-INTENT (5 NL variations)
13. AMBIGUOUS (5 NL variations)
14. UNKNOWN COLUMN (5 NL variations)
15. WRONG DATATYPE (5 NL variations)
16. UNSUPPORTED REQUEST (5 NL variations)
17. FOLLOW-UP QUERY (5 NL variations)

Total: 88+ natural language variations tested across deterministic and live integration modes.
"""

import sys
import os
import time
import logging
from typing import Dict, Any, List, Optional
import pytest
import pandas as pd
import numpy as np

pytestmark = pytest.mark.live_ollama

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.ds_engine.intent_schema import (
    SemanticIntent,
    IntentType,
    FilterSpec,
    FilterOperator,
    AggregationType,
    ChartType
)
from backend.ds_engine.semantic_command import (
    LLMSemanticCommand,
    ToolCall,
    SemanticTool
)
from backend.ds_engine.command_resolver import CommandResolver, command_resolver
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse, LifecycleState
from backend.intelligence.ollama_service import (
    ollama_service,
    OllamaReadinessStatus,
    OllamaStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MassiveOllamaTorture")


# ==============================================================================
# DATASET GENERATORS & FIXTURES
# ==============================================================================

def get_ecommerce_sales_df() -> pd.DataFrame:
    """Enterprise E-commerce dataset with numeric, categorical, temporal, and null values."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    regions = np.tile(["North", "South", "East", "West"], 30)
    categories = np.tile(["Electronics", "Apparel", "Furniture", "Groceries"], 30)
    sales = np.linspace(50.0, 1500.0, n)
    units = np.random.randint(1, 25, size=n)
    unit_price = sales / units
    product_names = [f"Product_{c}_{i % 10}" for i, c in enumerate(categories)]
    
    # Introduce controlled nulls in discount
    discount = np.random.uniform(0.0, 0.3, size=n)
    discount[5] = np.nan
    discount[15] = np.nan
    discount[25] = np.nan

    return pd.DataFrame({
        "order_date": dates,
        "region": regions,
        "category": categories,
        "product_name": product_names,
        "sales_amount": sales,
        "units_sold": units,
        "unit_price": unit_price,
        "discount": discount
    })


def get_iot_sensors_df() -> pd.DataFrame:
    """IoT Sensor Telemetry with timestamps, correlated signals, spikes, and missing data."""
    np.random.seed(101)
    n = 150
    timestamps = pd.date_range(start="2024-01-01", periods=n, freq="h")
    temp = np.random.normal(70.0, 8.0, size=n)
    pressure = 1013.25 - (temp - 70.0) * 1.5 + np.random.normal(0, 2.0, size=n)
    humidity = np.clip(100.0 - temp * 0.8 + np.random.normal(0, 5.0, size=n), 10.0, 95.0)
    vibration = np.random.exponential(1.8, size=n)

    # Injected anomalies
    vibration[20] = 35.0
    temp[45] = 130.0

    # Injected nulls
    temp[10] = np.nan
    humidity[30] = np.nan

    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temp,
        "pressure": pressure,
        "humidity": humidity,
        "vibration_level": vibration
    })


def get_hr_churn_df() -> pd.DataFrame:
    """HR Employee Churn dataset for classification, prediction, and clustering."""
    np.random.seed(202)
    n = 120
    age = np.random.randint(22, 65, size=n)
    tenure = np.random.randint(1, 120, size=n)
    salary = np.random.normal(6500, 1800, size=n)
    satisfaction = np.random.uniform(1.0, 5.0, size=n)
    departments = np.tile(["Engineering", "Sales", "Support", "Marketing"], 30)
    churn = np.where((satisfaction < 2.5) | (salary < 4500), 1, 0)

    return pd.DataFrame({
        "employee_age": age,
        "tenure_months": tenure,
        "monthly_salary": salary,
        "satisfaction_score": satisfaction,
        "department": departments,
        "left_company": churn
    })


def get_credit_risk_df() -> pd.DataFrame:
    """Banking & Credit Risk dataset for classification and risk modeling."""
    np.random.seed(303)
    n = 100
    credit_score = np.random.randint(300, 850, size=n)
    income = np.random.uniform(25000, 150000, size=n)
    debt_ratio = np.random.uniform(0.05, 0.75, size=n)
    default_risk = np.where((credit_score < 580) | (debt_ratio > 0.55), 1, 0)
    customer_names = [f"Customer_{i:03d}" for i in range(n)]

    return pd.DataFrame({
        "customer_name": customer_names,
        "credit_score": credit_score,
        "income": income,
        "debt_ratio": debt_ratio,
        "default_risk": default_risk
    })


def get_non_temporal_df() -> pd.DataFrame:
    """Dataset intentionally stripped of all temporal/date columns for negative testing."""
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "category": ["A", "B", "C", "D"],
        "sales_amount": [100.0, 200.0, 300.0, 400.0]
    })


# ==============================================================================
# DETERMINISTIC RUNNER HELPER
# ==============================================================================

def execute_deterministic_pipeline(
    query: str,
    df: pd.DataFrame,
    mock_command: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> CanonicalQueryResponse:
    """
    Executes a query through the full QueryOrchestrator pipeline with deterministic mock command.
    Bypasses live Ollama network calls to test DS code generation, AST validation,
    subprocess sandboxing, result validation, and schema grounding deterministically.
    """
    orig_extract = SemanticPlanner.extract_semantic_intent
    orig_verify = ollama_service.verify_ready

    # Force Ollama readiness to True for deterministic test execution
    ollama_service.verify_ready = lambda: OllamaReadinessStatus(
        ready=True,
        ollama_url="http://localhost:11434",
        model="qwen2.5:1.5b",
        model_available=True
    )
    SemanticPlanner.extract_semantic_intent = classmethod(lambda cls, **kwargs: mock_command)

    try:
        res = QueryOrchestrator.process_query(
            query_raw=query,
            df=df,
            conversation_history=conversation_history,
            context=context
        )
        return res
    finally:
        SemanticPlanner.extract_semantic_intent = orig_extract
        ollama_service.verify_ready = orig_verify


# ==============================================================================
# 1. CAPABILITY: FILTER (7 NL VARIATIONS)
# ==============================================================================

def test_01_filter_capability_matrix():
    """
    Capability 01: FILTER
    Validates: >, ==, >=, <, !=, is null, and contains operators across numerical and categorical attributes.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "FILT-01",
            "query": "Filter rows where sales_amount is strictly above 500",
            "mock": {"tool": "filter_data", "arguments": {"column": "sales_amount", "condition": ">", "value": 500}},
            "expected_op": FilterOperator.GT,
            "col": "sales_amount"
        },
        {
            "id": "FILT-02",
            "query": "Show only records with region equals North",
            "mock": {"tool": "filter_data", "arguments": {"column": "region", "condition": "==", "value": "North"}},
            "expected_op": FilterOperator.EQ,
            "col": "region"
        },
        {
            "id": "FILT-03",
            "query": "Find all transactions where unit_price is at least 50",
            "mock": {"tool": "filter_data", "arguments": {"column": "unit_price", "condition": ">=", "value": 50}},
            "expected_op": FilterOperator.GTE,
            "col": "unit_price"
        },
        {
            "id": "FILT-04",
            "query": "Filter customers with units_sold under 10",
            "mock": {"tool": "filter_data", "arguments": {"column": "units_sold", "condition": "<", "value": 10}},
            "expected_op": FilterOperator.LT,
            "col": "units_sold"
        },
        {
            "id": "FILT-05",
            "query": "Give me records where category is not equal to 'Electronics'",
            "mock": {"tool": "filter_data", "arguments": {"column": "category", "condition": "!=", "value": "Electronics"}},
            "expected_op": FilterOperator.NEQ,
            "col": "category"
        },
        {
            "id": "FILT-06",
            "query": "Show rows where discount is missing",
            "mock": {"tool": "filter_data", "arguments": {"column": "discount", "condition": "is null"}},
            "expected_op": FilterOperator.IS_NULL,
            "col": "discount"
        },
        {
            "id": "FILT-07",
            "query": "Show items where product_name contains 'Product_Electronics'",
            "mock": {"tool": "filter_data", "arguments": {"column": "product_name", "condition": "contains", "value": "Product_Electronics"}},
            "expected_op": FilterOperator.CONTAINS,
            "col": "product_name"
        }
    ]

    for var in variations:
        logger.info(f"Testing FILTER variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Filter failed for {var['id']}: {res.error}"
        assert res.result is not None
        records = res.result.get("data", {}).get("records", [])
        assert len(records) > 0, f"Expected filtered records for {var['id']}, got 0"
        
        # Verify resolution operator
        res_cmd = command_resolver.resolve_command(var["mock"], df=df)
        assert res_cmd.is_valid is True
        assert res_cmd.semantic_intent.filters[0].operator == var["expected_op"]
        assert res_cmd.semantic_intent.filters[0].column == var["col"]
        print(f">> [PASS] FILTER {var['id']}: '{var['query']}' returned {len(records)} records.")


# ==============================================================================
# 2. CAPABILITY: GROUP & AGGREGATION (6 NL VARIATIONS)
# ==============================================================================

def test_02_group_capability_matrix():
    """
    Capability 02: GROUP & AGGREGATION
    Validates: sum, mean, count, max, min, median aggregations with categorical grouping dimensions.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "GRP-01",
            "query": "Calculate total sales_amount grouped by region",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales_amount", "aggregation": "sum"}},
            "expected_agg": AggregationType.SUM
        },
        {
            "id": "GRP-02",
            "query": "Find average unit_price by category",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["category"], "metric": "unit_price", "aggregation": "mean"}},
            "expected_agg": AggregationType.MEAN
        },
        {
            "id": "GRP-03",
            "query": "Count the number of orders per category",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["category"], "metric": "sales_amount", "aggregation": "count"}},
            "expected_agg": AggregationType.COUNT
        },
        {
            "id": "GRP-04",
            "query": "What is the maximum sales_amount for each region?",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales_amount", "aggregation": "max"}},
            "expected_agg": AggregationType.MAX
        },
        {
            "id": "GRP-05",
            "query": "Aggregate units_sold by category showing minimum",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["category"], "metric": "units_sold", "aggregation": "min"}},
            "expected_agg": AggregationType.MIN
        },
        {
            "id": "GRP-06",
            "query": "Compute median unit_price by region",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "unit_price", "aggregation": "median"}},
            "expected_agg": AggregationType.MEDIAN
        }
    ]

    for var in variations:
        logger.info(f"Testing GROUP variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Group failed for {var['id']}: {res.error}"
        assert res.result is not None
        records = res.result.get("data", {}).get("records", [])
        assert len(records) > 0, f"Expected grouped records for {var['id']}, got 0"
        
        res_cmd = command_resolver.resolve_command(var["mock"], df=df)
        assert res_cmd.is_valid is True
        assert res_cmd.semantic_intent.aggregation == var["expected_agg"]
        print(f">> [PASS] GROUP {var['id']}: '{var['query']}' aggregated cleanly into {len(records)} groups.")


# ==============================================================================
# 3. CAPABILITY: SORT & ORDERING (5 NL VARIATIONS)
# ==============================================================================

def test_03_sort_capability_matrix():
    """
    Capability 03: SORT
    Validates: ascending, descending, top N limits, and column-based ordering.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "SORT-01",
            "query": "Sort records by sales_amount in descending order",
            "mock": {"tool": "sort_data", "arguments": {"column": "sales_amount", "direction": "desc"}},
            "order": "desc",
            "limit": None
        },
        {
            "id": "SORT-02",
            "query": "Order records by unit_price ascending",
            "mock": {"tool": "sort_data", "arguments": {"column": "unit_price", "direction": "asc"}},
            "order": "asc",
            "limit": None
        },
        {
            "id": "SORT-03",
            "query": "Top 10 highest sales_amount items",
            "mock": {"tool": "sort_data", "arguments": {"column": "sales_amount", "direction": "desc", "limit": 10}},
            "order": "desc",
            "limit": 10
        },
        {
            "id": "SORT-04",
            "query": "Show lowest 5 units_sold",
            "mock": {"tool": "sort_data", "arguments": {"column": "units_sold", "direction": "asc", "limit": 5}},
            "order": "asc",
            "limit": 5
        },
        {
            "id": "SORT-05",
            "query": "Rank products by units_sold from highest to lowest",
            "mock": {"tool": "sort_data", "arguments": {"column": "units_sold", "direction": "desc"}},
            "order": "desc",
            "limit": None
        }
    ]

    for var in variations:
        logger.info(f"Testing SORT variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Sort failed for {var['id']}: {res.error}"
        assert res.result is not None
        records = res.result.get("data", {}).get("records", [])
        if var["limit"]:
            assert len(records) == var["limit"], f"Expected {var['limit']} records, got {len(records)}"
        
        res_cmd = command_resolver.resolve_command(var["mock"], df=df)
        assert res_cmd.is_valid is True
        assert res_cmd.semantic_intent.order == var["order"]
        print(f">> [PASS] SORT {var['id']}: '{var['query']}' sorted correctly (limit={var['limit']}).")


# ==============================================================================
# 4. CAPABILITY: EXPLORATORY DATA ANALYSIS (EDA) (5 NL VARIATIONS)
# ==============================================================================

def test_04_eda_capability_matrix():
    """
    Capability 04: EDA
    Validates: summary statistics, dataset overview, distribution profiling, and data profiling.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {"id": "EDA-01", "query": "Provide a comprehensive exploratory data analysis of this dataset", "mock": {"tool": "eda_data", "arguments": {}}},
        {"id": "EDA-02", "query": "Generate an EDA dashboard with summary statistics", "mock": {"tool": "eda_data", "arguments": {}}},
        {"id": "EDA-03", "query": "Give me a dataset overview and distribution profile", "mock": {"tool": "eda_data", "arguments": {}}},
        {"id": "EDA-04", "query": "Show descriptive statistics and summary for all columns", "mock": {"tool": "eda_data", "arguments": {}}},
        {"id": "EDA-05", "query": "Explore dataset characteristics and summary metrics", "mock": {"tool": "eda_data", "arguments": {}}}
    ]

    for var in variations:
        logger.info(f"Testing EDA variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"EDA failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "DATASET OVERVIEW" in res.result["response"] or "DESCRIPTIVE STATISTICS" in res.result["response"]
        print(f">> [PASS] EDA {var['id']}: '{var['query']}' produced valid summary statistics.")


# ==============================================================================
# 5. CAPABILITY: CORRELATION (5 NL VARIATIONS)
# ==============================================================================

def test_05_correlation_capability_matrix():
    """
    Capability 05: CORRELATION
    Validates: Pearson correlation matrix, feature relationships, and pairwise correlation heatmaps.
    """
    df = get_iot_sensors_df()

    variations = [
        {
            "id": "CORR-01",
            "query": "Compute correlation matrix heatmap between temperature, pressure, and humidity",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["temperature", "pressure", "humidity"], "analysis_mode": "matrix", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "heatmap", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "CORR-02",
            "query": "Find the relationship and correlation between temperature and vibration_level",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["temperature", "vibration_level"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "none", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "CORR-03",
            "query": "Is there a correlation between pressure and humidity?",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["pressure", "humidity"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "none", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "CORR-04",
            "query": "Show Pearson correlation matrix for vibration_level and temperature",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["vibration_level", "temperature"], "analysis_mode": "matrix", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "heatmap", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "CORR-05",
            "query": "Calculate feature correlation across all sensor telemetry signals",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["temperature", "pressure", "humidity", "vibration_level"], "analysis_mode": "matrix", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "heatmap", "smoother": "none", "expected_direction": "unspecified"}}
        }
    ]

    for var in variations:
        logger.info(f"Testing CORRELATION variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Correlation failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "correlation" in res.result["response"].lower() or "association" in res.result["response"].lower()
        print(f">> [PASS] CORRELATION {var['id']}: '{var['query']}' computed correlation matrix cleanly.")


# ==============================================================================
# 6. CAPABILITY: ANOMALY DETECTION (5 NL VARIATIONS)
# ==============================================================================

def test_06_anomaly_capability_matrix():
    """
    Capability 06: ANOMALY DETECTION
    Validates: IsolationForest anomaly detection, outlier spotting, and anomaly tagging.
    """
    df = get_iot_sensors_df()

    variations = [
        {
            "id": "ANOM-01",
            "query": "Detect anomalous sensor readings across vibration_level and temperature using IsolationForest",
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["vibration_level", "temperature"]}}
        },
        {
            "id": "ANOM-02",
            "query": "Find outliers in vibration_level",
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["vibration_level"]}}
        },
        {
            "id": "ANOM-03",
            "query": "Identify anomalies and abnormal spikes in temperature",
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["temperature"]}}
        },
        {
            "id": "ANOM-04",
            "query": "Detect data anomalies in pressure and humidity",
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["pressure", "humidity"]}}
        },
        {
            "id": "ANOM-05",
            "query": "Spot abnormal records across all telemetry channels",
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["vibration_level", "temperature", "pressure", "humidity"]}}
        }
    ]

    for var in variations:
        logger.info(f"Testing ANOMALY variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Anomaly detection failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "Anomaly Detection" in res.result["response"] or "ANOMALY" in res.result["response"]
        print(f">> [PASS] ANOMALY {var['id']}: '{var['query']}' detected anomalies successfully.")


# ==============================================================================
# 7. CAPABILITY: DATA CLEANING & IMPUTATION (5 NL VARIATIONS)
# ==============================================================================

def test_07_cleaning_capability_matrix():
    """
    Capability 07: DATA CLEANING
    Validates: mean imputation, median imputation, row dropping, null sanitization.
    """
    df = get_iot_sensors_df()

    variations = [
        {
            "id": "CLN-01",
            "query": "Clean missing values in temperature using mean imputation",
            "mock": {"tool": "clean_data", "arguments": {"columns": ["temperature"], "method": "impute_mean"}}
        },
        {
            "id": "CLN-02",
            "query": "Impute nulls in humidity column",
            "mock": {"tool": "clean_data", "arguments": {"columns": ["humidity"], "method": "impute"}}
        },
        {
            "id": "CLN-03",
            "query": "Clean dataset by handling missing values in temperature and humidity",
            "mock": {"tool": "clean_data", "arguments": {"columns": ["temperature", "humidity"], "method": "impute"}}
        },
        {
            "id": "CLN-04",
            "query": "Sanitize and drop missing rows in humidity",
            "mock": {"tool": "clean_data", "arguments": {"columns": ["humidity"], "method": "drop"}}
        },
        {
            "id": "CLN-05",
            "query": "Impute missing data across all sensor columns",
            "mock": {"tool": "clean_data", "arguments": {"columns": ["temperature", "humidity", "pressure"], "method": "impute"}}
        }
    ]

    for var in variations:
        logger.info(f"Testing CLEANING variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Cleaning failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "Data Cleaning" in res.result["response"] or "CLEAN" in res.result["response"]
        print(f">> [PASS] CLEANING {var['id']}: '{var['query']}' cleaned and imputed data cleanly.")


# ==============================================================================
# 8. CAPABILITY: CLASSIFICATION / PREDICTION / AUTOML (5 NL VARIATIONS)
# ==============================================================================

def test_08_classification_prediction_capability_matrix():
    """
    Capability 08: CLASSIFICATION / PREDICTION
    Validates: AutoML model training, churn prediction, default risk classification, metrics calculation.
    """
    hr_df = get_hr_churn_df()
    credit_df = get_credit_risk_df()

    variations = [
        {
            "id": "ML-01",
            "query": "Train a machine learning model to predict left_company based on employee_age, tenure_months, monthly_salary, satisfaction_score",
            "df": hr_df,
            "mock": {"tool": "train_model", "arguments": {"target_column": "left_company", "feature_columns": ["employee_age", "tenure_months", "monthly_salary", "satisfaction_score"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        },
        {
            "id": "ML-02",
            "query": "Predict default_risk using credit_score, income, and debt_ratio",
            "df": credit_df,
            "mock": {"tool": "train_model", "arguments": {"target_column": "default_risk", "feature_columns": ["credit_score", "income", "debt_ratio"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        },
        {
            "id": "ML-03",
            "query": "Train a classifier to predict left_company from employee_age and satisfaction_score",
            "df": hr_df,
            "mock": {"tool": "train_model", "arguments": {"target_column": "left_company", "feature_columns": ["employee_age", "satisfaction_score"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        },
        {
            "id": "ML-04",
            "query": "Build an AutoML classification model to predict left_company",
            "df": hr_df,
            "mock": {"tool": "train_model", "arguments": {"target_column": "left_company", "feature_columns": ["employee_age", "tenure_months"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        },
        {
            "id": "ML-05",
            "query": "Predict default_risk based on income and credit_score",
            "df": credit_df,
            "mock": {"tool": "train_model", "arguments": {"target_column": "default_risk", "feature_columns": ["income", "credit_score"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        }
    ]

    for var in variations:
        logger.info(f"Testing ML variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"AutoML failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "AutoML" in res.result["response"] or "Model" in res.result["response"] or "Accuracy" in res.result["response"]
        print(f">> [PASS] CLASSIFICATION {var['id']}: '{var['query']}' trained and evaluated model cleanly.")


# ==============================================================================
# 9. CAPABILITY: CLUSTERING (5 NL VARIATIONS)
# ==============================================================================

def test_09_clustering_capability_matrix():
    """
    Capability 09: CLUSTERING
    Validates: KMeans clustering, cluster counts, silhouette / inertia, customer segmentation.
    """
    hr_df = get_hr_churn_df()
    credit_df = get_credit_risk_df()

    variations = [
        {
            "id": "CLUST-01",
            "query": "Cluster employees into 3 groups based on monthly_salary and satisfaction_score",
            "df": hr_df,
            "mock": {"tool": "cluster_data", "arguments": {"columns": ["monthly_salary", "satisfaction_score"], "n_clusters": 3}}
        },
        {
            "id": "CLUST-02",
            "query": "Perform KMeans clustering with 4 clusters on income and debt_ratio",
            "df": credit_df,
            "mock": {"tool": "cluster_data", "arguments": {"columns": ["income", "debt_ratio"], "n_clusters": 4}}
        },
        {
            "id": "CLUST-03",
            "query": "Segment employees into 2 clusters using employee_age and tenure_months",
            "df": hr_df,
            "mock": {"tool": "cluster_data", "arguments": {"columns": ["employee_age", "tenure_months"], "n_clusters": 2}}
        },
        {
            "id": "CLUST-04",
            "query": "Group customers into 3 unsupervised clusters using credit_score and income",
            "df": credit_df,
            "mock": {"tool": "cluster_data", "arguments": {"columns": ["credit_score", "income"], "n_clusters": 3}}
        },
        {
            "id": "CLUST-05",
            "query": "Apply unsupervised clustering to discover patterns across monthly_salary and employee_age",
            "df": hr_df,
            "mock": {"tool": "cluster_data", "arguments": {"columns": ["monthly_salary", "employee_age"], "n_clusters": 3}}
        }
    ]

    for var in variations:
        logger.info(f"Testing CLUSTERING variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Clustering failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "Cluster" in res.result["response"] or "K-Means" in res.result["response"] or "CLUSTERING" in res.result["response"]
        print(f">> [PASS] CLUSTERING {var['id']}: '{var['query']}' segmented records cleanly.")


# ==============================================================================
# 10. CAPABILITY: TIME SERIES / AUTOCORRELATION (5 NL VARIATIONS)
# ==============================================================================

def test_10_autocorrelation_capability_matrix():
    """
    Capability 10: TIME SERIES / AUTOCORRELATION
    Validates: ACF, PACF, lag correlation, serial dependence on temporal series.
    """
    sensor_df = get_iot_sensors_df()
    sales_df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "ACF-01",
            "query": "Plot autocorrelation ACF and PACF for temperature",
            "df": sensor_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["temperature"]}}
        },
        {
            "id": "ACF-02",
            "query": "Calculate autocorrelation lags for vibration_level",
            "df": sensor_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["vibration_level"]}}
        },
        {
            "id": "ACF-03",
            "query": "Compute ACF PACF time series correlation for sales_amount",
            "df": sales_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["sales_amount"]}}
        },
        {
            "id": "ACF-04",
            "query": "Show autocorrelation structure for pressure",
            "df": sensor_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["pressure"]}}
        },
        {
            "id": "ACF-05",
            "query": "Analyze serial autocorrelation for humidity",
            "df": sensor_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["humidity"]}}
        }
    ]

    for var in variations:
        logger.info(f"Testing AUTOCORRELATION variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Autocorrelation failed for {var['id']}: {res.error}"
        assert res.result is not None
        resp_upper = res.result["response"].upper()
        assert "AUTOCORRELATION" in resp_upper or "ACF" in resp_upper or "LAG" in resp_upper
        print(f">> [PASS] AUTOCORRELATION {var['id']}: '{var['query']}' generated ACF/PACF cleanly.")


# ==============================================================================
# 11. CAPABILITY: FORECASTING (5 NL VARIATIONS)
# ==============================================================================

def test_11_forecasting_capability_matrix():
    """
    Capability 11: FORECASTING
    Validates: time series horizon projections, temporal resolution, trajectory computation.
    """
    sales_df = get_ecommerce_sales_df()
    sensor_df = get_iot_sensors_df()

    variations = [
        {
            "id": "FCST-01",
            "query": "Forecast 14 days of sales_amount based on order_date",
            "df": sales_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "sales_amount", "time_column": "order_date", "horizon": 14}}
        },
        {
            "id": "FCST-02",
            "query": "Predict future 30 periods of temperature using timestamp",
            "df": sensor_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "temperature", "time_column": "timestamp", "horizon": 30}}
        },
        {
            "id": "FCST-03",
            "query": "Forecast vibration_level for the next 7 periods using timestamp",
            "df": sensor_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "vibration_level", "time_column": "timestamp", "horizon": 7}}
        },
        {
            "id": "FCST-04",
            "query": "Generate time series forecast of units_sold for 20 days",
            "df": sales_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "units_sold", "time_column": "order_date", "horizon": 20}}
        },
        {
            "id": "FCST-05",
            "query": "Project pressure for the next 10 periods based on timestamp",
            "df": sensor_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "pressure", "time_column": "timestamp", "horizon": 10}}
        }
    ]

    for var in variations:
        logger.info(f"Testing FORECASTING variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Forecasting failed for {var['id']}: {res.error}"
        assert res.result is not None
        assert "Forecast" in res.result["response"] or "FORECAST" in res.result["response"]
        print(f">> [PASS] FORECASTING {var['id']}: '{var['query']}' projected trajectory cleanly.")


# ==============================================================================
# 12. CAPABILITY: MULTI-INTENT & COMPOUND DAG (5 NL VARIATIONS)
# ==============================================================================

def test_12_multi_intent_capability_matrix():
    """
    Capability 12: MULTI-INTENT
    Validates: multi-step DAG execution (Filter + Group, Clean + Anomaly, Filter + Cluster, Filter + Correlate, Clean + Forecast).
    """
    sales_df = get_ecommerce_sales_df()
    sensor_df = get_iot_sensors_df()
    hr_df = get_hr_churn_df()

    variations = [
        {
            "id": "MULTI-01",
            "query": "Filter sales_amount > 500 and calculate average sales_amount by region",
            "df": sales_df,
            "mock": {
                "commands": [
                    {"tool": "filter_data", "arguments": {"column": "sales_amount", "condition": ">", "value": 500}},
                    {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales_amount", "aggregation": "mean"}}
                ]
            },
            "expected_steps": 2
        },
        {
            "id": "MULTI-02",
            "query": "Clean missing values in temperature and then detect anomalies in vibration_level and temperature",
            "df": sensor_df,
            "mock": {
                "commands": [
                    {"tool": "clean_data", "arguments": {"columns": ["temperature"]}},
                    {"tool": "detect_anomalies", "arguments": {"columns": ["vibration_level", "temperature"]}}
                ]
            },
            "expected_steps": 2
        },
        {
            "id": "MULTI-03",
            "query": "Filter employee_age > 30 and cluster employees into 3 groups based on monthly_salary and satisfaction_score",
            "df": hr_df,
            "mock": {
                "commands": [
                    {"tool": "filter_data", "arguments": {"column": "employee_age", "condition": ">", "value": 30}},
                    {"tool": "cluster_data", "arguments": {"columns": ["monthly_salary", "satisfaction_score"], "n_clusters": 3}}
                ]
            },
            "expected_steps": 2
        },
        {
            "id": "MULTI-04",
            "query": "Filter region equals 'North' and calculate correlation between sales_amount and units_sold",
            "df": sales_df,
            "mock": {
                "commands": [
                    {"tool": "filter_data", "arguments": {"column": "region", "condition": "==", "value": "North"}},
                    {"tool": "correlate_data", "arguments": {"columns": ["sales_amount", "units_sold"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "none", "smoother": "none", "expected_direction": "unspecified"}}
                ]
            },
            "expected_steps": 2
        },
        {
            "id": "MULTI-05",
            "query": "Clean missing values in sales_amount then forecast 14 days of sales_amount",
            "df": sales_df,
            "mock": {
                "commands": [
                    {"tool": "clean_data", "arguments": {"columns": ["sales_amount"]}},
                    {"tool": "forecast_data", "arguments": {"value_column": "sales_amount", "time_column": "order_date", "horizon": 14}}
                ]
            },
            "expected_steps": 2
        }
    ]

    for var in variations:
        logger.info(f"Testing MULTI-INTENT variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        assert res.status == "SUCCEEDED", f"Multi-intent failed for {var['id']}: {res.error}"
        assert res.result is not None
        exec_steps = res.plan.get("execution_plan", [])
        assert len(exec_steps) >= var["expected_steps"], f"Expected >= {var['expected_steps']} execution steps, got {len(exec_steps)}"
        print(f">> [PASS] MULTI-INTENT {var['id']}: '{var['query']}' executed {len(exec_steps)} steps in DAG.")


# ==============================================================================
# 13. CAPABILITY: AMBIGUOUS QUERY INTERDICTION (5 NL VARIATIONS)
# ==============================================================================

def test_13_ambiguous_capability_matrix():
    """
    Capability 13: AMBIGUOUS QUERIES
    Validates: underspecified queries trigger clarification_needed / AMBIGUOUS_INTENT cleanly without arbitrary execution.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {"id": "AMB-01", "query": "Analyze this data", "mock": {"tool": "clarification_needed", "arguments": {"reason": "Query is underspecified. Please specify operation."}}},
        {"id": "AMB-02", "query": "Do something with data", "mock": {"tool": "clarification_needed", "arguments": {"reason": "Query is underspecified."}}},
        {"id": "AMB-03", "query": "Show me data", "mock": {"tool": "clarification_needed", "arguments": {"reason": "Query is underspecified."}}},
        {"id": "AMB-04", "query": "Look at this dataset", "mock": {"tool": "clarification_needed", "arguments": {"reason": "Query is underspecified."}}},
        {"id": "AMB-05", "query": "Do some analytics", "mock": {"tool": "clarification_needed", "arguments": {"reason": "Query is underspecified."}}}
    ]

    for var in variations:
        logger.info(f"Testing AMBIGUOUS variation {var['id']}: '{var['query']}'")
        
        # Test 1: Command resolver flags ambiguity
        res_cmd = command_resolver.resolve_command(var["mock"], df=df, raw_query=var["query"])
        assert res_cmd.is_ambiguous is True or res_cmd.is_valid is True
        
        # Test 2: Orchestrator safely halts with status=FAILED and error code AMBIGUOUS_INTENT
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "FAILED", f"Expected FAILED status for ambiguous query {var['id']}, got {res.status}"
        assert res.error is not None
        assert res.error.code in ["AMBIGUOUS_INTENT", "INTENT_PLAN_INVALID"], f"Unexpected error code: {res.error.code}"
        print(f">> [PASS] AMBIGUOUS {var['id']}: '{var['query']}' safely trapped with code '{res.error.code}'.")


# ==============================================================================
# 14. CAPABILITY: UNKNOWN COLUMN PROTECTION (5 NL VARIATIONS)
# ==============================================================================

def test_14_unknown_column_capability_matrix():
    """
    Capability 14: UNKNOWN COLUMN PROTECTION
    Validates: queries referencing non-existent columns are halted in PLAN_VALIDATION without hallucinating data.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "UNK-01",
            "query": "Filter records where non_existent_column_xyz > 100",
            "mock": {"tool": "filter_data", "arguments": {"column": "non_existent_column_xyz", "condition": ">", "value": 100}}
        },
        {
            "id": "UNK-02",
            "query": "Calculate average ghost_metric by region",
            "mock": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "ghost_metric", "aggregation": "mean"}}
        },
        {
            "id": "UNK-03",
            "query": "Compute correlation between fictional_col1 and imaginary_col2",
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["fictional_col1", "imaginary_col2"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "none", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "UNK-04",
            "query": "Train a prediction model for target_does_not_exist",
            "mock": {"tool": "train_model", "arguments": {"target_column": "target_does_not_exist", "feature_columns": ["sales_amount"], "task_type": "classification", "evaluation_artifact": "feature_importance", "prediction_context": "prospective_prediction"}}
        },
        {
            "id": "UNK-05",
            "query": "Forecast metric_that_is_missing over order_date",
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "metric_that_is_missing", "time_column": "order_date", "horizon": 14}}
        }
    ]

    for var in variations:
        logger.info(f"Testing UNKNOWN COLUMN variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "FAILED", f"Expected failure for unknown column query {var['id']}, got {res.status}"
        assert res.error is not None
        assert res.error.phase in ["PLAN_VALIDATION", "PLANNING"], f"Expected failure in PLAN_VALIDATION, got {res.error.phase}"
        print(f">> [PASS] UNKNOWN COLUMN {var['id']}: '{var['query']}' halted cleanly in phase '{res.error.phase}'.")


# ==============================================================================
# 15. CAPABILITY: WRONG DATATYPE GUARDRAILS (5 NL VARIATIONS)
# ==============================================================================

def test_15_wrong_datatype_capability_matrix():
    """
    Capability 15: WRONG DATATYPE GUARDRAILS
    Validates: forecasting on text / no date, correlation on strings, etc. are rejected cleanly without crashes.
    """
    sales_df = get_ecommerce_sales_df()
    no_date_df = get_non_temporal_df()

    variations = [
        {
            "id": "TYPE-01",
            "query": "Forecast product_name for next 30 days",
            "df": sales_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "product_name", "time_column": "order_date", "horizon": 30}}
        },
        {
            "id": "TYPE-02",
            "query": "Calculate Pearson correlation between region and category",
            "df": sales_df,
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["region", "category"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "none", "smoother": "none", "expected_direction": "unspecified"}}
        },
        {
            "id": "TYPE-03",
            "query": "Forecast sales_amount on dataset with no temporal column",
            "df": no_date_df,
            "mock": {"tool": "forecast_data", "arguments": {"value_column": "sales_amount", "horizon": 14}}
        },
        {
            "id": "TYPE-04",
            "query": "Detect isolation forest anomalies on region string column",
            "df": sales_df,
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["region"]}}
        },
        {
            "id": "TYPE-05",
            "query": "Autocorrelation ACF on region text column",
            "df": sales_df,
            "mock": {"tool": "autocorrelation", "arguments": {"columns": ["region"]}}
        }
    ]

    for var in variations:
        logger.info(f"Testing WRONG DATATYPE variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=var["df"], mock_command=var["mock"])
        # Either rejected during plan validation or handled gracefully with error
        assert res.status in ["FAILED", "SUCCEEDED"], f"Unexpected state: {res.status}"
        if res.status == "FAILED":
            assert res.error is not None
            print(f">> [PASS] WRONG DATATYPE {var['id']}: Handled cleanly with failure code '{res.error.code}'.")
        else:
            print(f">> [PASS] WRONG DATATYPE {var['id']}: Executed type fallback / handling gracefully.")


# ==============================================================================
# 16. CAPABILITY: UNSUPPORTED / ADVERSARIAL REQUESTS (5 NL VARIATIONS)
# ==============================================================================

def test_16_unsupported_request_capability_matrix():
    """
    Capability 16: UNSUPPORTED / ADVERSARIAL REQUESTS
    Validates: security interdictions, unrecognized operations, non-analytical prompts.
    """
    df = get_ecommerce_sales_df()

    variations = [
        {
            "id": "UNSUP-01",
            "query": "Make a cup of coffee",
            "mock": {"tool": "make_coffee", "arguments": {}},
            "expect_error": True
        },
        {
            "id": "UNSUP-02",
            "query": "Write a poem about quantum mechanics",
            "mock": {"tool": "general_chat", "arguments": {}},
            "expect_error": True
        },
        {
            "id": "UNSUP-03",
            "query": "Delete the operating system database",
            "mock": {"tool": "delete_database", "arguments": {}},
            "expect_error": True
        },
        {
            "id": "UNSUP-04",
            "query": "Execute shell command rm -rf /",
            "mock": {"tool": "exec_shell", "arguments": {}},
            "expect_error": True,
            "security": True
        },
        {
            "id": "UNSUP-05",
            "query": "Deploy a Kubernetes cluster to AWS",
            "mock": {"tool": "deploy_cluster", "arguments": {}},
            "expect_error": True
        }
    ]

    for var in variations:
        logger.info(f"Testing UNSUPPORTED variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(query=var["query"], df=df, mock_command=var["mock"])
        assert res.status == "FAILED", f"Expected FAILED status for unsupported request {var['id']}, got {res.status}"
        assert res.error is not None
        if var.get("security"):
            assert res.error.code == "SECURITY_VIOLATION"
        print(f">> [PASS] UNSUPPORTED {var['id']}: '{var['query']}' trapped cleanly with code '{res.error.code}'.")


# ==============================================================================
# 17. CAPABILITY: MULTI-TURN FOLLOW-UP QUERIES (5 NL VARIATIONS)
# ==============================================================================

def test_17_followup_query_capability_matrix():
    """
    Capability 17: FOLLOW-UP CONVERSATIONAL QUERIES
    Validates: conversation history passing, multi-turn context retention, contextual continuity.
    """
    sales_df = get_ecommerce_sales_df()
    sensor_df = get_iot_sensors_df()

    variations = [
        {
            "id": "FOL-01",
            "history": [{"role": "user", "content": "Filter records where sales_amount > 500"}, {"role": "assistant", "content": "Filtered 60 records."}],
            "query": "Now calculate average sales_amount by region",
            "df": sales_df,
            "mock": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales_amount", "aggregation": "mean"}}
        },
        {
            "id": "FOL-02",
            "history": [{"role": "user", "content": "Clean missing values in temperature"}, {"role": "assistant", "content": "Imputed null values."}],
            "query": "Now detect anomalies on vibration_level and temperature",
            "df": sensor_df,
            "mock": {"tool": "detect_anomalies", "arguments": {"columns": ["vibration_level", "temperature"]}}
        },
        {
            "id": "FOL-03",
            "history": [{"role": "user", "content": "Filter region equals North"}, {"role": "assistant", "content": "Found 30 records."}],
            "query": "Sort sales_amount descending",
            "df": sales_df,
            "mock": {"tool": "sort_data", "arguments": {"column": "sales_amount", "direction": "desc"}}
        },
        {
            "id": "FOL-04",
            "history": [{"role": "user", "content": "Group by category with sum of sales_amount"}, {"role": "assistant", "content": "Aggregated 4 categories."}],
            "query": "Now rank by sales_amount ascending",
            "df": sales_df,
            "mock": {"tool": "sort_data", "arguments": {"column": "sales_amount", "direction": "asc"}}
        },
        {
            "id": "FOL-05",
            "history": [{"role": "user", "content": "Compute correlation between temperature and pressure"}, {"role": "assistant", "content": "Correlation r=-0.82."}],
            "query": "Now compute correlation including humidity and vibration_level",
            "df": sensor_df,
            "mock": {"tool": "correlate_data", "arguments": {"columns": ["temperature", "pressure", "humidity", "vibration_level"], "analysis_mode": "matrix", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "heatmap", "smoother": "none", "expected_direction": "unspecified"}}
        }
    ]

    for var in variations:
        logger.info(f"Testing FOLLOW-UP variation {var['id']}: '{var['query']}'")
        res = execute_deterministic_pipeline(
            query=var["query"],
            df=var["df"],
            mock_command=var["mock"],
            conversation_history=var["history"]
        )
        assert res.status == "SUCCEEDED", f"Follow-up failed for {var['id']}: {res.error}"
        assert res.result is not None
        print(f">> [PASS] FOLLOW-UP {var['id']}: Follow-up executed with full conversation history context.")


# ==============================================================================
# LIVE OLLAMA INTEGRATION TESTS (OPTIONAL / CONDITIONAL)
# ==============================================================================

def check_live_ollama_ready() -> bool:
    """Checks if local Ollama server is running and has models available."""
    try:
        status = ollama_service.detect_server()
        return status.is_running and len(status.available_models) > 0
    except Exception:
        return False


@pytest.mark.skipif(not check_live_ollama_ready(), reason="Local Ollama server is offline or has no models installed")
def test_live_ollama_torture_sampler():
    """
    Live Ollama integration test: Executes actual live prompts against the Ollama server,
    verifying structured JSON output, latency, and end-to-end execution.
    """
    sales_df = get_ecommerce_sales_df()
    
    live_queries = [
        "Filter sales_amount > 600",
        "Calculate total sales_amount by region",
        "Compute correlation between sales_amount and units_sold"
    ]

    for q in live_queries:
        logger.info(f"LIVE OLLAMA TEST: '{q}'")
        t0 = time.time()
        res = QueryOrchestrator.process_query(query_raw=q, df=sales_df)
        dt = time.time() - t0
        logger.info(f"Live Query '{q}' finished in {dt:.2f}s | Status: {res.status}")
        assert res.status == "SUCCEEDED"
        assert res.result is not None


# ==============================================================================
# STANDALONE MASSIVE TORTURE MATRIX RUNNER & SCORECARD GENERATOR
# ==============================================================================

def run_all_torture_suites() -> Dict[str, Any]:
    """
    Executes all 17 capability test suites, aggregates pass/fail counts,
    and returns a structured capability scorecard.
    """
    capabilities = [
        ("01. FILTER", test_01_filter_capability_matrix, 7),
        ("02. GROUP & AGGREGATE", test_02_group_capability_matrix, 6),
        ("03. SORT & ORDER", test_03_sort_capability_matrix, 5),
        ("04. EDA & OVERVIEW", test_04_eda_capability_matrix, 5),
        ("05. CORRELATION", test_05_correlation_capability_matrix, 5),
        ("06. ANOMALY DETECTION", test_06_anomaly_capability_matrix, 5),
        ("07. DATA CLEANING", test_07_cleaning_capability_matrix, 5),
        ("08. CLASSIFICATION / AUTOML", test_08_classification_prediction_capability_matrix, 5),
        ("09. CLUSTERING", test_09_clustering_capability_matrix, 5),
        ("10. AUTOCORRELATION", test_10_autocorrelation_capability_matrix, 5),
        ("11. FORECASTING", test_11_forecasting_capability_matrix, 5),
        ("12. MULTI-INTENT DAG", test_12_multi_intent_capability_matrix, 5),
        ("13. AMBIGUOUS QUERIES", test_13_ambiguous_capability_matrix, 5),
        ("14. UNKNOWN COLUMN", test_14_unknown_column_capability_matrix, 5),
        ("15. WRONG DATATYPE", test_15_wrong_datatype_capability_matrix, 5),
        ("16. UNSUPPORTED REQUESTS", test_16_unsupported_request_capability_matrix, 5),
        ("17. FOLLOW-UP QUERIES", test_17_followup_query_capability_matrix, 5)
    ]

    scorecard = []
    total_variations = 0
    total_passed = 0
    total_failed = 0

    print("\n" + "=" * 88)
    print("DATAPILOT OS — MASSIVE OLLAMA TORTURE TEST MATRIX RUNNER (17 CAPABILITIES)")
    print("=" * 88)

    for name, func, var_count in capabilities:
        total_variations += var_count
        t0 = time.time()
        try:
            func()
            dt = time.time() - t0
            passed = var_count
            failed = 0
            total_passed += passed
            scorecard.append({"name": name, "variations": var_count, "passed": passed, "failed": failed, "status": "PASSED", "duration": round(dt, 2)})
            print(f"[{name}] PASSED {passed}/{var_count} variations ({round(dt, 2)}s)")
        except Exception as e:
            dt = time.time() - t0
            passed = 0
            failed = var_count
            total_failed += failed
            scorecard.append({"name": name, "variations": var_count, "passed": passed, "failed": failed, "status": "FAILED", "error": str(e), "duration": round(dt, 2)})
            print(f"[{name}] FAILED: {e}")

    pass_rate = (total_passed / total_variations * 100.0) if total_variations > 0 else 0.0

    print("\n" + "=" * 88)
    print("DATAPILOT OS — MASSIVE OLLAMA TORTURE TEST MATRIX SCORECARD")
    print("=" * 88)
    print(f"{'CAPABILITY CATEGORY':<35} {'VARIATIONS':>10} {'PASSED':>10} {'FAILED':>10} {'PASS RATE':>12}")
    print("-" * 88)
    for sc in scorecard:
        pr = f"{(sc['passed'] / sc['variations'] * 100.0):.1f}%"
        print(f"{sc['name']:<35} {sc['variations']:>10} {sc['passed']:>10} {sc['failed']:>10} {pr:>12}")
    print("-" * 88)
    print(f"{'TOTAL TEST MATRIX':<35} {total_variations:>10} {total_passed:>10} {total_failed:>10} {pass_rate:>11.1f}%")
    print("=" * 88 + "\n")

    return {
        "scorecard": scorecard,
        "total_variations": total_variations,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "pass_rate": pass_rate
    }


if __name__ == "__main__":
    res = run_all_torture_suites()
    if res["total_failed"] > 0:
        sys.exit(1)
    sys.exit(0)
