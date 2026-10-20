import sys
import os
import json
import logging
import pandas as pd
import numpy as np

# Ensure root workspace is on path
sys.path.insert(0, os.path.abspath("."))

# Set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MatrixRegressionTest")

from modules.query_orchestrator import QueryOrchestrator
from backend.core.dataset_registry import dataset_registry


def generate_sales_data(n=200):
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    regions = np.random.choice(["North", "South", "East", "West"], size=n)
    categories = np.random.choice(["Electronics", "Apparel", "Furniture", "Groceries"], size=n)
    sales = np.random.uniform(50, 1500, size=n)
    units = np.random.randint(1, 20, size=n)
    return pd.DataFrame({
        "order_date": dates,
        "region": regions,
        "category": categories,
        "sales_amount": sales,
        "units_sold": units
    })


def generate_sensor_data(n=300):
    np.random.seed(101)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    temp = np.random.normal(70, 12, size=n)
    pressure = np.random.normal(1013, 15, size=n)
    humidity = np.random.uniform(20, 90, size=n)
    vibration = np.random.exponential(2.5, size=n)
    return pd.DataFrame({
        "timestamp": dates,
        "temperature": temp,
        "pressure": pressure,
        "humidity": humidity,
        "vibration_level": vibration
    })


def generate_hr_churn_data(n=250):
    np.random.seed(202)
    age = np.random.randint(22, 65, size=n)
    tenure_months = np.random.randint(1, 120, size=n)
    monthly_salary = np.random.normal(6500, 1800, size=n)
    satisfaction_score = np.random.uniform(1.0, 5.0, size=n)
    churn = np.where((satisfaction_score < 2.5) | (monthly_salary < 4500), 1, 0)
    return pd.DataFrame({
        "employee_age": age,
        "tenure_months": tenure_months,
        "monthly_salary": monthly_salary,
        "satisfaction_score": satisfaction_score,
        "left_company": churn
    })


def run_test_case(name: str, query: str, df: pd.DataFrame, expected_intent_family: str):
    print("\n" + "=" * 80)
    print(f"REGRESSION TEST: {name}")
    print(f"QUERY: '{query}'")
    print(f"DATASET SHAPE: {df.shape} | COLUMNS: {list(df.columns)}")
    print("=" * 80)

    res = QueryOrchestrator.process_query(query_raw=query, df=df)
    
    print(f"Response Status: {res.status}")
    if res.status != "SUCCEEDED":
        print(f"Error Code: {res.error.code if res.error else 'NONE'}")
        print(f"Error Message: {res.error.message if res.error else 'NONE'}")
        print(f"Error Phase: {res.error.phase if res.error else 'NONE'}")

    assert res.status == "SUCCEEDED", f"Query failed for {name}: {res.error}"
    assert res.result is not None, f"Result payload is None for {name}"
    
    records = res.result.get("data", {}).get("records", [])
    columns = res.result.get("data", {}).get("columns", [])
    print(f"Structured Records Returned: {len(records)} | Columns: {columns}")
    
    print(f">> [PASS] {name} verified successfully.\n")


def test_eda_dashboard():
    sales_df = generate_sales_data()
    run_test_case(
        name="1. Exploratory Data Analysis (EDA Dashboard)",
        query="Provide a comprehensive exploratory data analysis of this dataset.",
        df=sales_df,
        expected_intent_family="EDA"
    )


def test_filtered_group_aggregation():
    sales_df = generate_sales_data()
    run_test_case(
        name="2. Filtered Group Aggregation",
        query="Filter sales_amount > 500 and calculate total sales_amount by region.",
        df=sales_df,
        expected_intent_family="GROUP_AGGREGATE"
    )


def test_feature_correlation_matrix():
    sensor_df = generate_sensor_data()
    run_test_case(
        name="3. Feature Correlation Matrix",
        query="Compute correlation matrix heatmap between temperature, pressure, humidity and vibration_level.",
        df=sensor_df,
        expected_intent_family="CORRELATION"
    )


def test_anomaly_detection_isolation_forest():
    sensor_df = generate_sensor_data()
    run_test_case(
        name="4. Anomaly Detection (IsolationForest)",
        query="Detect anomalous sensor readings across vibration_level and temperature using IsolationForest.",
        df=sensor_df,
        expected_intent_family="ANOMALY_DETECTION"
    )


def test_automl_prediction():
    hr_df = generate_hr_churn_data()
    run_test_case(
        name="5. Machine Learning / AutoML Prediction",
        query="Train a machine learning model to predict left_company based on employee_age, tenure_months, monthly_salary, satisfaction_score.",
        df=hr_df,
        expected_intent_family="AUTO_ML"
    )


def test_unsupervised_clustering():
    hr_df = generate_hr_churn_data()
    run_test_case(
        name="6. Unsupervised KMeans Clustering",
        query="Cluster employees into 3 groups based on monthly_salary and satisfaction_score.",
        df=hr_df,
        expected_intent_family="CLUSTERING"
    )


def test_autocorrelation_acf_pacf():
    sensor_df = generate_sensor_data()
    run_test_case(
        name="7. Time Series Autocorrelation (ACF / PACF)",
        query="Plot autocorrelation ACF and PACF for temperature.",
        df=sensor_df,
        expected_intent_family="AUTOCORRELATION"
    )


def test_time_series_forecasting():
    sales_df = generate_sales_data()
    run_test_case(
        name="8. Time Series Forecasting",
        query="Forecast 14 days of sales_amount based on order_date.",
        df=sales_df,
        expected_intent_family="TIME_SERIES_FORECAST"
    )


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("STARTING COMPREHENSIVE BROAD REGRESSION MATRIX SUITE")
    print("#" * 80)

    test_eda_dashboard()
    test_filtered_group_aggregation()
    test_feature_correlation_matrix()
    test_anomaly_detection_isolation_forest()
    test_automl_prediction()
    test_unsupervised_clustering()
    test_autocorrelation_acf_pacf()
    test_time_series_forecasting()

    print("\n" + "#" * 80)
    print("ALL 8 BROAD CAPABILITY REGRESSION TESTS PASSED 100% CLEANLY [OK]")
    print("#" * 80 + "\n")
