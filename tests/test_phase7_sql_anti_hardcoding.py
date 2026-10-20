"""
Phase 7 Test Suite: SQL Studio Anti-Hardcoding Invariant.
Tests 3 structurally distinct domain datasets:
1. Genomics / Bio-Informatics: gene_id, expression_level, tissue_type, mutation_flag
2. Supply Chain / Freight: tracking_num, origin_port, transit_hours, cargo_weight_mt, carrier_code
3. IoT Telemetry: device_uuid, reading_timestamp, vibration_hz, temperature_c, pressure_bar

Verifies:
- All queries, filters, and aggregations are 100% schema-driven
- Zero hardcoded column names or assumptions in SQLValidator, GovernedDuckDBManager, or DeterministicSQLBuilder
"""
import pytest
import pandas as pd
import numpy as np

from backend.sql_engine.duckdb_manager import GovernedDuckDBManager
from backend.sql_engine.sql_validator import SQLValidator
from backend.sql_engine.sql_builder import DeterministicSQLBuilder
from backend.sql_engine.sql_models import SQLIntent


def test_domain_1_genomics_sql():
    """Domain 1: Genomics research dataset."""
    df = pd.DataFrame({
        "gene_id": [f"ENSG_{i:05d}" for i in range(50)],
        "expression_level": np.random.exponential(scale=25.0, size=50).round(2),
        "tissue_type": np.random.choice(["Liver", "Brain", "Kidney", "Heart"], size=50),
        "mutation_flag": np.random.choice([0, 1], size=50)
    })

    # Intent-based SQL
    intent = SQLIntent(
        operation="aggregate",
        selected_columns=["tissue_type"],
        aggregations=[{"column": "expression_level", "func": "avg", "alias": "mean_expr"}],
        group_by=["tissue_type"],
        order_by=[{"column": "mean_expr", "direction": "desc"}]
    )
    sql = DeterministicSQLBuilder.build(intent, df)
    
    val = SQLValidator.validate_query(sql, df)
    assert val.is_valid is True
    assert set(val.referenced_columns) == {"tissue_type", "expression_level"}

    res = GovernedDuckDBManager.execute_query(df, sql, dataset_id="genomics_ds")
    assert res.status == "SUCCEEDED"
    assert res.row_count == 4
    assert res.columns == ["tissue_type", "mean_expr"]


def test_domain_2_freight_logistics_sql():
    """Domain 2: Supply Chain Freight dataset."""
    df = pd.DataFrame({
        "tracking_num": [f"TRK_{i:06d}" for i in range(40)],
        "origin_port": np.random.choice(["Rotterdam", "Singapore", "Shanghai", "Los Angeles"], size=40),
        "transit_hours": np.random.randint(12, 120, size=40),
        "cargo_weight_mt": np.random.uniform(5.0, 85.0, size=40).round(1),
        "carrier_code": np.random.choice(["MAERSK", "MSC", "CMA_CGM"], size=40)
    })

    sql = """
    SELECT origin_port, carrier_code, AVG(transit_hours) as avg_hours, SUM(cargo_weight_mt) as total_cargo
    FROM active_dataset
    GROUP BY origin_port, carrier_code
    HAVING total_cargo > 20
    ORDER BY total_cargo DESC
    """
    val = SQLValidator.validate_query(sql, df)
    assert val.is_valid is True
    assert "cargo_weight_mt" in val.referenced_columns
    assert "transit_hours" in val.referenced_columns

    res = GovernedDuckDBManager.execute_query(df, sql, dataset_id="logistics_ds")
    assert res.status == "SUCCEEDED"
    assert "total_cargo" in res.columns


def test_domain_3_iot_telemetry_sql():
    """Domain 3: IoT Sensor Telemetry dataset."""
    df = pd.DataFrame({
        "device_uuid": [f"DEV_{i:03d}" for i in range(60)],
        "vibration_hz": np.random.normal(50.0, 8.5, size=60).round(3),
        "temperature_c": np.random.uniform(20.0, 95.0, size=60).round(2),
        "pressure_bar": np.random.uniform(1.0, 10.0, size=60).round(2)
    })

    # Window query for anomaly screening
    sql = """
    SELECT device_uuid, vibration_hz, temperature_c,
           AVG(vibration_hz) OVER () as global_avg_vib,
           vibration_hz - AVG(vibration_hz) OVER () as vib_deviation
    FROM active_dataset
    ORDER BY vib_deviation DESC
    LIMIT 5
    """
    val = SQLValidator.validate_query(sql, df)
    assert val.is_valid is True

    res = GovernedDuckDBManager.execute_query(df, sql, dataset_id="iot_ds")
    assert res.status == "SUCCEEDED"
    assert res.row_count == 5
    assert len(res.rows) == 5
    assert "vib_deviation" in res.columns
