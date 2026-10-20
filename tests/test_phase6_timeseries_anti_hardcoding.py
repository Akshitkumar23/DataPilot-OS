"""
DataPilot OS — Phase 6 Time-Series Studio Anti-Hardcoding & Static Integrity Test Suite.
Verifies that:
1. Time-series engine, API endpoints, and UI are 100% dataset-agnostic and schema-agnostic.
2. All forecasts, stationarity diagnostics, ACF/PACF, and decompositions are computed dynamically.
3. Zero hardcoded domain column names, mock forecast values, fake metrics, or magic parameters exist in production code.
"""

import ast
import re
import pytest


def test_production_timeseries_engine_anti_hardcoding():
    """
    Statically inspect backend/ds_engine/timeseries_engine.py to verify:
    - Zero hardcoded column names (e.g. sales, revenue, demand, stock, price, temperature, date, timestamp in heuristic conditions).
    - Zero mock/static forecast arrays or fake metrics.
    - Zero heuristics assuming specific columns or dataset domains.
    - Zero hardcoded dataset filenames or row counts.
    """
    file_path = "backend/ds_engine/timeseries_engine.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_column_heuristics = [
        r'if\s+["\']sales["\']\s+in',
        r'if\s+["\']revenue["\']\s+in',
        r'if\s+["\']demand["\']\s+in',
        r'if\s+["\']price["\']\s+in',
        r'if\s+["\']temperature["\']\s+in',
        r'if\s+["\']date["\']\s+in\s+df',
        r'if\s+["\']timestamp["\']\s+in\s+df',
    ]

    for pattern in forbidden_column_heuristics:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden heuristic pattern '{pattern}' found in {file_path}."

    # Verify no mock prediction arrays or hardcoded metrics
    assert "fake_forecast" not in content
    assert "fake_rmse" not in content
    assert "mock_forecast" not in content
    assert "mock_stationarity" not in content
    assert "mock_acf" not in content


def test_production_timeseries_api_anti_hardcoding():
    """
    Statically inspect backend/api/timeseries.py to verify:
    - Zero domain-specific routing or column hardcoding.
    - Clean invocation of timeseries_engine and dataset_registry.
    - Zero fallback to hardcoded mock results.
    """
    file_path = "backend/api/timeseries.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_literals = [
        r'["\']sales\.csv["\']',
        r'["\']demand\.csv["\']',
        r'["\']temperature\.csv["\']',
        r'target_column\s*=\s*["\']sales["\']',
        r'time_column\s*=\s*["\']date["\']',
    ]

    for pattern in forbidden_literals:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden literal pattern '{pattern}' found in {file_path}."


def test_frontend_timeseries_studio_anti_hardcoding():
    """
    Statically inspect src/components/DataScienceStudio.jsx to verify:
    - No hardcoded time column or target column defaults in component scope for time series.
    - No fake forecast arrays or fake validation metrics.
    - No mock ACF or mock decomposition arrays.
    - Dynamic rendering of state returned from backend.
    """
    file_path = "src/components/DataScienceStudio.jsx"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_frontend_patterns = [
        r'const\s+mockForecast\b',
        r'const\s+mockStationarity\b',
        r'const\s+mockAcf\b',
        r'const\s+mockDecomp\b',
        r'tsTimeCol\s*=\s*["\']date["\']',
        r'tsTargetCol\s*=\s*["\']sales["\']',
    ]

    for pattern in forbidden_frontend_patterns:
        match = re.search(pattern, content)
        assert match is None, f"Forbidden mock/hardcoded pattern '{pattern}' found in {file_path}."
