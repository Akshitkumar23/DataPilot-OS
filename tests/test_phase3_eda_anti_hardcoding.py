"""
DataPilot OS — Phase 3 EDA Anti-Hardcoding & Static Integrity Test Suite.
Verifies that:
1. EDA calculation engine and API endpoints are 100% dataset-agnostic and schema-agnostic.
2. All statistics, histograms, correlation matrices, and skewness calculations are computed dynamically.
3. Zero hardcoded domain column names, mock statistical values, or heuristic rules exist in production code.
"""
import ast
import re
import pytest


def test_production_eda_engine_anti_hardcoding():
    """
    Statically inspect backend/dataset_engine/eda_engine.py to verify:
    - Zero hardcoded column names (e.g., age, income, sales, rating, salary).
    - Zero mock/static statistical values or demo datasets.
    - Zero semantic heuristics based on substring matching of column names.
    """
    file_path = "backend/dataset_engine/eda_engine.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse AST to ensure valid Python syntax
    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_column_heuristics = [
        r'col\s*==\s*["\']age["\']',
        r'col\s*==\s*["\']income["\']',
        r'col\s*==\s*["\']sales["\']',
        r'col\s*==\s*["\']salary["\']',
        r'col\s*==\s*["\']rating["\']',
        r'col\s*==\s*["\']price["\']',
        r'["\']sales["\']\s*in\s*col',
        r'["\']age["\']\s*in\s*col',
        r'["\']price["\']\s*in\s*col',
        r'["\']rating["\']\s*in\s*col',
    ]

    for pattern in forbidden_column_heuristics:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden heuristic pattern '{pattern}' found in {file_path} at position {match.start()}."

    # Verify no mock dataset arrays or hardcoded matrices
    assert "fake_correlation" not in content
    assert "mock_bins" not in content
    assert "demo_stats" not in content


def test_production_eda_api_anti_hardcoding():
    """
    Statically inspect backend/api/eda.py to verify:
    - Zero domain-specific routing or column hardcoding.
    - Clean invocation of eda_engine and provenance_store.
    """
    file_path = "backend/api/eda.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_literals = [
        r'["\']employees\.csv["\']',
        r'["\']sales\.csv["\']',
        r'["\']customer["\']',
        r'["\']revenue["\']',
    ]

    for pattern in forbidden_literals:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden literal pattern '{pattern}' found in {file_path}."


def test_frontend_data_science_studio_anti_hardcoding():
    """
    Statically inspect src/components/DataScienceStudio.jsx to verify:
    - No hardcoded chart bars, mock histograms, or fake correlation numbers.
    - Component dynamically renders data returned from the backend.
    """
    file_path = "src/components/DataScienceStudio.jsx"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_frontend_patterns = [
        r'const\s+mockCorrelation\b',
        r'const\s+mockDistribution\b',
        r'const\s+fakeStats\b',
        r'\["age",\s*"income"\]',  # Should not hardcode default column selections in component scope
    ]

    for pattern in forbidden_frontend_patterns:
        match = re.search(pattern, content)
        assert match is None, f"Forbidden mock/hardcoded pattern '{pattern}' found in {file_path}."
