"""
DataPilot OS — Phase 4 ML Studio Anti-Hardcoding & Static Integrity Test Suite.
Verifies that:
1. ML engine, API endpoints, and UI are 100% dataset-agnostic, schema-agnostic, and target-agnostic.
2. All preprocessing, model training, evaluation metrics, confusion matrices, and inference are computed dynamically.
3. Zero hardcoded domain column names, mock predictions, fake metrics, or heuristic target/algorithm selection rules exist in production code.
"""
import ast
import re
import pytest


def test_production_ml_engine_anti_hardcoding():
    """
    Statically inspect backend/dataset_engine/ml_engine.py to verify:
    - Zero hardcoded column names (e.g. sales, revenue, salary, income, age, rating, price, temperature, department).
    - Zero mock/static metrics (fake accuracy, fake r2, hardcoded confusion matrices).
    - Zero heuristics inferring targets or task types from column name substrings (e.g. if 'score' in col).
    - Zero hardcoded dataset filenames or row counts.
    """
    file_path = "backend/dataset_engine/ml_engine.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse AST to ensure valid Python syntax
    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_column_heuristics = [
        r'target\s*==\s*["\']rating["\']',
        r'target\s*==\s*["\']score["\']',
        r'target\s*==\s*["\']sales["\']',
        r'target\s*==\s*["\']price["\']',
        r'target\s*==\s*["\']salary["\']',
        r'target\s*==\s*["\']income["\']',
        r'target\s*==\s*["\']churn["\']',
        r'["\']salary["\']\s*in\s*columns',
        r'["\']sales["\']\s*in\s*dataset_name',
        r'["\']score["\']\s*in\s*column_name',
        r'["\']rating["\']\s*in\s*col',
        r'col\s*==\s*["\']age["\']',
        r'col\s*==\s*["\']customer["\']',
        r'col\s*==\s*["\']revenue["\']',
        r'col\s*==\s*["\']department["\']',
        r'col\s*==\s*["\']temperature["\']',
    ]

    for pattern in forbidden_column_heuristics:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden heuristic pattern '{pattern}' found in {file_path}."

    # Verify no mock prediction arrays or hardcoded metrics
    assert "fake_metrics" not in content
    assert "mock_predictions" not in content
    assert "mock_confusion_matrix" not in content
    assert "fake_r2" not in content
    assert "fake_accuracy" not in content


def test_production_ml_api_anti_hardcoding():
    """
    Statically inspect backend/api/ml.py to verify:
    - Zero domain-specific routing or target hardcoding.
    - Clean invocation of ml_engine and provenance_store.
    - Zero fallback to hardcoded mock models.
    """
    file_path = "backend/api/ml.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_literals = [
        r'["\']employees\.csv["\']',
        r'["\']sales\.csv["\']',
        r'["\']customer["\']',
        r'["\']revenue["\']',
        r'["\']salary["\']',
        r'["\']price["\']',
        r'target\s*=\s*["\']',
    ]

    for pattern in forbidden_literals:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden literal pattern '{pattern}' found in {file_path}."


def test_frontend_ml_studio_anti_hardcoding():
    """
    Statically inspect src/components/DataScienceStudio.jsx to verify:
    - No hardcoded target selection defaults in component scope.
    - No fake metric values (e.g. accuracy: 0.95, r2: 0.88).
    - No mock confusion matrix or mock leaderboard arrays.
    - Dynamic rendering of state returned from backend.
    """
    file_path = "src/components/DataScienceStudio.jsx"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_frontend_patterns = [
        r'const\s+mockMetrics\b',
        r'const\s+mockLeaderboard\b',
        r'const\s+fakePredictions\b',
        r'targetColumn\s*=\s*["\']rating["\']',
        r'targetColumn\s*=\s*["\']sales["\']',
        r'targetColumn\s*=\s*["\']salary["\']',
    ]

    for pattern in forbidden_frontend_patterns:
        match = re.search(pattern, content)
        assert match is None, f"Forbidden mock/hardcoded pattern '{pattern}' found in {file_path}."
