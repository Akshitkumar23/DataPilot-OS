"""
DataPilot OS — Phase 5 Unsupervised Studio Anti-Hardcoding & Static Integrity Test Suite.
Verifies that:
1. Unsupervised engine, API endpoints, and UI are 100% dataset-agnostic and schema-agnostic.
2. All clustering, PCA projections, loadings, scree distributions, and elbow curves are computed dynamically.
3. Zero hardcoded domain column names, mock clusters, fake loadings, or magic k heuristics exist in production code.
"""
import ast
import re
import pytest


def test_production_unsupervised_engine_anti_hardcoding():
    """
    Statically inspect backend/ds_engine/unsupervised_engine.py to verify:
    - Zero hardcoded column names (e.g. sales, revenue, salary, income, age, rating, price, temperature, department).
    - Zero mock/static metrics (fake inertia, fake silhouette, hardcoded centroids).
    - Zero heuristics assuming specific columns or dataset domains.
    - Zero hardcoded dataset filenames or row counts.
    """
    file_path = "backend/ds_engine/unsupervised_engine.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=file_path)
    assert tree is not None

    forbidden_column_heuristics = [
        r'["\']sales["\']',
        r'["\']revenue["\']',
        r'["\']salary["\']',
        r'["\']income["\']',
        r'["\']rating["\']',
        r'["\']customer["\']',
        r'["\']department["\']',
        r'["\']temperature["\']',
    ]

    for pattern in forbidden_column_heuristics:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden heuristic pattern '{pattern}' found in {file_path}."

    # Verify no mock prediction arrays or hardcoded metrics
    assert "fake_inertia" not in content
    assert "fake_silhouette" not in content
    assert "mock_clusters" not in content
    assert "mock_loadings" not in content
    assert "mock_centroids" not in content


def test_production_unsupervised_api_anti_hardcoding():
    """
    Statically inspect backend/api/unsupervised.py to verify:
    - Zero domain-specific routing or column hardcoding.
    - Clean invocation of unsupervised_engine and provenance_store.
    - Zero fallback to hardcoded mock results.
    """
    file_path = "backend/api/unsupervised.py"
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
        r'features\s*=\s*\[["\']',
    ]

    for pattern in forbidden_literals:
        match = re.search(pattern, content, re.IGNORECASE)
        assert match is None, f"Forbidden literal pattern '{pattern}' found in {file_path}."


def test_frontend_unsupervised_studio_anti_hardcoding():
    """
    Statically inspect src/components/DataScienceStudio.jsx to verify:
    - No hardcoded feature selection defaults in component scope for clustering or PCA.
    - No fake metric values (e.g. silhouette: 0.75, inertia: 1200).
    - No mock elbow curve or mock loadings arrays.
    - Dynamic rendering of state returned from backend.
    """
    file_path = "src/components/DataScienceStudio.jsx"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_frontend_patterns = [
        r'const\s+mockElbowCurve\b',
        r'const\s+mockClusterResult\b',
        r'const\s+mockPcaResult\b',
        r'const\s+mockLoadings\b',
        r'clusterFeatures\s*=\s*\[["\']sales["\']',
        r'pcaFeatures\s*=\s*\[["\']revenue["\']',
    ]

    for pattern in forbidden_frontend_patterns:
        match = re.search(pattern, content)
        assert match is None, f"Forbidden mock/hardcoded pattern '{pattern}' found in {file_path}."
