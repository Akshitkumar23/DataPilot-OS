"""
DataPilot OS — Phase 2B Provenance Anti-Hardcoding & Cross-Dataset Audit Test Suite.
Verifies that:
1. Governed provenance tracking is 100% dataset-agnostic and domain-agnostic.
2. Provenance state snapshots (row_count, null_count, duplicate_count, fingerprint, columns)
   are derived purely and dynamically from runtime DataFrame state.
3. Cross-dataset isolation: Dataset A (HR/employees schema) and Dataset B (eCommerce schema)
   are processed through the exact same unspecialized provenance pipeline without hardcoded branching.
4. Static AST/regex scan verifies zero hardcoded test identifiers, column names, or mock counts
   in production provenance and governance code.
"""
import ast
import re
import os
import uuid
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.dataset_registry import dataset_registry, DatasetEntry
from backend.persistence.provenance_store import provenance_store, capture_dataset_state_snapshot
from backend.ds_engine.dataset_profiler import DatasetProfiler


def register_test_dataset(df: pd.DataFrame, name: str) -> DatasetEntry:
    """Helper to register a DataFrame into the active dataset_registry dynamically."""
    ds_id = f"ds_{uuid.uuid4().hex[:8]}"
    prof = DatasetProfiler.profile(df, dataset_name=name)
    entry = DatasetEntry(
        dataset_id=ds_id,
        name=name,
        file_path=f"/tmp/{name}",
        format="csv",
        row_count=len(df),
        column_count=len(df.columns),
        knowledge_model=prof.model_dump(),
        loaded_at="2026-09-28T00:00:00",
        file_hash=prof.dataset_hash,
    )
    dataset_registry.register(ds_id, entry)
    dataset_registry._dataframes_cache[ds_id] = df.copy(deep=True)
    dataset_registry.set_active(ds_id)
    return entry


@pytest.fixture(autouse=True)
def clean_test_environment():
    """Ensure clean registry and provenance store for each test."""
    provenance_store.clear()
    dataset_registry._datasets.clear()
    dataset_registry._dataframes_cache.clear()
    yield
    provenance_store.clear()
    dataset_registry._datasets.clear()
    dataset_registry._dataframes_cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_cross_dataset_generic_provenance_pipeline(client):
    """
    Test Dataset A and Dataset B through the identical governed operation & provenance pipeline.
    Verify all snapshots and fingerprints are calculated dynamically without any domain specialization.
    """
    # ── Dataset A: HR Domain (age, income, department) ──
    df_a = pd.DataFrame({
        "age": [None, 30.0, 45.0, 28.0, 28.0],
        "income": [50000.0, 60000.0, 80000.0, 55000.0, 55000.0],
        "department": ["HR", "Engineering", "Sales", "Engineering", "Engineering"]
    })
    entry_a = register_test_dataset(df_a, name="dataset_a_test.csv")
    id_a = entry_a.dataset_id

    # ── Dataset B: eCommerce Domain (product, price, region) ──
    df_b = pd.DataFrame({
        "product": ["Monitor", "Keyboard", "Mouse", "Laptop", "Monitor"],
        "price": [250.0, None, 30.0, 1200.0, 250.0],
        "region": ["North", "South", "East", "West", "North"]
    })
    entry_b = register_test_dataset(df_b, name="dataset_b_test.csv")
    id_b = entry_b.dataset_id

    # Verify initial dynamic state snapshots
    snap_a_init = capture_dataset_state_snapshot(df_a)
    assert snap_a_init.row_count == 5
    assert snap_a_init.column_count == 3
    assert snap_a_init.columns == ["age", "income", "department"]
    assert snap_a_init.null_count == 1
    assert snap_a_init.duplicate_count == 1
    assert snap_a_init.fingerprint == DatasetProfiler.compute_hash(df_a)

    snap_b_init = capture_dataset_state_snapshot(df_b)
    assert snap_b_init.row_count == 5
    assert snap_b_init.column_count == 3
    assert snap_b_init.columns == ["product", "price", "region"]
    assert snap_b_init.null_count == 1
    assert snap_b_init.duplicate_count == 1
    assert snap_b_init.fingerprint == DatasetProfiler.compute_hash(df_b)
    assert snap_a_init.fingerprint != snap_b_init.fingerprint

    # ── 1. Apply drop_nulls on Dataset A (target: age) ──
    resp_a1 = client.post("/api/operations/apply", json={
        "dataset_id": id_a,
        "operation": "drop_nulls",
        "column": "age"
    })
    assert resp_a1.status_code == 200, resp_a1.text
    data_a1 = resp_a1.json()
    assert data_a1["status"] == "SUCCEEDED"
    assert "provenance_event" in data_a1
    prov_a1 = data_a1["provenance_event"]

    # Verify dynamic provenance metrics for Dataset A operation 1
    assert prov_a1["dataset_id"] == id_a
    assert prov_a1["dataset_name"] == "dataset_a_test.csv"
    assert prov_a1["operation"] == "DROP_NULLS"
    assert prov_a1["affected_columns"] == ["age"]
    assert prov_a1["before_state"]["row_count"] == 5
    assert prov_a1["before_state"]["null_count"] == 1
    assert prov_a1["before_state"]["columns"] == ["age", "income", "department"]
    assert prov_a1["after_state"]["row_count"] == 4
    assert prov_a1["after_state"]["null_count"] == 0
    assert prov_a1["after_state"]["columns"] == ["age", "income", "department"]
    assert prov_a1["before_state"]["fingerprint"] != prov_a1["after_state"]["fingerprint"]
    assert prov_a1["after_state"]["fingerprint"] == DatasetProfiler.compute_hash(dataset_registry.get_dataframe(id_a))

    # ── 2. Apply normalize_minmax on Dataset A (target: income) ──
    resp_a2 = client.post("/api/operations/apply", json={
        "dataset_id": id_a,
        "operation": "normalize_minmax",
        "column": "income"
    })
    assert resp_a2.status_code == 200, resp_a2.text
    data_a2 = resp_a2.json()
    prov_a2 = data_a2["provenance_event"]
    assert prov_a2["affected_columns"] == ["income"]
    assert prov_a2["before_state"]["row_count"] == 4
    assert prov_a2["after_state"]["row_count"] == 4
    assert prov_a2["before_state"]["fingerprint"] == prov_a1["after_state"]["fingerprint"]
    assert prov_a2["after_state"]["fingerprint"] == DatasetProfiler.compute_hash(dataset_registry.get_dataframe(id_a))

    # ── 3. Apply drop_nulls on Dataset B (target: price) ──
    resp_b1 = client.post("/api/operations/apply", json={
        "dataset_id": id_b,
        "operation": "drop_nulls",
        "column": "price"
    })
    assert resp_b1.status_code == 200, resp_b1.text
    data_b1 = resp_b1.json()
    prov_b1 = data_b1["provenance_event"]

    # Verify dynamic provenance metrics for Dataset B operation 1
    assert prov_b1["dataset_id"] == id_b
    assert prov_b1["dataset_name"] == "dataset_b_test.csv"
    assert prov_b1["operation"] == "DROP_NULLS"
    assert prov_b1["affected_columns"] == ["price"]
    assert prov_b1["before_state"]["row_count"] == 5
    assert prov_b1["before_state"]["null_count"] == 1
    assert prov_b1["before_state"]["columns"] == ["product", "price", "region"]
    assert prov_b1["after_state"]["row_count"] == 4
    assert prov_b1["after_state"]["null_count"] == 0
    assert prov_b1["after_state"]["columns"] == ["product", "price", "region"]
    assert prov_b1["before_state"]["fingerprint"] != prov_b1["after_state"]["fingerprint"]
    assert prov_b1["after_state"]["fingerprint"] == DatasetProfiler.compute_hash(dataset_registry.get_dataframe(id_b))

    # ── 4. Apply drop_duplicates on Dataset B ──
    resp_b2 = client.post("/api/operations/apply", json={
        "dataset_id": id_b,
        "operation": "drop_duplicates"
    })
    assert resp_b2.status_code == 200, resp_b2.text
    data_b2 = resp_b2.json()
    prov_b2 = data_b2["provenance_event"]
    assert prov_b2["operation"] == "DROP_DUPLICATES"
    assert prov_b2["before_state"]["row_count"] == 4
    assert prov_b2["before_state"]["duplicate_count"] == 1
    assert prov_b2["after_state"]["row_count"] == 3
    assert prov_b2["after_state"]["duplicate_count"] == 0

    # ── 5. Audit Trail Verification & Dataset Isolation ──
    # Query audit for Dataset A
    audit_a = client.get(f"/api/governance/audit?dataset_id={id_a}").json()
    assert audit_a["status"] == "success"
    assert audit_a["total"] == 2
    assert audit_a["count"] == 2
    event_ids_a = [e["event_id"] for e in audit_a["events"]]
    assert prov_a1["event_id"] in event_ids_a
    assert prov_a2["event_id"] in event_ids_a
    assert prov_b1["event_id"] not in event_ids_a
    assert prov_b2["event_id"] not in event_ids_a

    # Query audit for Dataset B
    audit_b = client.get(f"/api/governance/audit?dataset_id={id_b}").json()
    assert audit_b["status"] == "success"
    assert audit_b["total"] == 2
    assert audit_b["count"] == 2
    event_ids_b = [e["event_id"] for e in audit_b["events"]]
    assert prov_b1["event_id"] in event_ids_b
    assert prov_b2["event_id"] in event_ids_b
    assert prov_a1["event_id"] not in event_ids_b
    assert prov_a2["event_id"] not in event_ids_b

    # Query global audit trail
    audit_all = client.get("/api/governance/audit").json()
    assert audit_all["status"] == "success"
    assert audit_all["total"] == 4

    # Query individual event by ID
    event_lookup = client.get(f"/api/governance/audit/{prov_a1['event_id']}")
    assert event_lookup.status_code == 200
    assert event_lookup.json()["event_id"] == prov_a1["event_id"]
    assert event_lookup.json()["affected_columns"] == ["age"]


def test_preview_never_records_provenance_event(client):
    """Verify that preview operations strictly do NOT write to the provenance store."""
    df = pd.DataFrame({"col_x": [1.0, 2.0, None, 4.0]})
    entry = register_test_dataset(df, name="test_preview.csv")

    assert len(provenance_store.get_events()) == 0

    resp = client.post("/api/operations/preview", json={
        "dataset_id": entry.dataset_id,
        "operation": "drop_nulls",
        "column": "col_x"
    })
    assert resp.status_code == 200
    assert len(provenance_store.get_events()) == 0
    assert len(client.get("/api/governance/audit").json()["events"]) == 0


def test_production_provenance_source_code_anti_hardcoding():
    """
    Exhaustive static inspection of production provenance and governance source files.
    Enforces that zero test column names, filenames, or mock metrics exist in production logic.
    """
    production_files = [
        "backend/persistence/provenance_store.py",
        "backend/api/governance.py",
        "backend/api/operations.py",
    ]

    forbidden_patterns = [
        r"\bemployees\.csv\b",
        r"\bemployees\b",
        r"\btemp_dataset_a\b",
        r"\btemp_dataset_b\b",
        r"\bdataset_a\b",
        r"\bdataset_b\b",
        r"\bage\b",
        r"\bincome\b",
        r"\bdepartment\b",
        r"\bproduct\b",
        r"\bprice\b",
        r"\bregion\b",
        r"rows_before\s*=\s*\d+",
        r"rows_after\s*=\s*\d+",
        r"fingerprint\s*=\s*['\"]abc123['\"]",
        r"dataset_version\s*=\s*['\"]v1['\"]",
    ]

    for file_path in production_files:
        assert os.path.exists(file_path), f"File {file_path} must exist"
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Parse AST to ensure valid syntax
        tree = ast.parse(code, filename=file_path)
        assert tree is not None

        # Check for forbidden patterns in source
        for pattern in forbidden_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            lines = code.split("\n")
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#") or clean_line.startswith('"""') or clean_line.startswith("'''") or clean_line.startswith("*"):
                    continue  # Ignore comment/docstring mentions
                match = regex.search(clean_line)
                assert not match, (
                    f"VIOLATION: Forbidden pattern '{pattern}' found in {file_path} at line {line_no}: {clean_line}"
                )
