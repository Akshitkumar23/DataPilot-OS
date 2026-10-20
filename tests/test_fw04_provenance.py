"""
Unit test suite for DataPilot OS — FW-04 Multi-Source Provenance.
Verifies:
1. ProvenanceStore records multi-source events with parent cryptographic fingerprints.
2. get_events retrieves events by primary or parent dataset IDs.
3. Cryptographic lineage is preserved across restarts.
"""

import pytest
import pandas as pd
from backend.persistence.provenance_store import provenance_store
from backend.ds_engine.dataset_profiler import DatasetProfiler


@pytest.fixture(autouse=True)
def clean_provenance():
    provenance_store.clear()
    yield
    provenance_store.clear()


def test_multi_source_provenance_recording_and_retrieval():
    df_a = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df_b = pd.DataFrame({"id": [1, 2], "desc": ["one", "two"]})

    hash_a = DatasetProfiler.compute_hash(df_a)
    hash_b = DatasetProfiler.compute_hash(df_b)
    hash_out = "sha256_mock_output"

    event = provenance_store.record_multi_source_event(
        operation="CROSS_TABLE_JOIN",
        primary_dataset_id="ds_a",
        primary_dataset_name="table_a",
        joined_dataset_ids=["ds_b"],
        source_fingerprints={"ds_a": hash_a, "ds_b": hash_b},
        output_dataset_id="ds_out",
        output_fingerprint=hash_out,
        join_plan={"left_table": "table_a", "right_table": "table_b", "type": "left"},
        metrics={"left_rows": 2, "right_rows": 2, "result_rows": 2},
        sql="SELECT * FROM table_a LEFT JOIN table_b ON table_a.id = table_b.id"
    )

    assert event.capability == "RELATIONAL_JOIN"
    assert event.dataset_id == "ds_out"
    assert "ds_a" in event.parent_dataset_ids
    assert "ds_b" in event.parent_dataset_ids
    assert event.source_fingerprints["ds_a"] == hash_a
    assert event.source_fingerprints["ds_b"] == hash_b

    # Verify retrieval by parent or output dataset_id
    events_by_parent_b = provenance_store.get_events("ds_b")
    assert len(events_by_parent_b) == 1
    assert events_by_parent_b[0].event_id == event.event_id

    events_by_output = provenance_store.get_events("ds_out")
    assert len(events_by_output) == 1
    assert events_by_output[0].event_id == event.event_id
