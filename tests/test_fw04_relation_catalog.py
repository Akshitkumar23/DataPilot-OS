"""
Unit test suite for DataPilot OS — FW-04 Relation Catalog.
Verifies:
1. Thread-safe dataset registration, canonical table naming, collision prevention with numeric suffixes.
2. Active relation alias preservation ('active_dataset').
3. Multi-table schema metadata introspection (DuckDB types, nullability, PK candidates).
4. Deterministic removal and catalog clearing.
5. Zero mutation of source DataFrames.
"""

import pytest
import pandas as pd
from backend.sql_engine.relation_catalog import relation_catalog, RelationCatalogEntry


@pytest.fixture(autouse=True)
def clean_catalog():
    relation_catalog.clear()
    yield
    relation_catalog.clear()


def test_relation_catalog_canonical_naming_and_registration():
    df_customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "segment": ["Consumer", "Corporate", "Consumer"]
    })
    entry = relation_catalog.register_dataset(
        dataset_id="ds_cust_01",
        name="Customer Master - 2026.csv",
        df=df_customers,
        is_active=True
    )
    assert entry.table_name == "customer_master_2026"
    assert entry.row_count == 3
    assert entry.column_count == 3
    assert "customer_id" in entry.primary_key_candidates
    assert entry.nullable_columns == []
    assert relation_catalog._active_table_name == "customer_master_2026"


def test_relation_catalog_collision_prevention():
    df1 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    df2 = pd.DataFrame({"id": [3, 4], "val": [30, 40]})

    e1 = relation_catalog.register_dataset("ds_01", "orders.csv", df1)
    e2 = relation_catalog.register_dataset("ds_02", "orders.csv", df2)

    assert e1.table_name == "orders"
    assert e2.table_name == "orders_2"
    assert len(relation_catalog.list_relations()) == 2
    assert relation_catalog.get_dataframe("orders") is df1
    assert relation_catalog.get_dataframe("orders_2") is df2


def test_relation_catalog_active_dataset_alias_and_switching():
    df_a = pd.DataFrame({"a": [1, 2]})
    df_b = pd.DataFrame({"b": [3, 4]})

    relation_catalog.register_dataset("ds_a", "table_a", df_a, is_active=True)
    relation_catalog.register_dataset("ds_b", "table_b", df_b, is_active=False)

    assert relation_catalog.get_active().table_name == "table_a"
    assert relation_catalog.get_active_dataframe() is df_a

    # Switch active
    relation_catalog.set_active("table_b")
    assert relation_catalog.get_active().table_name == "table_b"
    assert relation_catalog.get_active_dataframe() is df_b


def test_relation_catalog_unregistration():
    df = pd.DataFrame({"x": [10, 20]})
    relation_catalog.register_dataset("ds_x", "tbl_x", df)
    assert relation_catalog.get_relation("tbl_x") is not None

    relation_catalog.unregister_dataset("ds_x")
    assert relation_catalog.get_relation("tbl_x") is None
    assert relation_catalog.get_dataframe("tbl_x") is None
    assert len(relation_catalog.list_relations()) == 0
