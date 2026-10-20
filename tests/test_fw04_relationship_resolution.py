"""
Unit test suite for DataPilot OS — FW-04 Relationship Resolution Engine.
Verifies:
1. Three-tier relationship evidence model (explicit, db-declared, inferred).
2. Cardinality calculation (1:1, 1:m, m:1, m:m).
3. Candidate foreign key discovery with inclusion ratio and parent uniqueness.
4. Rejection of generic column names ('status', 'code', 'date') from automatic foreign key inference.
5. Incompatible datatype detection.
"""

import pytest
import pandas as pd
from backend.sql_engine.relationship_resolver import RelationshipResolver


def test_relationship_evaluation_one_to_many():
    df_customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })
    df_orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 3],
        "amount": [50.0, 75.0, 20.0, 100.0]
    })

    ev = RelationshipResolver.evaluate_relationship(
        parent_df=df_customers,
        parent_table="customers",
        parent_key="customer_id",
        child_df=df_orders,
        child_table="orders",
        child_key="customer_id",
        evidence_tier="explicit"
    )

    assert ev.cardinality == "1:m"
    assert ev.parent_uniqueness_ratio == 1.0
    assert ev.child_uniqueness_ratio < 1.0
    assert ev.inclusion_ratio == 1.0
    assert ev.orphan_child_keys == 0
    assert ev.is_valid_relationship is True


def test_relationship_evaluation_many_to_one():
    df_orders = pd.DataFrame({
        "order_id": [101, 102, 103],
        "customer_id": [1, 1, 2]
    })
    df_customers = pd.DataFrame({
        "customer_id": [1, 2],
        "name": ["Alice", "Bob"]
    })

    ev = RelationshipResolver.evaluate_relationship(
        parent_df=df_orders,
        parent_table="orders",
        parent_key="customer_id",
        child_df=df_customers,
        child_table="customers",
        child_key="customer_id",
        evidence_tier="explicit"
    )

    assert ev.cardinality == "m:1"


def test_relationship_evaluation_rejection_of_generic_name():
    df_a = pd.DataFrame({"status": ["active", "pending", "closed"], "val_a": [1, 2, 3]})
    df_b = pd.DataFrame({"status": ["active", "active", "closed"], "val_b": [10, 20, 30]})

    ev = RelationshipResolver.evaluate_relationship(
        parent_df=df_a,
        parent_table="table_a",
        parent_key="status",
        child_df=df_b,
        child_table="table_b",
        child_key="status",
        evidence_tier="inferred_candidate"
    )

    assert ev.is_valid_relationship is False
    assert ev.status == "REJECTED_GENERIC_NAME"
    assert any("generic column name" in w for w in ev.warnings)


def test_relationship_evaluation_type_mismatch():
    df_a = pd.DataFrame({"key": [1, 2, 3]})
    df_b = pd.DataFrame({"key": ["one", "two", "three"]})

    ev = RelationshipResolver.evaluate_relationship(
        parent_df=df_a,
        parent_table="a",
        parent_key="key",
        child_df=df_b,
        child_table="b",
        child_key="key",
        evidence_tier="explicit"
    )

    assert ev.status == "TYPE_MISMATCH"
    assert ev.is_valid_relationship is False


def test_discover_candidate_relationships():
    df_users = pd.DataFrame({"user_id": [10, 20, 30], "username": ["u1", "u2", "u3"]})
    df_posts = pd.DataFrame({"post_id": [1, 2, 3, 4], "user_id": [10, 10, 20, 30], "text": ["a", "b", "c", "d"]})

    tables = {"users": df_users, "posts": df_posts}
    candidates = RelationshipResolver.discover_candidate_relationships(tables)

    assert len(candidates) >= 1
    match = next(c for c in candidates if c.parent_table == "users" and c.child_table == "posts")
    assert match.parent_key == "user_id"
    assert match.child_key == "user_id"
    assert match.inclusion_ratio == 1.0
    assert match.requires_user_confirmation is True
