"""
Real-world validation suite for DataPilot OS — FW-04 Multi-Table Relational Schema Resolution.
Tests against four benchmark multi-table scenarios:
1. Online Retail Fact (Invoices) + Customer Dimension: m:1 enrichment preserving transaction grain.
2. Clinical Patients + Longitudinal Encounters: 1:m relationship with unmatched patient handling.
3. Students + Courses + Enrollments: 3-table associative bridge join.
4. Adversarial Name Collision: Table A (id, status, name) and Table B (id, status, rev) where status is NOT a key.
"""

import pytest
import pandas as pd
from backend.sql_engine.relation_catalog import relation_catalog
from backend.sql_engine.relationship_resolver import RelationshipResolver
from backend.sql_engine.join_planner import JoinPlanner, JoinPlan, JoinStep
from backend.sql_engine.duckdb_manager import GovernedDuckDBManager


@pytest.fixture(autouse=True)
def clean_catalog():
    relation_catalog.clear()
    yield
    relation_catalog.clear()


# ── SCENARIO 1: ONLINE RETAIL FACT + CUSTOMER DIMENSION (m:1 ENRICHMENT) ────

def test_real_world_online_retail_customer_enrichment():
    # Simulate Orders fact and Customers dim
    df_customers = pd.DataFrame({
        "CustomerID": [12345.0, 12346.0, 12347.0],
        "Country": ["United Kingdom", "France", "Germany"],
        "LoyaltyTier": ["Gold", "Silver", "Platinum"]
    })
    df_orders = pd.DataFrame({
        "InvoiceNo": [536365, 536366, 536367, 536368, 536369],
        "StockCode": ["85123A", "71053", "84406B", "22752", "21730"],
        "Quantity": [6, 6, 8, 2, 4],
        "UnitPrice": [2.55, 3.39, 2.75, 7.65, 1.25],
        "CustomerID": [12345.0, 12345.0, 12346.0, 12347.0, 99999.0]  # 4/5 unique keys = 80% inclusion
    })

    relation_catalog.register_dataset("ds_cust", "customers", df_customers)
    relation_catalog.register_dataset("ds_orders", "orders", df_orders)

    # 1. Discover candidates
    candidates = RelationshipResolver.discover_candidate_relationships(relation_catalog.get_all_dataframes())
    cust_cand = next(c for c in candidates if c.parent_table == "customers" and c.child_table == "orders")
    assert cust_cand.parent_key == "CustomerID"
    assert cust_cand.cardinality == "1:m"

    # 2. Execute Left Join Enrichment
    plan = JoinPlan(
        left_table="orders",
        steps=[
            JoinStep(
                right_table="customers",
                join_type="left",
                left_keys=["CustomerID"],
                right_keys=["CustomerID"]
            )
        ]
    )

    res = JoinPlanner.execute_join_plan(plan, relation_catalog.get_all_dataframes())
    assert res.status == "SUCCEEDED"
    # Preserves order grain: exactly 5 rows
    assert len(res.result_df) == 5
    assert res.expansion_ratio == 1.0
    # CustomerID 99999 is preserved with null LoyaltyTier
    unmatched_row = res.result_df[res.result_df["CustomerID"] == 99999.0]
    assert len(unmatched_row) == 1
    assert pd.isna(unmatched_row["LoyaltyTier"].iloc[0])


# ── SCENARIO 2: CLINICAL PATIENTS + LONGITUDINAL ENCOUNTERS (1:m PRESERVATION) ──

def test_real_world_clinical_patients_and_encounters():
    df_patients = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "age": [65, 72, 45],
        "gender": ["M", "F", "M"]
    })
    df_encounters = pd.DataFrame({
        "encounter_id": [101, 102, 103, 104, 105],
        "patient_id": ["P1", "P1", "P2", "P2", "P2"],
        "systolic_bp": [130, 135, 140, 142, 138],
        "test_day": [1, 14, 1, 30, 60]
    })

    relation_catalog.register_dataset("ds_patients", "patients", df_patients)
    relation_catalog.register_dataset("ds_encounters", "encounters", df_encounters)

    # Multi-relation SQL test via GovernedDuckDBManager
    sql = """
    SELECT 
        p.patient_id, 
        p.age, 
        COUNT(e.encounter_id) AS total_visits, 
        AVG(e.systolic_bp) AS avg_systolic
    FROM patients p
    LEFT JOIN encounters e ON p.patient_id = e.patient_id
    GROUP BY p.patient_id, p.age
    ORDER BY p.patient_id
    """
    res = GovernedDuckDBManager.execute_query(sql, df=df_patients, dataset_id="ds_patients", dataset_name="patients")
    assert res.status == "SUCCEEDED"
    assert len(res.rows) == 3
    # P3 has 0 visits and None avg_systolic
    p3 = next(r for r in res.rows if r["patient_id"] == "P3")
    assert p3["total_visits"] == 0
    assert p3["avg_systolic"] is None


# ── SCENARIO 3: STUDENTS + COURSES + ENROLLMENTS (3-TABLE BRIDGE PATH) ───────

def test_real_world_three_table_bridge_join():
    df_students = pd.DataFrame({"student_id": [1, 2], "name": ["Alice", "Bob"]})
    df_courses = pd.DataFrame({"course_id": [101, 102], "title": ["CS101", "MATH201"]})
    df_enrollments = pd.DataFrame({
        "enrollment_id": [1, 2, 3],
        "student_id": [1, 1, 2],
        "course_id": [101, 102, 101],
        "grade": ["A", "A", "B"]
    })

    relation_catalog.register_dataset("ds_s", "students", df_students)
    relation_catalog.register_dataset("ds_c", "courses", df_courses)
    relation_catalog.register_dataset("ds_e", "enrollments", df_enrollments)

    # Execute 3-table join via DuckDB
    sql = """
    SELECT s.name, c.title, e.grade
    FROM enrollments e
    JOIN students s ON e.student_id = s.student_id
    JOIN courses c ON e.course_id = c.course_id
    ORDER BY s.name, c.title
    """
    res = GovernedDuckDBManager.execute_query(sql, df=df_enrollments, dataset_id="ds_e", dataset_name="enrollments")
    assert res.status == "SUCCEEDED"
    assert len(res.rows) == 3
    assert res.rows[0]["name"] == "Alice"
    assert res.rows[0]["title"] == "CS101"


# ── SCENARIO 4: ADVERSARIAL SAME-NAME COLLISION (STATUS NOT A KEY) ────────────

def test_adversarial_name_collision_status_rejection():
    # Both tables have 'order_id' (real key) and 'status' (generic collision)
    df_orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "status": ["pending", "completed", "pending"],
        "order_total": [100.0, 200.0, 300.0]
    })
    df_shipping = pd.DataFrame({
        "order_id": [1, 2, 3],
        "status": ["shipped", "delivered", "shipped"],
        "tracking_num": ["TRK1", "TRK2", "TRK3"]
    })

    relation_catalog.register_dataset("ds_o", "orders", df_orders)
    relation_catalog.register_dataset("ds_s", "shipping", df_shipping)

    # 1. Inspect Candidate Relationships: 'status' MUST be rejected as generic
    candidates = RelationshipResolver.discover_candidate_relationships(relation_catalog.get_all_dataframes())
    status_cands = [c for c in candidates if c.parent_key == "status"]
    assert len(status_cands) == 0

    # 2. 'order_id' is correctly recognized as the 1:1 join key
    id_cand = next(c for c in candidates if c.parent_key == "order_id")
    assert id_cand.is_valid_relationship is True

    # 3. Join on 'order_id' applies deterministic suffixing to colliding 'status' column
    plan = JoinPlan(
        left_table="orders",
        steps=[JoinStep(right_table="shipping", join_type="left", left_keys=["order_id"], right_keys=["order_id"])]
    )
    res = JoinPlanner.execute_join_plan(plan, relation_catalog.get_all_dataframes())
    assert res.status == "SUCCEEDED"
    assert "status" in res.result_df.columns
    assert "status_shipping" in res.result_df.columns
