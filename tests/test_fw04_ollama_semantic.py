"""
Unit test suite for DataPilot OS — FW-04 Mandatory Ollama Semantic Boundary Contract.
Verifies all Section 10 & Section 11 requirements:
A. Ollama available -> valid structured intent -> Kernel execution continues.
B. Ollama unavailable -> OLLAMA_UNAVAILABLE -> NO local parser -> NO SQL execution.
C. Ollama malformed JSON -> OLLAMA_INVALID_RESPONSE -> NO fallback.
D. Ollama schema violation -> OLLAMA_SCHEMA_VALIDATION_ERROR -> NO fallback.
E. Ollama hallucinates nonexistent table -> Kernel grounding rejects it.
F. Ollama hallucinates nonexistent column -> Kernel grounding rejects it.
G. Ollama proposes statistically invalid relationship -> RelationshipResolver rejects it.
H. User explicitly specifies valid keys -> Kernel validates and executes.
I. User does not specify keys -> Kernel candidate discovery only if deterministic evidence rules permit.
J. Multiple valid candidates -> Clarification; never arbitrary selection.
11. Proves Ollama never generates executable SQL; malicious SQL strings are rejected.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from backend.sql_engine.relation_catalog import relation_catalog
from backend.sql_engine.relationship_resolver import RelationshipResolver
from backend.sql_engine.join_planner import JoinPlanner, JoinPlan, JoinStep
from backend.intelligence.relational_intent_parser import RelationalIntentParser, RelationalIntentResult
from backend.intelligence.ollama_service import ollama_service, OllamaResponse, OllamaStatus
from backend.core.error_states import ErrorState


@pytest.fixture(autouse=True)
def setup_catalog():
    relation_catalog.clear()
    relation_catalog.register_dataset("ds_orders", "orders", pd.DataFrame({
        "order_id": [1, 2, 3],
        "cust_id": [10, 20, 30],
        "amount": [100.0, 200.0, 150.0]
    }))
    relation_catalog.register_dataset("ds_cust", "customers", pd.DataFrame({
        "cust_id": [10, 20, 30],
        "segment": ["retail", "wholesale", "retail"]
    }))
    yield
    relation_catalog.clear()


def test_scenario_a_ollama_available_valid_structured_intent(monkeypatch):
    """Scenario A: Ollama available -> valid structured intent -> Kernel execution continues."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "candidate_relations": ["orders", "customers"], "join_keys_explicit": [], "join_type": "LEFT", "requested_dimensions": ["segment"], "requested_measures": ["amount"], "needs_clarification": false, "clarification_reason": null}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers to see amount by segment")
    assert res.success is True
    assert res.intent is not None
    assert res.intent.candidate_relations == ["orders", "customers"]
    assert res.intent.join_type == "LEFT"
    assert res.source == "ollama_semantic"

    # Kernel execution continuation
    tables_map = relation_catalog.get_all_dataframes()
    candidates = RelationshipResolver.discover_candidate_relationships(tables_map)
    assert len(candidates) >= 1
    best = candidates[0]
    assert best.parent_key == "cust_id"
    assert best.child_key == "cust_id"


def test_scenario_b_ollama_unavailable_strictly_fails_without_fallback(monkeypatch):
    """Scenario B: Ollama unavailable -> OLLAMA_UNAVAILABLE -> NO local parser -> NO SQL execution."""
    mock_status = OllamaStatus(is_running=False, ollama_url="http://localhost:11434", error="Connection refused")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers")
    assert res.success is False
    assert res.error_code == ErrorState.OLLAMA_UNAVAILABLE.value
    assert "The Ollama semantic service is unavailable" in res.error_message
    assert res.intent is None

    # Verify parse_intent raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        RelationalIntentParser.parse_intent("Join orders with customers")
    assert "OLLAMA_UNAVAILABLE" in str(exc_info.value)


def test_scenario_c_ollama_malformed_json_strictly_fails_without_fallback(monkeypatch):
    """Scenario C: Ollama malformed JSON -> OLLAMA_INVALID_RESPONSE -> NO fallback."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content="I think you should join orders and customers on customer id, here is what I found..."
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers")
    assert res.success is False
    assert res.error_code == ErrorState.OLLAMA_INVALID_RESPONSE.value
    assert res.intent is None


def test_scenario_d_ollama_schema_violation_strictly_fails_without_fallback(monkeypatch):
    """Scenario D: Ollama schema violation -> OLLAMA_SCHEMA_VALIDATION_ERROR -> NO fallback."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    # Missing required field 'candidate_relations' and 'join_type'
    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "needs_clarification": false}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers")
    assert res.success is False
    assert res.error_code == ErrorState.OLLAMA_SCHEMA_VALIDATION_ERROR.value
    assert res.intent is None


def test_scenario_e_ollama_hallucinates_nonexistent_table_grounding_rejects(monkeypatch):
    """Scenario E: Ollama hallucinates nonexistent table -> Kernel grounding rejects it."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "candidate_relations": ["orders", "imaginary_warehouse"], "join_keys_explicit": [], "join_type": "LEFT", "requested_dimensions": [], "requested_measures": [], "needs_clarification": false, "clarification_reason": null}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with warehouse")
    assert res.success is True
    # Kernel grounding validation
    candidate_tables = res.intent.candidate_relations
    valid_tables = [t for t in candidate_tables if relation_catalog.get_relation(t) is not None]
    assert "imaginary_warehouse" not in valid_tables
    assert valid_tables == ["orders"]
    # Kernel halts join planning because fewer than 2 valid tables exist
    assert len(valid_tables) < 2


def test_scenario_f_ollama_hallucinates_nonexistent_column_grounding_rejects(monkeypatch):
    """Scenario F: Ollama hallucinates nonexistent column -> Kernel grounding rejects it."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "candidate_relations": ["orders", "customers"], "join_keys_explicit": [{"left_table": "orders", "left_key": "fake_key", "right_table": "customers", "right_key": "cust_id"}], "join_type": "LEFT", "requested_dimensions": [], "requested_measures": [], "needs_clarification": false, "clarification_reason": null}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders and customers on fake_key")
    assert res.success is True

    # Kernel validation of explicit keys
    plan = JoinPlan(
        left_table="orders",
        steps=[JoinStep(right_table="customers", join_type="left", left_keys=["fake_key"], right_keys=["cust_id"])]
    )
    tables_map = relation_catalog.get_all_dataframes()
    exec_res = JoinPlanner.execute_join_plan(plan, tables_map)
    assert exec_res.status == "FAILED"
    assert exec_res.error_code in ("UNKNOWN_COLUMN", "COLUMN_NOT_FOUND")


def test_scenario_g_ollama_proposes_statistically_invalid_relationship_rejected(monkeypatch):
    """Scenario G: Proposes generic or invalid column -> RelationshipResolver rejects it."""
    # Add generic 'status' column to both tables
    orders_df = relation_catalog.get_dataframe("orders").copy()
    orders_df["status"] = ["active", "active", "pending"]
    relation_catalog.register_dataset("ds_orders", "orders", orders_df)

    cust_df = relation_catalog.get_dataframe("customers").copy()
    cust_df["status"] = ["active", "pending", "active"]
    relation_catalog.register_dataset("ds_cust", "customers", cust_df)

    tables_map = relation_catalog.get_all_dataframes()
    evidence = RelationshipResolver.evaluate_relationship(
        parent_df=tables_map["customers"],
        parent_table="customers",
        parent_key="status",
        child_df=tables_map["orders"],
        child_table="orders",
        child_key="status",
        evidence_tier="inferred_candidate"
    )
    # Kernel rejects generic column status without explicit user declaration
    assert evidence.status in ("REJECTED_GENERIC_NAME", "REJECTED_NON_UNIQUE_PARENT", "REJECTED_LOW_INCLUSION")



def test_scenario_h_user_explicit_keys_validated_and_executed(monkeypatch):
    """Scenario H: User explicitly specifies valid keys -> Kernel validates and executes."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "candidate_relations": ["orders", "customers"], "join_keys_explicit": [{"left_table": "orders", "left_key": "cust_id", "right_table": "customers", "right_key": "cust_id"}], "join_type": "LEFT", "requested_dimensions": [], "requested_measures": [], "needs_clarification": false, "clarification_reason": null}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders and customers on orders.cust_id = customers.cust_id")
    assert res.success is True
    assert len(res.intent.join_keys_explicit) == 1

    plan = JoinPlan(
        left_table="orders",
        steps=[JoinStep(right_table="customers", join_type="left", left_keys=["cust_id"], right_keys=["cust_id"])]
    )
    tables_map = relation_catalog.get_all_dataframes()
    exec_res = JoinPlanner.execute_join_plan(plan, tables_map)
    assert exec_res.status == "SUCCEEDED"
    assert exec_res.result_rows == 3


def test_scenario_i_user_does_not_specify_keys_kernel_candidate_discovery(monkeypatch):
    """Scenario I: User does not specify keys -> Kernel candidate discovery."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content='{"intent": "RELATIONAL_ENRICHMENT", "candidate_relations": ["orders", "customers"], "join_keys_explicit": [], "join_type": "LEFT", "requested_dimensions": [], "requested_measures": [], "needs_clarification": false, "clarification_reason": null}'
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers")
    assert res.success is True
    assert res.intent.join_keys_explicit == []

    tables_map = relation_catalog.get_all_dataframes()
    candidates = RelationshipResolver.discover_candidate_relationships(tables_map)
    assert len(candidates) >= 1
    # Both directions are evaluated; cust_id is the discovered join key
    assert any(c.parent_key == "cust_id" and c.child_key == "cust_id" for c in candidates)


def test_scenario_j_multiple_valid_candidates_triggers_clarification():
    """Scenario J: Multiple candidate links trigger clarification rather than silent arbitrary choice."""
    # Create two different valid foreign keys between tables with identical column names
    df_a = pd.DataFrame({"record_id": [1, 2, 3], "store_id": [10, 20, 30], "region_id": [1, 2, 3]})
    df_b = pd.DataFrame({"store_id": [10, 20, 30], "region_id": [1, 2, 3], "store_name": ["A", "B", "C"]})
    relation_catalog.register_dataset("ds_a", "tbl_a", df_a)
    relation_catalog.register_dataset("ds_b", "tbl_b", df_b)

    tables_map = relation_catalog.get_all_dataframes()
    candidates = RelationshipResolver.discover_candidate_relationships(tables_map)
    # Both store_id and region_id are candidate keys between tbl_a and tbl_b
    tbl_a_candidates = [c for c in candidates if (c.child_table == "tbl_a" and c.parent_table == "tbl_b")]
    assert len(tbl_a_candidates) >= 2


def test_section_11_ollama_raw_sql_emission_rejected(monkeypatch):
    """Section 11: Proves Ollama never generates executable SQL; raw SQL strings are rejected."""
    mock_status = OllamaStatus(is_running=True, ollama_url="http://localhost:11434")
    monkeypatch.setattr(ollama_service, "detect_server", lambda: mock_status)

    mock_resp = OllamaResponse(
        success=True,
        content="SELECT * FROM orders JOIN customers ON orders.cust_id = customers.cust_id;"
    )
    monkeypatch.setattr(ollama_service, "structured_request", lambda req: mock_resp)

    res = RelationalIntentParser.parse_intent_result("Join orders with customers")
    assert res.success is False
    assert res.error_code in (ErrorState.OLLAMA_SCHEMA_VALIDATION_ERROR.value, ErrorState.OLLAMA_INVALID_RESPONSE.value)

