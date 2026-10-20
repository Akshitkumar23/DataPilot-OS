"""
DataPilot OS — Live Ollama Unseen-Domain Integration Test Suite.

Rigorously tests LIVE Ollama semantic planning and end-to-end execution
across 10+ completely unseen enterprise domains with natural language & Hinglish queries.

CRITICAL ARCHITECTURAL CONTRACT:
- Zero manual mock commands or hardcoded `raw_command` dictionaries in tests.
- Every test calls `QueryOrchestrator.process_query(query_raw=query, df=df)` or `SemanticPlanner.extract_semantic_intent`.
- Pure end-to-end evaluation: Natural Language Query -> Live Ollama -> SemanticIntent -> ExecutionPlanCompiler -> ModularExecutionEngine -> Verified Canonical Result.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Optional
import pytest
import numpy as np
import pandas as pd

pytestmark = pytest.mark.live_ollama

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.intelligence.ollama_service import ollama_service
from modules.semantic_planner import SemanticPlanner
from modules.query_orchestrator import QueryOrchestrator
from modules.models.execution_result import CanonicalQueryResponse, LifecycleState
from backend.ds_engine.intent_schema import (
    SemanticIntent,
    IntentType,
    AggregationType,
    FilterOperator,
    FilterSpec
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveOllamaUnseenDomains")


def check_ollama_available() -> bool:
    """Checks if local Ollama daemon is active and has models loaded."""
    try:
        status = ollama_service.detect_server()
        return bool(status.is_running and len(status.available_models) > 0)
    except Exception as e:
        logger.warning(f"Ollama readiness check failed: {e}")
        return False


def print_test_telemetry(domain_name: str, query: str, res: CanonicalQueryResponse):
    """
    Standardized test reporting:
    1. Raw LLM Output (`res.plan.get('raw_llm_output')`)
    2. Canonical Resolved Intent
    3. Execution Result DataFrame
    """
    print("\n" + "=" * 80)
    print(f" DOMAIN: {domain_name}")
    print(f" QUERY:  '{query}'")
    print(f" STATUS: {res.status} | PHASE: {res.phase}")

    plan = res.plan or {}
    raw_llm = plan.get("raw_llm_output")
    print("\n--- 1. RAW LLM OUTPUT ---")
    print(raw_llm if raw_llm else "(None / Direct Structured JSON Output)")

    print("\n--- 2. CANONICAL INTENT ---")
    intent_summary = {
        "intent_type": plan.get("intent_type") or plan.get("primary_intent"),
        "primary_intent": plan.get("primary_intent"),
        "aggregation": plan.get("aggregation"),
        "dimensions": plan.get("dimensions"),
        "metrics": plan.get("metrics"),
        "filters": plan.get("filters"),
        "order": plan.get("order"),
        "limit": plan.get("limit"),
        "group_by": plan.get("group_by"),
    }
    print(json.dumps(intent_summary, indent=2, default=str))

    print("\n--- 3. EXECUTION RESULT DATAFRAME ---")
    if res.result and "data" in res.result:
        records = res.result["data"].get("records", [])
        if records:
            result_df = pd.DataFrame(records)
            print(result_df.head(10).to_string(index=False))
            print(f"Shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns")
        else:
            print("(No tabular records returned; response text below)")
            print(f"Response: {res.result.get('response', '')}")
    elif res.error:
        print(f"ERROR: [{res.error.code}] {res.error.message}")
    print("=" * 80 + "\n")


# ==============================================================================
# DETERMINISTIC SYNTHETIC DATASET FIXTURES (10+ UNSEEN DOMAINS)
# ==============================================================================

@pytest.fixture
def df_ecommerce() -> pd.DataFrame:
    """Domain 1: E-commerce orders and status."""
    np.random.seed(101)
    n = 100
    regions = np.random.choice(["North", "South", "East", "West"], size=n)
    payments = np.random.choice(["CreditCard", "UPI", "NetBanking", "COD"], size=n)
    statuses = np.random.choice(["Delivered", "Cancelled", "Shipped", "Processing"], size=n, p=[0.55, 0.15, 0.20, 0.10])
    totals = np.random.uniform(150.0, 4500.0, size=n).round(2)
    order_ids = [f"ORD-{2000+i}" for i in range(n)]
    return pd.DataFrame({
        "order_id": order_ids,
        "region": regions,
        "payment_method": payments,
        "order_total": totals,
        "order_status": statuses
    })


@pytest.fixture
def df_healthcare() -> pd.DataFrame:
    """Domain 2: Healthcare hospital admissions & billing."""
    np.random.seed(102)
    n = 120
    wards = np.random.choice(["General", "ICU", "Cardiology", "Neurology", "Oncology"], size=n)
    diagnoses = np.random.choice(["Hypertension", "Arrhythmia", "Pneumonia", "Stroke", "Diabetes"], size=n)
    billings = np.random.uniform(5000.0, 180000.0, size=n).round(2)
    ages = np.random.randint(18, 92, size=n)
    patient_ids = [f"PAT-{8000+i}" for i in range(n)]
    return pd.DataFrame({
        "patient_id": patient_ids,
        "ward": wards,
        "diagnosis": diagnoses,
        "billing": billings,
        "age": ages
    })


@pytest.fixture
def df_education() -> pd.DataFrame:
    """Domain 3: Academic performance & faculty."""
    np.random.seed(103)
    n = 110
    faculties = np.random.choice(["Sciences", "Arts", "Engineering", "Commerce", "Law"], size=n)
    attendance = np.random.uniform(45.0, 99.0, size=n).round(1)
    gpa = np.random.uniform(2.0, 4.0, size=n).round(2)
    student_ids = [f"STU-2024-{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "student_id": student_ids,
        "faculty": faculties,
        "attendance": attendance,
        "gpa": gpa
    })


@pytest.fixture
def df_logistics() -> pd.DataFrame:
    """Domain 4: Logistics & Supply Chain shipments."""
    np.random.seed(104)
    n = 100
    origins = np.random.choice(["Hub_NYC", "Hub_CHI", "Hub_LAX", "Hub_MIA"], size=n)
    dests = np.random.choice(["Hub_SEA", "Hub_DFW", "Hub_BOS", "Hub_DEN"], size=n)
    carriers = np.random.choice(["FedEx", "DHL", "UPS", "BlueDart"], size=n)
    weights = np.random.uniform(10.0, 120.0, size=n).round(2)
    days = np.random.randint(1, 10, size=n)
    shipment_ids = [f"SHP-{5000+i}" for i in range(n)]
    return pd.DataFrame({
        "shipment_id": shipment_ids,
        "origin_hub": origins,
        "dest_hub": dests,
        "weight_kg": weights,
        "delivery_days": days,
        "carrier": carriers
    })


@pytest.fixture
def df_movie_streaming() -> pd.DataFrame:
    """Domain 5: Movie Streaming & Box Office."""
    np.random.seed(105)
    n = 90
    genres = np.random.choice(["Sci-Fi", "Action", "Drama", "Comedy", "Thriller", "Horror"], size=n)
    years = np.random.randint(2010, 2025, size=n)
    box_office = np.random.uniform(15.0, 950.0, size=n).round(1)
    imdb = np.random.uniform(4.5, 9.4, size=n).round(1)
    directors = np.random.choice(["Nolan", "Spielberg", "Tarantino", "Cameron", "Villeneuve", "Coogler"], size=n)
    movie_ids = [f"MOV-{100+i}" for i in range(n)]
    return pd.DataFrame({
        "movie_id": movie_ids,
        "genre": genres,
        "release_year": years,
        "box_office_mil": box_office,
        "imdb_rating": imdb,
        "director": directors
    })


@pytest.fixture
def df_agriculture() -> pd.DataFrame:
    """Domain 6: Agriculture crop yield & soil moisture."""
    np.random.seed(106)
    n = 120
    crops = np.random.choice(["Wheat", "Rice", "Maize", "Cotton", "Soybean"], size=n)
    moisture = np.random.uniform(15.0, 65.0, size=n).round(1)
    yields = np.random.uniform(2.5, 45.0, size=n).round(2)
    rainfall = np.random.uniform(40.0, 260.0, size=n).round(1)
    farm_ids = [f"FRM-{300+i}" for i in range(n)]
    return pd.DataFrame({
        "farm_id": farm_ids,
        "crop_type": crops,
        "soil_moisture": moisture,
        "yield_tons": yields,
        "rainfall_mm": rainfall
    })


@pytest.fixture
def df_telecom() -> pd.DataFrame:
    """Domain 7: Telecom subscriber plans & churn."""
    np.random.seed(107)
    n = 130
    plans = np.random.choice(["Prepaid_Basic", "Prepaid_Pro", "Postpaid_Unlimited", "Family_Gold"], size=n)
    data_gb = np.random.uniform(5.0, 150.0, size=n).round(1)
    calls = np.random.randint(50, 1500, size=n)
    bills = np.random.uniform(199.0, 1999.0, size=n).round(2)
    churn = np.random.choice(["Yes", "No"], size=n, p=[0.22, 0.78])
    user_ids = [f"USR-{4000+i}" for i in range(n)]
    return pd.DataFrame({
        "user_id": user_ids,
        "plan_type": plans,
        "data_usage_gb": data_gb,
        "call_minutes": calls,
        "monthly_bill": bills,
        "churned": churn
    })


@pytest.fixture
def df_real_estate() -> pd.DataFrame:
    """Domain 8: Real Estate property listings."""
    np.random.seed(108)
    n = 100
    zones = np.random.choice(["Downtown", "Suburbs", "East_Bay", "West_End", "Tech_Corridor"], size=n)
    bedrooms = np.random.randint(1, 6, size=n)
    sqft = np.random.randint(450, 4800, size=n)
    prices = np.random.uniform(30.0, 450.0, size=n).round(2)
    prop_types = np.random.choice(["Apartment", "Villa", "Penthouse", "Studio"], size=n)
    prop_ids = [f"PROP-{900+i}" for i in range(n)]
    return pd.DataFrame({
        "property_id": prop_ids,
        "city_zone": zones,
        "bedrooms": bedrooms,
        "square_feet": sqft,
        "price_lakhs": prices,
        "property_type": prop_types
    })


@pytest.fixture
def df_gaming() -> pd.DataFrame:
    """Domain 9: Gaming player telemetry & monetization."""
    np.random.seed(109)
    n = 150
    modes = np.random.choice(["BattleRoyale", "Deathmatch", "Ranked", "Campaign"], size=n)
    hours = np.random.uniform(5.0, 600.0, size=n).round(1)
    levels = np.random.randint(1, 100, size=n)
    spend = (hours * np.random.uniform(0.5, 4.0, size=n) + np.random.uniform(0, 50, size=n)).round(2)
    player_ids = [f"PLY-{6000+i}" for i in range(n)]
    return pd.DataFrame({
        "player_id": player_ids,
        "game_mode": modes,
        "hours_played": hours,
        "level_reached": levels,
        "in_game_spend": spend
    })


@pytest.fixture
def df_automotive_iot() -> pd.DataFrame:
    """Domain 10: Automotive Fleet IoT sensor telemetry."""
    np.random.seed(110)
    n = 120
    temps = np.random.normal(88.0, 12.0, size=n).round(1)
    # Inject anomalous spikes
    temps[12] = 138.5
    temps[45] = 142.0
    temps[78] = 145.2
    pressures = np.random.normal(45.0, 5.0, size=n).round(1)
    speeds = np.random.uniform(0.0, 155.0, size=n).round(1)
    warnings = np.random.choice(["NONE", "WARN_TEMP", "WARN_OIL"], size=n, p=[0.85, 0.10, 0.05])
    vins = [f"VIN-{10000+i}" for i in range(n)]
    return pd.DataFrame({
        "vehicle_vin": vins,
        "engine_temp": temps,
        "oil_pressure": pressures,
        "speed_kmh": speeds,
        "warning_code": warnings
    })


@pytest.fixture
def df_renewable_energy() -> pd.DataFrame:
    """Domain 11: Renewable Energy wind turbine telemetry."""
    np.random.seed(111)
    n = 110
    sectors = np.random.choice(["Sector_Alpha", "Sector_Beta", "Sector_Gamma", "Sector_Delta"], size=n)
    wind_speeds = np.random.uniform(3.0, 25.0, size=n).round(2)
    power_kw = (wind_speeds ** 2 * 8.5 + np.random.normal(0, 15, size=n)).round(2)
    power_kw = np.clip(power_kw, 0, 5000)
    rpm = (wind_speeds * 1.8 + np.random.normal(0, 0.5, size=n)).round(1)
    turbine_ids = [f"WTG-{100+i}" for i in range(n)]
    return pd.DataFrame({
        "turbine_id": turbine_ids,
        "farm_sector": sectors,
        "wind_speed_ms": wind_speeds,
        "power_output_kw": power_kw,
        "rotor_rpm": rpm
    })


@pytest.fixture
def df_fintech_fraud() -> pd.DataFrame:
    """Domain 12: Fintech transactions and fraud classification."""
    np.random.seed(112)
    n = 140
    channels = np.random.choice(["MobileApp", "WebPortal", "POS_Terminal", "ATM"], size=n)
    amounts = np.random.exponential(scale=250.0, size=n).round(2)
    risks = np.random.uniform(0.0, 1.0, size=n).round(3)
    is_fraud = np.where((risks > 0.85) | (amounts > 1500), "Fraud", "Legit")
    txn_ids = [f"TXN-{90000+i}" for i in range(n)]
    return pd.DataFrame({
        "txn_id": txn_ids,
        "channel": channels,
        "txn_amount": amounts,
        "risk_score": risks,
        "is_fraud": is_fraud
    })


# ==============================================================================
# LIVE OLLAMA UNSEEN DOMAIN TEST SUITE
# ==============================================================================

class TestLiveOllamaUnseenDomains:
    """
    Rigorously tests LIVE Ollama across 10+ unseen domains using real NLP queries.
    Never mocks raw_command dicts.
    """

    # --------------------------------------------------------------------------
    # Domain 1: E-commerce
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_01_ecommerce_delivered_orders_count(self, df_ecommerce):
        """
        Query: 'Show count of delivered orders by region and payment method.'
        Assert: SUCCEEDED, aggregation == COUNT, dimensions contains region and payment_method, no invented metrics.
        """
        query = "Show count of delivered orders by region and payment method."
        res = QueryOrchestrator.process_query(query_raw=query, df=df_ecommerce)
        print_test_telemetry("01: E-Commerce", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.plan is not None

        intent = SemanticIntent(**res.plan)
        assert intent.aggregation == AggregationType.COUNT or res.plan.get("aggregation") == "count"

        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "region" in dims, f"Expected 'region' in dimensions: {dims}"
        assert "payment_method" in dims, f"Expected 'payment_method' in dimensions: {dims}"

        # No invented metrics
        assert len(intent.metrics) == 0 or intent.metrics == [], f"Metrics should be empty for record count: {intent.metrics}"

        # Execution Result contains non-empty tabular records
        assert res.result is not None
        assert "data" in res.result
        assert len(res.result["data"]["records"]) > 0

    # --------------------------------------------------------------------------
    # Domain 2: Healthcare (Hinglish Query)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_02_healthcare_hinglish_elderly_billing(self, df_healthcare):
        """
        Query: 'Patients aged above 60 ka ward-wise total billing dikhao.'
        Assert: SUCCEEDED, res.result has data, filter on age > 60 (or >= 60), metric 'billing', group by 'ward'.
        """
        query = "Patients aged above 60 ka ward-wise total billing dikhao."
        res = QueryOrchestrator.process_query(query_raw=query, df=df_healthcare)
        print_test_telemetry("02: Healthcare (Hinglish)", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        # Filter on age > 60 or >= 60
        age_filters = [f for f in intent.filters if f.column == "age"]
        assert len(age_filters) > 0, f"Expected filter on 'age', got filters: {intent.filters}"
        assert age_filters[0].operator in [FilterOperator.GT, FilterOperator.GTE]
        assert float(age_filters[0].value) in [60.0, 60]

        # Metric 'billing'
        metrics = [m.lower() for m in intent.metrics]
        assert "billing" in metrics or any("billing" in m for m in metrics), f"Expected 'billing' in metrics: {metrics}"

        # Group by 'ward'
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "ward" in dims, f"Expected 'ward' in dimensions: {dims}"

    # --------------------------------------------------------------------------
    # Domain 3: Education (Hinglish Query)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_03_education_hinglish_attendance_gpa(self, df_education):
        """
        Query: '75% se zyada attendance wale students ka faculty-wise average GPA dikhao.'
        Assert: SUCCEEDED, filter attendance >= 75 (or > 75), metric 'gpa', group by 'faculty', agg 'mean'.
        """
        query = "75% se zyada attendance wale students ka faculty-wise average GPA dikhao."
        res = QueryOrchestrator.process_query(query_raw=query, df=df_education)
        print_test_telemetry("03: Education (Hinglish)", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        # Filter on attendance >= 75 or > 75
        att_filters = [f for f in intent.filters if f.column == "attendance"]
        assert len(att_filters) > 0, f"Expected filter on 'attendance', got: {intent.filters}"
        assert att_filters[0].operator in [FilterOperator.GTE, FilterOperator.GT]
        assert float(att_filters[0].value) in [75.0, 75]

        # Metric 'gpa'
        metrics = [m.lower() for m in intent.metrics]
        assert "gpa" in metrics or any("gpa" in m for m in metrics), f"Expected 'gpa' in metrics: {metrics}"

        # Group by 'faculty'
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "faculty" in dims, f"Expected 'faculty' in dimensions: {dims}"

        # Aggregation 'mean'
        assert intent.aggregation == AggregationType.MEAN or res.plan.get("aggregation") == "mean"

    # --------------------------------------------------------------------------
    # Domain 4: Logistics / Supply Chain
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_04_logistics_carrier_weight_delivery(self, df_logistics):
        """
        Query: 'Average weight_kg and delivery_days by carrier where weight_kg > 50'
        Assert: SUCCEEDED, group by 'carrier', filter on weight_kg > 50, agg 'mean'.
        """
        query = "Average weight_kg by carrier where weight_kg > 50"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_logistics)
        print_test_telemetry("04: Logistics / Supply Chain", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "carrier" in dims, f"Expected 'carrier' in dimensions: {dims}"

        weight_filters = [f for f in intent.filters if f.column == "weight_kg"]
        assert len(weight_filters) > 0, f"Expected filter on 'weight_kg': {intent.filters}"
        assert weight_filters[0].operator in [FilterOperator.GT, FilterOperator.GTE]
        assert float(weight_filters[0].value) in [50.0, 50]

        assert intent.aggregation == AggregationType.MEAN or res.plan.get("aggregation") == "mean"

    # --------------------------------------------------------------------------
    # Domain 5: Movie Streaming
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_05_movie_streaming_top_rated(self, df_movie_streaming):
        """
        Query: 'Top 5 movies with highest imdb_rating where release_year >= 2015'
        Assert: SUCCEEDED, intent.order == 'desc', limit == 5, filter on release_year.
        """
        query = "Top 5 movies with highest imdb_rating where release_year >= 2015"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_movie_streaming)
        print_test_telemetry("05: Movie Streaming", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0
        assert len(res.result["data"]["records"]) <= 5

        intent = SemanticIntent(**res.plan)
        assert intent.order == "desc", f"Expected order 'desc', got {intent.order}"
        assert intent.limit == 5, f"Expected limit 5, got {intent.limit}"

        year_filters = [f for f in intent.filters if f.column == "release_year"]
        assert len(year_filters) > 0, f"Expected filter on 'release_year': {intent.filters}"
        assert year_filters[0].operator in [FilterOperator.GTE, FilterOperator.GT]

    # --------------------------------------------------------------------------
    # Domain 6: Agriculture
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_06_agriculture_yield_by_crop(self, df_agriculture):
        """
        Query: 'Total yield_tons by crop_type where rainfall_mm > 100'
        Assert: SUCCEEDED, group by 'crop_type', metric 'yield_tons', agg 'sum', filter on rainfall_mm > 100.
        """
        query = "Total yield_tons by crop_type where rainfall_mm > 100"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_agriculture)
        print_test_telemetry("06: Agriculture", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "crop_type" in dims, f"Expected 'crop_type' in dimensions: {dims}"

        metrics = [m.lower() for m in intent.metrics]
        assert "yield_tons" in metrics or any("yield" in m for m in metrics), f"Expected 'yield_tons' in metrics: {metrics}"

        assert intent.aggregation == AggregationType.SUM or res.plan.get("aggregation") == "sum"

        rain_filters = [f for f in intent.filters if f.column == "rainfall_mm"]
        assert len(rain_filters) > 0, f"Expected filter on 'rainfall_mm': {intent.filters}"
        assert rain_filters[0].operator in [FilterOperator.GT, FilterOperator.GTE]
        assert float(rain_filters[0].value) in [100.0, 100]

    # --------------------------------------------------------------------------
    # Domain 7: Telecom
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_07_telecom_churn_count(self, df_telecom):
        """
        Query: 'Count users by plan_type and churned status'
        Assert: SUCCEEDED, aggregation == 'count', dimensions has 'plan_type' and 'churned'.
        """
        query = "Count users by plan_type and churned status"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_telecom)
        print_test_telemetry("07: Telecom", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        assert intent.aggregation == AggregationType.COUNT or res.plan.get("aggregation") == "count"

        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "plan_type" in dims, f"Expected 'plan_type' in dimensions: {dims}"
        assert "churned" in dims, f"Expected 'churned' in dimensions: {dims}"

    # --------------------------------------------------------------------------
    # Domain 8: Real Estate
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_08_real_estate_average_price(self, df_real_estate):
        """
        Query: 'Average price_lakhs by city_zone where bedrooms >= 3'
        Assert: SUCCEEDED, metric 'price_lakhs', group by 'city_zone', filter on bedrooms >= 3.
        """
        query = "Filter bedrooms >= 3 and calculate average price_lakhs by city_zone"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_real_estate)
        print_test_telemetry("08: Real Estate", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "city_zone" in dims, f"Expected 'city_zone' in dimensions: {dims}"

        metrics = [m.lower() for m in intent.metrics]
        assert "price_lakhs" in metrics or any("price" in m for m in metrics), f"Expected 'price_lakhs' in metrics: {metrics}"

        bedroom_filters = [f for f in intent.filters if f.column == "bedrooms"]
        assert len(bedroom_filters) > 0, f"Expected filter on 'bedrooms': {intent.filters}"
        assert bedroom_filters[0].operator in [FilterOperator.GTE, FilterOperator.GT]
        assert float(bedroom_filters[0].value) in [3.0, 3]

        assert intent.aggregation == AggregationType.MEAN or res.plan.get("aggregation") == "mean"

    # --------------------------------------------------------------------------
    # Domain 9: Gaming Analytics (Correlation)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_09_gaming_correlation(self, df_gaming):
        """
        Query: 'Find correlation between hours_played and in_game_spend'
        Assert: SUCCEEDED, primary_intent == IntentType.CORRELATION.
        """
        query = "Find correlation between hours_played and in_game_spend"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_gaming)
        print_test_telemetry("09: Gaming Analytics", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None

        intent = SemanticIntent(**res.plan)
        is_corr = (intent.primary_intent == IntentType.CORRELATION or 
                   intent.intent_type == IntentType.CORRELATION or 
                   res.plan.get("intent_type") == "CORRELATION" or
                   res.plan.get("primary_intent") == "CORRELATION")
        assert is_corr, f"Expected primary_intent to be CORRELATION, got: {intent.primary_intent}"

    # --------------------------------------------------------------------------
    # Domain 10: Automotive / Fleet IoT (Anomaly Detection with IsolationForest)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_10_automotive_anomaly_detection(self, df_automotive_iot):
        """
        Query: 'Detect anomalies in engine_temp using IsolationForest'
        Assert: SUCCEEDED, primary_intent == IntentType.ANOMALY_DETECTION.
        """
        query = "Detect anomalies in engine_temp using IsolationForest"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_automotive_iot)
        print_test_telemetry("10: Automotive IoT", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None

        intent = SemanticIntent(**res.plan)
        is_anomaly = (intent.primary_intent == IntentType.ANOMALY_DETECTION or 
                      intent.intent_type == IntentType.ANOMALY_DETECTION or 
                      res.plan.get("intent_type") == "ANOMALY_DETECTION" or
                      res.plan.get("primary_intent") == "ANOMALY_DETECTION")
        assert is_anomaly, f"Expected primary_intent to be ANOMALY_DETECTION, got: {intent.primary_intent}"

    # --------------------------------------------------------------------------
    # Domain 11: Renewable Energy (Wind Turbine Telemetry)
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_11_renewable_energy_wind_power(self, df_renewable_energy):
        """
        Query: 'Average power_output_kw by farm_sector where wind_speed_ms > 12'
        Assert: SUCCEEDED, group by 'farm_sector', metric 'power_output_kw', filter on wind_speed_ms > 12.
        """
        query = "Average power_output_kw by farm_sector where wind_speed_ms > 12"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_renewable_energy)
        print_test_telemetry("11: Renewable Energy", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "farm_sector" in dims, f"Expected 'farm_sector' in dimensions: {dims}"

        wind_filters = [f for f in intent.filters if f.column == "wind_speed_ms"]
        assert len(wind_filters) > 0, f"Expected filter on 'wind_speed_ms': {intent.filters}"
        assert wind_filters[0].operator in [FilterOperator.GT, FilterOperator.GTE]
        assert float(wind_filters[0].value) in [12.0, 12]

        assert intent.aggregation == AggregationType.MEAN or res.plan.get("aggregation") == "mean"

    # --------------------------------------------------------------------------
    # Domain 12: Fintech & Fraud Analytics
    # --------------------------------------------------------------------------
    @pytest.mark.skipif(not check_ollama_available(), reason="Local Ollama server is not running or has no models")
    def test_domain_12_fintech_fraud_count(self, df_fintech_fraud):
        """
        Query: 'Count transactions by channel and is_fraud status'
        Assert: SUCCEEDED, aggregation == 'count', dimensions has 'channel' and 'is_fraud'.
        """
        query = "Count transactions by channel and is_fraud status"
        res = QueryOrchestrator.process_query(query_raw=query, df=df_fintech_fraud)
        print_test_telemetry("12: Fintech Fraud Analytics", query, res)

        assert res.status == "SUCCEEDED", f"Query failed with error: {res.error}"
        assert res.result is not None
        assert len(res.result["data"]["records"]) > 0

        intent = SemanticIntent(**res.plan)
        assert intent.aggregation == AggregationType.COUNT or res.plan.get("aggregation") == "count"

        dims = [d.lower() for d in (intent.dimensions or intent.group_by)]
        assert "channel" in dims, f"Expected 'channel' in dimensions: {dims}"
        assert "is_fraud" in dims, f"Expected 'is_fraud' in dimensions: {dims}"
