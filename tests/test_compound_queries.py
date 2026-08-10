import json
import pytest
import pandas as pd
from modules.capability_registry import capability_registry, split_multi_intent_query, score_query, validate_and_parse_ollama_json, ollama_semantic_decompose

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Order_ID": [f"ORD{i}" for i in range(100)],
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 10,
        "Income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000] * 10,
        "Gross_Sales": [100, 200, 150, 300, 250, 400, 350, 500, 450, 600] * 10,
        "Profit": [10, 25, 20, 40, 35, 60, 50, 80, 70, 90] * 10,
        "Category": ["Electronics", "Fashion", "Home", "Sports", "Beauty"] * 20,
        "Courier_Partner": ["DHL", "FedEx", "BlueDart", "Delhivery"] * 25
    })


class TestCompoundQuerySplitting:
    """Tests multi-intent query splitting rules and negative case safeguards."""

    def test_positive_compound_splitting(self):
        query = "correlation heatmap dikhao fir top 10 categories"
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert parts[0] == "correlation heatmap dikhao"
        assert parts[1] == "top 10 categories"

    def test_positive_and_then_splitting(self):
        query = "show me the distribution of Sales and then the correlation heatmap"
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "distribution" in parts[0]
        assert "heatmap" in parts[1]

    def test_positive_then_splitting(self):
        query = "clean missing values then show outlier boxplot for Price"
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "clean" in parts[0]
        assert "boxplot" in parts[1]

    def test_positive_aur_phir_splitting(self):
        query = "scatter plot Sales vs Profit aur phir feature importance"
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "scatter" in parts[0]
        assert "feature importance" in parts[1]

    def test_positive_complex_once_cleared(self):
        query = "I want to understand how sales relate to profit. Once that's clear, tell me which categories are doing the best."
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "sales relate to profit" in parts[0]
        assert "categories" in parts[1]

    def test_positive_complex_follow_up(self):
        query = "Start by comparing sales with profit, and follow that up with a category performance analysis."
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "sales" in parts[0]
        assert "category" in parts[1]

    def test_positive_complex_pehle_phir(self):
        query = "Pehle sales aur profit ke beech relation samjhao, phir categories mein sabse achha kaun perform kar raha hai wo dikhao."
        parts = split_multi_intent_query(query)
        assert len(parts) == 2
        assert "relation" in parts[0]
        assert "categories" in parts[1]

    def test_negative_scatter_and(self):
        # "and" joins two column names in one scatter plot request
        query = "show sales and profit scatter plot"
        parts = split_multi_intent_query(query)
        assert len(parts) == 1
        assert parts[0] == query

    def test_negative_correlation_and(self):
        # "and" joins two variables of one correlation request
        query = "correlation between Age and Income"
        parts = split_multi_intent_query(query)
        assert len(parts) == 1
        assert parts[0] == query

    def test_negative_impute_and(self):
        # "and" joins two column names in one cleaning operation
        query = "impute missing values in Age and Salary columns"
        parts = split_multi_intent_query(query)
        assert len(parts) == 1
        assert parts[0] == query


class TestCompoundQueryDispatching:
    """Tests execution and response formatting of compound queries."""

    def test_compound_heatmap_and_top(self, sample_df):
        query = "correlation heatmap dikhao fir top 10 categories"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert "multi_results" in res
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_heatmap"
        assert results[1]["intent"] == "sig_top"

    def test_compound_dist_and_heatmap(self, sample_df):
        query = "show me the distribution of Sales and then the correlation heatmap"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_hist"
        assert results[1]["intent"] == "sig_heatmap"

    def test_compound_clean_and_box(self, sample_df):
        query = "clean missing values then show outlier boxplot for Price"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_clean"
        assert results[1]["intent"] == "sig_box"

    def test_compound_scatter_and_ml(self, sample_df):
        query = "scatter plot Sales vs Profit aur phir feature importance"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_scatter"
        assert results[1]["intent"] == "sig_ml"

    def test_compound_top_and_pie(self, sample_df):
        query = "top 10 courier partners then payment method share"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_top"
        assert results[1]["intent"] == "sig_pie"

    def test_compound_forecast_and_cluster(self, sample_df):
        query = "forecast next 30 days aur clustering bhi karo"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_forecast"
        assert results[1]["intent"] == "sig_cluster"

    def test_compound_encode_and_scale(self, sample_df):
        query = "encode categorical columns then scale the features"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_encode"
        assert results[1]["intent"] == "sig_scale"

    def test_compound_eda_and_pca(self, sample_df):
        query = "EDA dashboard banao and show me PCA projection"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_eda_dashboard"
        assert results[1]["intent"] == "sig_pca"

    def test_compound_complex_once_cleared_dispatch(self, sample_df):
        query = "I want to understand how sales relate to profit. Once that's clear, tell me which categories are doing the best."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_scatter"
        assert results[1]["intent"] == "sig_top"

    def test_compound_complex_follow_up_dispatch(self, sample_df):
        query = "Start by comparing sales with profit, and follow that up with a category performance analysis."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        results = res["multi_results"]
        assert len(results) == 2
        assert results[0]["intent"] == "sig_scatter"
        assert results[1]["intent"] in ("sig_top", "sig_eda_dashboard")

    def test_unseen_query_1_compare_then_tell(self, sample_df):
        query = "Compare revenue and profit, then tell me which products lead."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_unseen_query_2_before_we_look(self, sample_df):
        query = "Before we look at product performance, examine revenue against profit."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_unseen_query_3_along_with(self, sample_df):
        query = "Revenue versus profit would be useful, along with the strongest categories."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_unseen_query_4_investigate_and_afterwards(self, sample_df):
        query = "Can you first investigate sales and profit and afterwards rank the categories?"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_unseen_query_5_once_we_have_that(self, sample_df):
        query = "Show me the relationship between sales and profit. Once we have that, I'd like the top categories too."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_unseen_query_6_plus(self, sample_df):
        query = "Sales and profit relationship plus the best performing categories."
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is True
        assert len(res["multi_results"]) == 2

    def test_single_intent_no_regression(self, sample_df):
        query = "show sales and profit scatter plot"
        res = capability_registry.route_and_dispatch(query_raw=query, df=sample_df)
        assert res.get("is_compound") is not True
        assert res["intent"] == "sig_scatter"


class TestOllamaSemanticValidationAndFallback:
    """Tests for Ollama JSON schema validation, malformed output rejection, and offline fallback."""

    def test_ollama_validation_rejection_malformed_json(self):
        # 1. Plain non-JSON string
        assert validate_and_parse_ollama_json("this is not json") is None
        
        # 2. Not a JSON object (list/number)
        assert validate_and_parse_ollama_json("[1, 2, 3]") is None
        
        # 3. is_compound is false (when require_compound=True)
        assert validate_and_parse_ollama_json('{"is_compound": false, "sub_queries": ["a", "b"]}', require_compound=True) is None
        
        # 4. sub_queries is not a list or has < 2 items (when require_compound=True)
        assert validate_and_parse_ollama_json('{"is_compound": true, "sub_queries": "only_one"}', require_compound=True) is None
        assert validate_and_parse_ollama_json('{"is_compound": true, "sub_queries": ["only_one"]}', require_compound=True) is None
        
        # 5. Rejected generic LLM fillers ("analyze everything", "do necessary work")
        assert validate_and_parse_ollama_json('{"is_compound": true, "sub_queries": ["analyze everything", "do necessary work"]}') is None

    def test_ollama_validation_acceptance_valid_json(self):
        valid_payload = json.dumps({
            "is_compound": True,
            "sub_queries": ["scatter plot of sales vs profit", "top 10 categories bar chart"]
        })
        res = validate_and_parse_ollama_json(valid_payload)
        assert len(res) == 2
        assert res[0]["sub_query"] == "scatter plot of sales vs profit"
        assert res[1]["sub_query"] == "top 10 categories bar chart"

    def test_ollama_offline_fallback(self, monkeypatch):
        # Point to unreachable port to simulate offline Ollama connection
        monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:59999/api/chat")
        res = ollama_semantic_decompose("Complex sentence query test")
        assert res is None  # Graceful fallback to deterministic parser
