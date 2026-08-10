import pytest
import os
import pandas as pd
import numpy as np

from ds_knowledge_engine import ds_engine, DataScienceKnowledgeEngine, compute_knowledge_hash
from modules.capability_registry import capability_registry, score_query, ROUTE_INTENT_LABELS

# Comprehensive 180 Test Cases covering all 18 Intent Routes (10 queries per route)
ROUTING_TEST_CASES = [
    # ── 1. Automated EDA Dashboard (sig_eda_dashboard) ──
    ("generate full eda report", "sig_eda_dashboard"),
    ("run automated eda dashboard", "sig_eda_dashboard"),
    ("show eda summary", "sig_eda_dashboard"),
    ("build eda dashboard for dataset", "sig_eda_dashboard"),
    ("complete analysis dashboard", "sig_eda_dashboard"),
    ("full automated dashboard", "sig_eda_dashboard"),
    ("give me a full eda report", "sig_eda_dashboard"),
    ("create automated eda analysis", "sig_eda_dashboard"),
    ("generate complete eda summary", "sig_eda_dashboard"),
    ("show full eda dashboard", "sig_eda_dashboard"),

    # ── 2. Time Series Forecast (sig_forecast) ──
    ("forecast sales for next 30 days", "sig_forecast"),
    ("predict future revenue trend", "sig_forecast"),
    ("time series forecast for next month", "sig_forecast"),
    ("predict sales for next 60 days", "sig_forecast"),
    ("run arima forecast model", "sig_forecast"),
    ("prophet forecast for future trend", "sig_forecast"),
    ("agle 30 din ka forecast dikhao", "sig_forecast"),
    ("predict next quarter sales", "sig_forecast"),
    ("holt winters forecast plot", "sig_forecast"),
    ("future sales trend prediction", "sig_forecast"),

    # ── 3. Machine Learning Model (sig_ml) ──
    ("train machine learning model", "sig_ml"),
    ("fit random forest classifier", "sig_ml"),
    ("predict target variable using ml", "sig_ml"),
    ("run automl model training", "sig_ml"),
    ("evaluate feature importances using random forest", "sig_ml"),
    ("train classification model on target", "sig_ml"),
    ("suggest engineered features for target", "sig_ml"),
    ("train regression model for profit", "sig_ml"),
    ("xgboost model comparison", "sig_ml"),
    ("build predictive ml model", "sig_ml"),

    # ── 4. Data Cleaning & Imputation (sig_clean) ──
    ("clean missing null values", "sig_clean"),
    ("impute nulls with median", "sig_clean"),
    ("fill missing values in numerical column", "sig_clean"),
    ("remove duplicate rows", "sig_clean"),
    ("run data quality audit", "sig_clean"),
    ("sanitize dataset inconsistencies", "sig_clean"),
    ("impute missing text categories with mode", "sig_clean"),
    ("clean up dataset null entries", "sig_clean"),
    ("fill missing null values using mean", "sig_clean"),
    ("handle null values and duplicates", "sig_clean"),

    # ── 5. SQL Query Engine (sig_sql) ──
    ("run window sql query", "sig_sql"),
    ("calculate rolling average sql", "sig_sql"),
    ("execute sql query select from table", "sig_sql"),
    ("sql row_number window calculation", "sig_sql"),
    ("group by sql query aggregation", "sig_sql"),
    ("lead lag sql window function", "sig_sql"),
    ("run sql join query", "sig_sql"),
    ("advanced sql query on table", "sig_sql"),
    ("sql window function ranking", "sig_sql"),
    ("select top records sql query", "sig_sql"),

    # ── 6. Data Profiling Report (sig_describe) ──
    ("generate full data profiling report", "sig_describe"),
    ("show dataset summary statistics", "sig_describe"),
    ("describe dataset shape and dtypes", "sig_describe"),
    ("print column statistics overview", "sig_describe"),
    ("dataset profiling summary", "sig_describe"),
    ("show statistical summary of columns", "sig_describe"),
    ("generate data profiling overview", "sig_describe"),
    ("describe numerical statistics", "sig_describe"),
    ("full report on dataset shape and nulls", "sig_describe"),
    ("show statistical profile", "sig_describe"),

    # ── 7. Distribution Histogram (sig_hist) ──
    ("plot distribution histogram", "sig_hist"),
    ("show price histogram with kde", "sig_hist"),
    ("draw bell curve distribution plot", "sig_hist"),
    ("plot frequency distribution of age", "sig_hist"),
    ("histplot of revenue feature", "sig_hist"),
    ("kde density plot for salary", "sig_hist"),
    ("plot feature frequency distribution", "sig_hist"),
    ("show histogram of continuous column", "sig_hist"),
    ("check bell curve distribution", "sig_hist"),
    ("draw histogram plot for column", "sig_hist"),

    # ── 8. Correlation Heatmap (sig_heatmap) ──
    ("show correlation heatmap matrix", "sig_heatmap"),
    ("plot seaborn feature correlation heatmap", "sig_heatmap"),
    ("calculate feature correlation matrix", "sig_heatmap"),
    ("check correlation between numeric features", "sig_heatmap"),
    ("generate correlation heat map", "sig_heatmap"),
    ("check vif multicollinearity heatmap", "sig_heatmap"),
    ("plot correlation matrix coolwarm", "sig_heatmap"),
    ("show correlation between all features", "sig_heatmap"),
    ("seaborn heatmap correlation plot", "sig_heatmap"),
    ("multicollinearity heatmap check", "sig_heatmap"),

    # ── 9. Outlier Boxplot (sig_box) ──
    ("plot outlier boxplot", "sig_box"),
    ("detect iqr outliers with box plot", "sig_box"),
    ("show boxplot outlier visualization", "sig_box"),
    ("detect extreme values outside iqr", "sig_box"),
    ("sns boxplot for salary feature", "sig_box"),
    ("five number summary boxplot", "sig_box"),
    ("outlier detection boxplot chart", "sig_box"),
    ("draw boxplot for numerical feature", "sig_box"),
    ("clip iqr outliers boxplot", "sig_box"),
    ("show whisker box plot", "sig_box"),

    # ── 10. Scatter Plot (sig_scatter) ──
    ("plot scatter plot of sales vs profit", "sig_scatter"),
    ("show bivariate scatter plot", "sig_scatter"),
    ("scatter plot of price versus quantity", "sig_scatter"),
    ("sns scatterplot feature x vs feature y", "sig_scatter"),
    ("bivariate relationship scatter plot", "sig_scatter"),
    ("draw scatter chart for two variables", "sig_scatter"),
    ("scatter plot revenue versus cost", "sig_scatter"),
    ("show relationship between x and y", "sig_scatter"),
    ("plot scatter points", "sig_scatter"),
    ("bivariate correlation scatter plot", "sig_scatter"),

    # ── 11. Top-N Ranking Chart (sig_top) ──
    ("show top 10 categories by sales", "sig_top"),
    ("plot top 5 regions by profit", "sig_top"),
    ("rank top 10 products", "sig_top"),
    ("show worst 5 performing products", "sig_top"),
    ("bar chart of highest revenue items", "sig_top"),
    ("count frequency of categories", "sig_top"),
    ("top 20 customers by total spend", "sig_top"),
    ("highest ranking items bar plot", "sig_top"),
    ("best 10 performing categories", "sig_top"),
    ("show top 10 list ranking chart", "sig_top"),

    # ── 12. One-Hot Encoding (sig_encode) ──
    ("apply one hot encoding to categorical feature", "sig_encode"),
    ("convert text column into dummy variables", "sig_encode"),
    ("onehot encode category column", "sig_encode"),
    ("pandas get_dummies encoding", "sig_encode"),
    ("categorical dummy variable encoding", "sig_encode"),
    ("encode string category column", "sig_encode"),
    ("create dummy binary columns", "sig_encode"),
    ("label encoding for text features", "sig_encode"),
    ("one-hot encode gender category", "sig_encode"),
    ("convert categories to dummy columns", "sig_encode"),

    # ── 13. Feature Scaling (sig_scale) ──
    ("apply standardscaler feature scaling", "sig_scale"),
    ("normalize numeric features zscore", "sig_scale"),
    ("minmax scaler feature normalization", "sig_scale"),
    ("scale continuous numeric features", "sig_scale"),
    ("standardize columns mean 0 std 1", "sig_scale"),
    ("z-score feature scaling normalization", "sig_scale"),
    ("scale features before training model", "sig_scale"),
    ("apply standard scaling to numeric data", "sig_scale"),
    ("minmax scale features", "sig_scale"),
    ("normalize column values to standard scale", "sig_scale"),

    # ── 14. Log Transformation (sig_skew) ──
    ("apply log transform to reduce skewness", "sig_skew"),
    ("yeo-johnson power transformation", "sig_skew"),
    ("log1p transform skewed feature", "sig_skew"),
    ("reduce non-normal distribution skewness", "sig_skew"),
    ("apply log transformation to skewed sales", "sig_skew"),
    ("log transform continuous skewed column", "sig_skew"),
    ("yeo johnson skewness reduction", "sig_skew"),
    ("normalize skewed distribution with log", "sig_skew"),
    ("power transform skewed feature", "sig_skew"),
    ("log transform revenue distribution", "sig_skew"),

    # ── 15. Pie Chart (sig_pie) ──
    ("plot pie chart of market share", "sig_pie"),
    ("show proportion donut pie chart", "sig_pie"),
    ("pie chart breakdown of category percentage", "sig_pie"),
    ("draw pie chart for product share", "sig_pie"),
    ("percentage share pie chart plot", "sig_pie"),
    ("composition breakdown pie chart", "sig_pie"),
    ("show share percentage pie plot", "sig_pie"),
    ("visualize pie chart distribution", "sig_pie"),
    ("pie chart of region sales share", "sig_pie"),
    ("proportion share pie chart", "sig_pie"),

    # ── 16. K-Means Clustering (sig_cluster) ──
    ("run k-means clustering customer segmentation", "sig_cluster"),
    ("plot kmeans elbow curve", "sig_cluster"),
    ("unsupervised clustering segmentation", "sig_cluster"),
    ("find optimal k clusters elbow method", "sig_cluster"),
    ("cluster observations with kmeans", "sig_cluster"),
    ("run customer segmentation clustering", "sig_cluster"),
    ("kmeans inertia elbow plot", "sig_cluster"),
    ("partition data into k clusters", "sig_cluster"),
    ("kmeans customer segmentation", "sig_cluster"),
    ("elbow curve for clustering", "sig_cluster"),

    # ── 17. PCA Dimensionality Reduction (sig_pca) ──
    ("run pca dimensionality reduction", "sig_pca"),
    ("compress features into 2d pca projection", "sig_pca"),
    ("principal component analysis pca plot", "sig_pca"),
    ("reduce high dimensions with pca", "sig_pca"),
    ("pca 2d projection scatter plot", "sig_pca"),
    ("fit pca 2 principal components", "sig_pca"),
    ("feature compression using pca", "sig_pca"),
    ("principal component analysis reduction", "sig_pca"),
    ("plot 2d pca components", "sig_pca"),
    ("dimensionality reduction via pca", "sig_pca"),

    # ── 18. Tabular Listing (sig_list) ──
    ("display tabular list preview", "sig_list"),
    ("show head preview of table rows", "sig_list"),
    ("preview list of 20 data records table", "sig_list"),
    ("sample preview of dataset table", "sig_list"),
    ("display table rows list", "sig_list"),
    ("show tabular preview of records", "sig_list"),
    ("list sample data rows", "sig_list"),
    ("head preview table listing", "sig_list"),
    ("preview tabular data records", "sig_list"),
    ("list first 25 dataset rows", "sig_list")
]


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "Category": np.random.choice(["Electronics", "Clothing", "Furniture", "Books"], size=100),
        "Sales": np.random.uniform(50, 5000, size=100),
        "Profit": np.random.uniform(-100, 1000, size=100),
        "Date": pd.date_range("2026-01-01", periods=100, freq="D"),
        "Region": np.random.choice(["North", "South", "East", "West"], size=100)
    })


def test_180_routing_regression_cases():
    """Verify that all 180 natural language queries route to their expected intent."""
    passed_count = 0
    total_count = len(ROUTING_TEST_CASES)

    for query, expected_intent in ROUTING_TEST_CASES:
        res = score_query(
            query,
            df=None,
            clf_model=getattr(ds_engine, 'clf_model', None),
            vectorizer=getattr(ds_engine, 'vectorizer', None),
            intent_parser=getattr(ds_engine, 'intent_parser', None)
        )
        top_intent = res["top_intent"]
        confidence = res["confidence"]

        assert top_intent == expected_intent, (
            f"Query '{query}' failed routing: expected '{expected_intent}', got '{top_intent}' (confidence: {confidence})"
        )
        assert 0.0 <= confidence <= 1.0, f"Confidence score {confidence} out of bounds for query '{query}'"
        passed_count += 1

    assert passed_count == total_count == 180


def test_alternative_intents_on_low_confidence_or_close_delta():
    """Verify that alternative_intents is populated when confidence < 0.70 or score delta < 0.15."""
    ambiguous_query = "show data metrics relationship correlation scatter"
    res = score_query(
        ambiguous_query,
        clf_model=getattr(ds_engine, 'clf_model', None),
        vectorizer=getattr(ds_engine, 'vectorizer', None),
        intent_parser=getattr(ds_engine, 'intent_parser', None)
    )

    confidence = res["confidence"]
    delta = res["score_delta"]
    alt_intents = res["alternative_intents"]

    if confidence < 0.70 or delta < 0.15:
        assert alt_intents is not None, "alternative_intents must be included for low confidence or close delta"
        assert len(alt_intents) >= 2, "alternative_intents must contain at least top 2 candidates"
        for item in alt_intents:
            assert "intent" in item and "label" in item


def test_capability_registry_dispatch(sample_df):
    """Verify that capability_registry.route_and_dispatch executes successfully for all 18 intent routes."""
    for intent_name in ROUTE_INTENT_LABELS.keys():
        test_query = f"test query for {intent_name} {ROUTE_INTENT_LABELS[intent_name]}"
        res = capability_registry.route_and_dispatch(
            query_raw=test_query,
            df=sample_df,
            clf_model=getattr(ds_engine, 'clf_model', None),
            vectorizer=getattr(ds_engine, 'vectorizer', None),
            intent_parser=getattr(ds_engine, 'intent_parser', None)
        )

        assert res.get("success") is True
        assert "intent" in res
        assert "confidence" in res
        assert "code" in res
        assert "response" in res


def test_ds_knowledge_engine_classifier_caching():
    """Verify that DataScienceKnowledgeEngine classifier caching loads from disk without retraining."""
    engine = DataScienceKnowledgeEngine()
    assert engine.clf_model is not None
    assert engine.vectorizer is not None
    
    current_hash = compute_knowledge_hash(engine.db_path)
    with open(engine.hash_path, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip()
    assert current_hash == stored_hash
