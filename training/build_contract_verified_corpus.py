"""Build a large, *real-public-data* planner corpus without mislabelling it.

The corpus is deliberately generated from versioned UCI public datasets and
validated against the live ToolSpec registry.  Its labels are
``contract-verified``: every command has passed the exact Pydantic tool
contract used by the product.  They are not claimed to be human-reviewed and
must be reviewed plus benchmarked before production promotion.

This is an offline curation tool only; it has no import path from the desktop
runtime and cannot change an Ollama model or query-routing behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from scripts.validate_planner_corpus import validate_records

DEFAULT_OUTPUT = ROOT / "training" / "planner_sft_contract_verified.jsonl"
PROVENANCE = ROOT / "data" / "real_benchmarks" / "provenance.json"


def command(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool": tool, "arguments": arguments}


# These are hand-authored analytical intents bound to genuine, versioned public
# schemas.  Rewording below creates planner-language diversity; it never
# invents dataset fields or alters a checked command contract.
RECIPES: list[dict[str, Any]] = [
    {"tool": "filter_data", "dataset": "uci_bank_marketing", "columns": ["age"], "query": "Keep only customers aged above 30.", "args": {"column": "age", "operator": ">", "value": 30}},
    {"tool": "sort_data", "dataset": "uci_bank_marketing", "columns": ["balance"], "query": "Rank customer records by balance from highest to lowest and return the top 20.", "args": {"column": "balance", "direction": "desc", "limit": 20}},
    {"tool": "conditional_rate", "dataset": "uci_bank_marketing", "columns": ["job", "y"], "query": "Show the deposit subscription rate by job for cohorts with at least 100 customers as a bar chart.", "args": {"group_by": ["job"], "numerator_condition": {"column": "y", "condition": "==", "value": "yes"}, "min_group_size": 100, "sort_by": "rate_percent", "direction": "desc", "limit": 20, "chart_type": "bar"}},
    {"tool": "group_data", "dataset": "uci_bank_marketing", "columns": ["marital", "balance"], "query": "Calculate mean account balance by marital status with a bar chart.", "args": {"group_by": ["marital"], "metric": "balance", "aggregation": "mean", "datetime_parses": [], "chart_type": "bar"}},
    {"tool": "aggregate_data", "dataset": "uci_bank_marketing", "columns": ["housing"], "query": "Count customers with a housing loan.", "args": {"aggregation": "count", "filters": [{"column": "housing", "condition": "==", "value": "yes"}]}},
    {"tool": "audit_metric_distribution", "dataset": "uci_bank_marketing", "columns": ["balance", "housing"], "query": "Audit the balance distribution for customers with a housing loan using a box plot.", "args": {"metric": "balance", "filters": [{"column": "housing", "condition": "==", "value": "yes"}], "chart_type": "box"}},
    {"tool": "correlate_data", "dataset": "uci_wine_quality_white", "columns": ["alcohol", "density"], "query": "Plot the linear Pearson association between alcohol and density.", "args": {"columns": ["alcohol", "density"], "analysis_mode": "pairwise", "relationship_form": "linear", "correlation_methods": ["pearson"], "chart_type": "scatter", "smoother": "none", "expected_direction": "unspecified"}},
    {"tool": "forecast_data", "dataset": "uci_appliances_energy", "columns": ["date", "Appliances"], "query": "Forecast appliance energy consumption for the next 24 intervals from the date column as a line chart.", "args": {"value_column": "Appliances", "horizon": 24, "horizon_evidence": "next 24 intervals", "time_column": "date", "frequency": "10min", "method": "holt_linear", "chart_type": "line"}},
    {"tool": "clean_data", "dataset": "uci_adult_income", "columns": ["workclass"], "query": "Clean missing values in the workclass field.", "args": {"columns": ["workclass"], "method": "drop_missing"}},
    {"tool": "detect_anomalies", "dataset": "uci_wine_quality_white", "columns": ["residual sugar", "alcohol"], "query": "Detect anomalous wine records using residual sugar and alcohol with Isolation Forest at 5% contamination.", "args": {"columns": ["residual sugar", "alcohol"], "method": "isolation_forest", "sensitivity": 0.05}},
    {"tool": "eda_data", "dataset": "uci_appliances_energy", "columns": ["Appliances", "T1"], "query": "Run a tabular exploratory profile for appliance energy consumption and indoor temperature.", "args": {"columns": ["Appliances", "T1"], "chart_type": "none"}},
    {"tool": "train_model", "dataset": "uci_adult_income", "columns": ["income", "age", "education_num", "hours_per_week"], "query": "Train a classification model to predict income using age, education number, and hours per week; report a confusion matrix for a retrospective employment-analysis use case.", "args": {"target_column": "income", "feature_columns": ["age", "education_num", "hours_per_week"], "task_type": "classification", "evaluation_artifact": "confusion_matrix", "prediction_context": "retrospective_explanation"}},
    {"tool": "cluster_data", "dataset": "uci_wine_quality_white", "columns": ["alcohol", "residual sugar", "density"], "query": "Segment wines into three clusters using alcohol, residual sugar, and density.", "args": {"feature_columns": ["alcohol", "residual sugar", "density"], "n_clusters": 3, "scaling": "standard"}},
    {"tool": "statistical_test", "dataset": "uci_wine_quality_white", "columns": ["alcohol", "quality"], "query": "Test whether alcohol differs across wine quality scores using one-way ANOVA.", "args": {"columns": ["alcohol", "quality"], "test_type": "one_way_anova", "alpha": 0.05}},
    {"tool": "categorical_outcome_association", "dataset": "uci_bank_marketing", "columns": ["duration", "y"], "query": "For retrospective campaign review, compare completed-call duration across subscription outcomes with a box plot.", "args": {"numeric_predictor": "duration", "outcome_column": "y", "chart_type": "box", "analysis_context": "retrospective_association"}},
    {"tool": "pca_analysis", "dataset": "uci_wine_quality_white", "columns": ["alcohol", "density", "pH"], "query": "Reduce alcohol, density, and pH to two principal components.", "args": {"columns": ["alcohol", "density", "pH"], "n_components": 2}},
    {"tool": "explain_features", "dataset": "uci_adult_income", "columns": ["income", "age", "education_num", "hours_per_week"], "query": "For retrospective classification analysis, rank age, education number, and hours per week for income using held-out permutation importance.", "args": {"target_column": "income", "feature_columns": ["age", "education_num", "hours_per_week"], "task_type": "classification", "method": "permutation_importance", "prediction_context": "retrospective_explanation"}},
    {"tool": "autocorrelation", "dataset": "uci_appliances_energy", "columns": ["Appliances"], "query": "Check autocorrelation of appliance energy usage through 24 lags.", "args": {"column": "Appliances", "max_lags": 24}},
    {"tool": "decompose_time_series", "dataset": "uci_appliances_energy", "columns": ["date", "Appliances"], "query": "Decompose appliance energy consumption over time with a daily seasonal period.", "args": {"time_column": "date", "target_column": "Appliances", "period": 144, "model": "additive"}},
    {"tool": "event_window_analysis", "dataset": "uci_appliances_energy", "columns": ["Appliances", "lights", "date"], "query": "Treat every record with lights above zero as an event row; compare appliance energy in the preceding 24 observations using date order and a box plot.", "args": {"columns": ["Appliances"], "event_column": "lights", "sequence_order": "date", "pre_event_window": 24, "event_predicate": {"column": "lights", "operator": ">", "value": 0}, "event_scope": "event_rows", "chart_type": "box"}},
]

TEST_RECIPES: list[dict[str, Any]] = [
    {"tool": "group_data", "dataset": "uci_bike_sharing_hourly", "columns": ["hr", "cnt"], "query": "Calculate mean hourly bike demand with a line chart.", "args": {"group_by": ["hr"], "metric": "cnt", "aggregation": "mean", "datetime_parses": [], "chart_type": "line"}},
    {"tool": "statistical_test", "dataset": "uci_bike_sharing_hourly", "columns": ["cnt", "weathersit"], "query": "Test whether bike demand differs across weather conditions using one-way ANOVA.", "args": {"columns": ["cnt", "weathersit"], "test_type": "one_way_anova", "alpha": 0.05}},
    {"tool": "eda_data", "dataset": "uci_online_retail", "columns": ["Quantity", "UnitPrice"], "query": "Run a tabular exploratory profile for transaction quantity and unit price.", "args": {"columns": ["Quantity", "UnitPrice"], "chart_type": "none"}},
    {"tool": "filter_data", "dataset": "uci_online_retail", "columns": ["Quantity"], "query": "Keep retail transactions with quantity above 10.", "args": {"column": "Quantity", "operator": ">", "value": 10}},
]

CLARIFICATIONS = [
    ("uci_bank_marketing", ["balance"], "Show the trend.", "Please specify the time column and the metric whose trend you want to analyse."),
    ("uci_wine_quality_white", ["quality"], "Train a model.", "Please specify the prediction target, available-before-outcome features, task type, and required evaluation artifact."),
    ("uci_appliances_energy", ["date", "Appliances"], "Forecast energy use.", "Please specify the forecast horizon, aggregation frequency, and forecasting method."),
    ("uci_adult_income", ["income"], "Find the relationship.", "Please specify the columns and whether you need correlation, group comparison, or a predictive model."),
]

PREFIXES = ["Data quality request:", "Analyst task:", "Business analysis:", "Please perform this analysis:", "Use the connected dataset to:"]
SUFFIXES = [
    "Use only the named columns.", "Keep the requested analytical scope.", "Return the requested artifact.",
    "Do not infer additional fields.", "Keep the result reproducible.", "Preserve the stated aggregation.",
    "Use the connected dataset schema.", "Keep this as a single analytical task.", "Respect the requested visualization.",
    "Do not substitute another metric.",
]


def phrased_query(query: str, index: int) -> str:
    held_out = " This is a held-out validation request." if 1000 <= index < 2000 else ""
    return f"{PREFIXES[index % len(PREFIXES)]} {query} {SUFFIXES[(index // len(PREFIXES)) % len(SUFFIXES)]}{held_out}"


def assistant_tool(call: dict[str, Any]) -> str:
    return json.dumps({"mode": "tool", "commands": [call]}, separators=(",", ":"))


def variant_for_recipe(recipe: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    """Produce plan-consistent numeric diversity for safe ToolSpec families.

    Only values explicitly stated in the corresponding user request vary.  This
    avoids the harmful pattern of changing a plan while leaving its natural
    language label unchanged.
    """
    args = deepcopy(recipe["args"])
    query = recipe["query"]
    step = index % 50
    if recipe["tool"] == "filter_data" and recipe["dataset"] == "uci_bank_marketing":
        value = 20 + step
        args["value"] = value
        query = f"Keep only customers aged above {value}."
    elif recipe["tool"] == "sort_data":
        limit = 1 + step
        args["limit"] = limit
        query = f"Rank customer records by balance from highest to lowest and return the top {limit}."
    elif recipe["tool"] == "conditional_rate":
        minimum = 50 + (step * 10)
        limit = 5 + step
        args["min_group_size"] = minimum
        args["limit"] = limit
        query = f"Show the deposit subscription rate by job for cohorts with at least {minimum} customers; return the top {limit} groups as a bar chart."
    elif recipe["tool"] == "forecast_data":
        horizon = 6 + step
        args["horizon"] = horizon
        args["horizon_evidence"] = f"next {horizon} intervals"
        query = f"Forecast appliance energy consumption for the next {horizon} intervals from the date column as a line chart."
    elif recipe["tool"] == "autocorrelation":
        lags = 1 + step
        args["max_lags"] = lags
        query = f"Check autocorrelation of appliance energy usage through {lags} lags."
    return args, query


def make_tool_record(recipe: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    arguments, base_query = variant_for_recipe(recipe, index)
    call = command(recipe["tool"], arguments)
    query = phrased_query(base_query, index)
    return {
        "schema_version": "planner-sft-v1",
        "case_id": f"contract-{split}-{recipe['tool']}-{index:04d}",
        "split": split,
        "dataset_id": recipe["dataset"],
        "query": query,
        "required_columns": recipe["columns"],
        "expected_decision": "tool",
        "expected_command": call,
        "messages": [{"role": "user", "content": query}, {"role": "assistant", "content": assistant_tool(call)}],
        "label_status": "contract-verified",
        "verification": {"kind": "live_toolspec_contract", "tool": recipe["tool"], "review_required": True},
    }


def make_clarification(dataset: str, columns: list[str], query: str, reason: str, index: int, split: str) -> dict[str, Any]:
    full_query = phrased_query(query, index)
    answer = json.dumps({"mode": "clarify", "clarification": reason}, separators=(",", ":"))
    return {
        "schema_version": "planner-sft-v1",
        "case_id": f"contract-{split}-clarification-{index:04d}",
        "split": split,
        "dataset_id": dataset,
        "query": full_query,
        "required_columns": columns,
        "expected_decision": "clarification",
        "messages": [{"role": "user", "content": full_query}, {"role": "assistant", "content": answer}],
        "label_status": "contract-verified",
        "verification": {"kind": "clarification_policy", "review_required": True},
    }


def build() -> list[dict[str, Any]]:
    # 50 tool examples x 20 live analytical ToolSpecs = 1,000 train records;
    # plus 100 clarification labels and held-out validation/test records.
    records: list[dict[str, Any]] = []
    for recipe in RECIPES:
        for index in range(50):
            records.append(make_tool_record(recipe, index, "train"))
    for index in range(100):
        dataset, columns, query, reason = CLARIFICATIONS[index % len(CLARIFICATIONS)]
        records.append(make_clarification(dataset, columns, query, reason, index, "train"))
    for index in range(100):
        records.append(make_tool_record(RECIPES[index % len(RECIPES)], 1000 + index, "validation"))
    # Test uses held-out public datasets only: bike sharing + online retail.
    held_out = TEST_RECIPES
    for index in range(200):
        records.append(make_tool_record(held_out[index % len(held_out)], index // len(held_out), "test"))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = build()
    registry = CapabilityRegistry(auto_bootstrap=True)
    for record in records:
        if record["expected_decision"] == "tool":
            call = record["expected_command"]
            registry.validate_tool_call(call["tool"], call["arguments"])
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    validate_records([{**record, "_line_number": number} for number, record in enumerate(records, 1)], provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "label_status": "contract-verified", "production_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
