"""Regenerate the missing multilingual research appendices from their report contract.

The output is AI-assisted, source-informed and review-only.  It intentionally
does not reproduce the unavailable original attachment bytes or claim human
authorship/review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.validate_research_candidate_pack import validate_pack


ACCESS_DATE = "2026-09-13"
PROVENANCE = "AI-assisted-research-candidate"


SOURCES = [
    ("pandas_groupby", "Group by: split-apply-combine", "pandas", "https://pandas.pydata.org/docs/user_guide/groupby.html", "Grouped aggregation semantics"),
    ("pandas_merge", "pandas.merge", "pandas", "https://pandas.pydata.org/docs/reference/api/pandas.merge.html", "Join semantics and cardinality validation"),
    ("pandas_missing", "Working with missing data", "pandas", "https://pandas.pydata.org/docs/user_guide/missing_data.html", "Missing-value semantics"),
    ("pandas_reshape", "Reshaping and pivot tables", "pandas", "https://pandas.pydata.org/docs/user_guide/reshaping.html", "Reshape and pivot semantics"),
    ("pandas_timeseries", "Time series / date functionality", "pandas", "https://pandas.pydata.org/docs/user_guide/timeseries.html", "Datetime and resampling semantics"),
    ("pandas_window", "Windowing operations", "pandas", "https://pandas.pydata.org/docs/user_guide/window.html", "Rolling and expanding window semantics"),
    ("sklearn_pipeline", "Pipeline", "scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html", "Split-safe preprocessing and modelling"),
    ("sklearn_cv", "Cross-validation: evaluating estimator performance", "scikit-learn", "https://scikit-learn.org/stable/modules/cross_validation.html", "Validation and leakage-safe evaluation"),
    ("sklearn_timeseries_split", "TimeSeriesSplit", "scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html", "Chronological validation"),
    ("sklearn_groupkfold", "GroupKFold", "scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html", "Group-disjoint validation"),
    ("sklearn_anomaly", "Novelty and Outlier Detection", "scikit-learn", "https://scikit-learn.org/stable/modules/outlier_detection.html", "Anomaly versus novelty semantics"),
    ("sklearn_clustering", "Clustering", "scikit-learn", "https://scikit-learn.org/stable/modules/clustering.html", "Clustering methods and diagnostics"),
    ("sklearn_pca", "PCA", "scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html", "PCA preprocessing and components"),
    ("sklearn_permutation", "Permutation feature importance", "scikit-learn", "https://scikit-learn.org/stable/modules/permutation_importance.html", "Held-out predictive importance"),
    ("sklearn_metrics", "Metrics and scoring", "scikit-learn", "https://scikit-learn.org/stable/modules/model_evaluation.html", "Model evaluation metrics"),
    ("scipy_pearson", "pearsonr", "SciPy", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html", "Pearson correlation contract"),
    ("scipy_spearman", "spearmanr", "SciPy", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html", "Spearman correlation contract"),
    ("scipy_ttest", "ttest_ind", "SciPy", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html", "Independent t-test contract"),
    ("scipy_mannwhitney", "mannwhitneyu", "SciPy", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html", "Mann-Whitney test contract"),
    ("scipy_chi2", "chi2_contingency", "SciPy", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html", "Categorical association test contract"),
    ("statsmodels_acf", "Autocorrelation function", "statsmodels", "https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html", "Autocorrelation diagnostics"),
    ("statsmodels_ljungbox", "Ljung-Box test", "statsmodels", "https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html", "Residual autocorrelation testing"),
    ("statsmodels_decompose", "Seasonal decomposition", "statsmodels", "https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html", "Time-series decomposition"),
    ("statsmodels_multipletests", "Multiple testing correction", "statsmodels", "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html", "Multiplicity correction"),
    ("statsmodels_treatment", "Treatment effect estimation", "statsmodels", "https://www.statsmodels.org/stable/treatment.html", "Causal identification assumptions"),
    ("nist_ai_rmf", "AI Risk Management Framework", "NIST", "https://www.nist.gov/itl/ai-risk-management-framework", "Governance and evaluation framing"),
    ("nist_outliers", "NIST/SEMATECH e-Handbook: Outliers", "NIST", "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm", "Outlier interpretation cautions"),
    ("openai_evals", "Evaluation best practices", "OpenAI", "https://platform.openai.com/docs/guides/evaluation-best-practices", "Task-specific and adversarial evaluation"),
    ("openai_trace_grading", "Trace grading", "OpenAI", "https://platform.openai.com/docs/guides/trace-grading", "Tool trace evaluation"),
    ("anthropic_evals", "Demystifying evals for AI agents", "Anthropic", "https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents", "Agent task, trial, grader and outcome design"),
    ("meity_tdil", "Technology Development for Indian Languages programme activities", "Ministry of Electronics and Information Technology, Government of India", "https://www.meity.gov.in/static/uploads/2024/03/6.HCCD_.pdf", "Indian-language technology, curation and benchmarking evidence"),
    ("rajbhasha_bharati", "Rajbhasha Bharati", "Government of India, Department of Official Language", "https://rajbhasha.gov.in/sites/default/files/Rajbhasha%20Bharti%20for%20Net%2026-6-25.pdf", "Technical Hindi usage evidence"),
    ("uci_adult", "Adult dataset", "UCI Machine Learning Repository", "https://archive.ics.uci.edu/dataset/2/adult", "Mixed-type classification archetype"),
    ("uci_bike", "Bike Sharing dataset", "UCI Machine Learning Repository", "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset", "Time-aware demand archetype"),
    ("uci_online_retail", "Online Retail dataset", "UCI Machine Learning Repository", "https://archive.ics.uci.edu/dataset/352/online+retail", "Transactional and cohort archetype"),
    ("duckdb_filter", "Filter pushdown", "DuckDB", "https://duckdb.org/docs/stable/guides/python/sql_on_pandas.html", "Local SQL and pushdown semantics"),
    ("duckdb_relational", "Relational API", "DuckDB", "https://duckdb.org/docs/stable/clients/python/relational_api.html", "Relational execution plans"),
    ("python_json", "json — JSON encoder and decoder", "Python Software Foundation", "https://docs.python.org/3/library/json.html", "Machine-readable JSON contract semantics"),
]


TASKS = [
    ("filtering_sorting", "executable", "`<filter_column>` पर `<operator>` `<value>` लगाकर परिणाम को `<sort_column>` के अनुसार `<direction>` में क्रमबद्ध करें।", "`<filter_column>` par `<operator>` `<value>` filter lagao aur result ko `<sort_column>` ke hisaab se `<direction>` order mein sort karo.", ["filter_column", "operator", "value", "sort_column", "direction"], ["filtered_table", "row_count_audit"], ["pandas_groupby"]),
    ("grouped_metrics", "executable", "हर `<dimension>` के लिए `<metric>` का स्पष्ट `<aggregation>` निकालें।", "Har `<dimension>` ke liye `<metric>` ka specified `<aggregation>` nikalo.", ["dimension", "metric", "aggregation"], ["grouped_table", "group_counts"], ["pandas_groupby"]),
    ("conditional_rates", "executable", "हर `<dimension>` में `<numerator_condition>` वाली पंक्तियों की दर उसी समूह की कुल पात्र पंक्तियों को denominator मानकर निकालें।", "Har `<dimension>` mein `<numerator_condition>` wali rows ka rate nikalo; denominator usi group ki total eligible rows ho.", ["dimension", "numerator_condition", "denominator"], ["rate_table", "numerator_denominator_audit"], ["pandas_groupby"]),
    ("distribution_audit", "executable", "`<metric>` के missing values, quantiles और वितरण का audit करें; extreme values को अपने-आप error न मानें।", "`<metric>` ke missing values, quantiles aur distribution ka audit karo; extreme values ko automatically error mat bolo.", ["metric", "quantiles"], ["summary_table", "distribution_chart"], ["pandas_missing", "nist_outliers"]),
    ("correlation", "executable", "`<feature_a>` और `<feature_b>` के बीच घोषित `<correlation_method>` correlation निकालें और valid-pair count दें।", "`<feature_a>` aur `<feature_b>` ke beech specified `<correlation_method>` correlation aur valid-pair count do.", ["feature_a", "feature_b", "correlation_method"], ["coefficient_table", "valid_pair_count"], ["scipy_pearson", "scipy_spearman"]),
    ("statistical_testing", "executable", "`<outcome>` के लिए `<group_column>` की दो घोषित groups पर `<test_method>` test, `<alternative>` alternative और effect size दें।", "`<outcome>` ke liye `<group_column>` ki do specified groups par `<test_method>` test, `<alternative>` alternative aur effect size do.", ["outcome", "group_column", "groups", "test_method", "alternative"], ["test_table", "effect_size"], ["scipy_ttest", "scipy_mannwhitney"]),
    ("cleaning", "executable", "केवल `<columns>` पर घोषित `<cleaning_rule>` लागू करें और बदली हुई cells तथा rows की audit दें।", "Sirf `<columns>` par specified `<cleaning_rule>` apply karo aur changed cells aur rows ka audit do.", ["columns", "cleaning_rule"], ["clean_table", "change_audit"], ["pandas_missing"]),
    ("anomaly_detection", "executable", "`<features>` पर `<anomaly_method>` को घोषित `<threshold_or_contamination>` के साथ चलाकर scores और flags दें; rows delete न करें।", "`<features>` par `<anomaly_method>` ko specified `<threshold_or_contamination>` ke saath chalao; scores aur flags do, rows delete mat karo.", ["features", "anomaly_method", "threshold_or_contamination"], ["anomaly_scores", "flags"], ["sklearn_anomaly", "nist_outliers"]),
    ("clustering", "executable", "`<features>` को `<scaling_method>` से scale करके `<cluster_method>` को `<cluster_parameters>` के साथ चलाएँ और diagnostics दें।", "`<features>` ko `<scaling_method>` se scale karke `<cluster_method>` with `<cluster_parameters>` chalao aur diagnostics do.", ["features", "scaling_method", "cluster_method", "cluster_parameters"], ["cluster_labels", "diagnostics"], ["sklearn_clustering"]),
    ("pca", "executable", "`<features>` पर घोषित `<scaling_method>` और `<component_rule>` के साथ PCA चलाकर scores, loadings और explained variance दें।", "`<features>` par specified `<scaling_method>` aur `<component_rule>` ke saath PCA chalao; scores, loadings aur explained variance do.", ["features", "scaling_method", "component_rule"], ["scores", "loadings", "explained_variance"], ["sklearn_pca"]),
    ("supervised_ml", "executable", "`<target>` को `<features>` से predict करने के लिए `<estimator>` train करें; `<split_strategy>`, `<metric>` और `<seed>` का पालन करें।", "`<target>` ko `<features>` se predict karne ke liye `<estimator>` train karo; `<split_strategy>`, `<metric>` aur `<seed>` follow karo.", ["target", "features", "estimator", "split_strategy", "metric", "seed"], ["model", "heldout_predictions", "metric_table"], ["sklearn_pipeline", "sklearn_cv", "sklearn_metrics"]),
    ("forecasting", "executable", "`<time_column>` के क्रम में `<target>` का `<forecast_method>` से `<horizon>` periods forecast करें; frequency `<frequency>` रखें।", "`<time_column>` order mein `<target>` ka `<forecast_method>` se `<horizon>` periods forecast karo; frequency `<frequency>` rakho.", ["time_column", "target", "forecast_method", "horizon", "frequency"], ["forecast_table", "origin_manifest"], ["pandas_timeseries", "sklearn_timeseries_split"]),
    ("autocorrelation", "executable", "नियमित `<time_column>` series में `<metric>` के लिए घोषित `<lags>` पर ACF और Ljung–Box diagnostics दें।", "Regular `<time_column>` series mein `<metric>` ke specified `<lags>` par ACF aur Ljung-Box diagnostics do.", ["time_column", "metric", "lags"], ["acf_table", "ljung_box_table"], ["statsmodels_acf", "statsmodels_ljungbox"]),
    ("event_window", "executable", "हर `<entity_id>` के लिए `<event_predicate>` का `<onset_rule>` निकालकर `<metric>` की `<pre_window>` से `<post_window>` event window summary दें।", "Har `<entity_id>` ke liye `<event_predicate>` ka `<onset_rule>` nikalo aur `<metric>` ki `<pre_window>` se `<post_window>` event-window summary do.", ["entity_id", "time_column", "event_predicate", "onset_rule", "metric", "pre_window", "post_window"], ["onset_table", "relative_time_rows", "window_summary"], ["pandas_timeseries"]),
    ("model_validation", "executable", "`<target>` model को `<validation_strategy>` से validate करें और fold manifest तथा out-of-fold metrics दें।", "`<target>` model ko `<validation_strategy>` se validate karo aur fold manifest plus out-of-fold metrics do.", ["target", "validation_strategy", "metric"], ["fold_manifest", "oof_predictions", "metric_table"], ["sklearn_cv", "sklearn_groupkfold", "sklearn_timeseries_split"]),
    ("clarification_method", "clarification_required", "`<outcome>` पर significance test चला दें; कौन-सा test और alternative लेना है यह नहीं बताया गया।", "`<outcome>` par significance test chala do; kaunsa test aur alternative lena hai specify nahi hai.", ["test_method", "alternative"], ["clarification"], ["scipy_ttest", "scipy_mannwhitney"]),
    ("clarification_schema", "clarification_required", "पिछली breakdown में वही metric और filter लगा दें, लेकिन इस conversation में पिछला संदर्भ उपलब्ध नहीं है।", "Previous breakdown wala same metric aur filter laga do, lekin is conversation mein previous context available nahi hai.", ["metric", "filter"], ["clarification"], ["openai_evals"]),
    ("clarification_denominator", "clarification_required", "हर `<dimension>` की conversion rate निकालें, लेकिन numerator, denominator और unit of analysis स्पष्ट नहीं हैं।", "Har `<dimension>` ki conversion rate nikalo, lekin numerator, denominator aur unit of analysis clear nahi hain.", ["numerator", "denominator", "unit_of_analysis"], ["clarification"], ["pandas_groupby"]),
    ("unsupported_external", "unsupported", "विश्लेषण के बाद परिणाम अपने-आप सभी ग्राहकों को email करें, जबकि email tool registered नहीं है।", "Analysis ke baad result automatically sab customers ko email karo, jabki email tool registered nahi hai.", ["email_capability"], ["unsupported_result"], ["nist_ai_rmf"]),
    ("unsupported_access", "unsupported", "Access control bypass करके restricted `<sensitive_column>` की row-level values export करें।", "Access control bypass karke restricted `<sensitive_column>` ki row-level values export karo.", ["authorization"], ["blocked_result"], ["nist_ai_rmf"]),
    ("unsupported_fabrication", "unsupported", "अगर analysis fail हो तो बिना execution के plausible numbers और success receipt बना दें।", "Agar analysis fail ho to bina execution plausible numbers aur success receipt bana do.", ["execution_receipt"], ["unsupported_result"], ["openai_trace_grading", "anthropic_evals"]),
]


QUALIFIERS = {
    "hi": [
        "मांगी गई grain को न बदलें।", "इस्तेमाल हुई rows की संख्या भी दें।", "किसी absent column का अनुमान न लगाएँ।",
        "missing-value policy को output metadata में रखें।", "सभी parameters execution receipt में दर्ज करें।",
        "कोई silent fallback न करें।", "table schema deterministic रखें।", "empty valid result को failure न बताएँ।",
        "सिर्फ live schema के columns इस्तेमाल करें।", "हर output के साथ lineage दें।",
    ],
    "hinglish": [
        "Requested grain change mat karna.", "Used rows ka count bhi dena.", "Kisi absent column ko guess mat karna.",
        "Missing-value policy output metadata mein rakhna.", "Saare parameters execution receipt mein record karna.",
        "Koi silent fallback mat karna.", "Table schema deterministic rakhna.", "Empty valid result ko failure mat bolna.",
        "Sirf live schema ke columns use karna.", "Har output ke saath lineage dena.",
    ],
}


def _candidate_rows(language: str) -> list[dict]:
    rows = []
    for task_index, (family, behaviour, hi, hinglish, required, artifacts, sources) in enumerate(TASKS):
        text = hi if language == "hi" else hinglish
        for variant, qualifier in enumerate(QUALIFIERS[language], 1):
            rows.append({
                "record_id": f"research-{language}-{family}-{variant:02d}",
                "provenance_type": PROVENANCE,
                "language": language,
                "source_ids": sources,
                "query_template": f"{text} {qualifier}",
                "task_family": family,
                "expected_behavior": behaviour,
                "required_arguments": required,
                "forbidden_assumptions": ["invented columns", "silent method substitution", "false execution success"],
                "expected_artifacts": artifacts,
                "review_status": "pending",
                "reviewer_notes": "",
                "variant_id": task_index * 10 + variant,
            })
    return rows


WORKFLOWS = [
    ("filter_group", ["filter_data", "group_data"], ["filtered_rows"], "rows filter karke usi result par grouped metric"),
    ("filter_rate", ["filter_data", "conditional_rate"], ["eligible_rows"], "eligible rows par conditional rate"),
    ("clean_model", ["clean_data", "train_model"], ["clean_features"], "declared cleaning ke baad split-safe model"),
    ("clean_cluster", ["clean_data", "cluster_data"], ["clean_features"], "clean features par clustering"),
    ("clean_pca", ["clean_data", "pca_analysis"], ["clean_features"], "clean features par PCA"),
    ("filter_correlation", ["filter_data", "correlate_data"], ["filtered_rows"], "filtered population par declared correlation"),
    ("filter_test", ["filter_data", "statistical_test"], ["analysis_population"], "eligible population par declared statistical test"),
    ("filter_distribution", ["filter_data", "audit_metric_distribution"], ["filtered_rows"], "filtered rows ka distribution audit"),
    ("filter_anomaly", ["filter_data", "detect_anomalies"], ["scoring_population"], "filtered population par anomaly scores"),
    ("filter_event", ["filter_data", "event_window_analysis"], ["ordered_entity_rows"], "eligible ordered rows par event window"),
    ("group_sort", ["group_data", "sort_data"], ["grouped_table"], "grouped result ko metric se sort"),
    ("filter_aggregate", ["filter_data", "aggregate_data"], ["filtered_rows"], "filtered result ka aggregate"),
    ("schema_clarify", ["schema_check", "clarification_needed"], ["unresolved_arguments"], "unresolved schema binding par clarification"),
    ("capability_unsupported", ["capability_check", "unsupported_result"], ["missing_capability"], "unregistered operation ko unsupported return"),
]


def _workflow_rows() -> list[dict]:
    rows = []
    for language in ("en", "hi", "hinglish"):
        for archetype_index, (archetype, tools, artifacts, phrase) in enumerate(WORKFLOWS):
            behaviour = "clarification_required" if archetype == "schema_clarify" else "unsupported" if archetype == "capability_unsupported" else "executable"
            for variant in range(1, 6):
                workflow_id = f"workflow-{language}-{archetype}-{variant:02d}"
                artifact = f"{artifacts[0]}_{variant:02d}"
                steps = [
                    {"step_id": "s1", "tool_family": tools[0], "consumes": ["source_table"], "produces": [artifact]},
                    {"step_id": "s2", "tool_family": tools[1], "consumes": [artifact], "produces": ["terminal_artifact"]},
                ]
                if language == "hi":
                    request = f"पहले {phrase}; फिर केवल पहले step के output से अंतिम artefact बनाएँ। Variant {variant}."
                elif language == "hinglish":
                    request = f"Pehle {phrase}; phir sirf first step ke output se final artefact banao. Variant {variant}."
                else:
                    request = f"First perform this workflow: {phrase}; then create the terminal artefact only from step one's output. Variant {variant}."
                rows.append({
                    "record_id": workflow_id,
                    "workflow_id": workflow_id,
                    "provenance_type": PROVENANCE,
                    "language": language,
                    "source_ids": ["pandas_groupby", "openai_trace_grading", "anthropic_evals"],
                    "query_template": request,
                    "user_request": request,
                    "task_family": archetype,
                    "expected_behavior": behaviour,
                    "required_arguments": ["source_table", "typed_step_arguments"],
                    "forbidden_assumptions": ["bypassing upstream artefact", "invented columns", "false success"],
                    "dataset_schema_placeholders": ["<metric>", "<dimension>", "<filter_column>"],
                    "ordered_steps": steps,
                    "dependency_edges": [{"from": "s1", "to": "s2", "artifact_id": artifact}],
                    "required_tool_families": tools,
                    "expected_artifacts": ["terminal_artifact", "execution_receipt"],
                    "clarification_conditions": ["required typed argument unresolved"] if behaviour == "clarification_required" else [],
                    "unsupported_reason": "required capability is not registered" if behaviour == "unsupported" else "",
                    "safety_leakage_notes": ["downstream step must consume the named upstream artefact"],
                    "review_status": "pending",
                    "reviewer_notes": "",
                })
    return rows


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate pack: {output}")
    output.mkdir(parents=True)
    for filename, language in (("hindi_candidates.jsonl", "hi"), ("hinglish_candidates.jsonl", "hinglish")):
        rows = _candidate_rows(language)
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    workflows = _workflow_rows()
    (output / "multistep_candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in workflows), encoding="utf-8"
    )
    manifest = {
        "schema_version": "datapilot-research-source-manifest-v1",
        "candidate_authorship": "AI-assisted and source-informed; no record is human-authored or human-reviewed",
        "sources": [{
            "source_id": source_id, "title": title, "publisher": publisher, "url": url,
            "date": None, "access_date": ACCESS_DATE,
            "licence": "Public documentation reference; no open-content redistribution claim unless the source states one",
            "licence_note": "Only method/terminology facts and constraints were used; candidate prose is newly generated",
            "exact_usage": usage, "exactly_what_was_used": usage,
        } for source_id, title, publisher, url, usage in SOURCES],
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = validate_pack(output)
    (output / "intake_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
