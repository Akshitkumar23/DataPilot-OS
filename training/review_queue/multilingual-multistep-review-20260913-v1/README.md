# Corpus review worklist

Records: 240; reference families: 120.

Reference families are exact dataset/plan groups, not certified independent business problems.

## Priorities

- P0: inspect execution/result-contract failures; do not train on these until the cause is resolved.
- P1: redesign overlapping splits; keep related variants together and curate independent held-out cases.
- P2: verify query meaning, evidence, numerical ground truth and clarification behavior.

A passing execution does not approve a query or all its paraphrases. Keep actual human review distinct.

## Capability coverage

| Tool | Train families | Validation families | Test families |
|---|---:|---:|---:|
| aggregate_data | 0 | 0 | 0 |
| audit_metric_distribution | 0 | 0 | 0 |
| autocorrelation | 0 | 0 | 0 |
| categorical_outcome_association | 0 | 0 | 0 |
| clarification | 0 | 0 | 0 |
| clean_data | 0 | 0 | 0 |
| cluster_data | 0 | 0 | 0 |
| conditional_rate | 0 | 0 | 0 |
| correlate_data | 0 | 0 | 0 |
| decompose_time_series | 0 | 0 | 0 |
| detect_anomalies | 0 | 0 | 0 |
| eda_data | 0 | 0 | 0 |
| event_window_analysis | 0 | 0 | 0 |
| explain_features | 0 | 0 | 0 |
| filter_data | 0 | 0 | 0 |
| forecast_data | 0 | 0 | 0 |
| group_data | 0 | 0 | 0 |
| pca_analysis | 0 | 0 | 0 |
| sort_data | 0 | 0 | 0 |
| statistical_test | 0 | 0 | 0 |
| train_model | 0 | 0 | 0 |

## Execution failures

None in the supplied execution sample; unexecuted families remain unverified.

Full query variants, case IDs, reference plans and review fields are in review_tasks.json.
Complete every row in variant_review_template.csv. Review identity is self-declared unless your workflow adds an external signature.
Import only with import_human_reviews.py and the bound review_manifest.json; the importer rejects stale or edited immutable fields.
Do not approve source labels based only on this report or automated execution.
