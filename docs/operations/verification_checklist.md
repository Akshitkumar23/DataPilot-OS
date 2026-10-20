# DataPilot OS — Verification & Quality Checklist

## Complete Verification Commands

Run this sequence before every release, staging deployment, or pull request to ensure 100% contract adherence.

### 1. Frontend & Static Analysis
```bash
# 1.1 Oxlint linting (must pass with 0 errors)
npm run lint

# 1.2 Vite production build (must exit code 0)
npm run build

# 1.3 Query result presentation node test
node tests/test_query_result_presentation.mjs
```

### 2. Python Architecture & Contract Test Gates (Phases 1-11)
```bash
# Set PYTHONPATH to project root
# Windows PowerShell:
$env:PYTHONPATH="."
# Linux / macOS:
export PYTHONPATH="."

# Run Canonical Regression Suite (142 tests across 24 files)
pytest tests/test_canonical_query_response.py \
       tests/test_tool_contract_completeness.py \
       tests/test_dataset_grounding.py \
       tests/test_column_ambiguity_contract.py \
       tests/test_planner_candidate_confinement.py \
       tests/test_planner_tool_retrieval.py \
       tests/test_runtime_prompt_parity.py \
       tests/test_planner_prompt_injection_resistance.py \
       tests/test_execution_graph_contract.py \
       tests/test_execution_graph_invalid_dependencies.py \
       tests/test_production_sequence_execution.py \
       tests/test_time_series_contract_hardening.py \
       tests/test_target_leakage_guard.py \
       tests/test_scientific_semantics_hardening.py \
       tests/test_semantic_feasibility.py \
       tests/test_time_series_grain_clarification.py \
       tests/test_statistical_assumption_contract.py \
       tests/test_operator_numerical_oracles.py \
       tests/test_ds_os_operators_and_validator.py \
       tests/test_conditional_rate_capability.py \
       tests/test_filtered_metric_audit_contract.py \
       tests/test_result_verification_registry.py \
       tests/test_chart_artifact_parity.py \
       tests/test_finite_repair_budget.py -v
```

---

## Acceptance Criteria Matrix

| Phase | Milestone | Acceptance Criteria | Verified |
|---|---|---|:---:|
| **Phase 1** | Canonical API Contract | `{request_id, status, decision, analysis_spec, execution_graph, result, manifest, validation, error}` unified shape | PASS |
| **Phase 2** | Strict ToolSpec Registry | 22 registered tools with exact schemas, no `RequestedArtifact.MODEL` | PASS |
| **Phase 3** | Dataset Grounding Engine | Exact/case-insensitive/normalized match, candidate discovery, no column guesses | PASS |
| **Phase 4** | Constrained Ollama Planner | Bounded history, cell value isolation, TF-IDF retriever, planner audit | PASS |
| **Phase 5** | Execution DAG Compiler | Unique op IDs, depends_on chains, read-only intermediate validation | PASS |
| **Phase 6** | Scientific Feasibility Gates | Time-series grain diagnosis, target leakage prevention, statistical assumption checks | PASS |
| **Phase 7** | Typed Operator Audit | 22 deterministic Python operators, execution manifests, numerical invariant oracles | PASS |
| **Phase 8** | Result Verification Registry | Independent row count, aggregation accuracy, chart artifact parity, mathematical bounds | PASS |
| **Phase 9** | Finite Repair Budget | Max 2 repair iterations, progress verification (diff check), audit history | PASS |
| **Phase 10** | Strict Frontend State Machine | Pure decision-driven UI transitions, ClarificationPrompt component, 0 Oxlint errors, Vite build exit 0 | PASS |
| **Phase 11** | Integration Regression Gate | 142 canonical integration and contract tests passing with 100% pass rate in < 20s | PASS |
| **Phase 12** | Deployment & Verification Runbook | Complete operations manuals and offline setup guides | PASS |
