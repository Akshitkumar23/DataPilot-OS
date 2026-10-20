# DataPilot OS — Future Work & Capability Backlog

**Notice:** Phase 1 through Phase 7 represent the complete, production-ready scope of DataPilot OS. This backlog outlines potential architectural directions for post-1.0 enterprise deployments without altering the current frozen kernel.

---

## 1. Advanced Tree-Based Ensemble Drivers (XGBoost / LightGBM)
* **Description:** Add optional GPU-accelerated gradient boosting backends alongside current scikit-learn Random Forests and Linear/Logistic models.
* **Scope Constraint:** Deferred to preserve minimal binary footprint and pure CPU portability in offline packaging.

## 2. Distributed Out-of-Core Processing via Apache Arrow
* **Description:** Direct Arrow memory mapping between DuckDB and memory-constrained environments for datasets exceeding physical RAM (>16 GB).
* **Scope Constraint:** Current DuckDB engine and Pandas chunking natively satisfy 99% of desktop analytical workflows without external cluster daemons.

## 3. Real-Time Streaming & Sensor Telemetry Ingestion
* **Description:** WebSocket ingestion pipeline for live time-series streaming and continuous sliding-window anomaly detection.
* **Scope Constraint:** DataPilot OS is targeted at desktop analytical exploration, dataset governance, and offline forensic modeling.

## 4. Multi-Table Relational Schema Visualizer
* **Description:** Interactive entity-relationship diagram (ERD) canvas in the Connect Hub for multi-file SQLite / PostgreSQL database schemas.
* **Scope Constraint:** Current DuckDB schema tree and SQL Studio provide complete interactive table inspection and join capability.

## 5. Structural Causal Discovery & Directed Acyclic Graph (DAG) Estimation
* **Description:** Integration of formal causal discovery algorithms (e.g. PC algorithm, FCI, LiNGAM) for learning causal graphs directly from observational data with do-calculus intervention effects.
* **Scope Constraint:** DataPilot currently enforces rigorous non-causal warnings and associations; formal structural DAG learning is deferred to post-1.0 research.

## 6. Small Language Model (SLM) Fine-Tuning Re-Evaluation
* **Description:** Exploration of parameter-efficient fine-tuning (LoRA / QLoRA) on small 1.5B–3B models for specialized domain taxonomy translation.
* **Scope Constraint:** Currently classified as NOT YET JUSTIFIED based on 100% routing and execution accuracy achieved with contracted zero-shot/few-shot prompts without fine-tuning maintenance overhead.

## 7. Explicit Missingness-Indicator Pattern Mining (MNAR Diagnostics)
* **Description:** Integration of formal Missing Completely At Random (MCAR) statistical hypothesis testing (e.g. Little's MCAR test) and automated pattern mixture analysis for complex survey and clinical non-response.
* **Scope Constraint:** DataPilot currently isolates target missingness, applies governed median/mode imputation on features, and logs all transformations in the provenance graph. High-dimensional pattern mixture modeling is reserved for post-1.0 scientific extensions.

