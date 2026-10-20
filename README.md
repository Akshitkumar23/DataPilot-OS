# DataPilot OS

DataPilot OS is an AI-guided data operating system combining local, privacy-first semantic planning with governed analytical execution and a fail-closed workspace agent. It bridges high-level business queries and engineering tasks to deterministic, verifiable execution pipelines without cloud dependencies or hallucinated analytical outcomes.

---

## Architectural Principles

1. **Semantic Planning vs. Trusted Execution:** LLMs are restricted to proposing structured, typed plans (`AnalysisSpecV3` and `WorkspaceTaskPlan`). The model is never permitted to fabricate analytical statistics or execute arbitrary uninspected code against datasets.
2. **Deterministic Execution Layer:** Analytical calculations, data transformations, and ML evaluations are carried out by dedicated engines (DuckDB, Pandas, Scikit-Learn, SciPy) and verified by numerical oracles.
3. **Fail-Closed Security & Zero Silent Fallback:** If an LLM is offline, a dataset schema is ambiguous, or an operation lacks required parameters, DataPilot OS halts and explicitly reports the requirement or raises a clarification request (`NEEDS_CLARIFICATION`). It never silently guesses column mappings or substitutes synthetic defaults.
4. **Governed Workspace Agent:** File reads, file writes, and command executions are constrained by path containment checks, atomic modifications with rollback, concurrent modification guards, and process tree termination.
5. **Cryptographic Provenance & Traceability:** Every transformation and model training step records input/output dataset fingerprints (SHA-256), parameters, execution timings, and operator contracts into an immutable provenance store.

---

## Capabilities & Implementation Status

| Capability Domain | Implemented & Verified | Experimental / Backlog |
| :--- | :--- | :--- |
| **Relational SQL & Storage** | Embedded DuckDB engine, multi-table joins (`join_planner.py`), relation catalog, schema introspection, fail-closed SQL validator. | Distributed out-of-core processing via Apache Arrow memory mapping. |
| **Data Cleaning & Engineering** | Governed type casting, missing value imputation, deduplication, filtering, outlier isolation, and provenance tracking. | Formal Little's MCAR statistical testing and automated pattern mixture modeling. |
| **Exploratory Data Analysis** | Automated column profiling, distribution analysis, correlation matrices (Pearson/Spearman), LOWESS curves. | Interactive multi-table ERD schema visualizer canvas. |
| **Inferential Statistics** | Two-sample t-tests, One-Way ANOVA with effect size ($\eta^2$), Chi-Square test of independence with minimum expected count checks. | Structural causal discovery DAG algorithms (PC, FCI, LiNGAM). |
| **Supervised Machine Learning** | Automated task inference (classification/regression), target leakage isolation, stratified train/test split, cross-validation, Decision Tree, Logistic/Linear Regression, Random Forest. | Optional GPU-accelerated gradient boosting drivers (XGBoost, LightGBM). |
| **Unsupervised Learning** | K-Means clustering with explicit cluster counts, Principal Component Analysis (PCA) with explained variance ratios. | High-dimensional UMAP / t-SNE embeddings. |
| **Time-Series Analysis** | Additive/multiplicative decomposition, autocorrelation (ACF/PACF), rolling statistics, trend forecasting with horizon verification. | Streaming real-time sensor WebSocket ingestion. |
| **Workspace Agent** | Path containment verification, atomic file writes with automatic rollback, process execution with tree kill (`taskkill`/`killpg`), secret redaction. | Reverse Engineering Anything (REA) provider integration. |
| **LLM Semantic Planning** | Local Ollama client (`qwen2.5:3b-instruct`), structured prompt contracts, bounded self-repair budget (max 2 retries), clarification request generation. | Parameter-efficient fine-tuned (PEFT/LoRA) specialized domain planner adapters. |

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 React 19 Frontend                                 |
|   (DataScienceStudio, SqlStudio, AgentStudio, ReportingStudio, AICoPilot, HUD)    |
+-----------------------------------------------------------------------------------+
                                          |
                                 IPC / Authenticated HTTP
                                          |
+-----------------------------------------------------------------------------------+
|                        Desktop Shell / Electron 41 Kernel                         |
|      (ContextIsolation, Single Instance Lock, Sandboxed Preload, Health Monitor)   |
+-----------------------------------------------------------------------------------+
                                          |
                                 FastAPI Backend (8011)
                                          |
+-----------------------------------------------------------------------------------+
|                            Semantic Planning Layer                                |
|   - Local Ollama Service (Qwen 2.5 Instruct)                                      |
|   - AnalysisSpecV3 Compiler & Structured Prompt Contracts                         |
|   - Finite Repair Budget Engine (Bounded self-repair)                             |
|   - Clarification & Ambiguity Resolver (Fail-closed on ambiguous schemas)         |
+-----------------------------------------------------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
+------------------------------------+        +------------------------------------+
|       Governed DS Engine           |        |      Governed Workspace Agent      |
| - Typed Analytical Operators       |        | - Fail-Closed Tool Registry        |
| - DAG Execution Engine             |        | - Path Containment Boundary        |
| - Numerical Oracles & Validators   |        | - Atomic Writes & Rollback Store   |
| - DuckDB OLAP & SQL Engine         |        | - Tree-Kill Process Runner         |
| - Cryptographic Provenance Store   |        | - Secret Redaction Filter          |
+------------------------------------+        +------------------------------------+
```

### Component Breakdown

- **`backend/ds_engine/`:** The trusted analytical kernel. Implements `AnalysisSpecV3`, typed data operators (`typed_operators.py`), dataset schema grounding (`dataset_grounding.py`), numerical verification oracles (`numerical_oracles.py`), and target leakage guards (`target_leakage.py`).
- **`backend/sql_engine/`:** Embeds DuckDB for zero-copy queries on Parquet and CSV files, manages relational catalogs, generates join plans, and enforces fail-closed SQL validation.
- **`backend/agent/`:** The local workspace agent. Enforces strict directory containment (`security.py`), validates file paths against workspace root, provisions atomic edits with rollback backups (`orchestrator.py`), and executes external processes with strict timeouts and tree-kill capability (`tools.py`).
- **`backend/intelligence/`:** Interacts with local Ollama daemons, formats system prompt contracts, parses structured JSON outputs, and isolates ambiguous intents.
- **`backend/persistence/`:** Records immutable dataset provenance (`provenance_store.py`), execution history, and session telemetry without cloud telemetry calls.
- **`src/`:** Modular React 19 UI styled with CSS variables and Lucide icons. Includes studios for data science exploration, DuckDB SQL execution, workspace agent management, and automated report generation.

---

## Trusted Validation & Execution

Traditional LLM analytics frameworks prompt a language model to produce executable code that directly touches production data, risking arbitrary code execution, target leakage, and hallucinated statistical conclusions. 

DataPilot OS eliminates this failure mode through a strict tripartite separation:

```
[ User Request ] 
       │
       ▼
1. Semantic Planner (LLM) ──► Produces JSON AnalysisSpecV3 (Zero Data Access)
       │
       ▼
2. Validation & Grounding ──► Verifies Column Existence, Type Contracts, Repair Budget
       │
       ▼
3. Deterministic Operators ──► Executes DuckDB/Pandas/Scikit-Learn & Validates via Oracles
```

1. **Zero Raw Execution:** The LLM does not generate executable strings for analytical tasks. It generates structured parameter specifications mapped to verified physical operators.
2. **Dataset Grounding:** `dataset_grounding.py` verifies all requested dimensions, metrics, and targets against the physical dataset schema. If a requested column does not exist or matches multiple candidates ambiguously, planning halts with HTTP 422 / `NEEDS_CLARIFICATION`.
3. **Target Leakage Interception:** When configuring supervised models, feature lists are cross-checked against the target variable and retrospective features. Any leakage causes hard rejection before model initialization.
4. **Result Verification Registry:** Once an operator runs, `result_verification_registry.py` and `numerical_oracles.py` assert that result shapes, distributions, and metrics satisfy mathematical invariants (e.g., probability sums to 1.0, metric values match oracle recomputations).

---

## Workspace Agent Security & Containment

The workspace agent operates under explicit security invariants:

- **Path Containment:** Revalidates every target path using realpath resolution. Access outside the active workspace directory raises `SecurityContainmentError`.
- **Protected Paths:** Files inside `.git`, `.venv`, `node_modules`, and environment configuration files (`.env`) cannot be modified or inspected by the agent.
- **Atomic Modifications & Rollback:** File writes are written to temporary staging files before replacement. Original file states are backed up in-memory (`TaskOrchestrator._backups`) and restored automatically if any subsequent step fails.
- **Process Execution Guardrails:** Commands run in isolated subprocesses with configurable timeouts, tree-kill process termination (`taskkill /F /T /PID` on Windows or POSIX process group signals), output size limits, and regex-based secret redaction (API tokens, private keys).
- **Default-Deny Approval Policies:** File writes, file deletions, and shell commands flag `requires_approval = True` by default in standard agent policy modes.

---

## Technology Stack

- **Desktop Shell:** Electron 41.7.1, Vite 8.1.5, React 19.2.7, Lucide Icons
- **Backend API:** Python 3.10+ (tested on Python 3.14.5), FastAPI, Uvicorn, Pydantic v2
- **Data Engine:** DuckDB 1.x (columnar SQL OLAP), Pandas, NumPy, PyArrow
- **Statistical & ML Computing:** Scikit-Learn, SciPy, Matplotlib, Seaborn
- **Local Model Serving:** Ollama (default model: `qwen2.5:3b-instruct` or `qwen2.5:7b-instruct`)
- **Testing & Quality:** Pytest 9.x, Oxlint

---

## Installation & Setup

### Prerequisites

- **Node.js:** v20.19.0 or higher
- **Python:** v3.10 or higher
- **Ollama:** [Ollama](https://ollama.com/) installed and running locally for semantic planning.

### 1. Clone the Repository

```bash
git clone https://github.com/Akshitkumar23/DataPilot-OS.git
cd DataPilot-OS
```

### 2. Configure Python Virtual Environment

```bash
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
npm install
```

### 4. Configure Local Ollama Model

Ensure Ollama is running and pull the default lightweight planner model:

```bash
ollama pull qwen2.5:3b-instruct
```

*Optional environment variables:*
- `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `OLLAMA_PLANNER_MODEL` (default: `qwen2.5:3b-instruct`)
- `OLLAMA_TIMEOUT` (default: `60`)
- `DATAPILOT_BACKEND_PORT` (default: `8011`)

---

## Running the Application

### Option A: One-Click Windows Launchers

- **Desktop App:** Double-click `DataPilot-OS.bat` (starts Ollama, backend engine, and packaged desktop app).
- **Silent Launcher:** Double-click `DataPilot-OS.vbs` (launches Electron without keeping a persistent console window open).

### Option B: Local Developer Mode

Run backend and frontend independently:

```bash
# Terminal 1: Backend Server (starts FastAPI on port 8011)
python server.py

# Terminal 2: Frontend & Electron Shell
npm run electron:dev
```

### Option C: Production Executable Build

```bash
# Builds React bundle, packages Python backend, and creates Electron binary
npm run dist
```
The packaged standalone application is output to `dist-electron/win-unpacked/`.

---

## Verification & Testing

DataPilot OS maintains a comprehensive test suite of 885 unit, integration, and contract tests in `tests/`.

### Run Core Contract & Architectural Tests

```bash
pytest tests/test_analysis_spec_v3_contract.py tests/test_canonical_query_response.py tests/test_engine_architectural_contract.py
```

### Run Workspace Agent Tests

```bash
pytest tests/test_local_filesystem_agent.py
```

### Run Full Offline Regression Suite

```bash
pytest -m "not live_ollama"
```

*Note: Tests marked `live_ollama` require a live Ollama daemon with the configured model pulled and responsive.*

### Run Static Linters & Frontend Checks

```bash
# Frontend build
npm run build

# Fast JavaScript/React linter
npm run lint

# Python AST syntax check
python -m compileall backend modules tools
```

---

## Current Boundaries & Limitations

1. **Classical Machine Learning Focus:** DataPilot OS supports deterministic classical statistical learning (Decision Trees, Random Forests, Linear/Logistic regression, K-Means, PCA). It does not embed deep learning frameworks (PyTorch, TensorFlow) to keep the packaging footprint lightweight.
2. **Memory Footprint:** In-memory Pandas operations are subject to host RAM limits. Large analytical queries should be executed via DuckDB SQL pushdown.
3. **No Distributed Execution:** DataPilot OS is designed as an edge/desktop workstation platform; it does not deploy to Apache Spark or Ray clusters.
4. **Reverse Engineering Anything (REA):** Extension points for external tool providers exist in `backend/agent/registry.py`, but REA tools are not bundled in this release.

---

## Future Architectural Directions

See [`FUTURE_WORK.md`](file:///FUTURE_WORK.md) for the complete post-1.0 architectural backlog:
- Direct Apache Arrow zero-copy memory mapping between DuckDB and UI components.
- Structural causal discovery and directed acyclic graph (DAG) estimation.
- Parameter-efficient fine-tuning (PEFT/LoRA) for compact (1.5B–3B) offline models on enterprise-specific taxonomies.

---

## License

This project is licensed under the [MIT License](LICENSE).
