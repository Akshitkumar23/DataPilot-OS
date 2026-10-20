# DataPilot OS — Professional Data Scientist Decision Boundary Matrix

**Audit Date**: October 2026  
**Evaluation Scope**: Workflow Autonomy & Scientific Decision Boundaries across Phase 1–7 Capabilities  
**Methodological Authorities**: Scikit-Learn (v1.7+), Pandas (v2.2+), SciPy (v1.15+), Statsmodels (v0.14+), DuckDB (v1.1+)  
**Classification Schema**:
* **[A] AUTOMATICALLY JUSTIFIED**: Mathematically or statistically determined directly from schema, types, or data distribution without human ambiguity.
* **[B] AUTOMATICALLY EXECUTABLE AFTER EXPLICIT INTENT**: Fully governed execution once the analyst specifies the objective, target, or design role.
* **[C] REQUIRES CLARIFICATION / DOMAIN JUDGMENT**: Inherently underdetermined by data alone; requires business, clinical, or semantic policy input.
* **[D] NOT CURRENTLY SUPPORTED**: Outside the verified capability boundary of DataPilot OS (documented in `docs/FUTURE_WORK.md`).

---

## 1. Data Understanding & Profiling

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Numeric vs Categorical Dtype** | **[A]** | `DatasetProfiler.profile()` inspecting pandas dtypes | Pure type-theoretic property. Numeric columns (`int64`, `float64`) are separated from strings/objects automatically. |
| **Nominal vs Ordinal Classification** | **[C]** | Categorical candidates identified; ordering unverified | Data values (e.g. `[1, 2, 3]` or `["low", "med", "high"]`) cannot be proved ordinal without domain semantic specification. |
| **Temporal Detection** | **[A]** | `datetime_parse_ratio >= 0.8` on candidate strings | Columns parsing successfully as timestamps are identified autonomously without keyword matching. |
| **Entity / Group Identifier Detection** | **[A]** | `cardinality == row_count` or high-cardinality discrete IDs | High-cardinality nominal keys are flagged autonomously as entity/leakage risks. |
| **Measurement Units & Scale** | **[C]** | Parsed as numeric arrays | Numbers alone cannot reveal physical units (e.g. `kg/m^3` vs `psi`, `Celsius` vs `Fahrenheit`). Domain context required. |
| **Target Identification** | **[C]** | Ranked target candidates in `DatasetProfile` | In unsupervised or exploratory contexts, any column could be the outcome of interest. Analyst must declare the target. |
| **Business / Domain Meaning** | **[C]** | Semantic schema metadata | Column headers like `feat_1` or `x12` require domain knowledge to understand business implications. |

---

## 2. Data Quality & Cleaning

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Missingness Detection** | **[A]** | Exact null counts via `df.isnull().sum()` | Factually determined from DataFrame memory representation. |
| **Missingness Mechanism (MCAR / MAR / MNAR)** | **[C]** | Correlation of null indicators with observed features | Statistical tests can suggest dependency, but proving MNAR (missing not at random) requires unobserved domain context. |
| **Duplicate Row Detection** | **[A]** | `df.duplicated().sum()` | Exact byte/value equality across all feature columns. |
| **Outlier Detection (Statistical)** | **[A]** | IQR (1.5x / 3x), Z-score ($> 3\sigma$), Cook's distance | Mathematical properties of distributions can be computed autonomously. |
| **Outlier Deletion / Imputation Policy** | **[C]** | Governed cleaning transformations | Whether an extreme reading is sensor noise (invalid) or a rare genuine failure (critical signal) is a domain decision. |
| **Invalid / Physical Boundary Violations** | **[C]** | Schema constraint checking | Detecting negative ages, negative blood pressures, or speeds exceeding physical limits requires domain boundary rules. |
| **Placeholder Value Discovery (e.g. 999, -1, "?")** | **[B]** | Replaced during cleaning after explicit pattern specification | High frequency isolated spikes can be flagged, but treating `-1` as missing vs valid code requires explicit intent. |

---

## 3. Exploratory Data Analysis (EDA)

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Univariate Distribution Summary** | **[A]** | Mean, std, median, IQR, skewness, kurtosis in `profiler.py` | Direct mathematical moments computed autonomously without user input. |
| **Bivariate Scatter & Correlation Matrix** | **[A]** | Pearson, Spearman, Kendall matrices in `eda_engine.py` | Pairwise numerical relationships computed automatically on all numeric columns. |
| **Nonlinear Association Detection** | **[A]** | Spearman rank correlation vs Pearson comparison | Divergence between linear and monotonic rank correlation is computed autonomously. |
| **Multimodality & Distribution Shape** | **[A]** | Kernel density estimation / histogram binning | Distribution topology is visualized directly from empirical data. |
| **Subgroup Cross-Tabulations** | **[B]** | `crosstab` / `groupby` execution in typed operators | Executed autonomously once grouping dimension and metric are specified. |
| **Causal Feature Attribution** | **[D]** | **UNSUPPORTED** (Explicitly blocked) | Observational data cannot establish causality without explicit DAG / instrumental variable specifications. |

---

## 4. Inferential Statistics & Hypothesis Testing

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Welch vs Student t-Test Selection** | **[A]** | Default to Welch unequal variance (`equal_var=False`) | Scipy & modern statistical literature recommend Welch by default; does not assume homoscedasticity. |
| **Observational Independence vs Paired Design** | **[C]** | Diagnostic warning `UNVERIFIED_OBSERVATIONAL_INDEPENDENCE` | Physical row alignment in wide format cannot distinguish independent cohorts from paired measures. |
| **Parametric vs Non-Parametric Alternatives** | **[B]** | `welch_ttest`, `mann_whitney_u`, `wilcoxon_signed_rank` | Analyst selects test family explicitly; engine reports normality and skewness diagnostics. |
| **One-Way ANOVA vs Kruskal-Wallis** | **[B]** | Executed via typed operators with explicit columns | Multi-group mean vs rank sum differences evaluated autonomously once columns are designated. |
| **Effect Size Calculation (Cohen's d, $\eta^2$, Cramer's V)**| **[A]** | Standardized effect sizes computed automatically | Computed directly alongside p-values to prevent p-hacking and sample-size distortions. |
| **Multiple Testing Correction (FDR / Bonferroni)** | **[A]** | Benjamini-Hochberg FDR adjustment | When multiple hypotheses are tested, adjusted p-values are calculated deterministically. |

---

## 5. Supervised Machine Learning

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Task Determination (Regression vs Classification)** | **[A]** | Resolved from target dtype and cardinality ($\le 5$ unique $\rightarrow$ clf) | Inferred deterministically by `ml_engine.validate_target()`. |
| **Target Column Selection** | **[C]** | Target must be declared by analyst | Unsupervised datasets contain multiple candidate outcomes; analyst must declare the business target. |
| **Feature / Target Leakage Prevention** | **[A]** | Strict exclusion of target from $X$, prospective leakage signal check | Enforced deterministically in `ml_engine.validate_features()` and `feasibility_rules.py`. |
| **Imbalanced Classification Metric Selection** | **[A]** | If imbalance ratio $\ge 3:1$, auto-switch to Balanced Accuracy & PR-AUC | Accuracy is mathematically deceptive under heavy imbalance; engine promotes balanced accuracy automatically. |
| **Stratified Grouped Validation (D-01)** | **[B]** | `StratifiedGroupKFold` when `group_column` + classification | Enforces zero group overlap while stratifying class ratios across folds. |
| **Chronological TimeSeriesSplit (D-03)** | **[B]** | `TimeSeriesSplit` on ascending timestamps | Eliminates future-to-past look-ahead leakage; preprocessor fit solely on historical folds. |
| **Asymmetric Cost Matrix / Threshold Tuning** | **[C]** | Prompts for analyst policy in `evaluate_model` | Business cost of False Positives vs False Negatives cannot be inferred from raw data alone. |
| **Probability Calibration (Platt / Isotonic)** | **[B]** | Brier score reported; calibration available | Brier loss evaluated autonomously; re-calibration requires analyst model refinement intent. |

---

## 6. Time Series Analysis & Forecasting

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Timestamp Validation & Ordering** | **[A]** | `pd.to_datetime` and chronological sort | Mathematical requirement for temporal modeling; executed automatically. |
| **Stationarity Testing (ADF / KPSS)** | **[A]** | Augmented Dickey-Fuller and KPSS in `timeseries_engine.py` | Both parametric unit root and non-parametric stationarity evaluated autonomously. |
| **ACF / PACF Significant Lag Discovery** | **[A]** | Bartlett standard error 95% confidence bands | Significant autoregressive and moving average lags detected mathematically. |
| **Temporal Frequency & Grain Detection** | **[A]** | `pd.infer_freq` and interval distribution | Detects daily, monthly, irregular timestamps automatically. |
| **Irregular Temporal Grid Resolution** | **[C]** | `TIME_SERIES_GRAIN_CLARIFICATION` gate | When timestamps are irregularly spaced, aggregation frequency (e.g. mean vs sum) requires business intent. |
| **Forecast Horizon Selection** | **[B]** | Specified via intent or defaults to 1 seasonal period | Horizon length depends on the planning cycle of the organization. |
| **Automated Lag & Rolling Window Construction** | **[D]** | **UNSUPPORTED** (Explicitly deferred to Phase 2) | Tabular ML requires pre-computed feature tables; automated feature synthesis is deferred. |

---

## 7. Relational Data & DuckDB SQL Engine

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **In-Memory OLAP Query Execution** | **[A]** | Zero-copy Arrow / DuckDB SQL execution | Standard SQL queries executed directly with ACID isolation. |
| **Natural Language to SQL Translation** | **[B]** | Governed NL-to-SQL compiler with schema validation | Queries generated from natural language intent, validated against active tables. |
| **Multi-Table Relational Schema Resolution** | **[D]** | **UNSUPPORTED** (Single active table architecture) | Cross-table foreign key discovery across arbitrary external databases is deferred. |

---

## 8. Reporting & Governance

| Analytical Decision | Boundary | Implementation / Mechanism | Scientific / Methodological Rationale |
| :--- | :---: | :--- | :--- |
| **Factual Metric Synthesis** | **[A]** | Template-free deterministic extraction of test statistics | Values extracted directly from fitted pipeline / operator output. |
| **Reproducible Python Code Generation** | **[A]** | Generates runnable self-contained Python scripts | Emits exact code reproducing all preprocessing, splits, and calculations. |
| **Provenance Logging** | **[A]** | Immutable event logs in SQLite provenance store | Records dataset hashes, parameters, execution timing, and operators. |
| **Causal Claims & Strategic Business Advice** | **[C]** | Explicitly blocked / flagged as observational limitation | Predictive associations must not be reported as causal levers without experimental confirmation. |
