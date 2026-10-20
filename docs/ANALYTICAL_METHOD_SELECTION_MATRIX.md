# DataPilot OS — Analytical Method Selection Matrix

This matrix maps incoming analytical problem structures, data profiles, and statistical assumptions to the mathematically correct analytical methods enforced by DataPilot OS.

**Authoritative Standards**: Scikit-Learn Estimator Guidelines, Statsmodels Diagnostic Frameworks, SciPy Statistical Distributions, ANSI SQL Standards.

---

## 1. Comprehensive Method Selection Matrix

| Problem Type | Target / Dependent Var | Features / Independent Vars | Data Characteristics & Assumptions | Primary Selected Method | Diagnostic Checks | Invalidation Trigger & Alternative |
|---|---|---|---|---|---|---|
| **Two-Group Comparison** | Numeric continuous | 1 Categorical (2 levels) | Normality holds ($p \ge 0.05$ Shapiro-Wilk) & Equal Variance ($p \ge 0.05$ Levene) | **Two-Sample Student's $t$-test** | Shapiro-Wilk, Levene test | Normality violated ($p < 0.05$) $\implies$ Route to **Mann-Whitney $U$ test** |
| **Two-Group Comparison (Skewed/Ordinal)** | Numeric skewed or ordinal | 1 Categorical (2 levels) | Non-normal distribution or small sample with outliers | **Mann-Whitney $U$ test** | Rank sum distribution, asymptotic $Z$ | Paired samples $\implies$ Route to **Wilcoxon Signed-Rank test** |
| **Multi-Group Comparison** | Numeric continuous | 1 Categorical ($k \ge 3$ levels) | Normality within groups & Homoscedasticity | **One-Way ANOVA ($F$-test)** | Residual normality, Levene test | Non-normal $\implies$ Route to **Kruskal-Wallis $H$-test** |
| **Categorical Association** | Categorical discrete | Categorical discrete | Contingency table counts; expected cells $\ge 5$ | **Pearson $\chi^2$ Test of Independence** | Expected frequency matrix | Cell count $< 5$ $\implies$ Recommend **Fisher's Exact Test** |
| **Continuous Linear Association** | Numeric continuous | Numeric continuous | Bivariate normality, linear relationship, no severe outliers | **Pearson Correlation ($r$)** | Scatter linearity, outlier count | Skewed / monotonic / ordinal $\implies$ Route to **Spearman Rank ($\rho$)** |
| **Continuous Monotonic Association** | Numeric monotonic/ranked | Numeric continuous/ordinal | Non-linear monotonic relationship or outlier presence | **Spearman Rank Correlation ($\rho$)** | Rank tie frequency | Non-monotonic relationship $\implies$ Non-linear mutual information |
| **Continuous Prediction (Linear)** | Continuous numeric | Continuous / Encoded categorical | Linear relationship, homoscedasticity, independent residuals | **Ordinary Least Squares (OLS) / Ridge Regression** | Spearman test on $\|e_i\|$ vs $\hat{y}_i$, residual skewness, VIF | Heteroscedasticity ($p < 0.05$) $\implies$ Log-transform target or alert for robust HC3 errors |
| **Continuous Prediction (Collinear)** | Continuous numeric | Highly collinear features ($VIF > 5.0$) | High dimensionality or strong correlation among predictors | **Ridge Regression ($L_2$) or Lasso ($L_1$)** | Cross-validated $\alpha$ tuning, coefficient shrinkage | Sparse feature importance required $\implies$ **Lasso ($L_1$)** |
| **Binary Classification (Balanced)** | Binary discrete (0/1) | Numeric / Encoded categorical | Linearly separable or smooth log-odds, balanced classes | **Logistic Regression / Random Forest** | ROC-AUC, Log-loss, F1-Score | Non-linear decision boundary $\implies$ **Random Forest Classifier** |
| **Binary Classification (Imbalanced)** | Binary discrete (0/1) | Numeric / Encoded categorical | Severe class ratio ($\ge 3:1$, e.g. fraud, churn) | **Class-Weighted Random Forest / Logistic Regression** | Balanced Accuracy, Confusion Matrix, PR-AUC | Raw accuracy misleading $\implies$ Prioritize **Balanced Accuracy** |
| **Multiclass Classification** | Discrete ($k \ge 3$ classes) | Numeric / Encoded categorical | Distinct class boundaries, mutually exclusive classes | **Multinomial Logistic / Random Forest Classifier** | Macro-averaged F1, Confusion Matrix | Extreme minority classes $\implies$ Macro/Balanced metric evaluation |
| **Unsupervised Grouping** | None (Unsupervised) | Multi-column numeric | Convex, isotropic clusters; uniform variance | **$K$-Means Clustering (Standardized)** | Silhouette Score, Inertia Elbow Curve | Non-convex / arbitrary cluster shapes $\implies$ Density-based methods (DBSCAN) |
| **Dimensionality Reduction** | None (Unsupervised) | High-dimensional numeric | Linear variance preservation; continuous variables | **Principal Component Analysis (PCA)** | Explained variance ratio, Scree plot, Loadings | Non-linear manifold $\implies$ Manifold learning (t-SNE / UMAP for visualization only) |
| **Time-Series Stationarity** | Single chronological series | Timestamp index + numeric | Equispaced observations, chronological ordering | **Augmented Dickey-Fuller (ADF) Test** | Test statistic vs critical values ($1\%, 5\%, 10\%$) | Non-stationary ($p \ge 0.05$) $\implies$ Enforce 1st-order differencing ($\Delta y_t$) |
| **Time-Series Forecasting** | Numeric continuous | Timestamp index + lag values | Stationary or difference-stationary series | **Autoregressive (AR) / Exponential Smoothing** | AIC/BIC, Residual ACF/PACF, Rolling Forward MAE | Autocorrelated residuals $\implies$ Adjust lag order / seasonal terms |
| **Relational Aggregation** | Relational columns | Grouping dimensions | Structured tabular data, exact key relationships | **Governed DuckDB SQL Engine** | Execution plan, row count preservation | Cartesian row explosion $\implies$ Strict join cardinality assertion |

---

## 2. Decision Logic Flowcharts

### 2.1 Bivariate Continuous vs Categorical Selection
```text
Is Dependent Variable Numeric Continuous?
  │
  ├── YES: Check Number of Factor Levels
  │     ├── 2 Levels:
  │     │     ├── Check Shapiro-Wilk Normality (Both Groups)
  │     │     │     ├── Both Normal (p >= 0.05) & Equal Variance -> Student's t-test
  │     │     │     └── Either Non-Normal (p < 0.05)            -> Mann-Whitney U test
  │     └── >= 3 Levels:
  │           ├── Normal & Equal Variance -> One-Way ANOVA
  │           └── Non-Normal              -> Kruskal-Wallis Test
  │
  └── NO (Categorical):
        └── Contingency Table Analysis -> Pearson Chi-Square Test
```

### 2.2 Supervised Learning Target Branching
```text
Evaluate Target Variable Distribution & Cardinality
  │
  ├── Unique Values == 1:
  │     └── REJECT: [SINGLE_CLASS_TARGET]
  │
  ├── Continuous Floating-Point / High Cardinality:
  │     ├── Check Sample Size >= 5
  │     ├── Fit Regressor (Ridge / OLS / Random Forest Regressor)
  │     └── Run Residual Diagnostics (Spearman Rank Test for Heteroscedasticity)
  │
  └── Discrete Binary / Multi-Class:
        ├── Calculate Imbalance Ratio (Majority / Minority)
        │     ├── Ratio < 3.0: Standard Stratified CV (Accuracy, F1, ROC-AUC)
        │     └── Ratio >= 3.0: Imbalance Protocol (Balanced Accuracy, PR-AUC, Confusion Matrix)
        └── Guard Against Target Leakage via Out-of-Fold Validation
```

---

## 3. Strict Operating Invariants
1. **Never substitute an algorithm silently**: If an OLS regression violates assumptions, report the diagnostics explicitly rather than quietly switching to a different loss function.
2. **Never evaluate models on training data alone**: All reported performance metrics must derive from out-of-fold or held-out test partitions.
3. **Never apply distance metrics to unscaled features**: Features submitted to K-Means or PCA must pass through standard scaling to ensure scale invariance.
