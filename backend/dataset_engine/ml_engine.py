"""
DataPilot OS — Governed Machine Learning Engine (Phase 4).
100% Production-Grade, Schema-Agnostic, Dataset-Agnostic.
Zero Hardcoding: Zero domain assumptions, zero column heuristics, zero fake metrics.
"""

import os
import math
import time
import uuid
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
)

logger = logging.getLogger(__name__)


def sanitize_float(val: Any) -> Optional[float]:
    """Safely convert numeric values to JSON-compliant floats (NaN/Inf -> None)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def sanitize_obj(obj: Any) -> Any:
    """Recursively convert NumPy/Pandas types and non-finite floats to JSON-safe native Python types."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        return sanitize_float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [sanitize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v) for k, v in obj.items()}
    if pd.isna(obj):
        return None
    return str(obj)


class MLEngine:
    """
    Deterministic analytical Machine Learning engine.
    Supports Supervised Classification, Regression, AutoML Tournament,
    Feature Importance attribution, and Model Artifact inference.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base_dir = os.getenv("DATAPILOT_DATA_DIR")
            if not base_dir:
                app_data = os.getenv("APPDATA")
                base_dir = (
                    os.path.join(app_data, "DataPilot OS", "backend-data")
                    if app_data
                    else os.path.join(os.getcwd(), "backend-data")
                )
            self.models_dir = os.path.join(base_dir, "models")
        else:
            self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def validate_target(
        self,
        df: pd.DataFrame,
        target: str,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate target column strictly against the active DataFrame schema.
        Zero hardcoding: target semantics are evaluated from runtime dtypes and distributions.
        """
        if target not in df.columns:
            raise ValueError(f"COLUMN_NOT_FOUND: Target column '{target}' not found in active dataset.")

        series = df[target].dropna()
        n_valid = len(series)
        if n_valid < 5:
            raise ValueError(
                f"INSUFFICIENT_SAMPLES: Target column '{target}' has only {n_valid} non-null values. "
                "Supervised modeling requires at least 5 samples."
            )

        n_unique = int(series.nunique())
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))

        task_norm = (task or "auto").strip().lower()

        if task_norm in ("classification", "classify", "clf"):
            if n_unique < 2:
                raise ValueError(
                    f"SINGLE_CLASS_TARGET: Classification target '{target}' has only {n_unique} unique class. "
                    "At least 2 distinct classes are required."
                )
            resolved_task = "classification"

        elif task_norm in ("regression", "regress", "reg"):
            if not is_numeric:
                raise ValueError(
                    f"NON_NUMERIC_REGRESSION_TARGET: Regression target '{target}' has non-numeric dtype '{series.dtype}'. "
                    "Regression requires a continuous numeric target."
                )
            resolved_task = "regression"

        elif task_norm == "auto":
            # Deterministic inference based solely on dtype and distribution
            if not is_numeric:
                if n_unique < 2:
                    raise ValueError(
                        f"SINGLE_CLASS_TARGET: Target column '{target}' has only {n_unique} unique value. "
                        "Classification requires at least 2 distinct classes."
                    )
                resolved_task = "classification"
            else:
                # Numeric column
                if n_unique <= 5 and n_valid >= 10:
                    # Discrete low-cardinality values indicate classification labels
                    resolved_task = "classification"
                else:
                    resolved_task = "regression"
        else:
            raise ValueError(
                f"UNSUPPORTED_TASK: Task '{task}' is unsupported. Supported tasks: 'classification', 'regression', 'auto'."
            )

        return {
            "target": target,
            "resolved_task": resolved_task,
            "n_samples": n_valid,
            "n_unique": n_unique,
            "is_numeric": is_numeric,
        }

    def validate_features(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Validate feature list against dataset schema.
        Prevents target leakage, deduplicates, and rejects nonexistent columns.
        """
        if features is None or len(features) == 0:
            # Default to all columns except target
            candidate_features = [c for c in df.columns if c != target]
        else:
            # Target leakage guard: reject if user explicitly included target
            if target in features:
                raise ValueError(
                    f"TARGET_LEAKAGE: Target column '{target}' is present in the feature list. "
                    "Including the target as a feature causes severe data leakage."
                )

            # Check existence of every requested feature
            for f in features:
                if f not in df.columns:
                    raise ValueError(f"COLUMN_NOT_FOUND: Feature column '{f}' not found in active dataset.")

            candidate_features = features

        # Deduplicate preserving order
        seen = set()
        deduped = [f for f in candidate_features if not (f in seen or seen.add(f))]

        # Filter out constant columns with zero variance across the whole dataset
        usable_features = []
        for f in deduped:
            col_series = df[f].dropna()
            if len(col_series) > 0 and col_series.nunique() > 1:
                usable_features.append(f)

        if len(usable_features) == 0:
            raise ValueError(
                "NO_VALID_FEATURES: No usable predictive features found. "
                "Ensure at least one non-constant feature column is selected."
            )

        return usable_features

    def build_preprocessor(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> Tuple[ColumnTransformer, List[str], List[str]]:
        """
        Build an isolated scikit-learn ColumnTransformer.
        Applies median imputation + standard scaling for numeric features,
        and frequent imputation + one-hot encoding for categorical features.
        """
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
        categorical_features = [f for f in features if f not in numeric_features]

        transformers = []
        if numeric_features:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, numeric_features))

        if categorical_features:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, categorical_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        return preprocessor, numeric_features, categorical_features

    def get_supported_algorithms(self, task: str) -> List[Dict[str, str]]:
        """List verified algorithms available for a task."""
        if task == "classification":
            return [
                {"id": "random_forest", "name": "Random Forest Classifier", "family": "Ensemble"},
                {"id": "decision_tree", "name": "Decision Tree Classifier", "family": "Tree"},
                {"id": "logistic_regression", "name": "Logistic Regression", "family": "Linear"},
            ]
        elif task == "regression":
            return [
                {"id": "random_forest", "name": "Random Forest Regressor", "family": "Ensemble"},
                {"id": "decision_tree", "name": "Decision Tree Regressor", "family": "Tree"},
                {"id": "linear_regression", "name": "Linear Regression", "family": "Linear"},
            ]
        return []

    def create_estimator(self, task: str, algorithm: str, random_state: int = 42) -> Any:
        """Create a deterministic scikit-learn estimator instance."""
        alg = algorithm.strip().lower()

        if task == "classification":
            if alg in ("random_forest", "rf", "random_forest_classifier"):
                return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
            elif alg in ("decision_tree", "dt", "decision_tree_classifier"):
                return DecisionTreeClassifier(max_depth=6, random_state=random_state)
            elif alg in ("logistic_regression", "logistic", "lr"):
                return LogisticRegression(max_iter=1000, random_state=random_state)
            else:
                raise ValueError(
                    f"UNSUPPORTED_ALGORITHM: Unknown classification algorithm '{algorithm}'. "
                    "Supported: random_forest, decision_tree, logistic_regression."
                )

        elif task == "regression":
            if alg in ("random_forest", "rf", "random_forest_regressor"):
                return RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state)
            elif alg in ("decision_tree", "dt", "decision_tree_regressor"):
                return DecisionTreeRegressor(max_depth=6, random_state=random_state)
            elif alg in ("linear_regression", "linear", "lr", "ols"):
                return LinearRegression()
            else:
                raise ValueError(
                    f"UNSUPPORTED_ALGORITHM: Unknown regression algorithm '{algorithm}'. "
                    "Supported: random_forest, decision_tree, linear_regression."
                )
        else:
            raise ValueError(f"UNSUPPORTED_TASK: Task '{task}' is not supported.")

    def evaluate_model(
        self,
        task: str,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Compute real, verified evaluation metrics on held-out test data."""
        t0 = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time_ms = (time.time() - t0) * 1000.0

        if task == "classification":
            # Handle string conversion for classification labels
            classes = sorted(list(set(y_test.astype(str)).union(set(pd.Series(y_pred).astype(str)))))
            y_test_str = y_test.astype(str)
            y_pred_str = pd.Series(y_pred).astype(str)

            acc = float(accuracy_score(y_test_str, y_pred_str))
            p_macro = float(precision_score(y_test_str, y_pred_str, average="macro", zero_division=0))
            r_macro = float(recall_score(y_test_str, y_pred_str, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_test_str, y_pred_str, average="macro", zero_division=0))
            p_weighted = float(precision_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
            r_weighted = float(recall_score(y_test_str, y_pred_str, average="weighted", zero_division=0))
            f1_weighted = float(f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0))

            from sklearn.metrics import balanced_accuracy_score, average_precision_score, brier_score_loss
            bal_acc = float(balanced_accuracy_score(y_test_str, y_pred_str))

            cm = confusion_matrix(y_test_str, y_pred_str, labels=classes)
            matrix_list = [[int(x) for x in row] for row in cm]

            class_dist = {str(k): int(v) for k, v in y_test_str.value_counts().items()}
            class_counts = list(class_dist.values())
            imbalance_ratio = float(max(class_counts) / max(min(class_counts), 1)) if class_counts else 1.0
            is_imbalanced = bool(imbalance_ratio >= 3.0)

            # Advanced classification diagnostics: PR-AUC, Brier score, asymmetric costs
            pr_auc = None
            brier = None
            try:
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_test)
                    if len(classes) == 2 and probs.shape[1] == 2:
                        pos_label = classes[1]
                        y_test_bin = (y_test_str == pos_label).astype(int)
                        pr_auc = float(average_precision_score(y_test_bin, probs[:, 1]))
                        brier = float(brier_score_loss(y_test_bin, probs[:, 1]))
            except Exception:
                pass

            return {
                "primary_score": acc if not is_imbalanced else bal_acc,
                "primary_metric": "accuracy" if not is_imbalanced else "balanced_accuracy",
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "is_imbalanced": is_imbalanced,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "precision_macro": p_macro,
                "recall_macro": r_macro,
                "f1_macro": f1_macro,
                "precision_weighted": p_weighted,
                "recall_weighted": r_weighted,
                "f1_weighted": f1_weighted,
                "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
                "brier_score": round(brier, 4) if brier is not None else None,
                "asymmetric_cost_status": "UNSPECIFIED (Analyst policy input required for business threshold tuning)",
                "confusion_matrix": {
                    "classes": classes,
                    "matrix": matrix_list,
                },
                "class_distribution": class_dist,
                "prediction_time_ms": pred_time_ms,
            }

        else:
            # Regression metrics
            y_test_num = y_test.astype(float)
            y_pred_num = np.array(y_pred, dtype=float)

            r2 = float(r2_score(y_test_num, y_pred_num))
            mse = float(mean_squared_error(y_test_num, y_pred_num))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test_num, y_pred_num))

            residuals = y_test_num.values - y_pred_num
            res_series = pd.Series(residuals)
            res_skew = float(res_series.skew()) if len(residuals) > 2 else 0.0

            # 1. Non-parametric rank-based heteroscedasticity test
            import scipy.stats as sp_stats
            r_het, p_het = sp_stats.spearmanr(np.abs(residuals), y_pred_num) if len(residuals) > 3 else (0.0, 1.0)
            is_heteroscedastic = bool(pd.notna(p_het) and p_het < 0.05 and abs(r_het) > 0.25)

            # 2. Statsmodels official parametric diagnostics
            bp_stat = None
            bp_pvalue = None
            dw_stat = None
            jb_stat = None
            jb_pvalue = None
            try:
                import statsmodels.api as sm
                from statsmodels.stats.diagnostic import het_breuschpagan
                from statsmodels.stats.stattools import durbin_watson

                if len(residuals) >= 5:
                    X_aux = sm.add_constant(y_pred_num)
                    bp_out = het_breuschpagan(residuals, X_aux)
                    bp_stat = float(bp_out[0])
                    bp_pvalue = float(bp_out[1])
                    if bp_pvalue < 0.05:
                        is_heteroscedastic = True

                if len(residuals) >= 3:
                    dw_val = float(durbin_watson(residuals))
                    dw_stat = round(dw_val, 4)

                if len(residuals) >= 8:
                    jb_res = sp_stats.jarque_bera(residuals)
                    jb_stat = round(float(jb_res.statistic), 4)
                    jb_pvalue = round(float(jb_res.pvalue), 4)
            except Exception as sm_err:
                logger.debug(f"Statsmodels diagnostic exception: {sm_err}")

            # 3. Multicollinearity: VIF check on numeric features
            vif_summary = {}
            max_vif = None
            high_collinearity = False
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                num_features = X_test.select_dtypes(include=[np.number]).dropna()
                if num_features.shape[1] >= 2 and num_features.shape[0] >= 5:
                    for i, col_name in enumerate(num_features.columns[:10]):
                        v = float(variance_inflation_factor(num_features.values, i))
                        if np.isfinite(v):
                            vif_summary[col_name] = round(v, 2)
                    if vif_summary:
                        max_vif = max(vif_summary.values())
                        high_collinearity = bool(max_vif > 5.0)
            except Exception:
                pass

            # 4. Outlier & Influence check: Cook's distance
            influential_count = 0
            try:
                if len(residuals) >= 6:
                    import statsmodels.api as sm
                    from statsmodels.stats.outliers_influence import OLSInfluence
                    X_aux = sm.add_constant(y_pred_num)
                    ols_aux = sm.OLS(y_test_num.values, X_aux).fit()
                    infl = OLSInfluence(ols_aux)
                    cooks_d = infl.cooks_distance[0]
                    cutoff = 4.0 / len(residuals)
                    influential_count = int(np.sum(cooks_d > cutoff))
            except Exception:
                pass

            res_summary = {
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": round(res_skew, 4),
                "heteroscedasticity_detected": is_heteroscedastic,
                "heteroscedasticity_p_value": sanitize_float(round(float(p_het), 4)) if pd.notna(p_het) else None,
                "breusch_pagan_p_value": sanitize_float(round(float(bp_pvalue), 4)) if bp_pvalue is not None else None,
                "durbin_watson": dw_stat,
                "jarque_bera_p_value": jb_pvalue,
                "max_vif": max_vif,
                "vif_by_feature": vif_summary,
                "high_multicollinearity_detected": high_collinearity,
                "multicollinearity_note": "High VIF inflates coefficient variance (inferential instability) without necessarily invalidating out-of-fold prediction accuracy." if high_collinearity else "Acceptable collinearity bounds.",
                "influential_outlier_count": influential_count,
            }

            return {
                "primary_score": r2,
                "primary_metric": "r2",
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "residuals": res_summary,
                "prediction_time_ms": pred_time_ms,
            }

    def extract_feature_importances(
        self,
        pipeline: Pipeline,
        features: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract and normalize feature importance weights from fitted pipeline."""
        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")

        if model is None:
            return []

        raw_importances = None
        if hasattr(model, "feature_importances_"):
            raw_importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                raw_importances = np.mean(np.abs(coef), axis=0)
            else:
                raw_importances = np.abs(coef)

        if raw_importances is None or len(raw_importances) == 0:
            return []

        # Try to map back through preprocessor transformed column names
        try:
            transformed_names = preprocessor.get_feature_names_out()
            feature_weights = {}
            for name, imp in zip(transformed_names, raw_importances):
                # Clean prefix like 'num__feature' or 'cat__feature_val'
                orig_col = name.split("__", 1)[-1] if "__" in name else name
                base_col = orig_col.split("_", 1)[0] if orig_col not in features and "_" in orig_col else orig_col
                target_key = orig_col if orig_col in features else base_col if base_col in features else orig_col
                feature_weights[target_key] = feature_weights.get(target_key, 0.0) + float(imp)

            total_weight = sum(feature_weights.values()) or 1.0
            ranked = [
                {"feature": k, "importance": round(v / total_weight, 4)}
                for k, v in feature_weights.items()
            ]
            ranked.sort(key=lambda x: x["importance"], reverse=True)
            return ranked[:15]
        except Exception:
            # Fallback if names out not accessible
            total = float(np.sum(raw_importances)) or 1.0
            ranked = [
                {"feature": f, "importance": round(float(imp) / total, 4)}
                for f, imp in zip(features, raw_importances[:len(features)])
            ]
            ranked.sort(key=lambda x: x["importance"], reverse=True)
            return ranked[:15]

    def train_model(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        task: Optional[str] = "auto",
        algorithm: str = "auto",
        validation_strategy: str = "split",
        test_size: float = 0.2,
        cv_folds: int = 5,
        dataset_id: Optional[str] = None,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full supervised ML training and evaluation workflow.
        Returns validated metrics, confusion matrix / residuals, feature rankings, and model artifact ID.
        """
        t_start = time.time()

        # 1. Schema & Parameter Validation
        target_info = self.validate_target(df, target, task=task)
        resolved_task = target_info["resolved_task"]
        validated_features = self.validate_features(df, target, features=features)

        # 2. Prepare Clean Dataframe
        clean_df = df[[target] + validated_features].copy()
        if resolved_task == "regression":
            clean_df = clean_df[np.isfinite(clean_df[target])]
        else:
            clean_df = clean_df.dropna(subset=[target])

        if len(clean_df) < 5:
            raise ValueError("INSUFFICIENT_SAMPLES: Fewer than 5 valid training rows after target clean.")

        X = clean_df[validated_features].replace([np.inf, -np.inf], np.nan)
        y = clean_df[target]

        # Temporal Feature Leakage Guard (FW-03)
        temporal_indicators = ("__lag_", "__roll_", "__diff_")
        temporal_feats = [
            f for f in validated_features
            if any(ind in f for ind in temporal_indicators)
        ]
        if temporal_feats and validation_strategy != "time_series_split":
            raise ValueError(
                f"TEMPORAL_LEAKAGE_GUARD: Dataset contains temporally generated feature(s) {temporal_feats}. "
                "Random train/test split or random cross-validation introduces lookahead data leakage. "
                "Explicit 'time_series_split' validation strategy with a valid 'time_column' is strictly required."
            )

        # 3. Train/Test Split (Group, Temporal, or Stratified Split)
        if validation_strategy == "time_series_split":
            if not time_column or time_column not in df.columns:
                raise ValueError("TIME_COLUMN_REQUIRED: TimeSeriesSplit validation requires an explicit 'time_column' present in the dataset.")
            try:
                parsed_times = pd.to_datetime(df.loc[clean_df.index, time_column])
            except Exception as dt_err:
                raise ValueError(f"INVALID_TIME_COLUMN: Column '{time_column}' cannot be parsed as timestamps: {dt_err}")

            if parsed_times.nunique() < 3:
                raise ValueError(
                    f"INSUFFICIENT_TEMPORAL_GRAIN: Time column '{time_column}' has only {parsed_times.nunique()} unique timestamps. "
                    "TimeSeriesSplit requires at least 3 distinct time points."
                )

            clean_df["_temp_time_order"] = parsed_times
            clean_df = clean_df.sort_values(by="_temp_time_order", ascending=True)
            clean_df.drop(columns=["_temp_time_order"], inplace=True)
            X = clean_df[validated_features].replace([np.inf, -np.inf], np.nan)
            y = clean_df[target]

            n_samples = len(clean_df)
            n_test = max(1, int(round(n_samples * test_size)))
            if n_samples - n_test < 2:
                raise ValueError("INSUFFICIENT_SAMPLES: Train split would have fewer than 2 samples under chronological split.")
            n_train = n_samples - n_test
            X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
            y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        elif group_column and group_column in df.columns:
            group_vals = df.loc[clean_df.index, group_column]
            if resolved_task == "classification":
                class_group_counts = group_vals.groupby(clean_df[target]).nunique()
                impossible_classes = class_group_counts[class_group_counts < 2]
                if not impossible_classes.empty:
                    bad_class = impossible_classes.index[0]
                    bad_cnt = int(impossible_classes.iloc[0])
                    raise ValueError(
                        f"IMPOSSIBLE_GROUP_TOPOLOGY: Class '{bad_class}' appears in only {bad_cnt} group(s). "
                        "Stratified grouped validation requires each class to appear in at least 2 distinct groups."
                    )
                from sklearn.model_selection import StratifiedGroupKFold
                n_groups = int(group_vals.nunique())
                n_splits_split = max(2, min(int(round(1.0 / max(0.01, test_size))), n_groups))
                sgkf = StratifiedGroupKFold(n_splits=n_splits_split, shuffle=True, random_state=42)
                train_idx, test_idx = next(sgkf.split(X, y, groups=group_vals))
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                from sklearn.model_selection import GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
                train_idx, test_idx = next(gss.split(X, y, groups=group_vals))
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            stratify_arg = None
            if resolved_task == "classification":
                # Stratify only if every class has at least 2 samples
                counts = y.value_counts()
                if (counts >= 2).all() and len(counts) > 1:
                    stratify_arg = y

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_arg,
            )

        preprocessor, num_cols, cat_cols = self.build_preprocessor(clean_df, validated_features)

        # 4. Handle Algorithm Selection & AutoML Tournament
        alg_clean = (algorithm or "auto").strip().lower()
        tournament_results = []

        if alg_clean in ("auto", "automl", "tournament", "compare"):
            supported = self.get_supported_algorithms(resolved_task)
            best_model_info = None
            best_score = -float("inf")

            for item in supported:
                alg_id = item["id"]
                t0 = time.time()
                cand_estimator = self.create_estimator(resolved_task, alg_id, random_state=42)
                cand_pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", cand_estimator),
                ])
                cand_pipe.fit(X_train, y_train)
                fit_time_ms = (time.time() - t0) * 1000.0

                eval_metrics = self.evaluate_model(resolved_task, cand_pipe, X_test, y_test)
                score = eval_metrics["primary_score"]

                cand_record = {
                    "algorithm_id": alg_id,
                    "algorithm_name": item["name"],
                    "family": item["family"],
                    "primary_score": score,
                    "primary_metric": eval_metrics["primary_metric"],
                    "fit_time_ms": fit_time_ms,
                    "metrics": eval_metrics,
                }
                tournament_results.append(cand_record)

                if score > best_score:
                    best_score = score
                    best_model_info = {
                        "algorithm_id": alg_id,
                        "algorithm_name": item["name"],
                        "pipeline": cand_pipe,
                        "eval_metrics": eval_metrics,
                        "fit_time_ms": fit_time_ms,
                    }

            # Sort tournament rankings
            tournament_results.sort(key=lambda x: x["primary_score"], reverse=True)
            for idx, r in enumerate(tournament_results, 1):
                r["rank"] = idx

            chosen_algorithm = best_model_info["algorithm_id"]
            chosen_name = best_model_info["algorithm_name"]
            fitted_pipeline = best_model_info["pipeline"]
            metrics = best_model_info["eval_metrics"]
            fit_time_ms = best_model_info["fit_time_ms"]

        else:
            # Explicit algorithm requested
            chosen_algorithm = alg_clean
            estimator = self.create_estimator(resolved_task, alg_clean, random_state=42)
            chosen_name = estimator.__class__.__name__

            fitted_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            t0 = time.time()
            fitted_pipeline.fit(X_train, y_train)
            fit_time_ms = (time.time() - t0) * 1000.0

            metrics = self.evaluate_model(resolved_task, fitted_pipeline, X_test, y_test)

        # 5. Optional Cross-Validation
        cv_scores_list = []
        cv_mean = None
        cv_std = None
        cv_performed = validation_strategy in ("cv", "group_cv", "stratified_group_cv", "time_series_split")
        cv_strategy_used = validation_strategy

        if cv_performed:
            scoring_metric = "f1_weighted" if resolved_task == "classification" else "r2"
            try:
                cv_folds_safe = min(cv_folds, len(y_train))
                if cv_folds_safe >= 2:
                    if validation_strategy == "time_series_split":
                        from sklearn.model_selection import TimeSeriesSplit
                        n_splits = max(2, min(cv_folds_safe, len(clean_df) - 1))
                        cv_splitter = TimeSeriesSplit(n_splits=n_splits)
                        cv_strategy_used = "time_series_split"
                        raw_cv = cross_val_score(fitted_pipeline, X, y, cv=cv_splitter, scoring=scoring_metric)
                    elif group_column and group_column in df.columns:
                        group_vals = df.loc[clean_df.index, group_column]
                        n_groups = int(group_vals.nunique())
                        n_splits = max(2, min(cv_folds_safe, n_groups))
                        if resolved_task == "classification":
                            class_group_counts = group_vals.groupby(clean_df[target]).nunique()
                            impossible_classes = class_group_counts[class_group_counts < 2]
                            if not impossible_classes.empty:
                                bad_class = impossible_classes.index[0]
                                bad_cnt = int(impossible_classes.iloc[0])
                                raise ValueError(
                                    f"IMPOSSIBLE_GROUP_TOPOLOGY: Class '{bad_class}' appears in only {bad_cnt} group(s). "
                                    "Stratified grouped validation requires each class to appear in at least 2 distinct groups."
                                )
                            from sklearn.model_selection import StratifiedGroupKFold
                            cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                            cv_strategy_used = "stratified_group_cv"
                            raw_cv = cross_val_score(fitted_pipeline, X, y, cv=cv_splitter, groups=group_vals, scoring=scoring_metric)
                        else:
                            from sklearn.model_selection import GroupKFold
                            cv_splitter = GroupKFold(n_splits=n_splits)
                            cv_strategy_used = "group_cv"
                            raw_cv = cross_val_score(fitted_pipeline, X, y, cv=cv_splitter, groups=group_vals, scoring=scoring_metric)
                    else:
                        cv_strategy_used = "cv"
                        raw_cv = cross_val_score(fitted_pipeline, X, y, cv=cv_folds_safe, scoring=scoring_metric)

                    cv_scores_list = [float(s) for s in raw_cv]
                    cv_mean = float(np.mean(raw_cv))
                    cv_std = float(np.std(raw_cv))
            except Exception as cv_err:
                if isinstance(cv_err, ValueError) and "IMPOSSIBLE_GROUP_TOPOLOGY" in str(cv_err):
                    raise cv_err
                logger.warning(f"Cross-validation error: {cv_err}")

        # 6. Feature Importances
        feature_importances = self.extract_feature_importances(fitted_pipeline, validated_features)

        # 7. Model Artifact Serialization
        model_id = f"model_{uuid.uuid4().hex[:10]}"
        model_filename = f"{model_id}.joblib"
        model_filepath = os.path.join(self.models_dir, model_filename)

        metadata = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target": target,
            "task": resolved_task,
            "algorithm": chosen_algorithm,
            "algorithm_name": chosen_name,
            "features": validated_features,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "group_column": group_column,
            "time_column": time_column,
            "validation_strategy": cv_strategy_used if cv_performed else validation_strategy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": metrics,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifact_file": model_filename,
        }

        try:
            joblib.dump({"pipeline": fitted_pipeline, "metadata": metadata}, model_filepath)
        except Exception as dump_err:
            logger.error(f"Failed to dump model artifact: {dump_err}")

        total_time_ms = (time.time() - t_start) * 1000.0

        result = {
            "model_id": model_id,
            "task": resolved_task,
            "algorithm": chosen_algorithm,
            "algorithm_name": chosen_name,
            "target": target,
            "features": validated_features,
            "group_column": group_column,
            "time_column": time_column,
            "sample_counts": {
                "total": len(clean_df),
                "train": len(X_train),
                "test": len(X_test),
            },
            "metrics": metrics,
            "feature_importances": feature_importances,
            "tournament": tournament_results,
            "cross_validation": {
                "performed": cv_performed,
                "strategy": cv_strategy_used if cv_performed else ("stratified_group_cv" if (group_column and group_column in df.columns and resolved_task == "classification") else ("group_cv" if (group_column and group_column in df.columns) else validation_strategy)),
                "group_column": group_column,
                "time_column": time_column,
                "scores": cv_scores_list,
                "mean": cv_mean,
                "std": cv_std,
            },
            "timing": {
                "fit_time_ms": fit_time_ms,
                "total_time_ms": total_time_ms,
            },
            "preprocessing": {
                "numeric_features": num_cols,
                "categorical_features": cat_cols,
            },
        }

        return sanitize_obj(result)

    def predict_with_model(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Run inference using a saved model pipeline.
        Validates incoming feature schema against stored training schema.
        """
        model_filepath = os.path.join(self.models_dir, f"{model_id}.joblib")
        if not os.path.exists(model_filepath):
            raise ValueError(f"MODEL_NOT_FOUND: Model artifact '{model_id}' does not exist.")

        loaded = joblib.load(model_filepath)
        pipeline = loaded["pipeline"]
        metadata = loaded["metadata"]
        required_features = metadata["features"]

        if isinstance(input_data, list):
            df_in = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_in = input_data
        else:
            raise ValueError("INVALID_INPUT_DATA: Input must be a DataFrame or list of row dicts.")

        # Check for missing required features
        missing_feats = [f for f in required_features if f not in df_in.columns]
        if missing_feats:
            raise ValueError(
                f"MODEL_SCHEMA_MISMATCH: Input data is missing required feature columns: {missing_feats}. "
                f"Required features are: {required_features}."
            )

        X_input = df_in[required_features].replace([np.inf, -np.inf], np.nan)
        t0 = time.time()
        preds = pipeline.predict(X_input)
        pred_time_ms = (time.time() - t0) * 1000.0

        if metadata["task"] == "classification":
            predictions_out = [str(p) for p in preds]
        else:
            predictions_out = [sanitize_float(p) for p in preds]

        return {
            "model_id": model_id,
            "target": metadata["target"],
            "task": metadata["task"],
            "algorithm": metadata["algorithm_name"],
            "prediction_count": len(predictions_out),
            "predictions": predictions_out,
            "prediction_time_ms": pred_time_ms,
        }

    def optimize_classification_threshold(
        self,
        df: pd.DataFrame,
        target: str,
        positive_class: Any,
        false_positive_cost: float,
        false_negative_cost: float,
        true_positive_cost: float = 0.0,
        true_negative_cost: float = 0.0,
        features: Optional[List[str]] = None,
        algorithm: str = "auto",
        calibration_policy: str = "auto",
        grid_resolution: int = 100,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        FW-02: Dynamic Asymmetric Cost-Matrix Optimization & Decision-Threshold Tuning.
        Minimizes expected business loss under analyst cost policy with zero data leakage.
        """
        t0 = time.time()
        warnings: List[Dict[str, str]] = []

        # 1. Target Schema & Binary Validation
        target_info = self.validate_target(df, target, task="classification")
        unique_classes = sorted(list(df[target].dropna().unique()))
        if len(unique_classes) != 2:
            raise ValueError(
                f"MULTICLASS_NOT_SUPPORTED: Cost-sensitive threshold optimization is strictly supported for binary classification. "
                f"Target '{target}' has {len(unique_classes)} unique classes: {unique_classes}."
            )

        # 2. Positive Class Validation
        matching_classes = [c for c in unique_classes if str(c) == str(positive_class)]
        if not matching_classes:
            raise ValueError(
                f"MISSING_POSITIVE_CLASS: Positive class '{positive_class}' not found in target '{target}'. "
                f"Valid classes are: {unique_classes}."
            )
        pos_label = matching_classes[0]
        neg_label = [c for c in unique_classes if c != pos_label][0]

        # 3. Cost Parameters Validation & Generalized Bayes Threshold
        try:
            c_fp = float(false_positive_cost)
            c_fn = float(false_negative_cost)
            c_tp = float(true_positive_cost)
            c_tn = float(true_negative_cost)
        except (ValueError, TypeError):
            raise ValueError("INVALID_COST_VALUES: Cost parameters must be valid numeric values.")

        if c_fp <= 0.0 or c_fn <= 0.0:
            raise ValueError("INVALID_COST_VALUES: False positive cost (C_FP) and false negative cost (C_FN) must be strictly positive (> 0).")

        denom = (c_fp - c_tn) + (c_fn - c_tp)
        if denom <= 0.0:
            raise ValueError(
                f"INVALID_COST_RATIONALITY: Cost denominator (C_FP - C_TN) + (C_FN - C_TP) must be strictly positive (> 0), "
                f"got {denom}. Misclassifications must cost strictly more than correct classifications."
            )

        bayes_tau = (c_fp - c_tn) / denom
        if not (0.0 < bayes_tau < 1.0):
            raise ValueError(
                f"DEGENERATE_BAYES_THRESHOLD: Theoretical Bayes threshold tau* = {bayes_tau:.4f} lies outside (0, 1). "
                "Costs imply trivial constant classification."
            )

        if c_fp == c_fn and c_tp == c_tn:
            warnings.append({
                "code": "EQUAL_COSTS_INFO",
                "message": "Symmetric costs detected (C_FP == C_FN and C_TP == C_TN). Default threshold tau=0.50 is already the theoretical Bayes optimum."
            })

        loss_ratio = round(c_fn / c_fp, 4)

        # 4. Feature Validation & Data Cleaning (Preserving Original DataFrame)
        validated_features = self.validate_features(df, target, features=features)
        clean_df = df[[target] + validated_features].dropna(subset=[target]).copy()

        y = (clean_df[target] == pos_label).astype(int).values
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        if pos_count < 10 or neg_count < 10:
            raise ValueError(
                f"INSUFFICIENT_SAMPLES_FOR_CALIBRATION: Target contains only {pos_count} positive and {neg_count} negative samples. "
                "At least 10 samples per class are required for cross-validated threshold optimization."
            )

        X = clean_df[validated_features].replace([np.inf, -np.inf], np.nan)

        # 5. Strict Train / Test Split (Zero Test Leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        preprocessor, num_cols, cat_cols = self.build_preprocessor(clean_df.loc[X_train.index], validated_features)

        # 6. Algorithm Selection & Calibration Routing
        alg_clean = (algorithm or "auto").strip().lower()
        if alg_clean in ("auto", "automl", "compare"):
            resolved_alg = "random_forest"
        elif alg_clean in ("random_forest", "rf", "random_forest_classifier"):
            resolved_alg = "random_forest"
        elif alg_clean in ("decision_tree", "dt", "decision_tree_classifier"):
            resolved_alg = "decision_tree"
        elif alg_clean in ("logistic_regression", "logistic", "lr"):
            resolved_alg = "logistic_regression"
        else:
            raise ValueError(
                f"UNSUPPORTED_ALGORITHM: Unknown classification algorithm '{algorithm}'. "
                "Supported: random_forest, decision_tree, logistic_regression."
            )

        base_estimator = self.create_estimator("classification", resolved_alg, random_state=random_state)
        base_pipeline = Pipeline([("preprocessor", preprocessor), ("model", base_estimator)])

        cal_clean = (calibration_policy or "auto").strip().lower()
        if cal_clean == "auto":
            if resolved_alg in ("decision_tree", "random_forest"):
                resolved_cal = "sigmoid"
            else:
                resolved_cal = "none"
        elif cal_clean == "none":
            resolved_cal = "none"
            if resolved_alg == "decision_tree":
                warnings.append({
                    "code": "DISCRETE_STEP_PROBABILITY_WARNING",
                    "message": "Optimizing thresholds on uncalibrated Decision Trees operates over discrete leaf plateaus and may fail to find true economic optima."
                })
        elif cal_clean == "sigmoid":
            resolved_cal = "sigmoid"
        elif cal_clean == "isotonic":
            if len(X_train) < 2000:
                raise ValueError(
                    f"ISOTONIC_SAMPLE_SIZE_VIOLATION: Isotonic calibration requires N >= 2000 to prevent non-parametric step overfitting "
                    f"(current training N={len(X_train)}). Use Sigmoid calibration instead."
                )
            resolved_cal = "isotonic"
        else:
            raise ValueError(
                f"UNSUPPORTED_CALIBRATION_POLICY: Unknown calibration policy '{calibration_policy}'. "
                "Supported: auto, sigmoid, isotonic, none."
            )

        # 7. Internal Cross-Validation on Training Split (OOF Predictions)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if resolved_cal in ("sigmoid", "isotonic"):
            cal_estimator = CalibratedClassifierCV(estimator=clone(base_pipeline), method=resolved_cal, cv=3)
            oof_probs = cross_val_predict(cal_estimator, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
        else:
            oof_probs = cross_val_predict(clone(base_pipeline), X_train, y_train, cv=skf, method="predict_proba")[:, 1]

        p_min = float(np.min(oof_probs))
        p_max = float(np.max(oof_probs))
        if np.isclose(p_min, p_max):
            raise ValueError("CONSTANT_PREDICTIONS_ERROR: The model makes constant predictions. Therefore, it is impossible to optimize the decision threshold.")

        # 8. Empirical Candidate Threshold Generation (scikit-learn TunedThresholdClassifierCV methodology)
        unique_probs = np.unique(oof_probs)
        if len(unique_probs) <= grid_resolution:
            candidates = list((unique_probs[:-1] + unique_probs[1:]) / 2.0)
        else:
            candidates = list(np.linspace(max(0.001, p_min), min(0.999, p_max), num=grid_resolution))

        candidates.append(float(bayes_tau))
        candidates.append(0.50)
        candidates = sorted(list(set(round(float(np.clip(t, 0.001, 0.999)), 6) for t in candidates)))

        def calc_cost(y_true_arr: np.ndarray, y_prob_arr: np.ndarray, tau_val: float) -> Tuple[float, int, int, int, int]:
            y_pred_arr = (y_prob_arr >= tau_val).astype(int)
            tp_c = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
            fp_c = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
            fn_c = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
            tn_c = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
            tot_c = (fp_c * c_fp) + (fn_c * c_fn) + (tp_c * c_tp) + (tn_c * c_tn)
            return float(tot_c), tn_c, fp_c, fn_c, tp_c

        # 9. Deterministic Threshold Optimization on OOF Training Data
        scored_candidates = []
        for t in candidates:
            c, tn, fp, fn, tp = calc_cost(y_train, oof_probs, t)
            scored_candidates.append((c, t))

        min_c = min(sc[0] for sc in scored_candidates)
        best_candidates = [sc[1] for sc in scored_candidates if np.isclose(sc[0], min_c, atol=1e-7)]

        # Deterministic Tie-Breaking Protocol:
        # 1. Minimum distance to generalized Bayes threshold
        # 2. Minimum distance to default 0.50
        # 3. Higher threshold (precision bias)
        best_candidates.sort(key=lambda t: (round(abs(t - bayes_tau), 6), round(abs(t - 0.50), 6), -t))
        opt_threshold = float(best_candidates[0])

        # 10. Fold-by-Fold Threshold Stability
        fold_thresholds = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            val_probs = oof_probs[val_idx]
            val_y = y_train[val_idx]
            fold_scored = []
            for t in candidates:
                fc, _, _, _, _ = calc_cost(val_y, val_probs, t)
                fold_scored.append((fc, t))
            fold_min = min(fs[0] for fs in fold_scored)
            fold_bests = [fs[1] for fs in fold_scored if np.isclose(fs[0], fold_min, atol=1e-7)]
            fold_bests.sort(key=lambda t: (round(abs(t - bayes_tau), 6), round(abs(t - 0.50), 6), -t))
            fold_thresholds.append(round(float(fold_bests[0]), 4))

        cv_stability = {
            "mean": round(float(np.mean(fold_thresholds)), 4),
            "median": round(float(np.median(fold_thresholds)), 4),
            "std": round(float(np.std(fold_thresholds)), 4),
            "min": round(float(np.min(fold_thresholds)), 4),
            "max": round(float(np.max(fold_thresholds)), 4),
            "folds": fold_thresholds,
        }

        # 11. Final Model Fit on Full 80% Training Split
        base_pipeline.fit(X_train, y_train)
        if resolved_cal in ("sigmoid", "isotonic"):
            final_calibrated_model = CalibratedClassifierCV(estimator=base_pipeline, method=resolved_cal, cv=cv_folds)
            final_calibrated_model.fit(X_train, y_train)
            inference_model = final_calibrated_model
        else:
            inference_model = base_pipeline

        # 12. Final Out-of-Sample Evaluation on Pristine Held-Out Test Set (Untouched)
        test_probs = inference_model.predict_proba(X_test)[:, 1]
        base_cost, b_tn, b_fp, b_fn, b_tp = calc_cost(y_test, test_probs, 0.50)
        opt_cost, o_tn, o_fp, o_fn, o_tp = calc_cost(y_test, test_probs, opt_threshold)

        abs_savings = round(float(base_cost - opt_cost), 4)
        pct_savings = round(float((base_cost - opt_cost) / base_cost * 100.0), 2) if base_cost > 0 else 0.0

        # Quality Metrics
        def compute_ece(y_true_arr: np.ndarray, y_prob_arr: np.ndarray, n_bins: int = 5) -> float:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_idx = np.digitize(y_prob_arr, bin_edges, right=True)
            bin_idx = np.clip(bin_idx, 1, n_bins)
            ece_val = 0.0
            n_tot = len(y_true_arr)
            for b in range(1, n_bins + 1):
                mask = (bin_idx == b)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    mean_p = float(np.mean(y_prob_arr[mask]))
                    pos_f = float(np.mean(y_true_arr[mask]))
                    ece_val += (cnt / n_tot) * abs(mean_p - pos_f)
            return round(float(ece_val), 4)

        roc_auc = round(float(roc_auc_score(y_test, test_probs)), 4) if len(np.unique(y_test)) > 1 else None
        pr_auc = round(float(average_precision_score(y_test, test_probs)), 4) if len(np.unique(y_test)) > 1 else None
        brier = round(float(brier_score_loss(y_test, test_probs)), 4)
        ece = compute_ece(y_test, test_probs)

        y_pred_base = (test_probs >= 0.50).astype(int)
        y_pred_opt = (test_probs >= opt_threshold).astype(int)

        f1_base = round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4)
        f1_opt = round(float(f1_score(y_test, y_pred_opt, zero_division=0)), 4)
        bal_acc_base = round(float(balanced_accuracy_score(y_test, y_pred_base)), 4)
        bal_acc_opt = round(float(balanced_accuracy_score(y_test, y_pred_opt)), 4)

        # 13. Cost Curve for Visualization (50 evaluation points)
        curve_sweep = np.linspace(max(0.001, p_min), min(0.999, p_max), 50)
        cost_curve = []
        for cp in curve_sweep:
            cc, c_tn, c_fp, c_fn, c_tp = calc_cost(y_test, test_probs, float(cp))
            cost_curve.append({
                "threshold": round(float(cp), 4),
                "cost": round(float(cc), 2),
                "fp": c_fp,
                "fn": c_fn
            })

        # 14. Persist Artifact
        model_id = f"threshold_opt_{uuid.uuid4().hex[:10]}"
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        meta_dict = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "target": target,
            "positive_class": str(pos_label),
            "negative_class": str(neg_label),
            "task": "classification",
            "algorithm": resolved_alg,
            "algorithm_name": resolved_alg.replace("_", " ").title(),
            "optimal_threshold": opt_threshold,
            "theoretical_bayes_threshold": round(bayes_tau, 4),
            "calibration_method": resolved_cal,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "features": validated_features,
        }
        joblib.dump({"model": inference_model, "metadata": meta_dict}, model_path)

        exec_time_ms = round((time.time() - t0) * 1000.0, 2)

        return sanitize_obj({
            "status": "OPTIMIZATION_SUCCESSFUL",
            "model_id": model_id,
            "target": target,
            "positive_class": str(pos_label),
            "negative_class": str(neg_label),
            "algorithm": resolved_alg,
            "cost_policy": {
                "false_positive_cost": c_fp,
                "false_negative_cost": c_fn,
                "true_positive_cost": c_tp,
                "true_negative_cost": c_tn,
                "loss_ratio": loss_ratio,
                "theoretical_bayes_threshold": round(bayes_tau, 4),
            },
            "calibration": {
                "applied": resolved_cal != "none",
                "method": resolved_cal,
                "cv_folds": cv_folds,
                "brier_score": brier,
                "ece": ece,
            },
            "threshold_selection": {
                "baseline_threshold": 0.50,
                "optimized_threshold": round(opt_threshold, 4),
                "stability": cv_stability,
            },
            "business_decision_quality": {
                "baseline_cost": round(base_cost, 2),
                "optimized_cost": round(opt_cost, 2),
                "absolute_cost_reduction": abs_savings,
                "percentage_cost_reduction": pct_savings,
                "cost_per_sample_baseline": round(float(base_cost / len(y_test)), 4),
                "cost_per_sample_optimized": round(float(opt_cost / len(y_test)), 4),
                "baseline_confusion_matrix": {
                    "tn": b_tn, "fp": b_fp, "fn": b_fn, "tp": b_tp,
                    "classes": [str(neg_label), str(pos_label)],
                    "matrix": [[b_tn, b_fp], [b_fn, b_tp]]
                },
                "optimized_confusion_matrix": {
                    "tn": o_tn, "fp": o_fp, "fn": o_fn, "tp": o_tp,
                    "classes": [str(neg_label), str(pos_label)],
                    "matrix": [[o_tn, o_fp], [o_fn, o_tp]]
                },
            },
            "model_quality": {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1_baseline": f1_base,
                "f1_optimized": f1_opt,
                "balanced_accuracy_baseline": bal_acc_base,
                "balanced_accuracy_optimized": bal_acc_opt,
            },
            "sample_counts": {
                "total_valid": len(clean_df),
                "train": len(X_train),
                "test": len(X_test),
                "positives_test": int(np.sum(y_test == 1)),
                "negatives_test": int(np.sum(y_test == 0)),
            },
            "cost_curve": cost_curve,
            "warnings": warnings,
            "execution_time_ms": exec_time_ms,
        })

    def list_models(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List persisted model artifacts and their metadata."""
        models = []
        if not os.path.exists(self.models_dir):
            return models

        for fname in os.listdir(self.models_dir):
            if fname.endswith(".joblib"):
                try:
                    fpath = os.path.join(self.models_dir, fname)
                    loaded = joblib.load(fpath)
                    meta = loaded.get("metadata", {})
                    if dataset_id is None or meta.get("dataset_id") == dataset_id:
                        models.append(meta)
                except Exception:
                    continue

        models.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return sanitize_obj(models)


# Global singleton instance
ml_engine = MLEngine()
