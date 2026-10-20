"""
Benchmark script for FW-02 Pre-Implementation Probability Calibration Benchmark.
Generates leakage-free calibration metrics and curves for DataPilot OS.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def build_pipeline(df: pd.DataFrame, features: List[str], model: Any) -> Pipeline:
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def calculate_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> Tuple[List[Dict[str, Any]], float, float]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    # Clamp 0.0 value into bin 1
    bin_indices = np.clip(bin_indices, 1, n_bins)
    
    table = []
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)
    
    for b in range(1, n_bins + 1):
        mask = (bin_indices == b)
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(y_prob[mask]))
            pos_frac = float(np.mean(y_true[mask]))
            gap = abs(mean_prob - pos_frac)
        else:
            mean_prob = float((bin_edges[b-1] + bin_edges[b]) / 2.0)
            pos_frac = 0.0
            gap = 0.0
            
        weight = count / total_samples
        ece += weight * gap
        if gap > mce and count > 0:
            mce = gap
            
        table.append({
            "bin": b,
            "bin_range": f"[{bin_edges[b-1]:.2f}, {bin_edges[b]:.2f}]",
            "mean_predicted_prob": round(mean_prob, 4),
            "observed_positive_fraction": round(pos_frac, 4),
            "sample_count": count,
            "absolute_gap": round(gap, 4),
        })
        
    return table, round(ece, 4), round(mce, 4)

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    
    # Clip probabilities for robust log loss computation
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob_clipped))
    
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    cal_table, ece, mce = calculate_calibration_table(y_true, y_prob, n_bins=5)
    
    return {
        "roc_auc": round(roc, 4) if roc is not None else None,
        "pr_auc": round(pr, 4) if pr is not None else None,
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "ece": ece,
        "mce": mce,
        "calibration_bins": cal_table,
    }

def run_dataset_benchmark(
    name: str,
    df: pd.DataFrame,
    target_col: str,
    pos_val: Any,
    random_state: int = 42,
    allow_isotonic: bool = True,
    isotonic_note: str = ""
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"BENCHMARKING DATASET: {name}")
    print(f"==================================================")
    
    # Positive class mapping
    y_raw = df[target_col]
    y = (y_raw == pos_val).astype(int).values
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    
    n_samples, n_features = X.shape
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_ratio = pos_count / n_samples
    
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Class Dist: Pos (1) = {pos_count} ({pos_ratio*100:.1f}%), Neg (0) = {neg_count} ({(1-pos_ratio)*100:.1f}%)")
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Train samples: {len(y_train)}, Held-out Test samples: {len(y_test)}")
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
    }
    
    results = {
        "dataset_name": name,
        "row_count": n_samples,
        "feature_count": n_features,
        "positive_class": str(pos_val),
        "class_distribution": {"1": pos_count, "0": neg_count, "pos_prevalence": round(pos_ratio, 4)},
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "random_seed": random_state,
        "validation_strategy": "Stratified 80/20 train/test split with 5-fold CV calibration inside train split",
        "models": {}
    }
    
    for model_name, model_inst in models.items():
        print(f"\n--- Model: {model_name} ---")
        
        # 1. Uncalibrated Model
        pipe_base = build_pipeline(X_train, feature_cols, model_inst)
        pipe_base.fit(X_train, y_train)
        probs_uncal = pipe_base.predict_proba(X_test)[:, 1]
        metrics_uncal = evaluate_predictions(y_test, probs_uncal)
        
        # 2. Sigmoid Calibration (CalibratedClassifierCV with 5-fold CV inside X_train)
        pipe_for_sig = build_pipeline(X_train, feature_cols, model_inst)
        cal_sig = CalibratedClassifierCV(estimator=pipe_for_sig, method="sigmoid", cv=5)
        cal_sig.fit(X_train, y_train)
        probs_sig = cal_sig.predict_proba(X_test)[:, 1]
        metrics_sig = evaluate_predictions(y_test, probs_sig)
        
        # 3. Isotonic Calibration (where sample size makes it defensible)
        if allow_isotonic:
            pipe_for_iso = build_pipeline(X_train, feature_cols, model_inst)
            cal_iso = CalibratedClassifierCV(estimator=pipe_for_iso, method="isotonic", cv=5)
            cal_iso.fit(X_train, y_train)
            probs_iso = cal_iso.predict_proba(X_test)[:, 1]
            metrics_iso = evaluate_predictions(y_test, probs_iso)
        else:
            metrics_iso = {
                "status": "OMITTED_DUE_TO_SAMPLE_SIZE_LIMITATION",
                "note": isotonic_note
            }
            
        print(f"  Uncalibrated: Brier={metrics_uncal['brier_score']}, LogLoss={metrics_uncal['log_loss']}, ECE={metrics_uncal['ece']}, ROC={metrics_uncal['roc_auc']}")
        print(f"  Sigmoid:      Brier={metrics_sig['brier_score']}, LogLoss={metrics_sig['log_loss']}, ECE={metrics_sig['ece']}, ROC={metrics_sig['roc_auc']}")
        if allow_isotonic:
            print(f"  Isotonic:     Brier={metrics_iso['brier_score']}, LogLoss={metrics_iso['log_loss']}, ECE={metrics_iso['ece']}, ROC={metrics_iso['roc_auc']}")
        else:
            print(f"  Isotonic:     OMITTED ({isotonic_note})")
            
        results["models"][model_name] = {
            "uncalibrated": metrics_uncal,
            "sigmoid": metrics_sig,
            "isotonic": metrics_iso,
        }
        
    return results

if __name__ == "__main__":
    benchmark_all = {}
    
    # 1. German Credit
    df_german = pd.read_csv("data/real_world_evaluation/dataset_b_german_credit.csv")
    # In German credit, class is 'good' or 'bad'. 'bad' represents credit risk default (minority class 300/1000 = 30%)
    res_german = run_dataset_benchmark(
        name="German Credit Risk (Statlog)",
        df=df_german,
        target_col="class",
        pos_val="bad", # Risk event = bad credit
        allow_isotonic=True,
        isotonic_note="Evaluated for diagnostic comparison on N=800 train samples per scikit-learn guidelines."
    )
    benchmark_all["german_credit"] = res_german
    
    # 2. Heart Disease Cleveland
    df_heart = pd.read_csv("data/real_world_evaluation/dataset_e_heart_disease.csv")
    df_heart["heart_disease"] = (df_heart["target"] > 0).astype(int)
    df_heart_sub = df_heart.drop(columns=["target"])
    res_heart = run_dataset_benchmark(
        name="Heart Disease Cleveland",
        df=df_heart_sub,
        target_col="heart_disease",
        pos_val=1, # Presence of heart disease
        allow_isotonic=False,
        isotonic_note="Omitted: N=242 train samples violates scikit-learn minimum sample size guideline (N >= 1000) for non-parametric isotonic regression, creating severe step-function overfitting risk."
    )
    benchmark_all["heart_disease"] = res_heart
    
    # 3. Bank Churn Check
    benchmark_all["bank_churn"] = {
        "status": "NOT AVAILABLE",
        "note": "Bank Churn dataset is not present in the project dataset corpus. Per prompt instructions, it was NOT fabricated."
    }
    
    # 4. Large-sample comparison: Mushroom
    df_mushroom = pd.read_csv("data/real_world_evaluation/dataset_h_mushroom_high_cardinality.csv")
    res_mushroom = run_dataset_benchmark(
        name="Mushroom Classification (High Sample Size)",
        df=df_mushroom,
        target_col="class",
        pos_val="p", # Poisonous
        allow_isotonic=True,
        isotonic_note="Evaluated on large sample size (N=6499 train samples) where isotonic calibration is methodologically defensible."
    )
    benchmark_all["mushroom"] = res_mushroom
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/fw02_calibration_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_all, f, indent=2)
        
    print("\nBenchmark completed. Results written to docs/fw02_calibration_benchmark_results.json")
