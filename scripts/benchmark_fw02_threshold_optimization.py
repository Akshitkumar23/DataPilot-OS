"""
Benchmark Script for FW-02 Pre-Implementation Cost-Sensitive Threshold Benchmark.
Evaluates cost-sensitive threshold optimization and probability calibration
across multiple datasets, models, and asymmetric cost matrices without data leakage.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Cost Scenarios Definition
COST_SCENARIOS = {
    "Scenario A": {"name": "Equal Error Cost (A)", "c_fp": 1, "c_fn": 1, "tau_star": 0.5000},
    "Scenario B": {"name": "FN More Expensive (B)", "c_fp": 1, "c_fn": 5, "tau_star": round(1.0 / (1.0 + 5.0), 4)},
    "Scenario C": {"name": "FP More Expensive (C)", "c_fp": 5, "c_fn": 1, "tau_star": round(5.0 / (5.0 + 1.0), 4)},
    "Scenario D": {"name": "Strong FN Penalty (D)", "c_fp": 1, "c_fn": 10, "tau_star": round(1.0 / (1.0 + 10.0), 4)},
}

THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)


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


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, c_fp: float, c_fn: float) -> Tuple[float, int, int, int, int]:
    # Confusion matrix
    # y_true in {0, 1}, y_pred in {0, 1}
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    cost = float(c_fp * fp + c_fn * fn)
    return cost, fp, fn, tp, tn


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float,
    c_fn: float,
    tau_star: float
) -> Tuple[float, float]:
    best_cost = float("inf")
    best_tau = 0.50
    candidates = []

    for tau in THRESHOLD_GRID:
        y_pred = (y_prob >= tau).astype(int)
        cost, _, _, _, _ = compute_cost(y_true, y_pred, c_fp, c_fn)
        if cost < best_cost:
            best_cost = cost
            candidates = [tau]
        elif cost == best_cost:
            candidates.append(tau)

    # Tie-breaking: candidate closest to theoretical tau_star, then median
    if len(candidates) == 1:
        best_tau = candidates[0]
    else:
        # Sort by distance to tau_star
        candidates.sort(key=lambda t: (abs(t - tau_star), abs(t - 0.5)))
        best_tau = candidates[0]

    return round(float(best_tau), 4), float(best_cost)


def calculate_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> Tuple[List[Dict[str, Any]], float, float]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
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
            mean_prob = float((bin_edges[b - 1] + bin_edges[b]) / 2.0)
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


def evaluate_test_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float,
    c_fn: float
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cost, fp, fn, tp, tn = compute_cost(y_true, y_pred, c_fp, c_fn)

    # Probabilistic metrics
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob_clipped))

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    _, ece, mce = calculate_calibration_table(y_true, y_prob, n_bins=5)

    return {
        "threshold": round(float(threshold), 4),
        "total_cost": round(cost, 2),
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc, 4) if roc is not None else None,
        "pr_auc": round(pr, 4) if pr is not None else None,
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
    }


def run_benchmark_for_dataset(
    dataset_id: str,
    name: str,
    df: pd.DataFrame,
    target_col: str,
    pos_val: Any,
    random_state: int = 42
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"BENCHMARKING DATASET: {name} ({dataset_id})")
    print(f"=======================================================")

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

    # Strict 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"Train samples: {len(y_train)}, Held-out Test samples: {len(y_test)}")

    model_configs = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
    }

    dataset_result = {
        "dataset_id": dataset_id,
        "dataset_name": name,
        "row_count": n_samples,
        "feature_count": n_features,
        "positive_class": str(pos_val),
        "class_distribution": {"positive": pos_count, "negative": neg_count, "prevalence": round(pos_ratio, 4)},
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "random_seed": random_state,
        "models": {},
        "raw_probabilities": {},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_key, model_inst in model_configs.items():
        print(f"\n--- Model: {model_key} ---")

        oof_uncal_probs = np.zeros(len(y_train))
        oof_cal_probs = np.zeros(len(y_train))

        fold_uncal_thresholds = {scen_id: [] for scen_id in COST_SCENARIOS}
        fold_cal_thresholds = {scen_id: [] for scen_id in COST_SCENARIOS}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_v, y_v = X_train.iloc[val_idx], y_train[val_idx]

            # 1. Base uncalibrated model
            pipe_uncal = build_pipeline(X_tr, feature_cols, model_inst)
            pipe_uncal.fit(X_tr, y_tr)
            p_uncal_v = pipe_uncal.predict_proba(X_v)[:, 1]
            oof_uncal_probs[val_idx] = p_uncal_v

            # 2. Calibrated model (5-fold CV internally inside X_tr to avoid any leakage into val_idx)
            pipe_for_cal = build_pipeline(X_tr, feature_cols, model_inst)
            cal_clf = CalibratedClassifierCV(estimator=pipe_for_cal, method="sigmoid", cv=5)
            cal_clf.fit(X_tr, y_tr)
            p_cal_v = cal_clf.predict_proba(X_v)[:, 1]
            oof_cal_probs[val_idx] = p_cal_v

            # Measure optimal threshold on this fold's validation set for threshold stability analysis
            for scen_id, scen_cfg in COST_SCENARIOS.items():
                c_fp = scen_cfg["c_fp"]
                c_fn = scen_cfg["c_fn"]
                t_star = scen_cfg["tau_star"]

                best_t_uncal, _ = find_optimal_threshold(y_v, p_uncal_v, c_fp, c_fn, t_star)
                best_t_cal, _ = find_optimal_threshold(y_v, p_cal_v, c_fp, c_fn, t_star)

                fold_uncal_thresholds[scen_id].append(best_t_uncal)
                fold_cal_thresholds[scen_id].append(best_t_cal)

        # Full training on entire 80% train split
        final_pipe_uncal = build_pipeline(X_train, feature_cols, model_inst)
        final_pipe_uncal.fit(X_train, y_train)
        test_probs_uncal = final_pipe_uncal.predict_proba(X_test)[:, 1]

        final_cal_clf = CalibratedClassifierCV(
            estimator=build_pipeline(X_train, feature_cols, model_inst),
            method="sigmoid",
            cv=5
        )
        final_cal_clf.fit(X_train, y_train)
        test_probs_cal = final_cal_clf.predict_proba(X_test)[:, 1]

        dataset_result["raw_probabilities"][model_key] = {
            "test_probs_uncal": test_probs_uncal.tolist(),
            "test_probs_cal": test_probs_cal.tolist(),
            "y_test": y_test.tolist(),
        }

        model_scenarios_result = {}

        for scen_id, scen_cfg in COST_SCENARIOS.items():
            c_fp = scen_cfg["c_fp"]
            c_fn = scen_cfg["c_fn"]
            tau_star = scen_cfg["tau_star"]

            # Threshold selected across pooled OOF train predictions
            tau_uncal_opt, oof_cost_uncal = find_optimal_threshold(y_train, oof_uncal_probs, c_fp, c_fn, tau_star)
            tau_cal_opt, oof_cost_cal = find_optimal_threshold(y_train, oof_cal_probs, c_fp, c_fn, tau_star)

            # Threshold stability across 5 folds
            uncal_fold_list = fold_uncal_thresholds[scen_id]
            cal_fold_list = fold_cal_thresholds[scen_id]

            uncal_stability = {
                "mean": round(float(np.mean(uncal_fold_list)), 4),
                "median": round(float(np.median(uncal_fold_list)), 4),
                "std": round(float(np.std(uncal_fold_list, ddof=1)), 4),
                "min": round(float(np.min(uncal_fold_list)), 4),
                "max": round(float(np.max(uncal_fold_list)), 4),
                "folds": uncal_fold_list,
            }

            cal_stability = {
                "mean": round(float(np.mean(cal_fold_list)), 4),
                "median": round(float(np.median(cal_fold_list)), 4),
                "std": round(float(np.std(cal_fold_list, ddof=1)), 4),
                "min": round(float(np.min(cal_fold_list)), 4),
                "max": round(float(np.max(cal_fold_list)), 4),
                "folds": cal_fold_list,
            }

            # Out-of-sample Test Set Evaluations:
            # 1. Default threshold (tau = 0.50) on uncalibrated probabilities
            eval_default = evaluate_test_predictions(y_test, test_probs_uncal, 0.50, c_fp, c_fn)

            # 2. Empirically optimized threshold on uncalibrated probabilities
            eval_uncal_opt = evaluate_test_predictions(y_test, test_probs_uncal, tau_uncal_opt, c_fp, c_fn)

            # 3. Default threshold (tau = 0.50) on calibrated probabilities
            eval_cal_default = evaluate_test_predictions(y_test, test_probs_cal, 0.50, c_fp, c_fn)

            # 4. Calibrated + optimized threshold on calibrated probabilities
            eval_cal_opt = evaluate_test_predictions(y_test, test_probs_cal, tau_cal_opt, c_fp, c_fn)

            # 5. Theoretical Bayes optimal threshold (tau_star) on calibrated probabilities
            eval_cal_theoretical = evaluate_test_predictions(y_test, test_probs_cal, tau_star, c_fp, c_fn)

            # Calculate cost reductions
            base_cost = eval_default["total_cost"]

            uncal_abs_red = round(base_cost - eval_uncal_opt["total_cost"], 2)
            uncal_pct_red = round((uncal_abs_red / base_cost * 100) if base_cost > 0 else 0.0, 2)

            cal_abs_red = round(base_cost - eval_cal_opt["total_cost"], 2)
            cal_pct_red = round((cal_abs_red / base_cost * 100) if base_cost > 0 else 0.0, 2)

            cal_theo_abs_red = round(base_cost - eval_cal_theoretical["total_cost"], 2)
            cal_theo_pct_red = round((cal_theo_abs_red / base_cost * 100) if base_cost > 0 else 0.0, 2)

            model_scenarios_result[scen_id] = {
                "scenario_name": scen_cfg["name"],
                "cost_fp": c_fp,
                "cost_fn": c_fn,
                "tau_star": tau_star,
                "thresholds": {
                    "default": 0.50,
                    "uncalibrated_optimal": tau_uncal_opt,
                    "calibrated_optimal": tau_cal_opt,
                    "theoretical_tau_star": tau_star,
                },
                "threshold_stability": {
                    "uncalibrated": uncal_stability,
                    "calibrated": cal_stability,
                },
                "evaluations": {
                    "default_uncalibrated": eval_default,
                    "default_calibrated": eval_cal_default,
                    "optimized_uncalibrated": eval_uncal_opt,
                    "calibrated_optimized": eval_cal_opt,
                    "calibrated_theoretical_tau_star": eval_cal_theoretical,
                },
                "cost_reduction_vs_default": {
                    "uncalibrated_optimized": {
                        "absolute_reduction": uncal_abs_red,
                        "percentage_reduction": uncal_pct_red,
                    },
                    "calibrated_optimized": {
                        "absolute_reduction": cal_abs_red,
                        "percentage_reduction": cal_pct_red,
                    },
                    "calibrated_theoretical": {
                        "absolute_reduction": cal_theo_abs_red,
                        "percentage_reduction": cal_theo_pct_red,
                    }
                }
            }

            print(f"  [{scen_id}] Default Cost: {base_cost} -> Uncal Opt Cost: {eval_uncal_opt['total_cost']} ({uncal_pct_red}%), Cal Opt Cost: {eval_cal_opt['total_cost']} ({cal_pct_red}%) | tau_opt: uncal={tau_uncal_opt}, cal={tau_cal_opt}, tau*={tau_star}")

        dataset_result["models"][model_key] = model_scenarios_result

    return dataset_result


def generate_benchmark_plots(all_results: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "logistic_regression": "#1f77b4",
        "random_forest": "#2ca02c",
        "decision_tree": "#d62728",
    }
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "decision_tree": "Decision Tree",
    }

    dataset_ids = ["german_credit", "heart_disease", "mushroom"]
    scenario_ids = ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]

    # =========================================================================
    # PLOT 1: cost_vs_threshold.png
    # Grid of datasets (rows) x scenarios (columns)
    # Shows test set cost vs threshold tau for uncalibrated vs calibrated
    # =========================================================================
    print("Generating cost_vs_threshold.png...")
    fig, axes = plt.subplots(len(dataset_ids), len(scenario_ids), figsize=(20, 14), sharex=True)
    fig.suptitle("FW-02 Cost-Sensitive Threshold Optimization: Test Set Total Cost vs. Decision Threshold (τ)", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, ds_id in enumerate(dataset_ids):
        ds_data = all_results[ds_id]
        ds_name = ds_data["dataset_name"]

        for col_idx, scen_id in enumerate(scenario_ids):
            ax = axes[row_idx, col_idx]
            scen_cfg = COST_SCENARIOS[scen_id]
            c_fp, c_fn, t_star = scen_cfg["c_fp"], scen_cfg["c_fn"], scen_cfg["tau_star"]

            for model_key in ["logistic_regression", "random_forest", "decision_tree"]:
                raw_prob = ds_data["raw_probabilities"][model_key]
                y_test = np.array(raw_prob["y_test"])
                p_cal = np.array(raw_prob["test_probs_cal"])

                costs = []
                for tau in THRESHOLD_GRID:
                    y_pred = (p_cal >= tau).astype(int)
                    c, _, _, _, _ = compute_cost(y_test, y_pred, c_fp, c_fn)
                    costs.append(c)

                model_res = ds_data["models"][model_key][scen_id]
                cal_opt_tau = model_res["thresholds"]["calibrated_optimal"]
                cal_opt_cost = model_res["evaluations"]["calibrated_optimized"]["total_cost"]

                ax.plot(THRESHOLD_GRID, costs, label=f"{model_labels[model_key]} (Cal)", color=colors[model_key], lw=2)
                ax.scatter([cal_opt_tau], [cal_opt_cost], color=colors[model_key], s=50, zorder=5)

            # Reference lines: default tau=0.50 and tau*
            ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)" if (row_idx==0 and col_idx==0) else "")
            ax.axvline(t_star, color="black", linestyle=":", alpha=0.8, label=f"Theoretical τ* ({t_star:.2f})" if (row_idx==0 and col_idx==0) else "")

            if row_idx == 0:
                ax.set_title(f"{scen_cfg['name']}\n$C_{{FP}}={c_fp}, C_{{FN}}={c_fn}$ (τ*={t_star:.2f})", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}\nTotal Cost", fontsize=11, fontweight="bold")
            if row_idx == len(dataset_ids) - 1:
                ax.set_xlabel("Decision Threshold (τ)", fontsize=11)

    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot1_path = os.path.join(output_dir, "cost_vs_threshold.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # =========================================================================
    # PLOT 2: fp_fn_vs_threshold.png
    # Demonstrates the fundamental trade-off between FP and FN on test set
    # =========================================================================
    print("Generating fp_fn_vs_threshold.png...")
    fig, axes = plt.subplots(len(dataset_ids), 3, figsize=(18, 12), sharex=True)
    fig.suptitle("Error Trade-Off Profile: False Positives (FP) vs. False Negatives (FN) across Threshold (τ)", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, ds_id in enumerate(dataset_ids):
        ds_data = all_results[ds_id]
        ds_name = ds_data["dataset_name"]

        for col_idx, model_key in enumerate(["logistic_regression", "random_forest", "decision_tree"]):
            ax = axes[row_idx, col_idx]
            raw_prob = ds_data["raw_probabilities"][model_key]
            y_test = np.array(raw_prob["y_test"])
            p_cal = np.array(raw_prob["test_probs_cal"])

            fps = []
            fns = []
            for tau in THRESHOLD_GRID:
                y_pred = (p_cal >= tau).astype(int)
                _, fp, fn, _, _ = compute_cost(y_test, y_pred, 1, 1)
                fps.append(fp)
                fns.append(fn)

            ax.plot(THRESHOLD_GRID, fps, label="False Positives (FP)", color="#d95f02", lw=2)
            ax.plot(THRESHOLD_GRID, fns, label="False Negatives (FN)", color="#7570b3", lw=2)
            ax.axvline(0.50, color="gray", linestyle="--", alpha=0.6, label="Default τ=0.50")

            if row_idx == 0:
                ax.set_title(f"{model_labels[model_key]}", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}\nError Counts", fontsize=11, fontweight="bold")
            if row_idx == len(dataset_ids) - 1:
                ax.set_xlabel("Decision Threshold (τ)", fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="center right", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot2_path = os.path.join(output_dir, "fp_fn_vs_threshold.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # =========================================================================
    # PLOT 3: default_vs_optimized_threshold.png
    # Bar chart showing test cost: Default vs Uncalibrated Opt vs Calibrated Opt
    # =========================================================================
    print("Generating default_vs_optimized_threshold.png...")
    fig, axes = plt.subplots(len(dataset_ids), len(scenario_ids), figsize=(20, 13))
    fig.suptitle("Cost Reduction: Default (τ=0.50) vs. Uncalibrated Tuned vs. Calibrated Tuned Thresholds", fontsize=16, fontweight="bold", y=0.98)

    bar_width = 0.25
    x_indices = np.arange(3)  # 3 models

    for row_idx, ds_id in enumerate(dataset_ids):
        ds_data = all_results[ds_id]
        ds_name = ds_data["dataset_name"]

        for col_idx, scen_id in enumerate(scenario_ids):
            ax = axes[row_idx, col_idx]
            scen_cfg = COST_SCENARIOS[scen_id]

            def_costs = []
            uncal_opt_costs = []
            cal_opt_costs = []

            for model_key in ["logistic_regression", "random_forest", "decision_tree"]:
                res = ds_data["models"][model_key][scen_id]["evaluations"]
                def_costs.append(res["default_uncalibrated"]["total_cost"])
                uncal_opt_costs.append(res["optimized_uncalibrated"]["total_cost"])
                cal_opt_costs.append(res["calibrated_optimized"]["total_cost"])

            rects1 = ax.bar(x_indices - bar_width, def_costs, bar_width, label="Default (τ=0.50)", color="#999999", alpha=0.85)
            rects2 = ax.bar(x_indices, uncal_opt_costs, bar_width, label="Uncalibrated Opt", color="#e7298a", alpha=0.85)
            rects3 = ax.bar(x_indices + bar_width, cal_opt_costs, bar_width, label="Calibrated Opt", color="#1b9e77", alpha=0.9)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(["LogReg", "RandFor", "DecTree"], fontsize=10)

            if row_idx == 0:
                ax.set_title(f"{scen_cfg['name']}\n$C_{{FP}}={scen_cfg['c_fp']}, C_{{FN}}={scen_cfg['c_fn']}$", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}\nTotal Test Cost", fontsize=11, fontweight="bold")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot3_path = os.path.join(output_dir, "default_vs_optimized_threshold.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # =========================================================================
    # PLOT 4: calibration_curves.png
    # Reliability curves (uncalibrated vs calibrated) across datasets and models
    # =========================================================================
    print("Generating calibration_curves.png...")
    fig, axes = plt.subplots(len(dataset_ids), 3, figsize=(18, 13))
    fig.suptitle("Out-of-Sample Calibration Curves (Reliability Diagrams) on Pristine Test Set", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, ds_id in enumerate(dataset_ids):
        ds_data = all_results[ds_id]
        ds_name = ds_data["dataset_name"]

        for col_idx, model_key in enumerate(["logistic_regression", "random_forest", "decision_tree"]):
            ax = axes[row_idx, col_idx]
            raw_prob = ds_data["raw_probabilities"][model_key]
            y_test = np.array(raw_prob["y_test"])
            p_uncal = np.array(raw_prob["test_probs_uncal"])
            p_cal = np.array(raw_prob["test_probs_cal"])

            # Compute calibration curve
            frac_pos_uncal, mean_p_uncal = calibration_curve(y_test, p_uncal, n_bins=5, strategy="uniform")
            frac_pos_cal, mean_p_cal = calibration_curve(y_test, p_cal, n_bins=5, strategy="uniform")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
            ax.plot(mean_p_uncal, frac_pos_uncal, "s-", color="#e41a1c", lw=2, label="Uncalibrated")
            ax.plot(mean_p_cal, frac_pos_cal, "o-", color="#377eb8", lw=2, label="Sigmoid Calibrated")

            # Extract metrics from Scenario A (or model-level)
            uncal_metrics = ds_data["models"][model_key]["Scenario A"]["evaluations"]["default_uncalibrated"]
            cal_metrics = ds_data["models"][model_key]["Scenario A"]["evaluations"]["default_calibrated"]

            stats_text = (
                f"Uncal: Brier={uncal_metrics['brier_score']:.3f}, ECE={uncal_metrics['ece']:.3f}\n"
                f"Cal:     Brier={cal_metrics['brier_score']:.3f}, ECE={cal_metrics['ece']:.3f}"
            )
            ax.text(0.05, 0.78, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

            if row_idx == 0:
                ax.set_title(f"{model_labels[model_key]}", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}\nFraction of Positives", fontsize=11, fontweight="bold")
            if row_idx == len(dataset_ids) - 1:
                ax.set_xlabel("Mean Predicted Probability", fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right", fontsize=9)

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot4_path = os.path.join(output_dir, "calibration_curves.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

    # =========================================================================
    # PLOT 5: threshold_distribution_cv.png
    # 5-fold CV threshold stability across models, scenarios, datasets
    # =========================================================================
    print("Generating threshold_distribution_cv.png...")
    fig, axes = plt.subplots(len(dataset_ids), len(scenario_ids), figsize=(20, 13), sharey=True)
    fig.suptitle("Cross-Validation Threshold Stability: 5-Fold Empirical Optimal Threshold Distributions (τ)", fontsize=16, fontweight="bold", y=0.98)

    for row_idx, ds_id in enumerate(dataset_ids):
        ds_data = all_results[ds_id]
        ds_name = ds_data["dataset_name"]

        for col_idx, scen_id in enumerate(scenario_ids):
            ax = axes[row_idx, col_idx]
            scen_cfg = COST_SCENARIOS[scen_id]
            t_star = scen_cfg["tau_star"]

            # Collect fold thresholds for 3 models: uncalibrated and calibrated
            model_keys = ["logistic_regression", "random_forest", "decision_tree"]
            positions = []
            box_data = []
            box_colors = []

            for m_idx, m_key in enumerate(model_keys):
                st_data = ds_data["models"][m_key][scen_id]["threshold_stability"]
                uncal_folds = st_data["uncalibrated"]["folds"]
                cal_folds = st_data["calibrated"]["folds"]

                pos_uncal = m_idx * 3 + 1
                pos_cal = m_idx * 3 + 2

                positions.extend([pos_uncal, pos_cal])
                box_data.extend([uncal_folds, cal_folds])
                box_colors.extend(["#fc8d62", "#66c2a5"])

                # Plot scatter dots for each fold
                ax.scatter([pos_uncal] * len(uncal_folds), uncal_folds, color="#d95f02", alpha=0.7, s=25, zorder=4)
                ax.scatter([pos_cal] * len(cal_folds), cal_folds, color="#1b9e77", alpha=0.7, s=25, zorder=4)

            bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                            boxprops=dict(alpha=0.6), medianprops=dict(color="black", lw=1.5))
            for patch, col in zip(bp['boxes'], box_colors):
                patch.set_facecolor(col)

            # Theoretical tau_star line and default 0.50 line
            ax.axhline(t_star, color="black", linestyle=":", lw=1.5, alpha=0.85, label=f"τ* ({t_star:.2f})")
            ax.axhline(0.50, color="gray", linestyle="--", lw=1.0, alpha=0.6, label="Default (0.50)")

            ax.set_xticks([1.5, 4.5, 7.5])
            ax.set_xticklabels(["LogReg", "RandFor", "DecTree"], fontsize=10)
            ax.set_ylim([-0.05, 1.05])

            if row_idx == 0:
                ax.set_title(f"{scen_cfg['name']}\n$C_{{FP}}={scen_cfg['c_fp']}, C_{{FN}}={scen_cfg['c_fn']}$ (τ*={t_star:.2f})", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{ds_name}\nOptimal Threshold (τ)", fontsize=11, fontweight="bold")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot5_path = os.path.join(output_dir, "threshold_distribution_cv.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved: {plot5_path}")


def main():
    print("=" * 70)
    print("STARTING FW-02 COST-SENSITIVE THRESHOLD BENCHMARK ENGINE")
    print("=" * 70)

    all_results = {}

    # 1. German Credit
    df_german = pd.read_csv("data/real_world_evaluation/dataset_b_german_credit.csv")
    res_german = run_benchmark_for_dataset(
        dataset_id="german_credit",
        name="German Credit Risk (Statlog)",
        df=df_german,
        target_col="class",
        pos_val="bad",
        random_state=42,
    )
    all_results["german_credit"] = res_german

    # 2. Heart Disease Cleveland
    df_heart = pd.read_csv("data/real_world_evaluation/dataset_e_heart_disease.csv")
    df_heart["heart_disease"] = (df_heart["target"] > 0).astype(int)
    df_heart_sub = df_heart.drop(columns=["target"])
    res_heart = run_benchmark_for_dataset(
        dataset_id="heart_disease",
        name="Heart Disease Cleveland",
        df=df_heart_sub,
        target_col="heart_disease",
        pos_val=1,
        random_state=42,
    )
    all_results["heart_disease"] = res_heart

    # 3. Mushroom High Cardinality
    df_mushroom = pd.read_csv("data/real_world_evaluation/dataset_h_mushroom_high_cardinality.csv")
    res_mushroom = run_benchmark_for_dataset(
        dataset_id="mushroom",
        name="Mushroom Classification (Large Sample)",
        df=df_mushroom,
        target_col="class",
        pos_val="p",
        random_state=42,
    )
    all_results["mushroom"] = res_mushroom

    # Generate visual artifacts
    artifacts_dir = os.path.join("docs", "fw02_benchmark_artifacts")
    generate_benchmark_plots(all_results, artifacts_dir)

    # Save machine-readable JSON (strip raw probabilities to keep file clean and lightweight)
    json_export = {}
    for ds_id, ds_res in all_results.items():
        clean_res = dict(ds_res)
        clean_res.pop("raw_probabilities", None)
        json_export[ds_id] = clean_res

    json_export["metadata"] = {
        "benchmark_name": "FW-02 Pre-Implementation Cost-Sensitive Threshold Optimization Benchmark",
        "validation_strategy": "Strict 80% train / 20% untouched held-out test split. 5-fold Stratified CV inside train split for calibration and threshold selection.",
        "models_evaluated": ["logistic_regression", "random_forest", "decision_tree"],
        "cost_scenarios": COST_SCENARIOS,
        "artifacts_generated": [
            "cost_vs_threshold.png",
            "fp_fn_vs_threshold.png",
            "default_vs_optimized_threshold.png",
            "calibration_curves.png",
            "threshold_distribution_cv.png",
        ]
    }

    out_json_path = os.path.join("docs", "fw02_threshold_benchmark_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE!")
    print(f"Results JSON: {out_json_path}")
    print(f"Artifacts Directory: {artifacts_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
