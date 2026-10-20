"""Dataset-grounded target-leakage assessment for supervised learning.

This module deliberately does not infer domain meanings from column names.  It
only measures whether a *selected* predictor almost deterministically encodes
the selected target on held-out folds.  The caller decides whether that
evidence is acceptable for a retrospective explanation or must block a
prospective prediction workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageSignal:
    feature: str
    target: str
    non_null_coverage: float
    mapping_purity: float
    held_out_balanced_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageAssessment:
    signals: List[LeakageSignal]
    assessment_scope: str = "classification_selected_features_only"

    @property
    def has_deterministic_signal(self) -> bool:
        return bool(self.signals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_scope": self.assessment_scope,
            "has_deterministic_signal": self.has_deterministic_signal,
            "signals": [signal.to_dict() for signal in self.signals],
        }


class TargetLeakageDetector:
    """Detect high-confidence classification leakage without model heuristics.

    A feature is interdictable only when all of the following are true:
    - it covers essentially all target-bearing rows;
    - each observed feature value maps to exactly one target class; and
    - a small, capacity-limited one-feature decision tree has near-perfect
      *out-of-fold balanced accuracy*.

    The held-out check prevents us from classifying arbitrary high-cardinality
    identifiers as proven leakage simply because every value is unique in the
    training data.
    """

    MIN_COVERAGE = 0.995
    MIN_PURITY = 0.999999
    MIN_HELD_OUT_BALANCED_ACCURACY = 0.999

    @classmethod
    def assess_classification(
        cls,
        df: pd.DataFrame,
        *,
        target_column: str,
        feature_columns: List[str],
    ) -> LeakageAssessment:
        if target_column not in df.columns:
            return LeakageAssessment(signals=[])

        working = df[[target_column] + [c for c in feature_columns if c in df.columns]].dropna(subset=[target_column])
        if len(working) < 20 or working[target_column].nunique(dropna=True) < 2:
            return LeakageAssessment(signals=[])

        signals: List[LeakageSignal] = []
        for feature in feature_columns:
            if feature not in working.columns or feature == target_column:
                continue
            candidate = cls._candidate_mapping_metrics(working, feature, target_column)
            if candidate is None:
                continue
            coverage, purity = candidate
            if coverage < cls.MIN_COVERAGE or purity < cls.MIN_PURITY:
                continue
            held_out_score = cls._held_out_balanced_accuracy(working, feature, target_column)
            if held_out_score >= cls.MIN_HELD_OUT_BALANCED_ACCURACY:
                signals.append(LeakageSignal(
                    feature=feature,
                    target=target_column,
                    non_null_coverage=round(coverage, 6),
                    mapping_purity=round(purity, 6),
                    held_out_balanced_accuracy=round(held_out_score, 6),
                ))
        return LeakageAssessment(signals=signals)

    @staticmethod
    def _candidate_mapping_metrics(df: pd.DataFrame, feature: str, target: str):
        observed = df[[feature, target]].dropna(subset=[feature])
        if observed.empty:
            return None
        coverage = len(observed) / len(df)
        # Weighted purity: for each observed feature value, use its dominant
        # target class, then aggregate over rows.  It is vectorized and works
        # for numeric and categorical values alike.
        counts = observed.groupby([feature, target], dropna=False).size()
        dominant = counts.groupby(level=0).max().sum()
        purity = float(dominant / len(observed))
        return coverage, purity

    @staticmethod
    def _held_out_balanced_accuracy(df: pd.DataFrame, feature: str, target: str) -> float:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.tree import DecisionTreeClassifier

        X = df[[feature]]
        y = df[target]
        min_class_size = int(y.value_counts().min())
        n_splits = min(5, min_class_size)
        if n_splits < 2:
            return 0.0
        # Keep the one-feature probe deliberately low capacity.  Its purpose is
        # only to establish an out-of-fold deterministic mapping, not to build
        # the user's predictive model.
        min_leaf = max(5, ceil(len(df) * 0.001))
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max(2, min(16, int(y.nunique()) * 2)),
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        if pd.api.types.is_numeric_dtype(X[feature]):
            preprocessing = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), [feature]),
            ])
        else:
            preprocessing = ColumnTransformer([
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), [feature]),
            ])
        pipeline = Pipeline([("preprocess", preprocessing), ("probe", tree)])
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        predictions = cross_val_predict(pipeline, X, y, cv=folds)
        return float(balanced_accuracy_score(y, predictions))
