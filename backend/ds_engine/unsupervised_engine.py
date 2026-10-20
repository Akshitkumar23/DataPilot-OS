"""
DataPilot OS — Phase 5: Real Unsupervised Learning Engine.
Integrates existing ClusteringOperator and PCADimensionalityReductionOperator
into a production-grade, governed analytical service with zero hardcoding,
zero mutation, JSON-safe serialization, and immutable provenance logging.
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.ds_engine.typed_operators import (
    ClusteringOperator,
    PCADimensionalityReductionOperator,
    OperatorOutput
)
from backend.ds_engine.intent_schema import AnalysisIntentV2, TaskFamily
from backend.persistence.provenance_store import provenance_store

logger = logging.getLogger(__name__)


def _sanitize_val(val: Any) -> Any:
    """Recursively convert NumPy scalars and float NaNs/Infs to JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.ndarray):
        return [_sanitize_val(x) for x in val.tolist()]
    if isinstance(val, list):
        return [_sanitize_val(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_val(v) for k, v in val.items()}
    return val


sanitize_obj = _sanitize_val


class UnsupervisedEngine:
    """
    Production-grade Unsupervised Learning Engine.
    Executes K-Means Clustering and Principal Component Analysis (PCA)
    via existing deterministic typed operators.
    """

    def __init__(self):
        self._clustering_operator = ClusteringOperator()
        self._pca_operator = PCADimensionalityReductionOperator()

    def run_kmeans(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_clusters: int,
        scaling: str = "standard",
        dataset_id: Optional[str] = "unnamed_dataset",
        dataset_name: Optional[str] = "Dataset",
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes K-Means Clustering using ClusteringOperator.
        Returns validated, JSON-safe cluster centroids, metrics, sizes, and plot.
        """
        start_time = time.time()

        # Strict validation
        if not features or len(features) < 2:
            raise ValueError("[INSUFFICIENT_FEATURES] At least 2 feature columns are required for clustering.")
        for f in features:
            if f not in df.columns:
                raise ValueError(f"[COLUMN_NOT_FOUND] Feature column '{f}' not found in dataset.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for f in features:
            if f not in num_cols:
                raise ValueError(f"[NON_NUMERIC_FEATURE] Feature column '{f}' must be numeric.")

        clean_df = df[features].dropna()
        if len(clean_df) < 3:
            raise ValueError("[INSUFFICIENT_DATA] Clustering requires at least 3 valid observations.")

        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool) or n_clusters < 2:
            raise ValueError(f"[INVALID_CLUSTER_COUNT] Number of clusters ({n_clusters}) must be at least 2.")
        if n_clusters >= len(clean_df):
            raise ValueError(f"[INVALID_CLUSTER_COUNT] Number of clusters ({n_clusters}) must be less than number of samples ({len(clean_df)}).")
        if scaling not in {"standard", "none"}:
            raise ValueError("[INVALID_SCALING] Scaling must be 'standard' or 'none'.")

        read_only_df = df.copy(deep=True)

        intent = AnalysisIntentV2(
            task_family=TaskFamily.CLUSTERING,
            subjects=list(features),
            arguments={
                "n_clusters": n_clusters,
                "k": n_clusters,
                "scaling": scaling
            }
        )

        res: OperatorOutput = self._clustering_operator.execute(read_only_df, intent)
        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        if res.status != "SUCCEEDED":
            error_code = res.error_code or "CLUSTERING_FAILED"
            error_message = res.error_message or "Clustering operation failed."

            if dataset_id:
                try:
                    provenance_store.record_event(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name or "Dataset",
                        operation="KMEANS_CLUSTERING",
                        capability="unsupervised_learning",
                        affected_columns=features,
                        parameters={"n_clusters": n_clusters, "scaling": scaling},
                        df_before=df,
                        df_after=df,
                        status="FAILED",
                        source={"type": "UNSUPERVISED_STUDIO", "tab": "clustering"},
                        error_message=f"[{error_code}] {error_message}"
                    )
                except Exception as prov_err:
                    logger.warning(f"Provenance record error: {prov_err}")

            raise ValueError(f"[{error_code}] {error_message}")

        # Extract structured outputs
        data_dict = res.data or {}
        centroids = data_dict.get("records", [])
        inertia = data_dict.get("inertia")
        silhouette_score = data_dict.get("silhouette_score")
        cluster_sizes = data_dict.get("cluster_sizes", {})

        # Sample preview of clustered observations (up to 50 rows, never mutating active df)
        clustered_df = res.dataframe
        assigned_sample = []
        if clustered_df is not None and not clustered_df.empty:
            preview_cols = [c for c in features if c in clustered_df.columns] + ["cluster_label"]
            assigned_sample = clustered_df[preview_cols].head(50).to_dict(orient="records")

        # Record success in provenance
        event_id = None
        if dataset_id:
            try:
                event = provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or "Dataset",
                    operation="KMEANS_CLUSTERING",
                    capability="unsupervised_learning",
                    affected_columns=features,
                    parameters={"n_clusters": n_clusters, "scaling": scaling, "features": features},
                    df_before=df,
                    df_after=df,  # Active dataset is strictly NOT mutated
                    status="SUCCESS",
                    source={"type": "UNSUPERVISED_STUDIO", "tab": "clustering"},
                    finding=res.written_finding or "",
                    code=res.reproducible_code or ""
                )
                event_id = getattr(event, "event_id", None)
            except Exception as prov_err:
                logger.warning(f"Provenance record error: {prov_err}")

        payload = {
            "status": "SUCCEEDED",
            "success": True,
            "algorithm": "kmeans",
            "operation": "kmeans",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "selected_features": features,
            "features": features,
            "n_clusters": n_clusters,
            "scaling": scaling,
            "inertia": inertia,
            "silhouette_score": silhouette_score,
            "cluster_sizes": cluster_sizes,
            "centroids": centroids,
            "cluster_centroids": centroids,
            "sample_assignments": assigned_sample,
            "cluster_sample": assigned_sample,
            "chart": res.plot_image_base64,
            "plot_image_base64": res.plot_image_base64,
            "finding": res.written_finding,
            "code": res.reproducible_code,
            "reproducible_code": res.reproducible_code,
            "execution_time_ms": execution_time_ms,
            "audit_event_id": event_id,
            "provenance_event_id": event_id
        }

        return _sanitize_val(payload)

    def run_pca(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_components: int,
        scaling: str = "standard",
        dataset_id: Optional[str] = "unnamed_dataset",
        dataset_name: Optional[str] = "Dataset"
    ) -> Dict[str, Any]:
        """
        Executes Principal Component Analysis using PCADimensionalityReductionOperator.
        Returns validated, JSON-safe explained variance, scree plot, loadings, and coordinates.
        """
        start_time = time.time()

        # Strict validation
        if not features or len(features) < 2:
            raise ValueError("[INSUFFICIENT_FEATURES] At least 2 feature columns are required for PCA.")
        for f in features:
            if f not in df.columns:
                raise ValueError(f"[COLUMN_NOT_FOUND] Feature column '{f}' not found in dataset.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for f in features:
            if f not in num_cols:
                raise ValueError(f"[NON_NUMERIC_FEATURE] Feature column '{f}' must be numeric.")

        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError(f"[INVALID_COMPONENT_COUNT] n_components must be at least 1.")
        if n_components > len(features):
            raise ValueError(f"[INVALID_COMPONENT_COUNT] n_components ({n_components}) cannot exceed the number of selected features ({len(features)}).")

        clean_df = df[features].dropna()
        if len(clean_df) < 3:
            raise ValueError("[INSUFFICIENT_DATA] PCA requires at least 3 valid observations.")
        if scaling not in {"standard", "none"}:
            raise ValueError("[INVALID_SCALING] Scaling must be 'standard' or 'none'.")

        read_only_df = df.copy(deep=True)

        intent = AnalysisIntentV2(
            task_family=TaskFamily.DIMENSIONALITY_REDUCTION,
            subjects=list(features),
            arguments={
                "n_components": n_components,
                "scaling": scaling
            }
        )

        res: OperatorOutput = self._pca_operator.execute(read_only_df, intent)
        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        if res.status != "SUCCEEDED":
            error_code = res.error_code or "PCA_FAILED"
            error_message = res.error_message or "PCA dimensionality reduction failed."

            if dataset_id:
                try:
                    provenance_store.record_event(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name or "Dataset",
                        operation="PCA_DIMENSIONALITY_REDUCTION",
                        capability="unsupervised_learning",
                        affected_columns=features,
                        parameters={"n_components": n_components, "scaling": scaling},
                        df_before=df,
                        df_after=df,
                        status="FAILED",
                        source={"type": "UNSUPERVISED_STUDIO", "tab": "pca"},
                        error_message=f"[{error_code}] {error_message}"
                    )
                except Exception as prov_err:
                    logger.warning(f"Provenance record error: {prov_err}")

            raise ValueError(f"[{error_code}] {error_message}")

        # Extract structured outputs
        data_dict = res.data or {}
        explained_variance_ratio = data_dict.get("explained_variance_ratio", [])
        cumulative_variance_ratio = data_dict.get("cumulative_variance_ratio", [])
        explained_variance = data_dict.get("explained_variance", [])
        singular_values = data_dict.get("singular_values", [])
        loadings = data_dict.get("feature_loadings", [])

        # Sample preview of transformed principal coordinates (up to 50 rows)
        transformed_sample = data_dict.get("projections", [])[:50]
        if not transformed_sample and res.dataframe is not None and not res.dataframe.empty:
            transformed_sample = res.dataframe.head(50).to_dict(orient="records")

        # Record success in provenance
        event_id = None
        if dataset_id:
            try:
                event = provenance_store.record_event(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name or "Dataset",
                    operation="PCA_DIMENSIONALITY_REDUCTION",
                    capability="unsupervised_learning",
                    affected_columns=features,
                    parameters={"n_components": n_components, "scaling": scaling, "features": features},
                    df_before=df,
                    df_after=df,  # Active dataset is strictly NOT mutated
                    status="SUCCESS",
                    source={"type": "UNSUPERVISED_STUDIO", "tab": "pca"},
                    finding=res.written_finding or "",
                    code=res.reproducible_code or ""
                )
                event_id = getattr(event, "event_id", None)
            except Exception as prov_err:
                logger.warning(f"Provenance record error: {prov_err}")

        payload = {
            "status": "SUCCEEDED",
            "success": True,
            "algorithm": "pca",
            "operation": "pca",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "selected_features": features,
            "features": features,
            "n_components": n_components,
            "scaling": scaling,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": cumulative_variance_ratio,
            "explained_variance": explained_variance,
            "singular_values": singular_values,
            "loadings": loadings,
            "feature_loadings": loadings,
            "transformed_sample": transformed_sample,
            "chart": res.plot_image_base64,
            "plot_image_base64": res.plot_image_base64,
            "finding": res.written_finding,
            "code": res.reproducible_code,
            "reproducible_code": res.reproducible_code,
            "execution_time_ms": execution_time_ms,
            "audit_event_id": event_id,
            "provenance_event_id": event_id
        }

        return _sanitize_val(payload)

    def compute_elbow(
        self,
        df: pd.DataFrame,
        features: List[str],
        min_k: int = 2,
        max_k: int = 8,
        scaling: str = "standard",
        dataset_id: Optional[str] = "unnamed_dataset",
        dataset_name: Optional[str] = "Dataset"
    ) -> Dict[str, Any]:
        """
        Computes deterministic inertia and silhouette curve across k in [min_k, max_k]
        by iteratively invoking the existing ClusteringOperator.
        """
        start_time = time.time()

        if not features or len(features) < 2:
            raise ValueError("[INSUFFICIENT_FEATURES] At least 2 feature columns are required for elbow evaluation.")
        if min_k < 2:
            raise ValueError("[INVALID_K_RANGE] min_k must be at least 2.")
        if min_k >= max_k:
            raise ValueError("[INVALID_K_RANGE] min_k must be strictly less than max_k.")

        for f in features:
            if f not in df.columns:
                raise ValueError(f"[COLUMN_NOT_FOUND] Feature column '{f}' not found in dataset.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for f in features:
            if f not in num_cols:
                raise ValueError(f"[NON_NUMERIC_FEATURE] Feature column '{f}' must be numeric.")

        clean_df = df[features].dropna()
        if len(clean_df) < 3:
            raise ValueError("[INSUFFICIENT_DATA] Elbow evaluation requires at least 3 valid observations.")

        if max_k >= len(clean_df):
            raise ValueError(f"[INSUFFICIENT_DATA] max_k ({max_k}) must be less than number of samples ({len(clean_df)}).")

        read_only_df = df.copy(deep=True)
        curve = []
        for k in range(min_k, max_k + 1):
            intent = AnalysisIntentV2(
                task_family=TaskFamily.CLUSTERING,
                subjects=list(features),
                arguments={"n_clusters": k, "scaling": scaling}
            )
            res = self._clustering_operator.execute(read_only_df, intent)
            if res.status == "SUCCEEDED" and res.data:
                curve.append({
                    "k": k,
                    "inertia": res.data.get("inertia"),
                    "silhouette_score": res.data.get("silhouette_score")
                })

        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        payload = {
            "status": "SUCCEEDED",
            "success": True,
            "operation": "elbow_curve",
            "dataset_id": dataset_id,
            "selected_features": features,
            "features": features,
            "scaling": scaling,
            "curve": curve,
            "execution_time_ms": execution_time_ms
        }

        return _sanitize_val(payload)


unsupervised_engine = UnsupervisedEngine()
