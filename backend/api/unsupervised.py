"""
DataPilot OS — Phase 5: Governed Unsupervised Learning API Router.
Provides structured endpoints for K-Means Clustering, PCA Dimensionality Reduction,
and deterministic Elbow Curve analysis.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from backend.core.dataset_registry import dataset_registry
from backend.ds_engine.unsupervised_engine import unsupervised_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/unsupervised", tags=["unsupervised"])


class KMeansRequest(BaseModel):
    dataset_id: str = Field(..., description="ID of the active dataset")
    features: List[str] = Field(..., min_length=1, description="Numerical features for clustering")
    n_clusters: int = Field(..., ge=2, description="Target cluster count k")
    scaling: Optional[str] = Field("standard", description="Feature scaling ('standard' or 'none')")


class PCARequest(BaseModel):
    dataset_id: str = Field(..., description="ID of the active dataset")
    features: List[str] = Field(..., min_length=2, description="Numerical features for PCA (minimum 2)")
    n_components: int = Field(..., ge=1, description="Number of principal components to extract")
    scaling: Optional[str] = Field("standard", description="Feature scaling ('standard' or 'none')")


class ElbowRequest(BaseModel):
    dataset_id: str = Field(..., description="ID of the active dataset")
    features: List[str] = Field(..., min_length=1, description="Numerical features to evaluate")
    max_k: Optional[int] = Field(8, ge=2, le=20, description="Maximum k for elbow search")
    scaling: Optional[str] = Field("standard", description="Feature scaling ('standard' or 'none')")


def _get_dataset(dataset_id: str):
    """Retrieve and validate dataframe from dataset registry."""
    if not dataset_id:
        raise HTTPException(status_code=400, detail={"error_code": "DATASET_ID_REQUIRED", "error_message": "dataset_id cannot be empty."})

    entry = dataset_registry.get(dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail={"error_code": "DATASET_NOT_FOUND", "error_message": f"Dataset '{dataset_id}' not found."})

    df = dataset_registry.get_dataframe(dataset_id)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail={"error_code": "DATASET_EMPTY", "error_message": f"Dataset '{dataset_id}' is empty or unavailable."})

    return df, entry.name or dataset_id


@router.post("/kmeans")
def kmeans_clustering(req: KMeansRequest):
    """
    Executes K-Means Clustering on active dataset numerical features.
    Returns cluster centroids, inertia, silhouette score, cluster sizes, and visualization.
    """
    df, dataset_name = _get_dataset(req.dataset_id)

    try:
        result = unsupervised_engine.run_kmeans(
            df=df,
            features=req.features,
            n_clusters=req.n_clusters,
            scaling=req.scaling or "standard",
            dataset_id=req.dataset_id,
            dataset_name=dataset_name,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "FAILED",
                "error_message": str(ve)
            }
        )


@router.post("/pca")
def pca_decomposition(req: PCARequest):
    """
    Executes Principal Component Analysis on active dataset numerical features.
    Returns scree plot, explained variance ratios, loadings, and coordinate projections.
    """
    df, dataset_name = _get_dataset(req.dataset_id)

    try:
        result = unsupervised_engine.run_pca(
            df=df,
            features=req.features,
            n_components=req.n_components,
            scaling=req.scaling or "standard",
            dataset_id=req.dataset_id,
            dataset_name=dataset_name,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "FAILED",
                "error_message": str(ve)
            }
        )


@router.post("/elbow")
def elbow_evaluation(req: ElbowRequest):
    """
    Calculates inertia and silhouette scores across a candidate range of k.
    Assists in data-driven cluster count determination without hardcoded heuristics.
    """
    df, dataset_name = _get_dataset(req.dataset_id)

    try:
        result = unsupervised_engine.compute_elbow(
            df=df,
            features=req.features,
            min_k=2,
            max_k=req.max_k or 8,
            scaling=req.scaling or "standard",
            dataset_id=req.dataset_id,
            dataset_name=dataset_name,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "FAILED",
                "error_message": str(ve)
            }
        )
