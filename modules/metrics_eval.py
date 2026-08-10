"""
metrics_eval.py - Metrics Evaluation Engine for DataPilot OS.

This module provides comprehensive metrics evaluation capabilities for classification,
regression, and clustering models. It also generates code snippets for formatted
evaluation reports.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsEvaluationEngine:
    """
    Engine for evaluating machine learning model metrics.
    
    Supports evaluation of Classification, Regression, and Clustering models.
    """

    def __init__(self) -> None:
        """Initialize the MetricsEvaluationEngine."""
        pass

    def evaluate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate classification metrics.
        
        Calculates Accuracy, Precision, Recall, Specificity, Sensitivity, F1-Score, 
        Balanced Accuracy, Matthews Correlation Coefficient (MCC), Cohen Kappa,
        Zero-One Loss, ROC-AUC, Log Loss, Brier Score, and PR-Curve data.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated targets as returned by a classifier.
            y_prob (Optional[np.ndarray], optional): Probabilities of the positive class. Needed for ROC-AUC and PR-Curve. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary containing the evaluated classification metrics.
        """
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                         balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
                                         precision_recall_curve, confusion_matrix, log_loss,
                                         cohen_kappa_score, zero_one_loss, brier_score_loss)
            
            logger.info("Evaluating classification metrics.")
            
            metrics: Dict[str, Any] = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "Recall (Sensitivity)": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
                "Zero-One Loss": zero_one_loss(y_true, y_pred)
            }
            
            # Calculate Specificity if binary classification
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                metrics["Specificity"] = specificity
            else:
                metrics["Specificity"] = "N/A (Multiclass)"

            if y_prob is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
                        metrics["Log Loss"] = log_loss(y_true, y_prob)
                        metrics["Brier Score"] = brier_score_loss(y_true, y_prob)
                        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
                        metrics["PR-Curve"] = {
                            "precision": precision.tolist(),
                            "recall": recall.tolist(),
                            "thresholds": thresholds.tolist()
                        }
                    else:
                        metrics["ROC-AUC"] = "N/A (Binary expected)"
                        metrics["Log Loss"] = log_loss(y_true, y_prob)
                        metrics["Brier Score"] = "N/A (Binary expected)"
                        metrics["PR-Curve"] = "N/A (Binary expected)"
                except Exception as e:
                    logger.warning("Could not calculate ROC-AUC or PR-Curve: %s", e)
                    metrics["ROC-AUC"] = None
                    metrics["Log Loss"] = None
                    metrics["Brier Score"] = None
                    metrics["PR-Curve"] = None
            
            logger.info("Classification metrics evaluation successful.")
            return metrics
        except ImportError as e:
            logger.error("Missing dependency for classification metrics: %s", e)
            raise
        except Exception as e:
            logger.error("Error evaluating classification metrics: %s", e)
            raise

    def evaluate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, num_features: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate regression metrics.
        
        Calculates MAE, MSE, RMSE, MAPE, R2 Score, Adjusted R2 Score, Explained Variance,
        Max Error, and Median Absolute Error.

        Args:
            y_true (np.ndarray): Ground truth (correct) target values.
            y_pred (np.ndarray): Estimated target values.
            num_features (Optional[int], optional): Number of predictors for Adjusted R2. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary containing the evaluated regression metrics.
        """
        try:
            from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                                         mean_absolute_percentage_error, explained_variance_score,
                                         max_error, median_absolute_error)
            
            logger.info("Evaluating regression metrics.")
            
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics: Dict[str, float] = {
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "MAPE": mean_absolute_percentage_error(y_true, y_pred),
                "R2 Score": r2,
                "Explained Variance": explained_variance_score(y_true, y_pred),
                "Max Error": max_error(y_true, y_pred),
                "Median Absolute Error": median_absolute_error(y_true, y_pred)
            }
            
            if num_features is not None:
                n = len(y_true)
                if n > num_features + 1:
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - num_features - 1)
                    metrics["Adjusted R2 Score"] = adj_r2
                else:
                    metrics["Adjusted R2 Score"] = float('nan')
            
            logger.info("Regression metrics evaluation successful.")
            return metrics
        except ImportError as e:
            logger.error("Missing dependency for regression metrics: %s", e)
            raise
        except Exception as e:
            logger.error("Error evaluating regression metrics: %s", e)
            raise

    def evaluate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering metrics.
        
        Calculates Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.

        Args:
            X (np.ndarray): A feature array.
            labels (np.ndarray): Index of the cluster each sample belongs to.

        Returns:
            Dict[str, float]: Dictionary containing the evaluated clustering metrics.
        """
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            
            logger.info("Evaluating clustering metrics.")
            
            if len(np.unique(labels)) < 2:
                logger.warning("Clustering metrics require at least 2 distinct clusters.")
                return {
                    "Silhouette Score": float('nan'),
                    "Davies-Bouldin Index": float('nan'),
                    "Calinski-Harabasz Index": float('nan')
                }

            metrics: Dict[str, float] = {
                "Silhouette Score": silhouette_score(X, labels),
                "Davies-Bouldin Index": davies_bouldin_score(X, labels),
                "Calinski-Harabasz Index": calinski_harabasz_score(X, labels)
            }
            
            logger.info("Clustering metrics evaluation successful.")
            return metrics
        except ImportError as e:
            logger.error("Missing dependency for clustering metrics: %s", e)
            raise
        except Exception as e:
            logger.error("Error evaluating clustering metrics: %s", e)
            raise

    def generate_metrics_report_code(self, metrics_type: str = "classification") -> str:
        """
        Generates executable Python snippet printing all metrics in a formatted evaluation table.

        Args:
            metrics_type (str, optional): Type of metrics report. Defaults to "classification".

        Returns:
            str: Python code snippet for printing the report.
        """
        logger.info("Generating metrics report code for type: %s", metrics_type)
        
        code = f'''def print_{metrics_type}_report(metrics_dict: dict):
    """
    Prints a formatted evaluation table for {metrics_type} metrics.
    """
    print("=" * 50)
    print(f"{{'{metrics_type.capitalize()} Metrics Evaluation Report':^50}}")
    print("=" * 50)
    print(f"{{'Metric':<30}} | {{'Value':<15}}")
    print("-" * 50)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{{key:<30}} | {{value:<15.4f}}")
        elif isinstance(value, str):
            print(f"{{key:<30}} | {{value:<15}}")
        elif isinstance(value, (dict, list)):
            print(f"{{key:<30}} | [Complex Data Not Shown]")
        else:
            print(f"{{key:<30}} | {{str(value):<15}}")
    print("=" * 50)

# Example usage:
# metrics = engine.evaluate_{metrics_type}_metrics(...)
# print_{metrics_type}_report(metrics)
'''
        return code

# Global Singleton Instance
metrics_engine = MetricsEvaluationEngine()
