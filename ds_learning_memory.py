import os
import json
import time
import threading
from typing import Dict, List, Any, Optional

class AutonomousLearningMemory:
    """
    360-Degree Self-Learning Memory Engine for DataPilot OS.
    Remembers cleaning patterns, model hyperparameter tuning, user preferences,
    and historical dataset transformations to continuously train and optimize DataPilot OS.
    """

    def __init__(self, memory_file: Optional[str] = None):
        if memory_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            memory_file = os.path.join(base_dir, "ds_learning_memory.json")
        self.memory_file = memory_file
        self._lock = threading.RLock()
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        data = None
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        if not isinstance(data, dict):
            return {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_datasets_processed": 0,
                "total_operations_executed": 0,
                "learned_rules": {
                    "numeric_imputation": "mean_or_median_based_on_skewness",
                    "categorical_imputation": "mode_or_unknown",
                    "encoding_strategy": "one_hot_if_cardinality_under_20",
                    "outlier_strategy": "iqr_capping_1.5x",
                    "model_strategy": "random_forest_ensemble"
                },
                "history": []
            }

        # Auto-migrate legacy keys
        if "memory_history" in data:
            legacy_hist = data.pop("memory_history")
            if "history" not in data or not data["history"]:
                data["history"] = legacy_hist if isinstance(legacy_hist, list) else []
            elif isinstance(legacy_hist, list):
                data["history"].extend(legacy_hist)

        if "total_learned_tasks" in data:
            legacy_tasks = data.pop("total_learned_tasks")
            if "total_operations_executed" not in data or data["total_operations_executed"] == 0:
                data["total_operations_executed"] = legacy_tasks

        # Ensure expected default keys exist
        if "history" not in data:
            data["history"] = []
        if "total_operations_executed" not in data:
            data["total_operations_executed"] = 0
        if "total_datasets_processed" not in data:
            data["total_datasets_processed"] = 0
        if "learned_rules" not in data:
            data["learned_rules"] = {
                "numeric_imputation": "mean_or_median_based_on_skewness",
                "categorical_imputation": "mode_or_unknown",
                "encoding_strategy": "one_hot_if_cardinality_under_20",
                "outlier_strategy": "iqr_capping_1.5x",
                "model_strategy": "random_forest_ensemble"
            }

        return data

    def save_memory(self):
        with self._lock:
            try:
                dir_name = os.path.dirname(self.memory_file) or "."
                os.makedirs(dir_name, exist_ok=True)
                temp_file = self.memory_file + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, indent=2)
                os.replace(temp_file, self.memory_file)
            except Exception as e:
                print(f"Memory save error: {e}")

    def record_learning_event(self, dataset_name: str, operation: str, column: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None):
        """
        Records an analyst execution event into the AI memory bank to train future recommendations.
        Supports DB import operation logging, file dataset loading, data cleaning, and model training.
        """
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "operation": operation,
            "column": column,
            "metrics": metrics or {}
        }
        with self._lock:
            self.memory["history"].append(event)
            self.memory["total_operations_executed"] += 1
            op_lower = (operation or "").lower()
            if any(k in op_lower for k in ["import", "connect", "load_data", "load_dataset"]):
                self.memory["total_datasets_processed"] = self.memory.get("total_datasets_processed", 0) + 1
        self.save_memory()

    def get_memory_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_datasets_processed": self.memory.get("total_datasets_processed", 0),
                "total_operations_executed": self.memory.get("total_operations_executed", 0),
                "learned_rules_count": len(self.memory.get("learned_rules", {})),
                "history_entries": len(self.memory.get("history", []))
            }

# Global Instance
learning_memory = AutonomousLearningMemory()
