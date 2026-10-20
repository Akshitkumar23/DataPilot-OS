"""
Telemetry and Metrics Collector module for DataPilot OS.
Tracks query lifecycle statistics, semantic planning success, repair rates,
schema resolution accuracy, latency benchmarks, and error rates.
"""

import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TelemetryCollector:
    """
    Collects performance telemetry and quality metrics across all executed queries.
    """

    def __init__(self):
        self.total_queries = 0
        self.semantic_plan_successes = 0
        self.plan_validation_failures = 0
        self.repair_attempts = 0
        self.repair_successes = 0
        self.schema_ambiguities_detected = 0
        self.capability_matches = 0
        self.dynamic_code_generations = 0
        self.execution_failures = 0
        self.result_validation_failures = 0

        self.latencies: List[Dict[str, float]] = []

    def record_query(
        self,
        semantic_success: bool,
        val_failed: bool,
        repaired: bool,
        ambiguity: bool,
        capability_matched: bool,
        dynamic_code: bool,
        exec_failed: bool,
        result_val_failed: bool,
        latency_breakdown: Dict[str, float]
    ):
        self.total_queries += 1
        if semantic_success: self.semantic_plan_successes += 1
        if val_failed: self.plan_validation_failures += 1
        if repaired: self.repair_successes += 1
        if ambiguity: self.schema_ambiguities_detected += 1
        if capability_matched: self.capability_matches += 1
        if dynamic_code: self.dynamic_code_generations += 1
        if exec_failed: self.execution_failures += 1
        if result_val_failed: self.result_validation_failures += 1

        self.latencies.append(latency_breakdown)

    def get_summary(self) -> Dict[str, Any]:
        avg_total_lat = (
            sum(l.get("total_latency", 0) for l in self.latencies) / len(self.latencies)
            if self.latencies else 0.0
        )
        avg_ollama_lat = (
            sum(l.get("ollama_planning", 0) for l in self.latencies) / len(self.latencies)
            if self.latencies else 0.0
        )

        return {
            "total_queries": self.total_queries,
            "semantic_plan_success_rate": (self.semantic_plan_successes / max(1, self.total_queries)) * 100,
            "plan_validation_failure_rate": (self.plan_validation_failures / max(1, self.total_queries)) * 100,
            "repair_success_rate": (self.repair_successes / max(1, max(1, self.plan_validation_failures))) * 100,
            "schema_ambiguity_count": self.schema_ambiguities_detected,
            "capability_match_rate": (self.capability_matches / max(1, self.total_queries)) * 100,
            "dynamic_code_rate": (self.dynamic_code_generations / max(1, self.total_queries)) * 100,
            "execution_failure_rate": (self.execution_failures / max(1, self.total_queries)) * 100,
            "result_validation_failure_rate": (self.result_validation_failures / max(1, self.total_queries)) * 100,
            "average_ollama_latency_ms": round(avg_ollama_lat * 1000, 2),
            "average_total_latency_ms": round(avg_total_lat * 1000, 2)
        }

telemetry_collector = TelemetryCollector()
