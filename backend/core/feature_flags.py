"""
DataPilot OS — Feature Flags & Rollout Management.
Supports dynamic feature toggles, canary progressive rollout (0-100%),
and Shadow Mode dual-run comparison between pipelines.
"""

import os
import random
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RolloutConfig(BaseModel):
    enable_minimal_semantic_command: bool = True
    canary_percentage: float = 100.0  # 0.0 to 100.0%
    enable_shadow_mode: bool = False
    enable_strict_json_schema: bool = True
    log_normalization_telemetry: bool = True


class FeatureFlagManager:
    """
    Centralized Feature Flag & Progressive Rollout Manager.
    Reads from environment variables with safe defaults.
    """

    def __init__(self):
        self._config = RolloutConfig(
            enable_minimal_semantic_command=os.getenv("FLAG_MINIMAL_SEMANTIC_COMMAND", "true").lower() == "true",
            canary_percentage=float(os.getenv("FLAG_CANARY_PERCENTAGE", "100.0")),
            enable_shadow_mode=os.getenv("FLAG_SHADOW_MODE", "false").lower() == "true",
            enable_strict_json_schema=os.getenv("FLAG_STRICT_JSON_SCHEMA", "true").lower() == "true",
            log_normalization_telemetry=os.getenv("FLAG_LOG_NORMALIZATION", "true").lower() == "true"
        )

    @property
    def config(self) -> RolloutConfig:
        return self._config

    def set_config(
        self,
        enable_minimal_semantic_command: Optional[bool] = None,
        canary_percentage: Optional[float] = None,
        enable_shadow_mode: Optional[bool] = None,
        enable_strict_json_schema: Optional[bool] = None,
        log_normalization_telemetry: Optional[bool] = None
    ):
        """Allows dynamic runtime configuration in tests or admin settings."""
        if enable_minimal_semantic_command is not None:
            self._config.enable_minimal_semantic_command = enable_minimal_semantic_command
        if canary_percentage is not None:
            self._config.canary_percentage = max(0.0, min(100.0, canary_percentage))
        if enable_shadow_mode is not None:
            self._config.enable_shadow_mode = enable_shadow_mode
        if enable_strict_json_schema is not None:
            self._config.enable_strict_json_schema = enable_strict_json_schema
        if log_normalization_telemetry is not None:
            self._config.log_normalization_telemetry = log_normalization_telemetry

    def should_use_minimal_command(self, request_id: Optional[str] = None) -> bool:
        """
        Determines whether the incoming query uses the Minimal Semantic Command pipeline
        based on active flag and canary percentage.
        """
        if not self._config.enable_minimal_semantic_command:
            return False

        if self._config.canary_percentage >= 100.0:
            return True
        if self._config.canary_percentage <= 0.0:
            return False

        # Deterministic hashing if request_id is provided, else random
        if request_id:
            val = abs(hash(request_id)) % 100
        else:
            val = random.uniform(0.0, 100.0)
        return val < self._config.canary_percentage

    def is_shadow_mode_enabled(self) -> bool:
        return self._config.enable_shadow_mode

    def log_shadow_comparison(
        self,
        request_id: str,
        query: str,
        primary_intent: Dict[str, Any],
        shadow_intent: Optional[Dict[str, Any]],
        primary_latency_ms: float,
        shadow_latency_ms: float
    ) -> Dict[str, Any]:
        """
        Logs telemetry comparison between active pipeline and shadow pipeline.
        """
        intents_match = False
        if shadow_intent and primary_intent:
            p_type = primary_intent.get("intent_type")
            s_type = shadow_intent.get("intent_type")
            intents_match = (p_type == s_type)

        audit_record = {
            "event": "SHADOW_MODE_DUAL_RUN_COMPARISON",
            "request_id": request_id,
            "query": query,
            "intents_match": intents_match,
            "primary_intent_type": primary_intent.get("intent_type") if primary_intent else None,
            "shadow_intent_type": shadow_intent.get("intent_type") if shadow_intent else None,
            "latency_delta_ms": round(shadow_latency_ms - primary_latency_ms, 2),
            "timestamp": time.time()
        }
        logger.info(f"SHADOW_DUAL_RUN | {audit_record}")
        return audit_record


feature_flag_manager = FeatureFlagManager()
