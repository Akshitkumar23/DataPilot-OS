"""
DataPilot OS — Workspace Agent Centralized Configuration.
Consolidates all agent-related timeouts, model identifiers, context limits,
and operational defaults with environment variable overrides.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AgentConfig:
    # Versioning
    PROMPT_VERSION: str = "1.0"
    SCHEMA_VERSION: str = "1.0"

    # Ollama Intelligence Settings
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    PLANNER_MODEL: str = os.getenv("OLLAMA_PLANNER_MODEL", "qwen2.5:3b-instruct")
    PLANNER_TIMEOUT_SEC: float = float(os.getenv("OLLAMA_PLANNER_TIMEOUT", "90.0"))
    NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "60m")
    TEMPERATURE: float = float(os.getenv("OLLAMA_PLANNER_TEMPERATURE", "0.0"))

    # Operational & Safety Limits
    MAX_REPAIR_ATTEMPTS: int = int(os.getenv("AGENT_MAX_REPAIR_ATTEMPTS", "1"))
    DEFAULT_COMMAND_TIMEOUT_SEC: int = int(os.getenv("AGENT_COMMAND_TIMEOUT_SEC", "30"))
    MAX_DIRECTORY_LIST_LIMIT: int = int(os.getenv("AGENT_MAX_DIRECTORY_LIST_LIMIT", "500"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("AGENT_MAX_SEARCH_RESULTS", "100"))
    COMMAND_MAX_OUTPUT_BYTES: int = int(os.getenv("AGENT_COMMAND_MAX_OUTPUT_BYTES", "65536"))


agent_config = AgentConfig()
