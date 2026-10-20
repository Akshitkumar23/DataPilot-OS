"""
DataPilot OS — Execution Plan & Execution Step Contract.
Represents multi-step sequential execution plans and step-by-step metadata.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
from backend.ds_engine.intent_schema import IntentType
from backend.ds_engine.intent_resolver import ResolvedIntent
from backend.ds_engine.code_validator import CodeVersion

class ExecutionStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    step_index: int
    intent_type: IntentType
    description: str
    resolved_intent: Optional[ResolvedIntent] = None
    code_version: Optional[CodeVersion] = None
    input_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    execution_time_sec: float = 0.0
    status: str = "PENDING"  # PENDING, SUCCEEDED, FAILED
    error: Optional[str] = None
    is_prerequisite: bool = False

class ExecutionPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    primary_intent: IntentType
    prerequisites: List[IntentType] = Field(default_factory=list)
    steps: List[ExecutionStep] = Field(default_factory=list)
    is_compiled: bool = False
    execution_status: str = "PENDING"
