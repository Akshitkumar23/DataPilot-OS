from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ExecutionStep(BaseModel):
    id: str
    capability: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    steps: List[ExecutionStep] = Field(default_factory=list)
