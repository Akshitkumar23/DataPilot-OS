from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Operation(BaseModel):
    operation: str
    target: Optional[str] = None
    aggregation: Optional[str] = None
    group_by: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Filter(BaseModel):
    column: str
    operator: str
    value: Any

class Visualization(BaseModel):
    requested: bool = True
    type: str = "auto"

class SemanticPlan(BaseModel):
    goal: str
    operations: List[Operation] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    visualization: Visualization = Field(default_factory=Visualization)
