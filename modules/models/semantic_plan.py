from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union

class Operation(BaseModel):
    """
    Represents an extensible single analytical operation in the plan.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    operation: str
    target: Optional[Union[str, List[str]]] = None
    aggregation: Optional[str] = None
    group_by: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[int] = Field(default_factory=list)

class Filter(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    column: str
    operator: str
    value: Any

class Visualization(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    requested: bool = True
    type: str = "auto"

class SemanticPlan(BaseModel):
    """
    Generic, Extensible Semantic Plan structure extracted by Ollama Semantic Brain.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    user_goal: str = Field(default="", alias="goal")
    intent: str = Field(default="analytics")
    operations: List[Operation] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    visualization: Visualization = Field(default_factory=Visualization)
    response_style: str = Field(default="analytical")
    confidence_score: float = Field(default=1.0)
