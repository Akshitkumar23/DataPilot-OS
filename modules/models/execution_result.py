from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    EMPTY_RESULT = "EMPTY_RESULT"
    INVALID_RESULT = "INVALID_RESULT"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"

class ExecutionResult(BaseModel):
    status: ExecutionStatus
    response: str
    code: Optional[str] = None
    plot_image: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
