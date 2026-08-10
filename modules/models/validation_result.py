from pydantic import BaseModel, Field
from typing import List, Optional

class ValidationResult(BaseModel):
    is_valid: bool
    error_msg: Optional[str] = None
    failed_checks: List[str] = Field(default_factory=list)
