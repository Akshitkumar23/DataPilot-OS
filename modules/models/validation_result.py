from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    is_valid: bool
    error_msg: Optional[str] = None
    failed_checks: List[str] = Field(default_factory=list)
