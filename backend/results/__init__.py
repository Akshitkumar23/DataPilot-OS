"""
Result engine module.
"""
from .validator import result_validator, ResultValidationReport, ResultValidationCheck
from .interpreter import result_interpreter, InsightPayload
from backend.core.error_states import DataPilotError
