"""
DataPilot OS — Domain-Agnostic DS Code Engine.
Handles intent resolution, ambiguity prevention, capability compilation, code generation, validation, error analysis, and repair.
"""

from .intent_schema import SemanticIntent, IntentType, ChartType, AggregationType
from .intent_resolver import IntentResolver, intent_resolver, IntentResolutionError, AmbiguousIntentError, ResolvedIntent
from .capability_drivers import DRIVER_REGISTRY
from .code_generator import CodeVersion, CodeGenerator, code_generator
from .code_validator import ValidationIssue, CodeValidationResult, CodeValidator
code_validator_instance = CodeValidator()
from .error_analyzer import ErrorCategory, ErrorAnalysis, ErrorAnalyzer, error_analyzer
from .repair_engine import RepairAttempt, RepairEngine, repair_engine

__all__ = [
    'SemanticIntent', 'IntentType', 'ChartType', 'AggregationType',
    'IntentResolver', 'intent_resolver', 'IntentResolutionError', 'AmbiguousIntentError', 'ResolvedIntent',
    'DRIVER_REGISTRY',
    'CodeVersion', 'CodeGenerator', 'code_generator',
    'ValidationIssue', 'CodeValidationResult', 'CodeValidator', 'code_validator_instance',
    'ErrorCategory', 'ErrorAnalysis', 'ErrorAnalyzer', 'error_analyzer',
    'RepairAttempt', 'RepairEngine', 'repair_engine'
]
