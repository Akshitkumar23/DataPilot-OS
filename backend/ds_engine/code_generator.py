"""
DataPilot OS — Domain-Agnostic Deterministic DS Code Generator Engine.
Translates resolved semantic intents & execution plans into inspectable Python/SQL code.
Ollama ONLY interprets NLP -> ResolvedIntent.
DataPilot DS Engine owns 100% of code generation, dependencies, output contracts, and 7-step validation.
Zero domain/ecommerce assumptions. Works on any dataset.
"""

import ast
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .intent_schema import SemanticIntent, IntentType, ChartType, AggregationType
from .intent_resolver import ResolvedIntent, intent_resolver
from .capability_drivers import EDADriver

logger = logging.getLogger(__name__)

class DataPilotError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

class CodeVersion(BaseModel):
    version_id: str
    workflow_id: str
    node_id: str
    purpose: str
    language: str = 'python'
    code: str
    dependencies: List[str] = Field(default_factory=list)
    expected_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_columns_referenced: List[str] = Field(default_factory=list)
    validation_requirements: List[str] = Field(default_factory=list)
    created_at: str
    is_user_edited: bool = False
    parent_version_id: Optional[str] = None

class CodeGenerator:
    def __init__(self):
        pass

    def generate_code(
        self,
        resolved: ResolvedIntent,
        node_id: str = "node_01",
        workflow_id: str = "wf_01"
    ) -> CodeVersion:
        return self.generate_from_resolved_intent(resolved, node_id=node_id, workflow_id=workflow_id)

    def generate_from_resolved_intent(
        self,
        resolved: ResolvedIntent,
        node_id: str = "node_01",
        workflow_id: str = "wf_01"
    ) -> CodeVersion:
        """
        Authoritative primary entry point:
        Translates ResolvedIntent via Capability Drivers into executable code, dependencies, and contracts.
        Mandates 7-step CodeValidator check prior to returning.
        """
        intent = resolved.intent
        intent_key = intent.intent_type.value if hasattr(intent.intent_type, "value") else str(intent.intent_type)

        # Lookup driver in CapabilityRegistry (Single Source of Truth)
        from modules.capability_registry import capability_registry
        tool_name = resolved.tool_name or (resolved.intent.tool_name if hasattr(resolved.intent, "tool_name") else None)
        if tool_name:
            driver = capability_registry.get_driver(tool_name)
        else:
            driver = capability_registry.get_driver_for_intent(intent_key)

        if not driver:
            from backend.core.error_states import DataPilotError, ErrorState
            raise DataPilotError(
                code=ErrorState.CAPABILITY_NOT_SUPPORTED,
                message=f"Analytical capability or tool '{tool_name or intent_key}' is not currently supported in DataPilot DS Engine."
            )
        code_body, dependencies, expected_outputs = driver.compile(resolved)

        preamble = """import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')"""

        final_code = f"{preamble}\n\n{code_body}"
        referenced_cols = self._identify_referenced_columns(
            final_code,
            resolved.resolved_dimensions + resolved.resolved_metrics +
            ([resolved.resolved_time_column] if resolved.resolved_time_column else []) +
            ([resolved.resolved_target_column] if resolved.resolved_target_column else [])
        )

        code_version = CodeVersion(
            version_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            node_id=node_id,
            purpose=intent_key,
            code=final_code,
            dependencies=list(set(dependencies)),
            expected_outputs=expected_outputs,
            dataset_columns_referenced=referenced_cols,
            validation_requirements=["syntax", "ast", "imports", "column_refs", "security"],
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # Mandatory 7-step validation pipeline check
        from .code_validator import CodeValidator
        validator = CodeValidator()
        val_res = validator.validate(code_version, {"columns": referenced_cols})
        if not val_res.is_valid and val_res.can_autofix and val_res.autofixed_code:
            logger.info("CODE_GENERATOR | Auto-fixing code via CodeValidator...")
            code_version.code = val_res.autofixed_code

        return code_version

    def generate(self, node_id: str, capability: str, intent: dict, dataset_context: dict, workflow_id: str) -> Union[CodeVersion, DataPilotError]:
        """
        Legacy adapter wrapper. Wraps legacy calls into ResolvedIntent pipeline.
        """
        try:
            cap_upper = capability.upper()
            intent_type = IntentType.EDA
            if "FORECAST" in cap_upper:
                intent_type = IntentType.TIME_SERIES_FORECAST
            elif "ML" in cap_upper:
                intent_type = IntentType.AUTO_ML
            elif "CLEAN" in cap_upper:
                intent_type = IntentType.DATA_CLEAN
            elif "GROUP" in cap_upper or "TOP" in cap_upper:
                intent_type = IntentType.GROUP_AGGREGATE

            intent_obj = SemanticIntent(
                intent_type=intent_type,
                dimensions=intent.get("target_columns", []),
                metrics=intent.get("metrics", []),
                prompt_summary=capability
            )
            resolved = intent_resolver.resolve(intent_obj, dataset_context=dataset_context)
            return self.generate_from_resolved_intent(resolved, node_id=node_id, workflow_id=workflow_id)
        except Exception as e:
            logger.error(f"Legacy code generation adapter error: {e}")
            return DataPilotError(code="GENERATION_FAILED", message=str(e))

    def _identify_referenced_columns(self, code: str, dataset_columns: List[str]) -> List[str]:
        referenced = set()
        for col in dataset_columns:
            if col and (f"'{col}'" in code or f'"{col}"' in code):
                referenced.add(col)
        return list(referenced)

    def create_user_edit_version(self, original: CodeVersion, new_code: str) -> CodeVersion:
        columns = original.dataset_columns_referenced
        referenced_columns = self._identify_referenced_columns(new_code, columns)
        
        return CodeVersion(
            version_id=str(uuid.uuid4()),
            workflow_id=original.workflow_id,
            node_id=original.node_id,
            purpose=original.purpose,
            language=original.language,
            code=new_code,
            dependencies=original.dependencies,
            expected_outputs=original.expected_outputs,
            dataset_columns_referenced=referenced_columns,
            validation_requirements=original.validation_requirements,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_user_edited=True,
            parent_version_id=original.version_id
        )

code_generator = CodeGenerator()
