"""
DataPilot OS — Code Repair Engine with Diff Check.
Repairs failed code via Ollama + root cause analysis.

CRITICAL: Diff check prevents LOOP_NO_PROGRESS.
If the proposed repair is identical to the failed version, stop immediately.
Max 3 repair attempts per code version.
"""

import hashlib
from typing import Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel

from .code_generator import CodeVersion
from .code_validator import CodeValidationResult, code_validator_instance
from .error_analyzer import ErrorAnalysis, error_analyzer
from backend.core.error_states import DataPilotError, ErrorState

class RepairAttempt(BaseModel):
    attempt_number: int
    original_code_hash: str
    proposed_code_hash: str
    is_same_as_original: bool
    error_analysis: ErrorAnalysis
    validation_result: Optional[CodeValidationResult]
    succeeded: bool
    timestamp: str

class RepairEngine:
    def attempt_repair(self, failed_code_version: CodeVersion, error_output: str, dataset_context: dict, attempt_number: int) -> Union[CodeVersion, DataPilotError]:
        if attempt_number > 3:
            return DataPilotError(
                code=ErrorState.LOOP_BUDGET_EXCEEDED,
                message="Maximum repair attempts exceeded.",
                stage="repair_engine",
                recoverable=False,
                required_action="Modify query or inspect error details."
            )

        code = failed_code_version.code
        orig_hash = self._compute_code_hash(code)
        
        analysis = error_analyzer.analyze(error_output, code, dataset_context)
        
        fixed_code = self._apply_deterministic_fix(code, analysis, dataset_context)
        if not fixed_code:
            fixed_code = code
            
        prop_hash = self._compute_code_hash(fixed_code)
        
        if orig_hash == prop_hash:
            return DataPilotError(
                code=ErrorState.LOOP_NO_PROGRESS,
                message="Proposed repair is identical to original code.",
                stage="repair_engine",
                recoverable=False,
                required_action="Modify query or schema definition."
            )
            
        # Create tentative new version
        new_version = CodeVersion(
            version_id=failed_code_version.version_id + f"-rep{attempt_number}",
            workflow_id=failed_code_version.workflow_id,
            node_id=failed_code_version.node_id,
            purpose=failed_code_version.purpose,
            language=failed_code_version.language,
            code=fixed_code,
            dependencies=failed_code_version.dependencies,
            expected_outputs=failed_code_version.expected_outputs,
            dataset_columns_referenced=failed_code_version.dataset_columns_referenced,
            validation_requirements=failed_code_version.validation_requirements,
            created_at=datetime.now(timezone.utc).isoformat(),
            is_user_edited=False,
            parent_version_id=failed_code_version.version_id
        )

        val_result = code_validator_instance.validate(new_version, dataset_context)
        if not val_result.is_valid:
            if val_result.can_autofix and val_result.autofixed_code:
                new_version.code = val_result.autofixed_code
            else:
                return DataPilotError(
                    code=ErrorState.CODE_INVALID,
                    message="Repaired code failed validation.",
                    stage="code_validator",
                    recoverable=False,
                    required_action="Inspect code validator issues."
                )
            
        return new_version

    def _compute_code_hash(self, code: str) -> str:
        # Strip whitespace for normalized hash
        normalized = code.strip().replace("\r\n", "\n")
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _apply_deterministic_fix(self, code: str, error_analysis: ErrorAnalysis, dataset_context: dict) -> Optional[str]:
        if error_analysis.repair_type == "column_name" and error_analysis.available_alternatives:
            correct_col = error_analysis.available_alternatives[0]
            wrong_col = error_analysis.affected_symbol
            if wrong_col:
                fixed = code.replace(f"['{wrong_col}']", f"['{correct_col}']")
                fixed = fixed.replace(f'["{wrong_col}"]', f'["{correct_col}"]')
                if fixed != code:
                    return fixed
        return None

repair_engine = RepairEngine()
