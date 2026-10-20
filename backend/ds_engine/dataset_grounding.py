"""
DataPilot OS — Strict Dataset Grounding & Column Ambiguity Engine.
Enforces exact column existence, role validation, candidate evidence,
clear typo suggestions without silent auto-substitution, and no silent defaults.
100% Zero domain keyword hardcoding.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from backend.ds_engine.dataset_profiler import DatasetProfile, DatasetProfiler
from backend.ds_engine.analysis_decision import (
    AnalysisDecision,
    DecisionRecord,
    DecisionType,
    make_clarify_decision,
    make_execute_decision,
    make_unsupported_decision,
)

logger = logging.getLogger(__name__)


class GroundingMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_name: str
    matched_column: Optional[str] = None
    is_exact: bool = False
    is_case_insensitive: bool = False
    is_normalized: bool = False
    suggestions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def found(self) -> bool:
        return self.matched_column is not None


class ColumnValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    is_valid: bool
    detected_role: Optional[str] = None
    expected_role: Optional[str] = None
    is_role_valid: bool = True
    is_ambiguous: bool = False
    ambiguity_roles: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class DatasetGrounding:
    """
    Authoritative dataset grounding engine for DataPilot OS.
    Eliminates guessing, substring bleeding, and silent column substitution.
    """

    @staticmethod
    def normalize_identifier(name: str) -> str:
        """Strip whitespace and common punctuation delimiters for comparison."""
        return (
            name.strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    @classmethod
    def find_column_match(
        cls,
        raw_name: str,
        available_columns: List[str],
    ) -> GroundingMatch:
        """
        Resolve raw_name against available_columns with strict precedence:
        1. Exact case-sensitive match
        2. Exact case-insensitive match
        3. Strict character-normalized match (spaces/underscores/hyphens removed)
           ONLY IF exactly one column matches.
        If no match is found, compute typo suggestions without auto-substituting.
        """
        if not raw_name or not str(raw_name).strip():
            return GroundingMatch(raw_name="", matched_column=None)

        clean_name = str(raw_name).strip()

        # Priority 1: Exact case-sensitive match
        if clean_name in available_columns:
            return GroundingMatch(
                raw_name=raw_name,
                matched_column=clean_name,
                is_exact=True,
                confidence=1.0,
            )

        # Priority 2: Exact case-insensitive match
        lower_map = {c.lower(): c for c in available_columns}
        if clean_name.lower() in lower_map:
            return GroundingMatch(
                raw_name=raw_name,
                matched_column=lower_map[clean_name.lower()],
                is_case_insensitive=True,
                confidence=0.98,
            )

        # Priority 3: Character normalization (spaces, underscores, hyphens)
        norm_clean = cls.normalize_identifier(clean_name)
        norm_matches = [
            c for c in available_columns if cls.normalize_identifier(c) == norm_clean
        ]
        if len(norm_matches) == 1:
            return GroundingMatch(
                raw_name=raw_name,
                matched_column=norm_matches[0],
                is_normalized=True,
                confidence=0.92,
            )
        elif len(norm_matches) > 1:
            # Ambiguous normalized match -> surface candidates for clarification
            return GroundingMatch(
                raw_name=raw_name,
                matched_column=None,
                suggestions=norm_matches,
                confidence=0.0,
            )

        # No match found -> Generate typo suggestions without auto-substitution
        suggestions = difflib.get_close_matches(
            clean_name, available_columns, n=3, cutoff=0.55
        )
        if not suggestions:
            # Fall back to case-insensitive prefix/suffix search if close matches miss
            suggestions = [
                c
                for c in available_columns
                if clean_name.lower() in c.lower() or c.lower() in clean_name.lower()
            ][:3]

        return GroundingMatch(
            raw_name=raw_name,
            matched_column=None,
            suggestions=suggestions,
            confidence=0.0,
        )

    @classmethod
    def validate_column_role(
        cls,
        column_name: str,
        expected_role: str,
        profile: DatasetProfile,
    ) -> ColumnValidationResult:
        """
        Validate whether column_name physically and semantically satisfies expected_role.
        Expected roles: 'numeric', 'temporal', 'categorical', 'binary_flag', 'entity_id'.
        """
        if column_name not in profile.columns:
            return ColumnValidationResult(
                column=column_name,
                is_valid=False,
                error_message=f"Column '{column_name}' does not exist in dataset profile.",
            )

        col_profile = profile.columns[column_name]
        detected = col_profile.semantic_role
        ambiguities = profile.role_ambiguities.get(column_name, [])
        is_ambiguous = len(ambiguities) > 1

        role_valid = False
        exp = expected_role.lower().strip()

        if exp in ("numeric", "continuous_numeric", "metric"):
            role_valid = (
                detected in ("continuous_numeric", "discrete_numeric")
                or col_profile.stats is not None
            )
        elif exp in ("temporal", "datetime", "date", "time"):
            role_valid = (
                detected == "temporal"
                or col_profile.datetime_parse_ratio >= 0.8
            )
        elif exp in ("categorical", "dimension", "group_by"):
            role_valid = detected in ("categorical_nominal", "binary_flag", "text", "discrete_numeric")
        elif exp in ("binary_flag", "boolean"):
            role_valid = detected == "binary_flag" or col_profile.cardinality == 2
        elif exp in ("entity_id", "id", "key"):
            role_valid = detected == "entity_id" or col_profile.is_unique
        else:
            role_valid = detected == exp

        err = None
        if not role_valid:
            err = (
                f"Column '{column_name}' has detected role '{detected}' ({col_profile.dtype}), "
                f"which is incompatible with required role '{expected_role}'."
            )

        return ColumnValidationResult(
            column=column_name,
            is_valid=True,
            detected_role=detected,
            expected_role=expected_role,
            is_role_valid=role_valid,
            is_ambiguous=is_ambiguous,
            ambiguity_roles=ambiguities,
            error_message=err,
        )

    @classmethod
    def get_role_candidates(
        cls,
        role: str,
        profile: DatasetProfile,
    ) -> List[str]:
        """Return deduplicated list of candidate column names for a given role."""
        role_clean = role.lower().strip()
        if role_clean in ("temporal", "time", "date", "datetime"):
            return list(dict.fromkeys(profile.temporal_column_candidates))
        elif role_clean in ("numeric", "metric", "measure"):
            return list(dict.fromkeys(profile.continuous_numeric_candidates))
        elif role_clean in ("categorical", "dimension", "group_by"):
            return list(dict.fromkeys(profile.categorical_candidates))
        elif role_clean in ("target", "label", "outcome"):
            return list(dict.fromkeys(profile.target_column_candidates))
        elif role_clean in ("entity", "entity_id", "id"):
            return list(dict.fromkeys(profile.entity_id_candidates))
        elif role_clean in ("binary", "flag", "binary_flag"):
            return list(dict.fromkeys(profile.binary_flag_candidates))
        elif role_clean in ("sequence", "order"):
            return list(dict.fromkeys(profile.sequence_column_candidates))
        return []

    @classmethod
    def ground_columns(
        cls,
        requested_columns: List[str],
        available_columns: List[str],
    ) -> Tuple[List[str], Optional[DecisionRecord]]:
        """
        Ground a list of requested column names.
        Returns (grounded_names, None) on complete success,
        or ([], clarify_decision) if any column is missing or ambiguous.
        """
        grounded: List[str] = []
        missing: List[str] = []
        suggestions_map: Dict[str, List[str]] = {}

        for req in requested_columns:
            match = cls.find_column_match(req, available_columns)
            if match.found:
                if match.matched_column not in grounded:
                    grounded.append(match.matched_column)
            else:
                missing.append(req)
                if match.suggestions:
                    suggestions_map[req] = match.suggestions

        if missing:
            bullets = []
            for col in missing:
                sug = suggestions_map.get(col, [])
                if sug:
                    bullets.append(f"• '{col}' (did you mean: {', '.join(repr(s) for s in sug)}?)")
                else:
                    bullets.append(f"• '{col}'")

            reason = (
                f"Column(s) not found in dataset:\n"
                + "\n".join(bullets)
                + "\nPlease confirm exact column name."
            )
            decision = make_clarify_decision(
                reason=reason,
                suggested_columns=suggestions_map,
                metadata={"missing_columns": missing},
            )
            return [], decision

        return grounded, None

    @classmethod
    def ground_or_clarify_role(
        cls,
        role: str,
        specified_column: Optional[str],
        profile: DatasetProfile,
        operation_name: str = "Analysis",
    ) -> Tuple[Optional[str], Optional[DecisionRecord]]:
        """
        Ground a column for a required role (e.g. time_column for forecast,
        target_column for classification, metric for aggregation).
        - If specified: validates exact existence and role compatibility.
        - If not specified:
          - If multiple candidates exist -> emits CLARIFY with candidate_columns.
          - If zero candidates exist -> emits UNSUPPORTED / CLARIFY.
          - Never silently picks the first column.
        """
        available_cols = list(profile.columns.keys())

        if specified_column:
            match = cls.find_column_match(specified_column, available_cols)
            if not match.found:
                reason = f"Specified {role} column '{specified_column}' was not found in dataset."
                if match.suggestions:
                    reason += f" Available similar columns: {', '.join(match.suggestions)}"
                return None, make_clarify_decision(
                    reason=reason,
                    suggested_columns={specified_column: match.suggestions} if match.suggestions else {},
                    candidate_columns=cls.get_role_candidates(role, profile),
                )

            # Role validation
            val_res = cls.validate_column_role(match.matched_column, role, profile)
            if not val_res.is_role_valid:
                return None, make_clarify_decision(
                    reason=val_res.error_message or f"Column '{match.matched_column}' is not suitable as a {role} column.",
                    candidate_columns=cls.get_role_candidates(role, profile),
                )
            return match.matched_column, None

        # Column was not specified by user/planner -> evaluate candidates
        candidates = cls.get_role_candidates(role, profile)
        if len(candidates) > 1:
            return None, make_clarify_decision(
                reason=(
                    f"{operation_name} requires a {role} column, but multiple candidate columns were found "
                    f"({', '.join(candidates)}). Please specify which column to use."
                ),
                candidate_columns=candidates,
            )
        elif len(candidates) == 1:
            # Exactly one unambiguous candidate exists for the dataset
            return candidates[0], None
        else:
            return None, make_unsupported_decision(
                reason=f"{operation_name} requires a valid {role} column, but none was detected in the dataset."
            )
