"""
DataPilot OS — Domain-Agnostic Intent Resolver & Schema Validator.
Validates requested time_column, target_column, feature_columns, dimensions, and metrics
against the active DatasetEntry schema.
NO SILENT GUESSING. Raises AmbiguousIntentError or IntentResolutionError when ambiguities or missing columns are detected.
"""

import logging
import difflib
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .intent_schema import SemanticIntent, IntentType, AggregationType
from backend.ds_engine.dataset_grounding import DatasetGrounding

logger = logging.getLogger(__name__)

class IntentResolutionError(Exception):
    def __init__(self, code: str, message: str, invalid_columns: Optional[List[str]] = None):
        self.code = code
        self.message = message
        self.invalid_columns = invalid_columns or []
        super().__init__(self.message)

class AmbiguousIntentError(Exception):
    def __init__(self, message: str, candidate_columns: Optional[List[str]] = None, code: str = "AMBIGUOUS_COLUMN"):
        self.code = code
        self.message = message
        self.candidate_columns = candidate_columns or []
        super().__init__(self.message)

class ResolvedIntent(BaseModel):
    is_valid: bool
    intent: SemanticIntent
    tool_name: Optional[str] = None
    resolved_time_column: Optional[str] = None
    resolved_target_column: Optional[str] = None
    resolved_feature_columns: List[str] = Field(default_factory=list)
    resolved_dimensions: List[str] = Field(default_factory=list)
    resolved_metrics: List[str] = Field(default_factory=list)
    resolution_source: Dict[str, str] = Field(default_factory=dict)
    dataset_name: str
    row_count: int
    column_count: int

class IntentResolver:
    def resolve(self, intent: SemanticIntent, df: Optional[pd.DataFrame] = None, dataset_context: Optional[dict] = None) -> ResolvedIntent:
        if df is None and not dataset_context:
            raise IntentResolutionError(
                code="NO_DATASET_LOADED",
                message="No active dataset connected to resolve intent.",
                invalid_columns=[]
            )

        available_cols = list(df.columns) if df is not None and hasattr(df, 'columns') else (dataset_context.get("columns", []) if dataset_context else [])
        col_lower_map = {str(c).lower(): str(c) for c in available_cols}

        invalid_cols = []

        def find_column_match(raw_name: str) -> Optional[str]:
            if not raw_name:
                return None
            
            grounding_match = DatasetGrounding.find_column_match(raw_name, available_cols)
            if grounding_match.found:
                logger.info(f"AUDIT_COLUMN_MATCH | Grounded: '{raw_name}' -> '{grounding_match.matched_column}'")
                return grounding_match.matched_column

            # Priority 2: Check for normalized ambiguity
            norm_clean = DatasetGrounding.normalize_identifier(raw_name)
            norm_matches = [
                c for c in available_cols if DatasetGrounding.normalize_identifier(c) == norm_clean
            ]
            if len(norm_matches) > 1:
                raise AmbiguousIntentError(
                    message=f"Ambiguous column reference '{raw_name}': matches multiple dataset columns ({', '.join(norm_matches)}). Please specify exact column name.",
                    candidate_columns=norm_matches,
                    code="AMBIGUOUS_COLUMN"
                )

            # Priority 3: Substring ambiguity check
            clean = raw_name.strip().lower()
            sub_matches = [
                orig_c for orig_c in available_cols
                if clean in orig_c.lower() or norm_clean in DatasetGrounding.normalize_identifier(orig_c)
            ]
            if len(sub_matches) > 1:
                raise AmbiguousIntentError(
                    message=f"Ambiguous column reference '{raw_name}': matches multiple dataset columns ({', '.join(sub_matches)}). Please specify exact column name.",
                    candidate_columns=sub_matches,
                    code="AMBIGUOUS_COLUMN"
                )
            elif len(sub_matches) == 1:
                matched = sub_matches[0]
                logger.info(f"AUDIT_COLUMN_MATCH | Substring: '{raw_name}' -> '{matched}'")
                return matched

            # Zero hardcoded target list and zero stem auto-substitution
            return None

        # 1. Resolve time_column
        resolved_time_col = None
        if intent.time_column:
            m = find_column_match(intent.time_column)
            if m:
                resolved_time_col = m
            else:
                invalid_cols.append(intent.time_column)

        # 2. Resolve target_column
        resolved_target_col = None
        if intent.target_column:
            if "," in intent.target_column:
                for sub_t in [s.strip() for s in intent.target_column.split(",") if s.strip()]:
                    m = find_column_match(sub_t)
                    if m:
                        if m not in resolved_features:
                            resolved_features.append(m)
                    else:
                        invalid_cols.append(sub_t)
            else:
                m = find_column_match(intent.target_column)
                if m:
                    resolved_target_col = m
                else:
                    invalid_cols.append(intent.target_column)

        # 3. Resolve feature_columns
        resolved_features = []
        for feat in intent.feature_columns:
            m = find_column_match(feat)
            if m:
                resolved_features.append(m)
            else:
                invalid_cols.append(feat)

        # 4. Resolve dimensions
        resolved_dims = []
        for dim in intent.dimensions + intent.group_by:
            m = find_column_match(dim)
            if m:
                if m not in resolved_dims:
                    resolved_dims.append(m)
            else:
                invalid_cols.append(dim)

        # 5. Resolve metrics
        resolved_mets = []
        metrics_candidates = list(intent.metrics)
        if resolved_target_col and resolved_target_col not in metrics_candidates and (intent.intent_type == IntentType.GROUP_AGGREGATE or intent.primary_intent == IntentType.GROUP_AGGREGATE):
            metrics_candidates.append(resolved_target_col)

        for met in metrics_candidates:
            m = find_column_match(met)
            if m:
                if m not in resolved_mets:
                    resolved_mets.append(m)
            else:
                invalid_cols.append(met)

        # 5b. Resolve filter column names against schema (Strict validation!)
        for f_item in intent.filters:
            f_col = None
            if hasattr(f_item, 'column'):
                f_col = f_item.column
            elif isinstance(f_item, dict):
                f_col = f_item.get("column") or f_item.get("col") or f_item.get("field")
            elif isinstance(f_item, str) and f_item.strip():
                import re
                m_f = re.match(r'^\s*([a-zA-Z0-9_\-]+)', f_item)
                if m_f:
                    f_col = m_f.group(1)
            
            if f_col:
                m = find_column_match(f_col)
                if not m:
                    invalid_cols.append(f_col)

        if invalid_cols:
            unique_invalid = list(dict.fromkeys(invalid_cols))
            all_close = []
            for col in unique_invalid:
                for match in difflib.get_close_matches(col, available_cols, n=3, cutoff=0.3):
                    if match not in all_close:
                        all_close.append(match)
            if not all_close and available_cols:
                all_close = available_cols[:3]

            bullets = "\n".join(f"• {c}" for c in all_close)
            if len(unique_invalid) == 1:
                missing_str = unique_invalid[0]
                if bullets:
                    err_msg = f"I couldn't find '{missing_str}' in the current dataset.\n\nAvailable similar columns:\n{bullets}"
                else:
                    err_msg = f"I couldn't find '{missing_str}' in the current dataset."
            else:
                missing_names = ", ".join(f"'{c}'" for c in unique_invalid)
                if bullets:
                    err_msg = f"I couldn't find {missing_names} in the current dataset.\n\nAvailable similar columns:\n{bullets}"
                else:
                    err_msg = f"I couldn't find {missing_names} in the current dataset."

            raise IntentResolutionError(
                code="COLUMN_NOT_FOUND",
                message=err_msg,
                invalid_columns=unique_invalid
            )

        # 6. Ambiguity Checks & Type-Aware Schema Resolution (Zero Keyword Guessing!)
        if df is not None:
            from modules.capability_registry import capability_registry
            selected_spec = capability_registry.get_tool_spec((intent.tool_name or "").lower().strip())
            requires_grouping = not bool(
                selected_spec
                and selected_spec.prerequisites
                and selected_spec.prerequisites.requires_grouping_or_dimension is False
            )
            # If GROUP_AGGREGATE requested without dimension specified, check categorical / dimension columns
            if intent.intent_type == IntentType.GROUP_AGGREGATE and requires_grouping and not resolved_dims:
                cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
                if not cat_cols:
                    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c].dtype) or df[c].nunique() <= 20]
                if len(cat_cols) > 1:
                    raise AmbiguousIntentError(
                        message=f"Ambiguous query: multiple dimension columns found ({', '.join(cat_cols)}). Please specify which dimension to group by.",
                        candidate_columns=cat_cols,
                        code="AMBIGUOUS_COLUMN"
                    )
                elif len(cat_cols) == 1:
                    resolved_dims = [cat_cols[0]]
                elif len(available_cols) > 1:
                    raise AmbiguousIntentError(
                        message=f"Ambiguous query: multiple candidate columns found ({', '.join(available_cols[:10])}). Please specify which dimension to group by.",
                        candidate_columns=available_cols,
                        code="AMBIGUOUS_COLUMN"
                    )
                elif len(available_cols) == 1:
                    resolved_dims = [available_cols[0]]
                else:
                    raise IntentResolutionError(
                        code="MISSING_REQUIRED_INFORMATION",
                        message="Group aggregation requires at least one dimension/grouping column.",
                        invalid_columns=[]
                    )

            # If GROUP_AGGREGATE requested without metric specified, check numeric columns (COUNT does not require a metric)
            if intent.intent_type == IntentType.GROUP_AGGREGATE and not resolved_mets:
                if intent.aggregation == AggregationType.COUNT:
                    # Count aggregation operates on record/row level — no specific numeric metric required
                    pass
                else:
                    num_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                    referenced_num = []
                    for f in intent.filters:
                        fc = getattr(f, 'column', None) or (f.get('column') if isinstance(f, dict) else None)
                        if fc and fc in num_cols and fc not in referenced_num:
                            referenced_num.append(fc)
                    if len(referenced_num) == 1:
                        resolved_mets = [referenced_num[0]]
                    elif len(num_cols) == 1:
                        resolved_mets = [num_cols[0]]
                    elif len(num_cols) > 1:
                        raise AmbiguousIntentError(
                            message=f"Ambiguous query: multiple metric columns found ({', '.join(num_cols)}). Please specify which metric to aggregate.",
                            candidate_columns=num_cols,
                            code="AMBIGUOUS_COLUMN"
                        )
                    else:
                        raise IntentResolutionError(
                            code="MISSING_REQUIRED_INFORMATION",
                            message="Group aggregation requires at least one numeric metric column.",
                            invalid_columns=[]
                        )

            # Time column resolution for time-series operations (Zero raw-query text guessing!)
            ts_intents = [IntentType.AUTOCORRELATION, IntentType.TIME_SERIES_FORECAST, IntentType.TIME_SERIES_DECOMPOSITION]
            if intent.intent_type in ts_intents and not resolved_time_col:
                dt_cols = df.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns.tolist()
                if not dt_cols:
                    for c in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[c].dtype):
                            sample_vals = df[c].dropna().head(10)
                            if len(sample_vals) > 0:
                                try:
                                    parsed_dt = pd.to_datetime(sample_vals, errors='coerce')
                                    if parsed_dt.notna().sum() >= len(sample_vals) * 0.8:
                                        dt_cols.append(c)
                                except Exception:
                                    pass
                if len(dt_cols) == 1:
                    resolved_time_col = dt_cols[0]
                elif len(dt_cols) > 1:
                    raise AmbiguousIntentError(
                        message=f"Ambiguous date column: multiple date columns exist ({', '.join(dt_cols)}). Please specify which date column to track.",
                        candidate_columns=dt_cols,
                        code="AMBIGUOUS_COLUMN"
                    )
                else:
                    raise IntentResolutionError(
                        code="NO_TEMPORAL_COLUMN",
                        message="Time-series operation requires a valid date/time column in the dataset.",
                        invalid_columns=[]
                    )

            target_required_intents = [IntentType.TIME_SERIES_FORECAST, IntentType.REGRESSION, IntentType.CLASSIFICATION, IntentType.AUTO_ML]
            if intent.intent_type in target_required_intents and not resolved_target_col:
                if resolved_mets:
                    resolved_target_col = resolved_mets[0]
                else:
                    if intent.intent_type == IntentType.CLASSIFICATION:
                        cand_cols = [c for c in df.columns if df[c].nunique() <= 20]
                    else:
                        cand_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                    
                    if len(cand_cols) == 1:
                        resolved_target_col = cand_cols[0]
                    elif len(cand_cols) > 1:
                        raise AmbiguousIntentError(
                            message=f"Ambiguous query: multiple target column candidates found ({', '.join(cand_cols)}). Please specify target column.",
                            candidate_columns=cand_cols,
                            code="AMBIGUOUS_COLUMN"
                        )
                    else:
                        raise IntentResolutionError(
                            code="MISSING_REQUIRED_INFORMATION",
                            message=f"Analytical operation '{intent.intent_type.value}' requires a target column.",
                            invalid_columns=[]
                        )

            # Type Guardrail: Validate numeric types for target column if operation requires numeric target
            if resolved_target_col and resolved_target_col in df.columns:
                numeric_target_intents = [IntentType.TIME_SERIES_FORECAST, IntentType.REGRESSION, IntentType.AUTOCORRELATION]
                if intent.intent_type in numeric_target_intents and not pd.api.types.is_numeric_dtype(df[resolved_target_col].dtype):
                    raise IntentResolutionError(
                        code="INVALID_COLUMN_TYPE",
                        message=f"Target column '{resolved_target_col}' must be numeric for operation '{intent.intent_type.value}'.",
                        invalid_columns=[resolved_target_col]
                    )

            # For AutoML, Regression, Classification, Clustering, PCA, Correlation, Autocorrelation: resolve feature_columns if omitted
            if intent.intent_type in [IntentType.AUTO_ML, IntentType.REGRESSION, IntentType.CLASSIFICATION, IntentType.CLUSTERING, IntentType.PCA, IntentType.CORRELATION, IntentType.AUTOCORRELATION] and not resolved_features:
                num_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                resolved_features = [c for c in num_cols if c != resolved_target_col and c != resolved_time_col]
                if not resolved_features:
                    other_cols = [c for c in df.columns if c != resolved_target_col]
                    if other_cols:
                        resolved_features = other_cols

            # For STATISTICAL_TEST: resolve dimensions and metrics if omitted
            if intent.intent_type == IntentType.STATISTICAL_TEST:
                if not resolved_dims:
                    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
                    if len(cat_cols) > 2:
                        raise AmbiguousIntentError(
                            message=f"Statistical test requires explicit columns to compare; multiple categorical columns found ({', '.join(cat_cols)}). Please specify columns.",
                            candidate_columns=cat_cols,
                            code="AMBIGUOUS_COLUMN",
                        )
                    elif len(cat_cols) in (1, 2):
                        resolved_dims = list(cat_cols)
                if not resolved_mets:
                    num_cols = df.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()
                    if len(num_cols) > 2:
                        raise AmbiguousIntentError(
                            message=f"Statistical test requires explicit metric columns; multiple numeric columns found ({', '.join(num_cols)}). Please specify columns.",
                            candidate_columns=num_cols,
                            code="AMBIGUOUS_COLUMN",
                        )
                    elif len(num_cols) in (1, 2):
                        resolved_mets = list(num_cols)

        return ResolvedIntent(
            is_valid=True,
            intent=intent,
            tool_name=intent.tool_name,
            resolved_time_column=resolved_time_col,
            resolved_target_column=resolved_target_col,
            resolved_feature_columns=resolved_features,
            resolved_dimensions=resolved_dims,
            resolved_metrics=resolved_mets,
            dataset_name=dataset_context.get("name", "Active Dataset") if dataset_context else "Active Dataset",
            row_count=len(df) if df is not None else (dataset_context.get("row_count", 0) if dataset_context else 0),
            column_count=len(available_cols)
        )

intent_resolver = IntentResolver()
