"""
DataPilot OS — Domain-Agnostic Semantic Intent Schema Contract.
Ollama translates raw user prompts/OCR into structured SemanticIntent JSON.
Zero domain assumptions (no e-commerce or hardcoded column names).
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict
from enum import Enum
import re


class FilterOperator(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    MISSING = "missing"
    CONTAINS = "contains"


_OP_LOOKUP = {
    "gt": FilterOperator.GT, ">": FilterOperator.GT, "greater": FilterOperator.GT, "greater_than": FilterOperator.GT, "above": FilterOperator.GT, "more than": FilterOperator.GT,
    "gte": FilterOperator.GTE, ">=": FilterOperator.GTE, "greater_or_equal": FilterOperator.GTE, "greater than or equal": FilterOperator.GTE, "at least": FilterOperator.GTE,
    "lt": FilterOperator.LT, "<": FilterOperator.LT, "less": FilterOperator.LT, "less_than": FilterOperator.LT, "below": FilterOperator.LT, "under": FilterOperator.LT, "lower than": FilterOperator.LT,
    "lte": FilterOperator.LTE, "<=": FilterOperator.LTE, "less_or_equal": FilterOperator.LTE, "less than or equal": FilterOperator.LTE, "at most": FilterOperator.LTE, "up to": FilterOperator.LTE,
    "eq": FilterOperator.EQ, "=": FilterOperator.EQ, "==": FilterOperator.EQ, "equal": FilterOperator.EQ, "equals": FilterOperator.EQ, "is": FilterOperator.EQ, "match": FilterOperator.EQ,
    "neq": FilterOperator.NEQ, "!=": FilterOperator.NEQ, "<>": FilterOperator.NEQ, "not_equal": FilterOperator.NEQ, "not equal": FilterOperator.NEQ, "is not": FilterOperator.NEQ,
    "in": FilterOperator.IN, "isin": FilterOperator.IN, "within": FilterOperator.IN, "one of": FilterOperator.IN,
    "not_in": FilterOperator.NOT_IN, "notin": FilterOperator.NOT_IN, "not in": FilterOperator.NOT_IN,
    "is_null": FilterOperator.IS_NULL, "isnull": FilterOperator.IS_NULL, "isna": FilterOperator.IS_NULL, "null": FilterOperator.IS_NULL, "missing": FilterOperator.IS_NULL,
    "is_not_null": FilterOperator.IS_NOT_NULL, "notnull": FilterOperator.IS_NOT_NULL, "notna": FilterOperator.IS_NOT_NULL, "not_null": FilterOperator.IS_NOT_NULL, "not null": FilterOperator.IS_NOT_NULL,
    "contains": FilterOperator.CONTAINS, "like": FilterOperator.CONTAINS, "includes": FilterOperator.CONTAINS
}


def _cast_numeric_if_possible(val: Any) -> Any:
    if isinstance(val, str):
        val_clean = val.strip()
        if re.match(r'^-?\d+$', val_clean):
            try:
                return int(val_clean)
            except ValueError:
                pass
        elif re.match(r'^-?\d+\.\d+$', val_clean):
            try:
                return float(val_clean)
            except ValueError:
                pass
    return val


def _get_first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely retrieves the value of the first key that is present in dictionary d.
    Crucially preserves 0, False, and empty strings ("").
    """
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    column: str
    operator: FilterOperator = FilterOperator.EQ
    value: Optional[Union[str, int, float, bool, List[Union[str, int, float]]]] = None

    @model_validator(mode="before")
    @classmethod
    def pre_validate_filter(cls, data: Any) -> Any:
        if isinstance(data, str):
            raise ValueError(f"FilterSpec requires a structured dictionary with 'column', 'operator', and 'value'. Got raw string: '{data}'")

        if isinstance(data, dict):
            col = _get_first_present(data, ["column", "col", "field", "name"])
            if not col or not str(col).strip():
                raise ValueError("FilterSpec requires a non-empty 'column' field.")

            raw_op = _get_first_present(data, ["operator", "condition", "op"], default="eq")
            if isinstance(raw_op, FilterOperator):
                norm_op = raw_op
            else:
                raw_op_str = str(raw_op).lower().strip()
                if "." in raw_op_str:
                    raw_op_str = raw_op_str.split(".")[-1]
                if raw_op_str not in _OP_LOOKUP:
                    raise ValueError(f"Invalid filter operator: '{raw_op_str}'. Supported operators: {list(set(_OP_LOOKUP.values()))}")
                norm_op = _OP_LOOKUP[raw_op_str]

            val = _get_first_present(data, ["value", "val", "target", "v"])
            val = _cast_numeric_if_possible(val)
            return {"column": str(col).strip(), "operator": norm_op, "value": val}

        return data


class IntentType(str, Enum):
    EDA = "EDA"
    GROUP_AGGREGATE = "GROUP_AGGREGATE"
    TIME_SERIES_FORECAST = "TIME_SERIES_FORECAST"
    AUTO_ML = "AUTO_ML"
    DATA_CLEAN = "DATA_CLEAN"
    CORRELATION = "CORRELATION"
    STATISTICAL_TEST = "STATISTICAL_TEST"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    CLUSTERING = "CLUSTERING"
    PCA = "PCA"
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    FILTER_QUERY = "FILTER_QUERY"
    GENERAL_CHAT = "GENERAL_CHAT"
    AUTOCORRELATION = "AUTOCORRELATION"
    TIME_SERIES_DECOMPOSITION = "TIME_SERIES_DECOMPOSITION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    CLARIFICATION = "CLARIFICATION"
    EVENT_WINDOW_ANALYSIS = "EVENT_WINDOW_ANALYSIS"
    SQL_QUERY = "SQL_QUERY"



class AggregationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HIST = "hist"
    BOX = "box"
    PIE = "pie"
    HEATMAP = "heatmap"
    NONE = "none"
    ACF = "acf"
    PACF = "pacf"


class SemanticIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    intent_type: IntentType
    primary_intent: Optional[IntentType] = None
    prerequisite_operations: List[IntentType] = Field(default_factory=list, description="Sequential prerequisite steps (e.g. FILTER_QUERY, DATA_CLEAN)")
    time_column: Optional[str] = Field(None, description="Explicit temporal/date column name for time-series operations")
    target_column: Optional[str] = Field(None, description="Explicit target column name for ML or forecast")
    feature_columns: List[str] = Field(default_factory=list, description="Explicit predictor/feature column names for ML")
    dimensions: List[str] = Field(default_factory=list, description="Categorical or grouping column names")
    metrics: List[str] = Field(default_factory=list, description="Numeric column names for calculation")
    aggregation: Optional[AggregationType] = None
    group_by: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    order: Optional[str] = Field(None, description="Sort order: asc or desc, or None if unspecified")
    limit: Optional[int] = Field(None, description="Explicit row limit, None unless user specifies limit/top-N")
    horizon: Optional[int] = None
    pre_event_window: Optional[int] = Field(None, description="Pre-event window size for event window analysis")
    event_predicate: Optional[Any] = Field(None, description="Event predicate or condition for event window analysis")
    sequence_column: Optional[str] = Field(None, description="Explicit ordered sequence column for sequence-based event analysis")
    time_context: Optional[Any] = Field(None, description="Time context details")
    chart_type: Optional[ChartType] = Field(ChartType.NONE, description="Chart visualization type")
    temporal_granularity: Optional[str] = None
    prompt_summary: str = ""
    raw_query: Optional[str] = None
    confidence: float = Field(1.0, description="Confidence score from 0.0 to 1.0")
    tool_name: Optional[str] = Field(None, description="The canonical or dynamic tool name executed")
    is_ambiguous: bool = Field(False, description="Flag indicating if user intent is underspecified or ambiguous")
    ambiguity_reason: Optional[str] = Field(None, description="Reason for ambiguity if is_ambiguous is True")
    schema_version: Optional[str] = None
    prompt_version: Optional[str] = None
    raw_llm_output: Optional[str] = None
    llm_latency_ms: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    repair_attempts: Optional[int] = None
    repair_history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    execution_plan: Optional[List[Dict[str, Any]]] = None
    slot_filling: Optional[Dict[str, Any]] = None
    inferred_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_columns: List[str] = Field(default_factory=list)
    is_dataset_wide: bool = False
    sql_query: Optional[str] = Field(None, description="Explicit SQL query string for SQL operations")

    @model_validator(mode="before")
    @classmethod
    def pre_validate_intent(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # Normalize intent_type and primary_intent
        raw_val = d.get("primary_intent") or d.get("intent_type")
        if raw_val is None:
            raise ValueError("Missing required field: 'intent_type' or 'primary_intent'")

        matched = None
        if isinstance(raw_val, IntentType):
            matched = raw_val
        else:
            raw_str = str(raw_val).upper().strip()
            if "." in raw_str:
                raw_str = raw_str.split(".")[-1]
            if raw_str in IntentType.__members__:
                matched = IntentType[raw_str]
            else:
                for member in IntentType:
                    if member.value.upper() == raw_str:
                        matched = member
                        break
        if matched is None:
            raise ValueError(f"Invalid intent_type: '{raw_val}'. Must be one of {[m.value for m in IntentType]}")
        d["intent_type"] = matched
        d["primary_intent"] = matched

        # Normalize prerequisite_operations
        raw_prereqs = d.get("prerequisite_operations") or []
        norm_prereqs = []
        if isinstance(raw_prereqs, list):
            for pr in raw_prereqs:
                if isinstance(pr, IntentType):
                    norm_prereqs.append(pr)
                else:
                    pr_str = str(pr.value if hasattr(pr, "value") else pr).upper().strip()
                    if "." in pr_str:
                        pr_str = pr_str.split(".")[-1]
                    if pr_str in IntentType.__members__:
                        norm_prereqs.append(IntentType[pr_str])
        d["prerequisite_operations"] = norm_prereqs

        # Normalize filters
        raw_filters = d.get("filters") or []
        norm_filters = []
        if isinstance(raw_filters, list):
            for f in raw_filters:
                if isinstance(f, FilterSpec):
                    norm_filters.append(f)
                elif isinstance(f, (dict, str)):
                    norm_filters.append(FilterSpec.model_validate(f))
                else:
                    raise ValueError(f"Invalid filter format: {f}")
        d["filters"] = norm_filters

        # Normalize aggregation
        raw_agg = d.get("aggregation")
        if raw_agg is not None and raw_agg != "":
            if isinstance(raw_agg, AggregationType):
                d["aggregation"] = raw_agg
            else:
                agg_str = str(raw_agg).lower().strip()
                agg_map = {
                    "mean": AggregationType.MEAN, "avg": AggregationType.MEAN, "average": AggregationType.MEAN,
                    "sum": AggregationType.SUM, "total": AggregationType.SUM,
                    "count": AggregationType.COUNT, "min": AggregationType.MIN,
                    "max": AggregationType.MAX, "median": AggregationType.MEDIAN
                }
                if agg_str not in agg_map:
                    raise ValueError(f"Invalid aggregation: '{raw_agg}'. Supported: {list(set(agg_map.values()))}")
                d["aggregation"] = agg_map[agg_str]

        # Normalize order
        raw_order = d.get("order")
        if raw_order is not None:
            ord_str = str(raw_order).lower().strip()
            if ord_str in ["desc", "descending", "highest", "top", "largest"]:
                d["order"] = "desc"
            elif ord_str in ["asc", "ascending", "lowest", "bottom", "smallest"]:
                d["order"] = "asc"
            else:
                d["order"] = None

        # Clean string nulls
        for k in ["time_column", "target_column"]:
            if isinstance(d.get(k), str) and d[k].lower().strip() in ["none", "null", "n/a", ""]:
                d[k] = None

        return d


class TaskFamily(str, Enum):
    DISTRIBUTION_COMPARISON = "distribution_comparison"
    EVENT_WINDOW_ANALYSIS = "event_window_analysis"
    GROUP_AGGREGATION = "group_aggregation"
    FILTER_SLICE = "filter_slice"
    CORRELATION_ANALYSIS = "correlation_analysis"
    CATEGORICAL_OUTCOME_ASSOCIATION = "categorical_outcome_association"
    TIME_SERIES_FORECAST = "time_series_forecast"
    ANOMALY_DETECTION = "anomaly_detection"
    SUPERVISED_ML = "supervised_ml"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    EXPLAINABILITY = "explainability"
    STATISTICAL_TEST = "statistical_test"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    GENERAL_EDA = "general_eda"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION = "clarification"
    AUTOCORRELATION = "autocorrelation"
    TIME_SERIES_DECOMPOSITION = "time_series_decomposition"
    SQL_QUERY = "sql_query"
    UNSUPPORTED = "unsupported"


class TimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    entity_id: Optional[str] = None
    time_column: Optional[str] = None
    sequence_column: Optional[str] = None
    event_column: Optional[str] = None
    event_predicate: Optional[Any] = None
    window: Optional[Union[str, int]] = None
    pre_event_window: Optional[int] = None
    post_event_window: Optional[int] = None
    frequency: Optional[str] = None
    horizon: Optional[int] = None


class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    type: str = "none"  # "box", "histogram", "bar", "line", "scatter", "heatmap", "pie", "violin", "none"
    x: Optional[str] = None
    y: Optional[Union[str, List[str]]] = None
    facet: Optional[str] = None
    color: Optional[str] = None


class CorrelationSpec(BaseModel):
    """Explicit analytical contract for a correlation request.

    These fields describe the requested statistical evidence and visualization;
    they are not inferred by the deterministic operator from query wording.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    analysis_mode: Literal["pairwise", "matrix"] = Field(description="pairwise for exactly two subjects; matrix for a multi-feature correlation matrix")
    relationship_form: Literal["linear", "monotonic", "nonlinear"] = Field(description="Requested association form; nonlinear requires LOWESS plus Pearson and Spearman")
    correlation_methods: List[Literal["pearson", "spearman", "kendall"]] = Field(min_length=1, description="Explicit statistical coefficient(s) to compute")
    chart_type: Literal["scatter", "heatmap", "none"] = Field(description="Requested visualization; scatter is required for pairwise nonlinear analysis")
    smoother: Literal["none", "lowess"] = Field(description="Descriptive smoother; LOWESS is required for a nonlinear scatter request")
    expected_direction: Literal["positive", "negative", "unspecified"] = Field(description="Direction explicitly claimed or requested by the user: positive for direct/increasing, negative for inverse/decreasing, unspecified only when no direction was stated")

    @model_validator(mode="after")
    def validate_analysis_contract(self):
        methods = set(self.correlation_methods)
        if self.analysis_mode == "matrix":
            if len(self.correlation_methods) != 1:
                raise ValueError("Matrix correlation analysis requires exactly one correlation method per request.")
            if self.chart_type not in {"heatmap", "none"}:
                raise ValueError("Matrix correlation analysis supports only 'heatmap' or 'none' chart types.")
            if self.smoother != "none":
                raise ValueError("Matrix correlation analysis cannot use a pairwise smoother.")
            if self.expected_direction != "unspecified":
                raise ValueError("A matrix request cannot claim one expected direction for all pairs.")
        else:
            if self.chart_type not in {"scatter", "none"}:
                raise ValueError("Pairwise correlation analysis supports only 'scatter' or 'none' chart types.")
            if self.smoother == "lowess" and self.chart_type != "scatter":
                raise ValueError("LOWESS requires a scatter chart.")

        if self.relationship_form == "linear" and "pearson" not in methods:
            raise ValueError("Linear correlation analysis requires the Pearson method.")
        if self.relationship_form == "monotonic" and "spearman" not in methods:
            raise ValueError("Monotonic correlation analysis requires the Spearman method.")
        if self.relationship_form == "nonlinear":
            if self.analysis_mode != "pairwise":
                raise ValueError("Nonlinear correlation analysis must be pairwise.")
            if self.chart_type != "scatter" or self.smoother != "lowess":
                raise ValueError("Nonlinear correlation analysis requires a LOWESS scatter chart.")
            if not {"pearson", "spearman"}.issubset(methods):
                raise ValueError("Nonlinear correlation analysis requires both Pearson and Spearman evidence.")
        return self


class ExecutionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    operator: str
    columns_used: List[str] = Field(default_factory=list)
    filters_applied: List[Dict[str, Any]] = Field(default_factory=list)
    statistics_computed: List[str] = Field(default_factory=list)
    chart_type: str = "none"
    row_count_in: int = 0
    row_count_out: int = 0
    row_count_after_filter: Optional[int] = None
    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    event_row_count: Optional[int] = None
    pre_event_row_count: Optional[int] = None
    predicate_applied: Optional[str] = None
    sorted_by: Optional[str] = None
    partitioned_by: Optional[str] = None


class AnalysisIntentV2(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    task_family: TaskFamily
    operations: List[str] = Field(default_factory=list, description="Sequence of operator names (e.g. ['filter', 'distribution_comparison'])")
    subjects: List[str] = Field(default_factory=list, description="Target column names to analyze")
    target_column: Optional[str] = Field(None, description="Primary target/value column")
    time_column: Optional[str] = Field(None, description="Primary temporal column")
    sequence_column: Optional[str] = Field(None, description="Primary explicit sequence-order column")
    horizon: Optional[int] = Field(None, description="Forecast horizon")
    conditions: List[FilterSpec] = Field(default_factory=list, description="Filtering conditions")
    grouping: List[str] = Field(default_factory=list, description="Grouping / dimension column names")
    requires_grouping: bool = Field(
        default=True,
        description="Execution contract derived from the selected ToolSpec; false permits one overall aggregate result.",
    )
    time_context: Optional[TimeContext] = None
    statistical_requirements: List[str] = Field(default_factory=list, description="e.g. ['mean', 'std', 'quantiles', 'iqr', 'variance', 'correlation', 'p_value']")
    visualization: Optional[VisualizationSpec] = None
    correlation_spec: Optional[CorrelationSpec] = None
    required_artifacts: List[str] = Field(default_factory=lambda: ["table", "chart", "written_finding"])
    assumptions: List[str] = Field(default_factory=list)
    prompt_summary: Optional[str] = None
    raw_query: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Dynamic tool call arguments")
