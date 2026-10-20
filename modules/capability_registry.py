import os
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.ds_engine.intent_schema import ChartType, CorrelationSpec, TaskFamily
from backend.ds_engine.analysis_spec_v3 import (
    AnalysisMode,
    AnalysisObjective,
    AnalysisSpecV3,
    RequestedArtifact,
)
from backend.ds_engine.time_frequency import canonicalize_pandas_frequency

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class RegistryContractError(ValueError):
    """Raised when a ToolSpec contract, completion policy, or metadata validation fails."""
    pass

class ToolPrerequisites(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requires_temporal_column: bool = False
    requires_ordering_column: bool = False
    requires_numeric_subjects: bool = False
    requires_event_predicate: bool = False
    requires_event_column: bool = False
    requires_pre_event_window: bool = False
    requires_grouping_or_dimension: bool = False
    requires_target_column: bool = False
    # Name of a ToolSpec argument that must contain a FilterSpec-compatible
    # structured condition.  This lets capabilities declare predicate
    # requirements without the feasibility engine knowing tool names.
    condition_argument_name: Optional[str] = None
    # A capability may require a numeric argument to be grounded in an exact
    # phrase from the user's request.  This is capability metadata, not a
    # query-keyword router: the resolver only proves that the model did not
    # manufacture a number that the user never supplied.
    requires_quantified_request_evidence: bool = False
    quantity_argument_name: Optional[str] = None
    quantity_evidence_argument_name: Optional[str] = None
    # Aggregating capabilities declare their measure arguments and operations
    # that do not need a numeric measure (such as record count). The resolver
    # can then validate types without knowing a tool name.
    requires_numeric_measure: bool = False
    measure_argument_names: List[str] = Field(default_factory=list)
    aggregation_argument_name: Optional[str] = None
    non_numeric_measure_aggregations: List[str] = Field(default_factory=list)
    min_numeric_columns: int = 0

class SlotPresentationPolicy(StrictBaseModel):
    """
    Declarative Slot Presentation and Candidate Sourcing Policy for Conversational Slot Cards.
    Specifies how a missing slot is labeled, sourced from profile, and presented in the UI.
    """
    label: str
    role_source: Optional[str] = None
    fallback_roles: List[str] = Field(default_factory=list)
    max_candidates: int = 5
    slot_type: str = "select"  # select | multiselect
    is_pairwise: bool = False
    pairwise_slots: List[Tuple[str, str]] = Field(default_factory=list)
    presentation_slot_name: Optional[str] = None
    default_value: Optional[str] = None
    options: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    prompt_template: Optional[str] = None
    suggested_queries: List[str] = Field(default_factory=list)

class CompletionPolicy(StrictBaseModel):
    """
    Declarative Intent Completion & Slot-Filling Policy for a ToolSpec.
    Determines how partial, implicit, or underspecified commands are completed
    using statistical evidence from DatasetProfile or conversational slot cards.
    """
    allowed_dataset_wide_mode: bool = False
    dataset_wide_role: Optional[str] = None
    dataset_wide_target_arg: Optional[str] = None
    dataset_wide_min_columns: int = 1
    dataset_wide_required_mode_arg: Optional[Tuple[str, str]] = None
    dataset_wide_default_args: Dict[str, Any] = Field(default_factory=dict)

    allow_role_candidate_fallback: bool = False
    fallback_role_name: Optional[str] = None
    argument_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    slot_presentations: Dict[str, SlotPresentationPolicy] = Field(default_factory=dict)
    allow_inferred_frequency: bool = False
    frequency_confidence_threshold: float = 0.8
    frequency_slot_name: Optional[str] = "frequency"

    required_slots: List[str] = Field(default_factory=list)
    inferable_slots: Dict[str, str] = Field(default_factory=dict)
    can_auto_infer_target: bool = False

    minimum_items_by_slot: Dict[str, int] = Field(default_factory=dict)
    requires_temporal_column: bool = False
    temporal_slot_name: Optional[str] = None

    requires_explicit_method: bool = False
    required_method_slot: str = "method"
    allowed_methods: List[str] = Field(default_factory=list)

    requires_explicit_sensitivity: bool = False
    required_sensitivity_slot: str = "sensitivity"
    allowed_sensitivities: List[float] = Field(default_factory=list)

    is_clarification_tool: bool = False
    clarification_category: Optional[str] = None

    slot_filling_type: Optional[str] = None
    slot_filling_title: Optional[str] = None
    slot_filling_prompt: Optional[str] = None
    default_query_template: str = ""

    confidence_threshold: float = 0.85
    ambiguity_margin: float = 0.15

    post_completion_defaults: Dict[str, Any] = Field(default_factory=dict)
    infers_frequency_from_temporal: bool = False
    multi_column_threshold: Optional[int] = None
    multi_column_overrides: Dict[str, Any] = Field(default_factory=dict)
    multi_column_first_item_slots: List[str] = Field(default_factory=list)

class FilterCondition(StrictBaseModel):
    column: str
    condition: Optional[str] = "=="
    operator: Optional[str] = None
    value: Optional[Union[str, int, float, bool, List[Union[str, int, float]]]] = None

class FilterArgs(StrictBaseModel):
    column: Optional[Union[str, List[str]]] = None
    columns: Optional[List[str]] = None
    condition: Optional[str] = None
    operator: Optional[str] = None
    value: Optional[Union[str, int, float, bool, List[Union[str, int, float]]]] = None
    conditions: Optional[List[FilterCondition]] = None


class DatetimeParseSpec(StrictBaseModel):
    """An explicit, non-mutating datetime conversion for an analysis plan."""
    column: str = Field(description="Exact source column to parse as datetimes.")
    format: str = Field(description="Required Python strptime format, for example %d/%m/%Y for DD/MM/YYYY values.")

    @field_validator("format", mode="after")
    @classmethod
    def validate_and_normalize_format(cls, v: str) -> str:
        from backend.ds_engine.datetime_parser import normalize_datetime_format
        return normalize_datetime_format(v)

class GroupArgs(StrictBaseModel):
    group_by: List[str] = Field(description="Requested output grouping dimensions / categorical columns.")
    metric: Optional[Union[str, List[str]]] = Field(default=None, description="Exact numeric measure column(s) being aggregated; omit only for an explicit record count.")
    metrics: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    aggregation: Optional[Literal["sum", "mean", "count", "min", "max", "median"]] = Field(
        default=None,
        description="Requested computation: 'mean' for average/mean, 'sum' for total/sum, 'count' for frequency, 'min'/'max' for extremes, 'median' for 50th percentile. When metric is supplied, aggregation is required.",
    )
    order: Optional[str] = None
    direction: Optional[str] = None
    ascending: Optional[bool] = None
    limit: Optional[int] = None
    filters: Optional[List[Union[FilterCondition, Dict[str, Any]]]] = None
    datetime_parses: List[DatetimeParseSpec] = Field(
        default_factory=list,
        description="Optional explicit datetime parsing steps applied to the analysis copy before aggregation. Preprocessing only: its source column must not be substituted into group_by unless explicitly requested.",
    )
    @field_validator("aggregation", mode="before")
    @classmethod
    def normalize_aggregation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            aliases = {
                "average": "mean",
                "avg": "mean",
                "total": "sum",
                "maximum": "max",
                "minimum": "min",
            }
            return aliases.get(v_lower, v_lower)
        return v

    chart_type: Literal["bar", "line", "none"] = Field(
        default="none",
        description="Requested output chart: 'line' for an ordered/temporal profile, 'bar' for categorical comparisons, or 'none' for table-only result.",
    )


class GroupPlannerArgs(StrictBaseModel):
    """Canonical, compact LLM contract for grouped aggregation planning.

    ``GroupArgs`` accepts several legacy/provider aliases because it also
    validates manual and API commands.  Those aliases create a very large
    grammar and invite a small local model to put the same column into several
    mutually incompatible fields.  The semantic planner instead receives this
    canonical projection.  It is still validated by ``GroupArgs`` before
    execution, so this narrows generation without adding a fallback path.
    """

    group_by: List[str] = Field(
        min_length=1,
        description=(
            "Exact output dimensions. Bind only the columns in the user's requested "
            "breakdown clause (for example, the columns after 'by', 'per', 'across', "
            "or their natural-language equivalent). A datetime source named only in a "
            "parse/format instruction belongs in datetime_parses, never here."
        ),
    )
    metric: str = Field(
        description="One exact numeric measure to aggregate. Never use a datetime parsing source as this metric unless the user explicitly requests it.",
    )
    aggregation: Literal["sum", "mean", "count", "min", "max", "median"] = Field(
        description="Required metric computation: mean for average, sum for total, count for non-missing metric values, min/max for extremes, median for 50th percentile.",
    )
    datetime_parses: List[DatetimeParseSpec] = Field(
        default_factory=list,
        description=(
            "Optional requested datetime preprocessing only. Map a parse/format clause "
            "to this field. Its source column must not be copied into group_by unless the "
            "query separately requests a breakdown by that same column."
        ),
    )
    chart_type: Literal["bar", "line", "none"] = Field(
        description=(
            "Required output decision: use the chart explicitly requested in the query; "
            "use none only when the user requests a table-only result or does not ask for a visualization."
        ),
    )
    order: Optional[Literal["asc", "desc"]] = Field(
        default=None,
        description="Optional output ordering only when the user explicitly requests it.",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional result limit only when the user explicitly requests it.",
    )
    filters: Optional[List[FilterCondition]] = Field(
        default=None,
        description="Optional typed row filters explicitly requested by the user.",
    )



class GroupCountPlannerArgs(StrictBaseModel):
    """Separate canonical branch for an explicitly requested grouped count."""

    group_by: List[str] = Field(
        min_length=1,
        description=(
            "Exact output dimensions. Bind only the columns in the user's requested "
            "breakdown clause, not a column mentioned solely for parsing or formatting."
        ),
    )
    aggregation: Literal["count"] = Field(
        description="Record-count computation. Select only when the user requests a count/frequency/number of records.",
    )
    datetime_parses: List[DatetimeParseSpec] = Field(
        default_factory=list,
        description="Optional requested datetime preprocessing. A parsed source column is not an output dimension unless explicitly requested.",
    )
    chart_type: Literal["bar", "line", "none"] = Field(
        description="Required output decision: bar, line, or none when no visualization is requested."
    )
    order: Optional[Literal["asc", "desc"]] = Field(default=None)
    limit: Optional[int] = Field(default=None, ge=1)
    filters: Optional[List[FilterCondition]] = Field(default=None)


class ConditionalRateArgs(StrictBaseModel):
    """A typed numerator/denominator aggregation over one or more segments.

    The denominator is every row in each ``group_by`` cohort.  The numerator
    is the subset matching ``numerator_condition``.  Keeping the predicate
    structured prevents query-string interpolation and makes conversion,
    defect, approval, churn, and response rates one data-agnostic capability.
    """
    group_by: List[str] = Field(min_length=1, description="Exact categorical segment column(s) that define each denominator cohort.")
    numerator_condition: FilterCondition = Field(description="Typed predicate defining a successful/positive numerator record.")
    min_group_size: Optional[int] = Field(default=None, ge=1, description="Only retain cohorts with at least this many denominator records.")
    sort_by: Optional[Literal["rate_percent", "matching_count", "total_count"]] = Field(default=None, description="Optional explicit ranking metric.")
    direction: Optional[Literal["asc", "desc"]] = Field(default=None, description="Ranking direction when sort_by is supplied.")
    limit: Optional[int] = Field(default=None, ge=1, description="Optional explicit maximum number of displayed cohorts.")
    chart_type: Literal["bar", "none"] = Field(description="Use bar for a percentage bar chart, or none for a table-only result.")

class AggregateArgs(StrictBaseModel):
    metric: Optional[str] = None
    column: Optional[str] = None
    columns: Optional[List[str]] = None
    aggregation: Literal["sum", "mean", "count", "min", "max", "median"] = Field(
        description="Exact aggregation operation: 'mean' for average/mean, 'sum' for total/sum, 'count' for frequency/count, 'min' for minimum, 'max' for maximum, 'median' for median."
    )
    order: Optional[str] = None
    direction: Optional[str] = None
    ascending: Optional[bool] = None
    limit: Optional[int] = None
    filters: Optional[List[Union[FilterCondition, Dict[str, Any]]]] = None

    @field_validator("aggregation", mode="before")
    @classmethod
    def normalize_aggregation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            aliases = {
                "average": "mean",
                "avg": "mean",
                "total": "sum",
                "maximum": "max",
                "minimum": "min",
            }
            return aliases.get(v_lower, v_lower)
        return v


class MetricAuditArgs(StrictBaseModel):
    """A filtered descriptive audit for one numeric measure.

    This is intentionally distinct from ``aggregate_data``.  An aggregate is
    one scalar; a metric audit returns the evidence needed to assess a
    distribution (including observed zeros) after an explicitly declared
    typed filter has been applied.
    """
    metric: str = Field(description="Exact numeric metric column to audit.")
    filters: List[Union[FilterCondition, Dict[str, Any]]] = Field(
        min_length=1,
        description="One or more typed predicates that define the audited subset."
    )
    chart_type: Literal["box", "none"] = Field(
        default="box",
        description="Use box for a distribution/variance chart or none for a table-only audit."
    )

class SortArgs(StrictBaseModel):
    column: Optional[str] = None
    sort_by: Optional[str] = None
    direction: Optional[str] = "desc"
    order: Optional[str] = None
    ascending: Optional[bool] = None
    limit: Optional[int] = None

class ForecastArgs(StrictBaseModel):
    value_column: Optional[str] = None
    target_column: Optional[str] = None
    column: Optional[str] = None
    metric: Optional[str] = None
    horizon: int = Field(ge=1, description="Explicit number of future periods to forecast.")
    horizon_evidence: Optional[str] = Field(
        default=None,
        description=(
            "Exact contiguous phrase from the user's request that states the numeric forecast horizon, "
            "for example 'next 30 days'. Never invent this value."
        ),
    )
    time_column: str = Field(description="Exact temporal column that defines the observed time grid.")
    frequency: str = Field(description="Explicit calendar interval for future timestamps, e.g. D, W, MS, h. Legacy pandas spellings are canonicalized and the observed grid must match exactly.")
    method: Literal["naive_last_value", "holt_linear"] = Field(description="Explicit forecasting method. naive_last_value is a baseline; holt_linear fits level and damped trend without claiming calibrated intervals.")
    chart_type: Literal["line", "none"] = Field(description="Use line for a forecast chart or none for table-only output.")
    datetime_format: Optional[str] = Field(default=None, description="Optional explicit datetime format for parsing the temporal column (e.g. DD/MM/YYYY or %d/%m/%Y).")

    @field_validator("datetime_format")
    @classmethod
    def validate_datetime_format(cls, value: Optional[str]) -> Optional[str]:
        if value:
            from backend.ds_engine.datetime_parser import normalize_datetime_format
            return normalize_datetime_format(value)
        return value

    @field_validator("frequency")
    @classmethod
    def canonicalize_frequency(cls, value: str) -> str:
        return canonicalize_pandas_frequency(value)

class CleanArgs(StrictBaseModel):
    columns: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    method: Literal[
        "drop_missing", "drop_nulls",
        "impute_mean", "impute_median", "impute_mode", "impute_constant",
        "remove_duplicates", "drop_duplicates",
        "clip_iqr",
        "minmax_scale", "normalize_minmax",
        "standardize", "z_score",
        "one_hot_encode",
        "yeo_johnson", "power_transform",
        "log_transform",
        "lag", "create_lag",
        "rolling", "create_rolling",
        "difference", "create_difference",
        "calendar_features", "create_calendar_features"
    ] = Field(
        description="Explicit cleaning or feature transformation operation: drop missing rows, imputation (mean/median/mode/constant), duplicate removal, IQR outlier capping, Min-Max scaling, standardization (z-score), one-hot encoding, Yeo-Johnson power transform, log transform, lag features, rolling window features, differencing, or calendar feature extraction."
    )
    fill_value: Optional[Union[str, int, float, bool]] = Field(
        default=None,
        description="Constant value to use when method is impute_constant."
    )
    n_bins: Optional[int] = Field(default=5, description="Number of bins for binning")
    time_column: Optional[str] = Field(default=None, description="Temporal or sequence ordering column for temporal feature engineering.")
    group_column: Optional[str] = Field(default=None, description="Optional entity grouping column to prevent cross-entity contamination.")
    lag_order: Optional[int] = Field(default=None, ge=1, description="Lag step order k (k >= 1).")
    window_size: Optional[int] = Field(default=None, ge=2, description="Trailing rolling window size W (W >= 2).")
    aggregation: Optional[Literal["mean", "std", "min", "max", "sum"]] = Field(default=None, description="Rolling window aggregation statistic.")
    min_periods: Optional[int] = Field(default=None, ge=1, description="Minimum observations required in window.")
    new_name: Optional[str] = Field(default=None, description="Optional explicit name for newly created feature column.")
    is_target: Optional[bool] = Field(default=None, description="Flag indicating if the feature being rolled is the supervised target (requires shift(1) leakage protection).")
    datetime_format: Optional[str] = Field(default=None, description="Optional explicit datetime format for parsing the temporal column.")

    @field_validator("datetime_format")
    @classmethod
    def validate_clean_datetime_format(cls, value: Optional[str]) -> Optional[str]:
        if value:
            from backend.ds_engine.datetime_parser import normalize_datetime_format
            return normalize_datetime_format(value)
        return value

    @model_validator(mode="after")
    def require_columns_when_needed(self):
        selected = self.columns or ([self.column] if isinstance(self.column, str) else self.column)
        if self.method in ("calendar_features", "create_calendar_features"):
            if not selected and not self.time_column:
                raise ValueError("Calendar feature extraction requires an explicit time_column or column.")
            return self
        if self.method not in ("remove_duplicates", "drop_duplicates") and not selected:
            raise ValueError(f"Cleaning or transformation method '{self.method}' requires explicit columns.")
        return self

TransformArgs = CleanArgs

class AnomalyArgs(StrictBaseModel):
    columns: List[str] = Field(min_length=1)
    method: Literal["isolation_forest", "iqr", "zscore"] = Field(
        description="Explicit anomaly definition; the engine never selects an algorithm from the dataset name or query keywords."
    )
    sensitivity: float = Field(
        description="Method parameter: contamination fraction for isolation_forest, IQR multiplier for iqr, or absolute z-score threshold for zscore.",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_sensitivity(self):
        if self.method == "isolation_forest" and not 0 < self.sensitivity < 0.5:
            raise ValueError("isolation_forest sensitivity must be a contamination fraction between 0 and 0.5")
        return self

class CorrelationArgs(CorrelationSpec):
    """Tool-level correlation request with explicit subjects and analysis semantics."""
    columns: List[str] = Field(min_length=2, description="Exact two columns for pairwise mode, or two or more numeric columns for matrix mode.")

    @model_validator(mode="before")
    @classmethod
    def reconcile_mode(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cols = data.get("columns")
            if isinstance(cols, list) and len(cols) > 2:
                data["analysis_mode"] = "matrix"
                if data.get("chart_type") not in ("heatmap", "none"):
                    data["chart_type"] = "heatmap"
                data["smoother"] = "none"
                data["expected_direction"] = "unspecified"
                methods = data.get("correlation_methods")
                if not methods:
                    data["correlation_methods"] = ["pearson"]
                elif len(methods) > 1:
                    data["correlation_methods"] = [methods[0]]
        return data

    @model_validator(mode="after")
    def validate_subject_cardinality(self):
        if self.analysis_mode == "pairwise" and len(self.columns) != 2:
            raise ValueError("Pairwise correlation analysis requires exactly two columns.")
        return self

class EDAArgs(StrictBaseModel):
    columns: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    target_column: Optional[str] = None
    metrics: Optional[List[str]] = None
    chart_type: Optional[ChartType] = Field(default=None, strict=False)
    chart: Optional[str] = None
    lag: Optional[int] = None
    lags: Optional[int] = None

class TrainModelArgs(StrictBaseModel):
    target_column: Optional[str] = Field(default=None, description="Exact existing label/target column to predict for supervised classification or regression.")
    target: Optional[str] = Field(default=None, description="Alias for target_column.")
    feature_columns: Optional[List[str]] = Field(default=None, description="Exact predictor feature columns for the supervised model.")
    features_columns: Optional[List[str]] = Field(default=None, description="Alias for feature_columns.")
    features: Optional[List[str]] = Field(default=None, description="Alias for feature_columns.")
    task_type: Literal["classification", "regression"] = Field(description="Requested supervised task: classification or regression.")
    evaluation_artifact: Literal["confusion_matrix", "feature_importance", "none"] = Field(
        description="Requested model-evaluation visualization: confusion_matrix for classification, feature_importance, or none.",
    )
    prediction_context: Literal["prospective_prediction", "retrospective_explanation"] = Field(
        description=(
            "Whether features must be available before the predicted outcome "
            "(prospective_prediction), or the task is a retrospective/explanatory analysis."
        ),
    )

class ClusterArgs(StrictBaseModel):
    columns: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    feature_columns: Optional[List[str]] = None
    group_by: Optional[List[str]] = None
    aggregation: Optional[Literal["sum", "mean", "count", "min", "max", "median"]] = Field(
        default=None,
        description="Aggregation method if clustering grouped/aggregated features: sum, mean, count, min, max, median.",
    )

    @field_validator("aggregation", mode="before")
    @classmethod
    def normalize_aggregation(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            aliases = {
                "average": "mean",
                "avg": "mean",
                "total": "sum",
                "maximum": "max",
                "minimum": "min",
            }
            return aliases.get(v_lower, v_lower)
        return v
    metric: Optional[str] = None
    metrics: Optional[List[str]] = None
    n_clusters: Optional[int] = Field(default=None, ge=2)
    k: Optional[int] = None
    scaling: Literal["standard", "none"] = Field(
        description="Explicit feature scaling for K-Means; standard and none are never chosen silently.",
    )

    @model_validator(mode="after")
    def require_cluster_contract(self):
        if self.n_clusters is None and self.k is None:
            raise ValueError("Clustering requires explicit n_clusters (or k).")
        if self.n_clusters is not None and self.k is not None and self.n_clusters != self.k:
            raise ValueError("n_clusters and k disagree.")
        return self

class ElbowAnalysisArgs(StrictBaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Feature columns to evaluate.")
    feature_columns: Optional[List[str]] = Field(default=None, description="Alias for columns.")
    column: Optional[Union[str, List[str]]] = Field(default=None, description="Alias for columns.")
    min_k: int = Field(default=2, ge=2, description="Minimum clusters to evaluate.")
    max_k: int = Field(default=8, ge=3, description="Maximum clusters to evaluate.")
    scaling: Literal["standard", "none"] = Field(
        default="standard",
        description="Feature scaling for K-Means: standard or none.",
    )

    @model_validator(mode="after")
    def validate_features(self):
        cols = self.columns or self.feature_columns or self.column
        if isinstance(cols, str):
            cols = [cols]
        if cols is None or (isinstance(cols, list) and len(cols) < 2):
            raise ValueError("Elbow analysis requires at least 2 feature columns.")
        if self.min_k >= self.max_k:
            raise ValueError("min_k must be strictly less than max_k.")
        return self

class StatisticalTestArgs(StrictBaseModel):
    # ``columns`` always has a deterministic positional meaning defined by
    # ``test_type``.  The aliases are normalized only at model-validation time
    # for provider compatibility; the operator never guesses a test from dtype.
    columns: List[str] = Field(min_length=2, description="Exactly two columns. For independent_ttest: two numeric samples; for one_way_anova: [numeric_measurement, categorical_group]; for chi_square_independence: two categorical variables; for stationarity: [time_column, target_column].")
    test_type: Literal["independent_ttest", "one_way_anova", "chi_square_independence", "stationarity"] = Field(description="The exact inferential test to run; DataPilot will not infer this from data types.")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Declared significance threshold. Defaults visibly to 0.05 when the user does not specify a study-specific threshold.")

    @model_validator(mode="before")
    @classmethod
    def normalize_provider_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "columns" not in normalized:
            raw_columns = normalized.get("metrics", normalized.get("column"))
            if isinstance(raw_columns, str):
                raw_columns = [raw_columns]
            if raw_columns is not None:
                normalized["columns"] = raw_columns
        aliases = {
            "t_test": "independent_ttest",
            "ttest": "independent_ttest",
            "anova": "one_way_anova",
            "chi_square": "chi_square_independence",
            "chi_square_test": "chi_square_independence",
            "adf": "stationarity",
            "kpss": "stationarity",
            "stationarity_test": "stationarity",
            "unit_root": "stationarity",
        }
        raw_type = normalized.get("test_type")
        if isinstance(raw_type, str):
            normalized["test_type"] = aliases.get(raw_type.strip().lower(), raw_type.strip().lower())
        normalized.pop("metrics", None)
        normalized.pop("column", None)
        return normalized

class StationarityTestArgs(StrictBaseModel):
    time_column: str = Field(description="Temporal column for chronological ordering.")
    target_column: Optional[str] = Field(default=None, description="Numeric series column to test.")
    value_column: Optional[str] = Field(default=None, description="Alias for target_column.")
    column: Optional[str] = Field(default=None, description="Alias for target_column.")
    max_lags: Optional[int] = Field(default=None, description="Maximum lags for ADF test.")
    datetime_format: Optional[str] = Field(default=None, description="Optional explicit datetime format for parsing the temporal column.")

    @field_validator("datetime_format")
    @classmethod
    def validate_stationarity_datetime_format(cls, value: Optional[str]) -> Optional[str]:
        if value:
            from backend.ds_engine.datetime_parser import normalize_datetime_format
            return normalize_datetime_format(value)
        return value

    @model_validator(mode="after")
    def require_target(self):
        if not (self.target_column or self.value_column or self.column):
            raise ValueError("Stationarity testing requires target_column, value_column, or column.")
        return self

class CategoricalOutcomeAssociationArgs(StrictBaseModel):
    """Descriptive association between one numeric measure and one class/outcome.

    This intentionally is not a predictive-model contract.  It is suitable for
    retrospective campaign, quality, clinical-research, and operations review
    where a numeric measurement is compared across outcome groups.
    """
    numeric_predictor: str = Field(description="Exact numeric measurement whose distribution is compared across outcome groups.")
    outcome_column: str = Field(description="Exact categorical or binary outcome/group column.")
    chart_type: Literal["box", "none"] = Field(
        default="box",
        description="Box gives a distribution comparison by outcome; none returns the summary table only.",
    )
    analysis_context: Literal["retrospective_association"] = Field(
        default="retrospective_association",
        description="This capability is descriptive and retrospective only; it does not establish causality or a prospective targeting model.",
    )

class AutocorrelationArgs(StrictBaseModel):
    time_column: Optional[str] = None
    target_column: Optional[str] = None
    value_column: Optional[str] = None
    columns: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    max_lags: Optional[int] = None
    lag: Optional[int] = None
    lags: Optional[int] = None
    datetime_format: Optional[str] = Field(default=None, description="Optional explicit datetime format for parsing the temporal column.")

    @field_validator("datetime_format")
    @classmethod
    def validate_autocorr_datetime_format(cls, value: Optional[str]) -> Optional[str]:
        if value:
            from backend.ds_engine.datetime_parser import normalize_datetime_format
            return normalize_datetime_format(value)
        return value

class PCAAnalysisArgs(StrictBaseModel):
    columns: Optional[List[str]] = None
    column: Optional[Union[str, List[str]]] = None
    n_components: Optional[int] = None
    feature_columns: Optional[List[str]] = None
    scaling: Optional[Literal["standard", "none"]] = "standard"

class ExplainFeaturesArgs(StrictBaseModel):
    target_column: Optional[str] = None
    target: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    features: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    task_type: Literal["classification", "regression"]
    method: Literal["permutation_importance"]
    prediction_context: Literal["prospective_prediction", "retrospective_explanation"]

    @model_validator(mode="after")
    def require_explicit_roles(self):
        target = self.target_column or self.target
        features = self.feature_columns or self.features or self.columns
        if not target:
            raise ValueError("Feature explanation requires an explicit target_column.")
        if not features:
            raise ValueError("Feature explanation requires explicit feature_columns.")
        if target in features:
            raise ValueError("The target column cannot also be an explanatory feature.")
        return self

class DecomposeTimeSeriesArgs(StrictBaseModel):
    time_column: str
    target_column: Optional[str] = None
    value_column: Optional[str] = None
    period: int = Field(ge=2)
    model: Literal["additive", "multiplicative"]
    datetime_format: Optional[str] = Field(default=None, description="Optional explicit datetime format for parsing the temporal column.")

    @field_validator("datetime_format")
    @classmethod
    def validate_decompose_datetime_format(cls, value: Optional[str]) -> Optional[str]:
        if value:
            from backend.ds_engine.datetime_parser import normalize_datetime_format
            return normalize_datetime_format(value)
        return value

    @model_validator(mode="after")
    def require_target(self):
        if not (self.target_column or self.value_column):
            raise ValueError("Time-series decomposition requires target_column or value_column.")
        return self

class GeneralChatArgs(StrictBaseModel):
    response: Optional[str] = None
    message: Optional[str] = None
    text: Optional[str] = None
    topic: Optional[str] = None

class ClarificationArgs(StrictBaseModel):
    reason: str

class SQLQueryArgs(StrictBaseModel):
    query: str = Field(description="DuckDB SQL query string to execute against the active dataset table 'active_dataset'.")

class EventWindowArgs(StrictBaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Numeric measurements to compare across normal, pre-event, and event regimes.")
    column: Optional[Union[str, List[str]]] = Field(default=None, description="One numeric measurement, or an alias for columns.")
    target_column: Optional[str] = Field(default=None, description="Event indicator column; use event_column when possible.")
    event_column: Optional[str] = Field(default=None, description="Column whose values mark the event, such as a failure or defect flag.")
    time_column: Optional[str] = Field(default=None, description="Temporal ordering column when the dataset contains timestamps.")
    chart_type: Optional[ChartType] = Field(default=None, strict=False, description="Requested visualization type, for example box.")
    entity_id: Optional[str] = Field(default=None, description="Optional entity or machine identifier; windows are calculated separately per entity.")
    pre_event_window: Optional[int] = Field(default=None, ge=1, description="Explicit count of preceding ordered records to include before each event. Ask for clarification when absent.")
    event_predicate: Optional[Dict[str, Any]] = Field(default=None, description="Typed event condition: {column, operator, value}; for a binary failure flag use operator == and value 1.")
    sequence_order: Optional[str] = Field(default=None, description="Non-temporal ordering column when row order must be explicit.")
    order_column: Optional[str] = Field(default=None, description="Alias for sequence_order.")
    event_scope: Literal["event_rows", "event_onsets"] = Field(
        description="event_rows analyzes every matching row; event_onsets analyzes only false-to-true transitions within each entity/order sequence."
    )

class PlanDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["tool", "general_chat", "clarify"]
    commands: List[Dict[str, Any]] = Field(default_factory=list)
    clarification: Optional[str] = None

@dataclass
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    objective: AnalysisObjective
    analysis_modes: List[AnalysisMode]
    supported_artifacts: List[RequestedArtifact]
    # Optional compact canonical contract used only for LLM generation.  The
    # execution input_model remains authoritative for validation and adapters.
    planner_input_model: Optional[Type[BaseModel]] = None
    # A capability may expose several mutually exclusive canonical planner
    # forms (for example measured aggregation versus record count).
    planner_input_models: Optional[List[Type[BaseModel]]] = None
    requires_dataset: bool = True
    compiler_or_driver: Optional[Any] = None
    adapter: Optional[Callable[[Any, Dict[str, Any]], None]] = None
    result_contract: str = "dataframe" # "dataframe" | "text" | "plot" | "metric"
    side_effect_level: str = "read_only" # "read_only" | "modifies_data" | "trains_model"
    allowed_predecessors: List[str] = field(default_factory=list)
    category: str = "eda"
    synonyms: List[str] = field(default_factory=list)
    task_family: Optional[Any] = None
    prerequisites: Optional[ToolPrerequisites] = None
    planner_always_available: bool = False
    planner_without_dataset_only: bool = False
    execution_priority: int = 2
    is_prerequisite_only: bool = False
    requires_filter_conditions: bool = False
    # The physical operator output contract. This is distinct from the LLM's
    # requested chart argument: execution is rejected before runtime whenever
    # a tool would be asked to promise a visualization it cannot create.
    # Every capability declares this explicitly; table-only tools use
    # ["none"].  An absent physical chart contract is not production-ready.
    supported_chart_types: Optional[List[str]] = None
    completion_policy: CompletionPolicy = field(default=None)  # type: ignore

    def validate_contract(self) -> None:
        """Reject incomplete or internally inconsistent capability metadata."""
        if not self.name.strip() or not self.description.strip():
            raise ValueError("ToolSpec name and description are required.")
        if not issubclass(self.input_model, BaseModel):
            raise ValueError(f"ToolSpec '{self.name}' input_model must be a subclass of BaseModel.")
        if self.completion_policy is None or not isinstance(self.completion_policy, CompletionPolicy):
            raise RegistryContractError(f"ToolSpec '{self.name}' must declare a valid CompletionPolicy.")
        if not self.analysis_modes:
            raise ValueError(f"ToolSpec '{self.name}' must declare at least one analysis mode.")
        if not self.supported_artifacts:
            raise ValueError(f"ToolSpec '{self.name}' must declare supported artifacts.")
        if RequestedArtifact.MODEL in self.supported_artifacts:
            raise ValueError(
                f"ToolSpec '{self.name}' cannot declare RequestedArtifact.MODEL; "
                f"ephemeral model training outputs evaluation tables and metrics, not a persisted model artifact."
            )
        if self.supported_chart_types is None or len(self.supported_chart_types) == 0:
            raise ValueError(f"ToolSpec '{self.name}' must declare supported chart types (use ['none'] for table-only tools).")
        if self.adapter is None:
            raise ValueError(f"ToolSpec '{self.name}' must declare an execution adapter.")
        if self.requires_dataset:
            if self.task_family is None:
                raise ValueError(f"Dataset ToolSpec '{self.name}' must declare a task family.")
            if self.allowed_predecessors is None:
                raise ValueError(f"Dataset ToolSpec '{self.name}' must declare allowed predecessors.")
        chart_values = set(self.supported_chart_types or []) - {"none"}
        if chart_values and RequestedArtifact.CHART not in self.supported_artifacts:
            raise ValueError(
                f"ToolSpec '{self.name}' declares chart types but not the chart artifact."
            )
        if RequestedArtifact.CHART in self.supported_artifacts and not chart_values:
            raise ValueError(
                f"ToolSpec '{self.name}' declares RequestedArtifact.CHART but no specific chart types in supported_chart_types."
            )
        if self.result_contract not in {"dataframe", "text", "plot", "metric"}:
            raise ValueError(f"ToolSpec '{self.name}' has invalid result_contract '{self.result_contract}'.")
        if self.side_effect_level not in {"read_only", "modifies_data", "trains_model"}:
            raise ValueError(f"ToolSpec '{self.name}' has invalid side_effect_level '{self.side_effect_level}'.")

class CapabilityContract(BaseModel):
    capability_id: str
    name: str
    description: str
    category: str
    input_types: List[str]
    output_types: List[str]
    preconditions: List[str]
    postconditions: List[str]
    side_effects: List[str]
    safety_level: str

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class Capability:
    """
    Represents a specific system capability or skill that can be executed.
    """
    name: str
    category: str
    description: str
    intent_tags: List[str]
    required_col_types: List[str]
    output_type: str
    handler_func: Callable[..., Any]
    difficulty: int = 1


class CapabilityRegistry:
    """
    Unified Single Source of Truth Registry for managing, validating, and executing system tools.
    """

    def __init__(self, auto_bootstrap: bool = False) -> None:
        self._registry: Dict[str, Capability] = {}
        self._tool_specs: Dict[str, ToolSpec] = {}
        self._canonical_tool_names: set[str] = set()
        self._tool_definitions: Dict[str, str] = {}
        self._tool_intent_map: Dict[str, Any] = {}
        self._driver_registry: Dict[str, Any] = {}
        self._planner_embedding_cache: Dict[Tuple[str, Tuple[str, ...]], List[List[float]]] = {}
        if auto_bootstrap:
            self._bootstrap_canonical_tools()
        logger.info("CapabilityRegistry initialized.")

    def _bootstrap_canonical_tools(self) -> None:
        """
        Authoritative bootstrapping of canonical ToolSpecs, execution drivers, and adapters.
        """
        from backend.ds_engine.capability_drivers import (
            FilterQueryDriver, GroupAggregateDriver, CorrelationDriver,
            TimeSeriesForecastDriver, DataCleanDriver, AnomalyDetectionDriver,
            EDADriver, AutoMLDriver, ClusteringDriver, AutocorrelationDriver,
            FeatureEngineeringDriver, StatisticalTestDriver, DRIVER_REGISTRY
        )
        filter_driver = FilterQueryDriver()
        group_driver = GroupAggregateDriver()
        corr_driver = CorrelationDriver()
        forecast_driver = TimeSeriesForecastDriver()
        clean_driver = DataCleanDriver()
        anomaly_driver = AnomalyDetectionDriver()
        eda_driver = EDADriver()
        automl_driver = AutoMLDriver()
        cluster_driver = ClusteringDriver()
        autocorr_driver = AutocorrelationDriver()
        feat_driver = FeatureEngineeringDriver()
        stat_driver = StatisticalTestDriver()

        # Tool argument adapters
        def adapt_filter(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.command_resolver import CommandResolver
            filters = state.setdefault("filters", [])
            has_filter = False
            if hasattr(args, "conditions") and args.conditions:
                for cond in args.conditions:
                    fs = CommandResolver.normalize_filter_spec(cond.model_dump() if hasattr(cond, "model_dump") else cond)
                    if fs:
                        filters.append(fs)
                        has_filter = True
                    else:
                        state["filter_error"] = f"Invalid filter condition: {cond}"
            elif hasattr(args, "column") and args.column:
                col_val = args.column
                if isinstance(col_val, list):
                    for c in col_val:
                        f_dict = {"column": c, "condition": getattr(args, "condition", None), "value": getattr(args, "value", None)}
                        fs = CommandResolver.normalize_filter_spec(f_dict)
                        if fs:
                            filters.append(fs)
                            has_filter = True
                else:
                    fs = CommandResolver.normalize_filter_spec(args.model_dump() if hasattr(args, "model_dump") else args)
                    if fs:
                        filters.append(fs)
                        has_filter = True
                    else:
                        state["filter_error"] = f"Invalid filter arguments: {args}"
            elif hasattr(args, "columns") and args.columns:
                for c in args.columns:
                    f_dict = {"column": c, "condition": getattr(args, "condition", None), "value": getattr(args, "value", None)}
                    fs = CommandResolver.normalize_filter_spec(f_dict)
                    if fs:
                        filters.append(fs)
                        has_filter = True
            state["has_filter_spec"] = has_filter

        def adapt_sort(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.command_resolver import CommandResolver
            col = getattr(args, "column", None) or getattr(args, "sort_by", None)
            if col:
                state["target_column"] = str(col).strip()
            direction = getattr(args, "order", None) or getattr(args, "direction", None)
            if direction:
                state["order"] = CommandResolver.resolve_order(direction)
            elif getattr(args, "ascending", None) is not None:
                state["order"] = "asc" if args.ascending else "desc"
            limit = getattr(args, "limit", None)
            if limit is not None:
                state["limit"] = limit

        def adapt_group(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.command_resolver import CommandResolver
            from backend.ds_engine.intent_schema import AggregationType
            dims = state.setdefault("dimensions", [])
            for d in getattr(args, "group_by", []):
                if d not in dims:
                    dims.append(d)
            mets = state.setdefault("metrics", [])
            if getattr(args, "metric", None):
                if isinstance(args.metric, list):
                    for m in args.metric:
                        if m not in mets:
                            mets.append(m)
                elif args.metric not in mets:
                    mets.append(args.metric)
            if getattr(args, "metrics", None):
                for m in args.metrics:
                    if m not in mets:
                        mets.append(m)
            col = getattr(args, "column", None) or getattr(args, "target_column", None)
            if isinstance(col, str) and col not in mets:
                mets.append(col)
            elif isinstance(col, list):
                for c in col:
                    if c not in mets:
                        mets.append(c)
            if getattr(args, "columns", None):
                for c in args.columns:
                    if c not in mets:
                        mets.append(c)
            if getattr(args, "aggregation", None):
                state["aggregation"] = CommandResolver.resolve_aggregation(args.aggregation)
            elif dims and not mets:
                state["aggregation"] = AggregationType.COUNT
            if getattr(args, "order", None):
                state["order"] = CommandResolver.resolve_order(args.order)
            elif getattr(args, "direction", None):
                state["order"] = CommandResolver.resolve_order(args.direction)
            elif getattr(args, "ascending", None) is not None:
                state["order"] = "asc" if args.ascending else "desc"
            if getattr(args, "limit", None) is not None:
                state["limit"] = args.limit
            state["chart_type"] = ChartType(getattr(args, "chart_type", "none"))
            if getattr(args, "datetime_parses", None):
                state.setdefault("arguments", {})["datetime_parses"] = [
                    parse.model_dump(mode="json") for parse in args.datetime_parses
                ]
            if getattr(args, "filters", None):
                filters = state.setdefault("filters", [])
                for f in args.filters:
                    fs = CommandResolver.normalize_filter_spec(f.model_dump() if hasattr(f, "model_dump") else f)
                    if fs and fs not in filters:
                        filters.append(fs)
                    elif not fs:
                        state["filter_error"] = f"Invalid embedded filter: {f}"

        def adapt_conditional_rate(args: Any, state: Dict[str, Any]):
            """Adapt a validated rate contract without treating its numerator as a pre-filter.

            A numerator predicate is deliberately kept in the primary tool
            arguments: applying it as a pipeline filter would incorrectly make
            the numerator and denominator identical and yield a 100% rate.
            """
            from backend.ds_engine.intent_schema import AggregationType

            dims = state.setdefault("dimensions", [])
            for dimension in args.group_by:
                if dimension not in dims:
                    dims.append(dimension)

            # This is a count-based group aggregation, but the typed operator
            # owns the numerator/denominator calculation.  Do not add the
            # predicate column to metrics or to state["filters"].
            state["aggregation"] = AggregationType.COUNT
            state["arguments"] = args.model_dump(mode="json")

        def adapt_aggregate(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.command_resolver import CommandResolver
            mets = state.setdefault("metrics", [])
            m = getattr(args, "metric", None) or getattr(args, "column", None)
            if m and m not in mets:
                mets.append(m)
            if getattr(args, "columns", None):
                for c in args.columns:
                    if c not in mets:
                        mets.append(c)
            if getattr(args, "aggregation", None):
                state["aggregation"] = CommandResolver.resolve_aggregation(args.aggregation)
            if getattr(args, "order", None):
                state["order"] = CommandResolver.resolve_order(args.order)
            elif getattr(args, "direction", None):
                state["order"] = CommandResolver.resolve_order(args.direction)
            elif getattr(args, "ascending", None) is not None:
                state["order"] = "asc" if args.ascending else "desc"
            if getattr(args, "limit", None) is not None:
                state["limit"] = args.limit
            if getattr(args, "filters", None):
                filters = state.setdefault("filters", [])
                for f in args.filters:
                    fs = CommandResolver.normalize_filter_spec(f.model_dump() if hasattr(f, "model_dump") else f)
                    if fs and fs not in filters:
                        filters.append(fs)
                    elif not fs:
                        state["filter_error"] = f"Invalid embedded filter: {f}"

        def adapt_metric_audit(args: Any, state: Dict[str, Any]):
            """Adapt a filtered numeric audit without inferring any columns.

            The planner must provide both the exact measure and the predicate;
            this adapter only transfers that already-validated contract to the
            V2 typed operator pipeline.
            """
            from backend.ds_engine.command_resolver import CommandResolver

            subjects = state.setdefault("feature_columns", [])
            if args.metric not in subjects:
                subjects.append(args.metric)
            state["chart_type"] = ChartType(args.chart_type)
            state["arguments"] = args.model_dump(mode="json")
            filters = state.setdefault("filters", [])
            for raw_filter in args.filters:
                fs = CommandResolver.normalize_filter_spec(
                    raw_filter.model_dump() if hasattr(raw_filter, "model_dump") else raw_filter
                )
                if fs and fs not in filters:
                    filters.append(fs)
                elif not fs:
                    state["filter_error"] = f"Invalid embedded filter: {raw_filter}"

        def adapt_correlate(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            for c in args.columns:
                if c not in feats:
                    feats.append(c)
            correlation_args = args.model_dump(exclude={"columns"})
            state.setdefault("arguments", {}).update(correlation_args)
            state["chart_type"] = ChartType(correlation_args["chart_type"])

        def adapt_forecast(args: Any, state: Dict[str, Any]):
            tgt = getattr(args, "value_column", None) or getattr(args, "target_column", None) or getattr(args, "column", None) or getattr(args, "metric", None)
            if tgt:
                state["target_column"] = tgt
            state["time_column"] = args.time_column
            state["horizon"] = args.horizon
            state["arguments"] = args.model_dump(mode="json")

        def adapt_clean(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            cols = getattr(args, "columns", None) or getattr(args, "column", None)
            if isinstance(cols, str):
                cols = [cols]
            if cols:
                for c in cols:
                    if c not in feats:
                        feats.append(c)
            state["arguments"] = args.model_dump(mode="json")

        def adapt_transform(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            cols = getattr(args, "columns", None) or getattr(args, "column", None)
            if isinstance(cols, str):
                cols = [cols]
            if cols:
                for c in cols:
                    if c not in feats:
                        feats.append(c)
            state["arguments"] = args.model_dump(mode="json")

        def adapt_anomaly(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            if getattr(args, "columns", None):
                for c in args.columns:
                    if c not in feats:
                        feats.append(c)
            state["arguments"] = args.model_dump(mode="json")

        def adapt_train_model(args: Any, state: Dict[str, Any]):
            tgt = getattr(args, "target_column", None) or getattr(args, "target", None)
            if tgt:
                state["target_column"] = tgt
            feats = state.setdefault("feature_columns", [])
            raw_feats = getattr(args, "feature_columns", None) or getattr(args, "features_columns", None) or getattr(args, "features", None)
            if raw_feats:
                for c in raw_feats:
                    if c not in feats:
                        feats.append(c)
            for arg_name in ("task_type", "evaluation_artifact", "prediction_context"):
                value = getattr(args, arg_name, None)
                if value is not None:
                    state.setdefault("arguments", {})[arg_name] = value

        def adapt_cluster(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            raw_cols = getattr(args, "columns", None) or getattr(args, "feature_columns", None) or getattr(args, "group_by", None) or getattr(args, "column", None)
            if isinstance(raw_cols, str):
                raw_cols = [raw_cols]
            if raw_cols:
                for c in raw_cols:
                    if c not in feats:
                        feats.append(c)
            k = getattr(args, "n_clusters", None) or getattr(args, "k", None)
            if k is not None:
                state["n_clusters"] = k
                state["k"] = k
            # Persist defaults as part of the canonical plan so typed execution
            # and generated code cannot silently disagree about scaling.
            state["arguments"] = args.model_dump(mode="json")

        def adapt_elbow_analysis(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            raw_cols = getattr(args, "columns", None) or getattr(args, "feature_columns", None) or getattr(args, "column", None)
            if isinstance(raw_cols, str):
                raw_cols = [raw_cols]
            if raw_cols:
                for c in raw_cols:
                    if c not in feats:
                        feats.append(c)
            state["arguments"] = {
                "mode": "elbow",
                "min_k": getattr(args, "min_k", 2),
                "max_k": getattr(args, "max_k", 8),
                "scaling": getattr(args, "scaling", "standard"),
            }

        def adapt_autocorr(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            raw_cols = getattr(args, "value_column", None) or getattr(args, "target_column", None) or getattr(args, "columns", None) or getattr(args, "column", None)
            if isinstance(raw_cols, str):
                raw_cols = [raw_cols]
            if raw_cols:
                for c in raw_cols:
                    if c not in feats:
                        feats.append(c)
                state["target_column"] = raw_cols[0]
                state["metrics"] = raw_cols
            if getattr(args, "time_column", None):
                state["time_column"] = args.time_column
            lags = getattr(args, "max_lags", None) or getattr(args, "lag", None) or getattr(args, "lags", None)
            if lags is not None:
                state["max_lags"] = lags
                state.setdefault("arguments", {})["max_lags"] = lags

        def adapt_stat_test(args: Any, state: Dict[str, Any]):
            mets = state.setdefault("metrics", [])
            for column in args.columns:
                if column not in mets:
                    mets.append(column)
            # Persist the validated, canonical test contract for the typed
            # operator.  It must never infer a test from column dtypes.
            state["arguments"] = args.model_dump(mode="json")

        def adapt_stationarity_test(args: Any, state: Dict[str, Any]):
            tgt = getattr(args, "target_column", None) or getattr(args, "value_column", None) or getattr(args, "column", None)
            if tgt:
                state["target_column"] = tgt
                state.setdefault("metrics", []).append(tgt)
            if getattr(args, "time_column", None):
                state["time_column"] = args.time_column
            state.setdefault("arguments", {})["test_type"] = "stationarity"
            state.setdefault("arguments", {})["time_column"] = getattr(args, "time_column", None)
            state.setdefault("arguments", {})["target_column"] = tgt
            if getattr(args, "max_lags", None) is not None:
                state.setdefault("arguments", {})["max_lags"] = args.max_lags

        def adapt_categorical_outcome_association(args: Any, state: Dict[str, Any]):
            # Preserve the two distinct semantic roles.  The numeric measure is
            # a subject; the categorical field is an outcome/group, never an
            # implicitly trained label.
            state["target_column"] = args.outcome_column
            state["metrics"] = [args.numeric_predictor]
            state["dimensions"] = [args.outcome_column]
            state["arguments"] = args.model_dump(mode="json")

        def adapt_pca(args: Any, state: Dict[str, Any]):
            feats = state.setdefault("feature_columns", [])
            cols = getattr(args, "columns", None) or getattr(args, "feature_columns", None) or getattr(args, "column", None)
            if isinstance(cols, str):
                cols = [cols]
            if cols:
                for c in cols:
                    if c not in feats:
                        feats.append(c)
            n_comp = getattr(args, "n_components", None)
            if n_comp is not None:
                state["n_components"] = n_comp
                state.setdefault("arguments", {})["n_components"] = n_comp
            state.setdefault("arguments", {})["scaling"] = getattr(args, "scaling", "standard") or "standard"

        def adapt_explain_features(args: Any, state: Dict[str, Any]):
            tgt = getattr(args, "target_column", None) or getattr(args, "target", None)
            if tgt:
                state["target_column"] = tgt
            feats = state.setdefault("feature_columns", [])
            raw_feats = getattr(args, "feature_columns", None) or getattr(args, "features", None) or getattr(args, "columns", None)
            if raw_feats:
                if isinstance(raw_feats, str):
                    raw_feats = [raw_feats]
                for c in raw_feats:
                    if c not in feats:
                        feats.append(c)
            state["arguments"] = args.model_dump(mode="json")

        def adapt_decompose_ts(args: Any, state: Dict[str, Any]):
            tgt = getattr(args, "target_column", None) or getattr(args, "value_column", None)
            if tgt:
                state["target_column"] = tgt
                state.setdefault("metrics", []).append(tgt)
            if getattr(args, "time_column", None):
                state["time_column"] = args.time_column
            if getattr(args, "period", None) is not None:
                state["period"] = args.period
                state.setdefault("arguments", {})["period"] = args.period
            if getattr(args, "model", None) is not None:
                state["model"] = args.model
                state.setdefault("arguments", {})["model"] = args.model

        def adapt_eda(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.intent_schema import ChartType
            cols = getattr(args, "columns", None) or getattr(args, "column", None)
            if isinstance(cols, str):
                cols = [cols]
            if cols:
                dims = state.setdefault("dimensions", [])
                feats = state.setdefault("feature_columns", [])
                for c in cols:
                    if c not in dims:
                        dims.append(c)
                    if c not in feats:
                        feats.append(c)
            chart_type = getattr(args, "chart_type", None)
            if chart_type:
                ct_upper = str(getattr(chart_type, "value", chart_type)).upper()
                if not hasattr(ChartType, ct_upper):
                    raise ValueError(f"Unsupported chart type: {chart_type}")
                state["chart_type"] = getattr(ChartType, ct_upper)

        def adapt_chat(args: Any, state: Dict[str, Any]):
            resp = getattr(args, "response", None) or getattr(args, "message", None) or getattr(args, "text", None)
            if resp:
                state["response_text"] = resp

        def adapt_clarify(args: Any, state: Dict[str, Any]):
            if getattr(args, "reason", None):
                state["clarification_reason"] = args.reason

        def adapt_event_window(args: Any, state: Dict[str, Any]):
            from backend.ds_engine.intent_schema import ChartType
            cols = getattr(args, "columns", None) or getattr(args, "column", None)
            if isinstance(cols, str):
                cols = [cols]
            if cols:
                mets = state.setdefault("metrics", [])
                feats = state.setdefault("feature_columns", [])
                for c in cols:
                    if c not in mets:
                        mets.append(c)
                    if c not in feats:
                        feats.append(c)
            tgt = getattr(args, "target_column", None) or getattr(args, "event_column", None)
            if tgt:
                state["target_column"] = tgt
            t_col = getattr(args, "time_column", None)
            if t_col:
                state["time_column"] = t_col
            seq_col = getattr(args, "sequence_order", None) or getattr(args, "order_column", None)
            if seq_col:
                state["sequence_column"] = seq_col
            win = getattr(args, "pre_event_window", None) or getattr(args, "window", None)
            if win is not None:
                state["pre_event_window"] = win
            pred = getattr(args, "event_predicate", None)
            if pred is not None:
                state["event_predicate"] = pred
            eid = getattr(args, "entity_id", None)
            if eid:
                state["entity_id"] = eid
            chart_type = getattr(args, "chart_type", None)
            if chart_type:
                ct_upper = str(getattr(chart_type, "value", chart_type)).upper()
                if not hasattr(ChartType, ct_upper):
                    raise ValueError(f"Unsupported chart type: {chart_type}")
                state["chart_type"] = getattr(ChartType, ct_upper)
            state["arguments"] = args.model_dump(mode="json")

        def adapt_sql(args: Any, state: Dict[str, Any]):
            query = getattr(args, "query", None) or getattr(args, "sql", None)
            if query:
                state["sql_query"] = str(query).strip()
                state.setdefault("arguments", {})["query"] = str(query).strip()

        # Register canonical ToolSpecs
        self.register_tool_spec(ToolSpec(
            name="filter_data",
            description="Filters dataset rows based on column conditions (==, !=, >, <, >=, <=, contains, in)",
            input_model=FilterArgs,
            objective=AnalysisObjective.TRANSFORM,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.TRANSFORMED_DATA],
            requires_dataset=True,
            compiler_or_driver=filter_driver,
            adapter=adapt_filter,
            result_contract="dataframe",
            side_effect_level="read_only",
            synonyms=["filter", "filter_query"],
            category="filter_query",
            task_family=TaskFamily.FILTER_SLICE,
            allowed_predecessors=["filter_data", "clean_data", "sort_data", "group_data", "conditional_rate"],
            prerequisites=ToolPrerequisites(),
            execution_priority=1,
            is_prerequisite_only=True,
            requires_filter_conditions=True,
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=[],
                argument_aliases={},
                slot_presentations={},
                slot_filling_type="generic",
                slot_filling_title="Specify Filter Conditions",
                slot_filling_prompt="Please specify filter conditions for the dataset",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="sort_data",
            description="Sorts dataset by column in ascending or descending order with optional limit",
            input_model=SortArgs,
            objective=AnalysisObjective.TRANSFORM,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.TRANSFORMED_DATA],
            requires_dataset=True,
            compiler_or_driver=filter_driver,
            adapter=adapt_sort,
            result_contract="dataframe",
            side_effect_level="read_only",
            synonyms=["sort"],
            category="filter_query",
            task_family=TaskFamily.FILTER_SLICE,
            allowed_predecessors=["filter_data", "clean_data", "sort_data", "group_data", "conditional_rate"],
            prerequisites=ToolPrerequisites(),
            execution_priority=3,
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["column"],
                inferable_slots={"column": "numeric_measure"},
                argument_aliases={"column": ["column", "columns", "metric"]},
                slot_presentations={
                    "column": SlotPresentationPolicy(
                        label="Sort Column",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Column to Sort",
                slot_filling_prompt="Which column would you like to sort by?",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="conditional_rate",
            description=(
                "Computes a conditional percentage/rate by categorical segment: "
                "the numerator is a typed condition (for example outcome equals yes) "
                "and the denominator is every record in the same group. Supports "
                "conversion, response, approval, defect, retention, and churn rates; "
                "optional minimum cohort size, explicit ranking, and a percentage bar chart."
            ),
            input_model=ConditionalRateArgs,
            objective=AnalysisObjective.COMPARE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=group_driver,
            adapter=adapt_conditional_rate,
            result_contract="dataframe",
            side_effect_level="read_only",
            synonyms=["conditional_percentage", "segment_rate", "conversion_rate", "response_rate", "cohort_rate"],
            category="group_aggregate",
            task_family=TaskFamily.GROUP_AGGREGATION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(
                requires_grouping_or_dimension=True,
                condition_argument_name="numerator_condition",
            ),
            supported_chart_types=["bar", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["group_by", "numerator_condition"],
                inferable_slots={"group_by": "categorical_dimension"},
                can_auto_infer_target=False,
                argument_aliases={
                    "group_by": ["group_by", "dimensions", "dimension", "categorical_feature"],
                    "numerator_condition": ["numerator_condition", "condition", "outcome_column"],
                },
                slot_presentations={
                    "group_by": SlotPresentationPolicy(
                        label="Category / Dimension",
                        role_source="ranked_categorical_dimensions",
                        presentation_slot_name="dimension",
                        slot_type="multiselect",
                    ),
                    "numerator_condition": SlotPresentationPolicy(
                        label="Numerator Condition",
                        role_source="columns",
                    ),
                },
                slot_filling_type="dimension_selector",
                slot_filling_title="Select Cohort and Numerator Condition",
                slot_filling_prompt="Select grouping dimension and numerator condition for conditional rate calculation",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="group_data",
            description=(
                "Creates historical grouped summaries and time-bucket profiles: group rows by one or more dimensions "
                "and compute an explicit aggregate of a numeric measure (sum, mean, count, min, max, median). "
                "It can explicitly parse declared datetime columns using a supplied strptime format before aggregation. "
                "A datetime parsing source is preprocessing context, never the numeric measure unless the user explicitly requests it. "
                "Use this for observed hourly/daily/seasonal profiles and comparisons; it does not predict future values."
            ),
            input_model=GroupArgs,
            objective=AnalysisObjective.COMPARE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            planner_input_model=GroupPlannerArgs,
            planner_input_models=[GroupPlannerArgs, GroupCountPlannerArgs],
            requires_dataset=True,
            compiler_or_driver=group_driver,
            adapter=adapt_group,
            result_contract="dataframe",
            side_effect_level="read_only",
            synonyms=["groupby", "group_aggregate"],
            category="group_aggregate",
            task_family=TaskFamily.GROUP_AGGREGATION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(
                requires_grouping_or_dimension=True,
                requires_numeric_measure=True,
                measure_argument_names=["metric", "metrics", "target_column", "column", "columns"],
                aggregation_argument_name="aggregation",
                non_numeric_measure_aggregations=["count"],
                min_numeric_columns=1,
            ),
            supported_chart_types=["bar", "line", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["group_by"],
                inferable_slots={"group_by": "categorical_dimension"},
                argument_aliases={
                    "group_by": ["group_by", "dimensions", "dimension", "group_columns", "categorical_feature"],
                },
                slot_presentations={
                    "group_by": SlotPresentationPolicy(
                        label="Category / Dimension",
                        role_source="ranked_categorical_dimensions",
                        presentation_slot_name="dimension",
                        slot_type="multiselect",
                    ),
                },
                slot_filling_type="dimension_selector",
                slot_filling_title="Select Grouping Dimension",
                slot_filling_prompt="Which categorical dimension would you like to group by?",
                default_query_template="Show total by {dimension}",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="aggregate_data",
            description=(
                "Calculates one overall summary metric across the active dataset or an explicitly filtered subset. "
                "Unlike group_data, it deliberately has no group_by dimension: use it for a single row count, "
                "total, mean, minimum, maximum, or median. Use group_data only when the user explicitly asks "
                "for a breakdown by one or more dimensions."
            ),
            input_model=AggregateArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=group_driver,
            adapter=adapt_aggregate,
            result_contract="dataframe",
            synonyms=["aggregate"],
            category="group_aggregate",
            task_family=TaskFamily.GROUP_AGGREGATION,
            allowed_predecessors=["filter_data", "clean_data"],
            # Overall aggregation is a distinct capability from grouped
            # aggregation.  Requiring a dimension here made valid filtered
            # row counts impossible and caused pointless planner retries.
            prerequisites=ToolPrerequisites(requires_grouping_or_dimension=False),
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["metric", "aggregation"],
                inferable_slots={"metric": "numeric_measure"},
                argument_aliases={
                    "metric": ["metric", "column", "columns", "target_column", "value_column"],
                },
                slot_presentations={
                    "metric": SlotPresentationPolicy(
                        label="Numeric Metric",
                        role_source="ranked_numeric_measures",
                    ),
                    "aggregation": SlotPresentationPolicy(
                        label="Aggregation Function",
                        role_source=None,
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Metric to Aggregate",
                slot_filling_prompt="Which metric and aggregation would you like to compute?",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="audit_metric_distribution",
            description=(
                "Audits the distribution of one exact numeric metric within an explicitly filtered subset. "
                "Returns the filtered row count, descriptive distribution statistics, observed zero count and zero rate, "
                "outlier evidence, and an optional box plot. Use this when a user asks what a metric looks like for a "
                "particular operational status, cohort, condition, or segment. It reports observed data only and does not "
                "make an unsupported causal claim about why zeros or values occurred."
            ),
            input_model=MetricAuditArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=eda_driver,
            adapter=adapt_metric_audit,
            result_contract="dataframe",
            synonyms=["metric_distribution_audit", "filtered_distribution_audit"],
            # SemanticIntent has no historic DISTRIBUTION intent enum.  EDA is
            # retained only as the compatibility classification; task_family
            # below is the authoritative typed execution dispatch.
            category="eda",
            task_family=TaskFamily.DISTRIBUTION_COMPARISON,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(
                requires_numeric_subjects=True,
                requires_numeric_measure=True,
                measure_argument_names=["metric"],
                min_numeric_columns=1,
            ),
            supported_chart_types=["box", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["metric"],
                inferable_slots={"metric": "numeric_measure"},
                argument_aliases={
                    "metric": ["metric", "column", "columns", "target_column"],
                },
                slot_presentations={
                    "metric": SlotPresentationPolicy(
                        label="Numeric Metric",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Metric for Distribution Audit",
                slot_filling_prompt="Which metric would you like to audit?",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="correlate_data",
            description=(
                "Performs an explicitly specified numeric correlation analysis. "
                "Choose pairwise for exactly two columns or matrix for multiple columns; "
                "state relationship form, methods, chart type, smoother, and any claimed direction. "
                "A nonlinear request must use pairwise scatter with LOWESS plus Pearson and Spearman."
            ),
            input_model=CorrelationArgs,
            objective=AnalysisObjective.INFER,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=corr_driver,
            adapter=adapt_correlate,
            result_contract="dataframe",
            synonyms=["correlation", "correlate", "correlated", "correlating", "relationship", "association"],
            category="correlation",
            task_family=TaskFamily.CORRELATION_ANALYSIS,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=2),
            supported_chart_types=["scatter", "heatmap", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_required_mode_arg=("analysis_mode", "matrix"),
                dataset_wide_role="ranked_numeric_measures",
                dataset_wide_target_arg="columns",
                dataset_wide_min_columns=2,
                allow_role_candidate_fallback=True,
                fallback_role_name="continuous_numeric_candidates",
                required_slots=["columns"],
                minimum_items_by_slot={"columns": 2},
                argument_aliases={
                    "columns": ["columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Variables to Correlate",
                        role_source="ranked_numeric_measures",
                        is_pairwise=True,
                        pairwise_slots=[("col_x", "First Variable (X)"), ("col_y", "Second Variable (Y)")],
                    ),
                },
                dataset_wide_default_args={
                    "analysis_mode": "matrix",
                    "relationship_form": "linear",
                    "correlation_methods": ["pearson"],
                    "chart_type": "heatmap",
                    "smoother": "none",
                    "expected_direction": "unspecified",
                },
                multi_column_threshold=2,
                multi_column_first_item_slots=["correlation_methods"],
                multi_column_overrides={
                    "analysis_mode": "matrix",
                    "chart_type": "heatmap",
                    "smoother": "none",
                    "expected_direction": "unspecified",
                },
                slot_filling_type="pairwise_slots",
                slot_filling_title="Select Variables for Correlation",
                slot_filling_prompt="Which two numeric variables would you like to correlate?",
                default_query_template="Show correlation between {col_x} and {col_y}",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="forecast_data",
            description=(
                "Predicts future values of one numeric time-series target on an explicitly declared regular time grid. "
                "Requires a horizon plus an exact user-query phrase proving that numeric horizon, pandas frequency, method "
                "(naive_last_value baseline or holt_linear), and chart choice. "
                "Use only when the user asks for a future projection; historical hourly/daily profiles and grouped averages use group_data. "
                "Does not fabricate calibrated confidence intervals or assume daily frequency."
            ),
            input_model=ForecastArgs,
            objective=AnalysisObjective.FORECAST,
            analysis_modes=[AnalysisMode.PROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=forecast_driver,
            adapter=adapt_forecast,
            result_contract="dataframe",
            synonyms=["forecast", "time_series_forecast"],
            category="time_series_forecast",
            task_family=TaskFamily.TIME_SERIES_FORECAST,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(
                requires_temporal_column=True,
                requires_numeric_subjects=True,
                requires_quantified_request_evidence=True,
                quantity_argument_name="horizon",
                quantity_evidence_argument_name="horizon_evidence",
                min_numeric_columns=1,
            ),
            supported_chart_types=["line", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                requires_temporal_column=True,
                temporal_slot_name="time_column",
                required_slots=["time_column", "target_column"],
                inferable_slots={"time_column": "temporal"},
                can_auto_infer_target=False,
                argument_aliases={
                    "time_column": ["time_column", "temporal_column", "date_column", "timestamp"],
                    "target_column": ["target_column", "target", "value_column", "metric", "column"],
                },
                slot_presentations={
                    "time_column": SlotPresentationPolicy(
                        label="Time / Date Column",
                        role_source="ranked_temporal_columns",
                        fallback_roles=["temporal_column_candidates"],
                    ),
                    "target_column": SlotPresentationPolicy(
                        label="Forecast Target",
                        role_source="ranked_target_candidates",
                        fallback_roles=["ranked_numeric_measures"],
                    ),
                    "frequency": SlotPresentationPolicy(
                        label="Calendar Frequency",
                        slot_type="select",
                        options=["D", "W", "MS", "h", "M"],
                        title="Specify Calendar Frequency",
                        prompt_template="Could not determine calendar frequency for temporal column '{time_col}'. Please specify a frequency (e.g., 'D', 'W', 'MS', 'h'):",
                        suggested_queries=[
                            "Forecast with daily frequency D",
                            "Forecast with weekly frequency W",
                            "Forecast with monthly frequency MS",
                        ],
                    ),
                },
                post_completion_defaults={
                    "horizon": 7,
                    "horizon_evidence": "default horizon",
                    "method": "holt_linear",
                    "chart_type": "line",
                },
                infers_frequency_from_temporal=True,
                allow_inferred_frequency=True,
                frequency_confidence_threshold=0.8,
                slot_filling_type="forecast_selector",
                slot_filling_title="Select Forecast Parameters",
                slot_filling_prompt="Which date and metric columns should I use for forecasting?",
                default_query_template="Forecast {target_column} over {time_column}",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="clean_data",
            description=(
                "Applies one explicit, column-scoped cleaning or transformation operation: drop missing rows, "
                "mean/median/mode/constant imputation, duplicate removal, IQR outlier capping, Min-Max scaling, "
                "standardization (z-score), one-hot encoding, Yeo-Johnson power transform, log transform, "
                "lag features, rolling window features, differencing, or calendar feature extraction. "
                "Never choose a cleaning or transformation strategy implicitly; request clarification when the method is not stated."
            ),
            input_model=CleanArgs,
            objective=AnalysisObjective.TRANSFORM,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.TRANSFORMED_DATA],
            requires_dataset=True,
            compiler_or_driver=clean_driver,
            adapter=adapt_clean,
            result_contract="dataframe",
            synonyms=["data_clean", "impute", "transform_data", "data_transform", "transform_features", "feature_engineering", "scale_data", "encode_data", "create_lag", "create_rolling", "create_difference", "create_calendar_features"],
            category="data_clean",
            task_family=TaskFamily.DATA_CLEANING,
            allowed_predecessors=["clean_data"],
            prerequisites=ToolPrerequisites(),
            execution_priority=0,
            is_prerequisite_only=True,
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_target_arg="columns",
                argument_aliases={
                    "columns": ["columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Columns to Clean or Transform",
                        role_source="columns",
                    ),
                },
                slot_filling_type="generic",
                slot_filling_title="Data Cleaning Specification",
                slot_filling_prompt="Specify cleaning or transformation method and columns",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="detect_anomalies",
            description=(
                "Detects anomalies in explicit numeric columns using a user-declared method "
                "(isolation_forest, iqr, or zscore) and an explicit method-specific sensitivity. "
                "The method and threshold are never inferred silently."
            ),
            input_model=AnomalyArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE, AnalysisMode.RETROSPECTIVE, AnalysisMode.PROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=anomaly_driver,
            adapter=adapt_anomaly,
            result_contract="dataframe",
            synonyms=["anomaly_detection", "outliers"],
            category="anomaly_detection",
            task_family=TaskFamily.ANOMALY_DETECTION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=1),
            supported_chart_types=["histogram", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_role="ranked_numeric_measures",
                dataset_wide_target_arg="columns",
                dataset_wide_min_columns=1,
                allow_role_candidate_fallback=True,
                fallback_role_name="continuous_numeric_candidates",
                requires_explicit_method=True,
                requires_explicit_sensitivity=True,
                allowed_methods=["iqr", "zscore", "isolation_forest"],
                allowed_sensitivities=[1.0, 1.5, 2.0, 3.0],
                argument_aliases={
                    "columns": ["columns", "column", "metric"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Numeric Metric",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="method_selector",
                slot_filling_title="Select Anomaly Detection Parameters",
                slot_filling_prompt="Which method and sensitivity threshold would you like to use for anomaly detection?",
                default_query_template="Detect anomalies using {method}",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="eda_data",
            description="Runs automated exploratory data analysis overview and distributions",
            input_model=EDAArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=eda_driver,
            adapter=adapt_eda,
            result_contract="dataframe",
            synonyms=["eda", "explore_data"],
            category="eda",
            task_family=TaskFamily.GENERAL_EDA,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(),
            # General EDA currently produces a profile table only. Declaring
            # that fact prevents a planner from promising a bar/line chart
            # which its typed operator cannot physically emit.
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                slot_filling_type="generic",
                slot_filling_title="Dataset Overview",
                slot_filling_prompt="Overview and exploratory summary of dataset",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="train_model",
            description=(
                "Trains and evaluates a supervised machine-learning prediction model using an explicit label column. "
                "Supports classification and regression, including Random Forest classifiers, held-out evaluation, "
                "class labels, accuracy metrics, and confusion-matrix evaluation when requested."
            ),
            input_model=TrainModelArgs,
            objective=AnalysisObjective.PREDICT,
            analysis_modes=[AnalysisMode.PROSPECTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=automl_driver,
            adapter=adapt_train_model,
            result_contract="dataframe",
            synonyms=["auto_ml", "automl", "ml"],
            category="auto_ml",
            task_family=TaskFamily.SUPERVISED_ML,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_target_column=True, min_numeric_columns=1),
            # Training is ephemeral: the current operator returns evaluation
            # evidence, not a persisted/reusable model artifact.
            side_effect_level="trains_model",
            supported_chart_types=["confusion_matrix", "feature_importance", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["target_column"],
                can_auto_infer_target=False,
                argument_aliases={
                    "target_column": ["target_column", "target", "value_column", "outcome_column", "column"],
                },
                slot_presentations={
                    "target_column": SlotPresentationPolicy(
                        label="Target Column",
                        role_source="ranked_target_candidates",
                        fallback_roles=["ranked_categorical_dimensions", "ranked_numeric_measures"],
                        max_candidates=6,
                    ),
                },
                slot_filling_type="target_selector",
                slot_filling_title="Select Target Variable to Predict",
                slot_filling_prompt="Which column would you like the model to predict?",
                default_query_template="Train a model to predict {target_column}",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="cluster_data",
            description="Segments records with K-Means using an explicit cluster count and explicitly declared feature scaling; use none only when the feature units are already comparable.",
            input_model=ClusterArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=cluster_driver,
            adapter=adapt_cluster,
            result_contract="dataframe",
            synonyms=["clustering", "cluster"],
            category="clustering",
            task_family=TaskFamily.CLUSTERING,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=1),
            supported_chart_types=["scatter", "hist", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_role="ranked_numeric_measures",
                dataset_wide_target_arg="columns",
                dataset_wide_min_columns=2,
                allow_role_candidate_fallback=True,
                fallback_role_name="continuous_numeric_candidates",
                requires_explicit_method=False,
                argument_aliases={
                    "columns": ["columns", "features", "feature_columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Features to Cluster",
                        role_source="ranked_numeric_measures",
                        slot_type="multiselect",
                    ),
                },
                slot_filling_type="column_selector",
                slot_filling_title="Select Clustering Parameters",
                slot_filling_prompt="Specify features and cluster count",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="elbow_analysis",
            description="Evaluates K-Means inertia and silhouette scores across cluster counts (k) to determine optimal clusters.",
            input_model=ElbowAnalysisArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=cluster_driver,
            adapter=adapt_elbow_analysis,
            result_contract="dataframe",
            synonyms=["elbow_method", "cluster_elbow", "optimal_clusters", "elbow_curve"],
            category="clustering",
            task_family=TaskFamily.CLUSTERING,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=2),
            supported_chart_types=["line", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_role="ranked_numeric_measures",
                dataset_wide_target_arg="columns",
                dataset_wide_min_columns=2,
                allow_role_candidate_fallback=True,
                fallback_role_name="continuous_numeric_candidates",
                argument_aliases={
                    "columns": ["columns", "features", "feature_columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Features for Elbow Evaluation",
                        role_source="ranked_numeric_measures",
                        slot_type="multiselect",
                    ),
                },
                slot_filling_type="column_selector",
                slot_filling_title="Select Features for Elbow Evaluation",
                slot_filling_prompt="Specify numeric features to compute optimal cluster count",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="statistical_test",
            description=(
                "Performs an explicitly selected inferential test. Use independent_ttest for two independent numeric samples, "
                "one_way_anova for one numeric measurement grouped by one categorical variable, or chi_square_independence "
                "for two categorical variables. The exact test and its column roles are required; DataPilot never guesses them."
            ),
            input_model=StatisticalTestArgs,
            objective=AnalysisObjective.INFER,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=stat_driver,
            adapter=adapt_stat_test,
            result_contract="dataframe",
            synonyms=["stat_test", "hypothesis_test"],
            category="statistical_test",
            task_family=TaskFamily.STATISTICAL_TEST,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(),
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["columns"],
                minimum_items_by_slot={"columns": 2},
                argument_aliases={
                    "columns": ["columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Variables to Compare",
                        role_source="ranked_numeric_measures",
                        is_pairwise=True,
                        pairwise_slots=[("col_x", "First Variable (X)"), ("col_y", "Second Variable (Y)")],
                    ),
                },
                slot_filling_type="pairwise_slots",
                slot_filling_title="Select Variables for Statistical Test",
                slot_filling_prompt="Which variables would you like to compare?",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="stationarity_test",
            description="Executes Augmented Dickey-Fuller (ADF) and KPSS unit root stationarity tests on a time series with explicit time_column and target_column.",
            input_model=StationarityTestArgs,
            objective=AnalysisObjective.INFER,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=stat_driver,
            adapter=adapt_stationarity_test,
            result_contract="dataframe",
            synonyms=["stationarity", "unit_root_test", "adf_test", "kpss_test"],
            category="statistical_test",
            task_family=TaskFamily.STATISTICAL_TEST,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_temporal_column=True, requires_target_column=True),
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                requires_temporal_column=True,
                temporal_slot_name="time_column",
                required_slots=["time_column", "target_column"],
                inferable_slots={"time_column": "temporal", "target_column": "numeric_measure"},
                can_auto_infer_target=False,
                argument_aliases={
                    "time_column": ["time_column", "temporal_column", "date_column", "timestamp"],
                    "target_column": ["target_column", "value_column", "metric", "column"],
                },
                slot_presentations={
                    "time_column": SlotPresentationPolicy(
                        label="Time / Date Column",
                        role_source="ranked_temporal_columns",
                        fallback_roles=["temporal_column_candidates"],
                    ),
                    "target_column": SlotPresentationPolicy(
                        label="Numeric Metric to Test",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Time Series for Stationarity Test",
                slot_filling_prompt="Select time column and numeric series to test for stationarity (ADF & KPSS)",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="categorical_outcome_association",
            description=(
                "Performs a retrospective descriptive association analysis between one numeric measurement "
                "and one categorical or binary outcome. Returns per-outcome count, mean, median, standard deviation, "
                "and a distribution box plot when requested. It is not a predictive model, causal claim, or prospective targeting feature."
            ),
            input_model=CategoricalOutcomeAssociationArgs,
            objective=AnalysisObjective.COMPARE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=stat_driver,
            adapter=adapt_categorical_outcome_association,
            result_contract="dataframe",
            synonyms=["numeric_categorical_association", "outcome_group_comparison", "retrospective_association"],
            # SemanticIntent retains the established statistical-test type for
            # compatibility; TaskFamily dispatch below is the authoritative
            # execution path.
            category="statistical_test",
            task_family=TaskFamily.CATEGORICAL_OUTCOME_ASSOCIATION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(),
            supported_chart_types=["box", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["numeric_predictor", "outcome_column"],
                can_auto_infer_target=False,
                argument_aliases={
                    "numeric_predictor": ["numeric_predictor", "predictor", "column", "metric"],
                    "outcome_column": ["outcome_column", "target_column", "target", "categorical_feature", "dimension"],
                },
                slot_presentations={
                    "numeric_predictor": SlotPresentationPolicy(
                        label="Numeric Predictor",
                        role_source="ranked_numeric_measures",
                    ),
                    "outcome_column": SlotPresentationPolicy(
                        label="Categorical Outcome",
                        role_source="ranked_categorical_dimensions",
                    ),
                },
                slot_filling_type="dimension_selector",
                slot_filling_title="Select Predictor and Outcome",
                slot_filling_prompt="Select numeric predictor and outcome column for association analysis",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="pca_analysis",
            description="Performs Principal Component Analysis (PCA) for dimensionality reduction",
            input_model=PCAAnalysisArgs,
            objective=AnalysisObjective.TRANSFORM,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=cluster_driver,
            adapter=adapt_pca,
            result_contract="dataframe",
            synonyms=["pca", "dimensionality_reduction"],
            category="pca",
            task_family=TaskFamily.DIMENSIONALITY_REDUCTION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=2),
            supported_chart_types=["bar", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=True,
                dataset_wide_role="ranked_numeric_measures",
                dataset_wide_target_arg="columns",
                dataset_wide_min_columns=2,
                allow_role_candidate_fallback=True,
                fallback_role_name="continuous_numeric_candidates",
                argument_aliases={
                    "columns": ["columns", "column"],
                },
                slot_presentations={
                    "columns": SlotPresentationPolicy(
                        label="Features for PCA",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="generic",
                slot_filling_title="PCA Analysis",
                slot_filling_prompt="Perform PCA dimensionality reduction across numeric features",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="explain_features",
            description=(
                "Computes held-out permutation importance for an explicit target and explicit "
                "predictor list. Requires classification/regression task type and prospective "
                "or retrospective context; preprocessing is fitted inside the training pipeline."
            ),
            input_model=ExplainFeaturesArgs,
            objective=AnalysisObjective.EXPLAIN,
            analysis_modes=[AnalysisMode.PROSPECTIVE, AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=automl_driver,
            adapter=adapt_explain_features,
            result_contract="dataframe",
            synonyms=["explainability", "shap", "feature_importance"],
            category="feature_engineering",
            task_family=TaskFamily.EXPLAINABILITY,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_target_column=True, min_numeric_columns=1),
            supported_chart_types=["bar", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                required_slots=["target_column"],
                can_auto_infer_target=False,
                argument_aliases={
                    "target_column": ["target_column", "target", "value_column", "column"],
                },
                slot_presentations={
                    "target_column": SlotPresentationPolicy(
                        label="Target Feature",
                        role_source="ranked_target_candidates",
                        fallback_roles=["ranked_numeric_measures", "ranked_categorical_dimensions"],
                    ),
                },
                slot_filling_type="target_selector",
                slot_filling_title="Select Target for Feature Importance",
                slot_filling_prompt="Which target variable should feature importance be computed for?",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="autocorrelation",
            description="Computes ACF and PACF lags for time series",
            input_model=AutocorrelationArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=autocorr_driver,
            adapter=adapt_autocorr,
            result_contract="dataframe",
            synonyms=["acf_pacf"],
            category="autocorrelation",
            task_family=TaskFamily.AUTOCORRELATION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_numeric_subjects=True, min_numeric_columns=1),
            supported_chart_types=["acf", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                requires_temporal_column=True,
                temporal_slot_name="time_column",
                required_slots=["time_column", "value_column"],
                inferable_slots={"time_column": "temporal"},
                can_auto_infer_target=False,
                argument_aliases={
                    "time_column": ["time_column", "temporal_column", "date_column", "timestamp"],
                    "value_column": ["value_column", "metric", "column", "target_column"],
                },
                slot_presentations={
                    "time_column": SlotPresentationPolicy(
                        label="Time / Date Column",
                        role_source="ranked_temporal_columns",
                        fallback_roles=["temporal_column_candidates"],
                    ),
                    "value_column": SlotPresentationPolicy(
                        label="Numeric Metric",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Time Series for Autocorrelation",
                slot_filling_prompt="Select time column and value series for ACF/PACF",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="decompose_time_series",
            description=(
                "Decomposes an explicit numeric time series into trend, seasonal and residual "
                "components using a declared additive/multiplicative model and explicit period."
            ),
            input_model=DecomposeTimeSeriesArgs,
            objective=AnalysisObjective.DESCRIBE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=forecast_driver,
            adapter=adapt_decompose_ts,
            result_contract="dataframe",
            synonyms=["ts_decomposition", "time_series_decomposition"],
            category="time_series_decomposition",
            task_family=TaskFamily.TIME_SERIES_DECOMPOSITION,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_temporal_column=True, requires_numeric_subjects=True, min_numeric_columns=1),
            supported_chart_types=["line", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                requires_temporal_column=True,
                temporal_slot_name="time_column",
                required_slots=["time_column", "value_column"],
                inferable_slots={"time_column": "temporal"},
                can_auto_infer_target=False,
                argument_aliases={
                    "time_column": ["time_column", "temporal_column", "date_column", "timestamp"],
                    "value_column": ["value_column", "metric", "column", "target_column"],
                },
                slot_presentations={
                    "time_column": SlotPresentationPolicy(
                        label="Time / Date Column",
                        role_source="ranked_temporal_columns",
                        fallback_roles=["temporal_column_candidates"],
                    ),
                    "value_column": SlotPresentationPolicy(
                        label="Numeric Metric",
                        role_source="ranked_numeric_measures",
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Time Series for Decomposition",
                slot_filling_prompt="Select time column and value series for trend/seasonal decomposition",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="event_window_analysis",
            description=(
                "Compares numeric measurements before versus during an explicitly marked event "
                "(failure, breakdown, defect, alert, churn). Use for pre-event patterns, variance spikes, "
                "or grouped box plots around an event. Requires an event flag/predicate, an explicit ordering "
                "column, lookback window, and event scope (every matching row versus false-to-true onset); "
                "request clarification if any is not supplied."
            ),
            input_model=EventWindowArgs,
            objective=AnalysisObjective.COMPARE,
            analysis_modes=[AnalysisMode.RETROSPECTIVE],
            supported_artifacts=[RequestedArtifact.TABLE, RequestedArtifact.METRICS, RequestedArtifact.CHART, RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=True,
            compiler_or_driver=eda_driver,
            adapter=adapt_event_window,
            result_contract="dataframe",
            # Capability metadata is multilingual so retrieval does not depend
            # on a separate query-time Hinglish keyword routing rule.
            synonyms=[
                "event_window", "pre_event_analysis", "failure_analysis",
                "before_event", "before_failure", "failure se pehle",
            ],
            category="event_window_analysis",
            task_family=TaskFamily.EVENT_WINDOW_ANALYSIS,
            allowed_predecessors=["filter_data", "clean_data"],
            prerequisites=ToolPrerequisites(requires_ordering_column=True, requires_event_column=True, requires_event_predicate=True, requires_pre_event_window=True, requires_numeric_subjects=True, min_numeric_columns=1),
            supported_chart_types=["box", "none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                requires_temporal_column=True,
                temporal_slot_name="time_column",
                required_slots=["time_column", "event_column"],
                inferable_slots={"time_column": "temporal"},
                can_auto_infer_target=False,
                argument_aliases={
                    "time_column": ["time_column", "temporal_column", "date_column", "timestamp"],
                    "event_column": ["event_column", "event", "column"],
                },
                slot_presentations={
                    "time_column": SlotPresentationPolicy(
                        label="Time / Date Column",
                        role_source="ranked_temporal_columns",
                        fallback_roles=["temporal_column_candidates"],
                    ),
                    "event_column": SlotPresentationPolicy(
                        label="Event Column",
                        role_source="ranked_categorical_dimensions",
                        fallback_roles=["binary_flag_candidates"],
                    ),
                },
                slot_filling_type="column_selection",
                slot_filling_title="Select Event Window Parameters",
                slot_filling_prompt="Select time column, event column, and lookback window",
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="general_chat",
            description="Handles conceptual explanations, general data science questions, educational inquiries, and conversational greetings",
            input_model=GeneralChatArgs,
            objective=AnalysisObjective.CONVERSE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=False,
            compiler_or_driver=None,
            adapter=adapt_chat,
            result_contract="text",
            side_effect_level="read_only",
            synonyms=["chat", "general_response", "conversation"],
            category="general_chat",
            task_family=TaskFamily.GENERAL_CHAT,
            allowed_predecessors=[],
            prerequisites=ToolPrerequisites(),
            planner_always_available=True,
            planner_without_dataset_only=True,
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
            ),
        ))

        self.register_tool_spec(ToolSpec(
            name="clarification_needed",
            description="Requests user clarification when intent or required column is ambiguous or missing",
            input_model=ClarificationArgs,
            objective=AnalysisObjective.CONVERSE,
            analysis_modes=[AnalysisMode.DESCRIPTIVE],
            supported_artifacts=[RequestedArtifact.WRITTEN_FINDING],
            requires_dataset=False,
            compiler_or_driver=None,
            adapter=adapt_clarify,
            result_contract="text",
            side_effect_level="read_only",
            synonyms=["clarify", "ambiguous"],
            category="clarification",
            task_family=TaskFamily.CLARIFICATION,
            allowed_predecessors=[],
            prerequisites=ToolPrerequisites(),
            planner_always_available=True,
            supported_chart_types=["none"],
            completion_policy=CompletionPolicy(
                allowed_dataset_wide_mode=False,
                is_clarification_tool=True,
                clarification_category="clarification",
            ),
        ))

        for intent_k, driver_obj in DRIVER_REGISTRY.items():
            self.register_driver(intent_k, driver_obj)
        self._canonical_tool_names = set(self._tool_specs.keys())

    def register_tool_spec(self, spec: ToolSpec) -> None:
        """
        Registers an authoritative ToolSpec into the single source of truth registry.
        """
        spec.validate_contract()
        t_clean = spec.name.lower().strip()
        self._planner_embedding_cache.clear()
        self._tool_specs[t_clean] = spec
        if spec.compiler_or_driver:
            self._driver_registry[t_clean] = spec.compiler_or_driver
        if spec.synonyms:
            for syn in spec.synonyms:
                self._tool_specs[syn.lower().strip()] = spec
                if spec.compiler_or_driver:
                    self._driver_registry[syn.lower().strip()] = spec.compiler_or_driver
        logger.info(f"Registered ToolSpec: '{spec.name}' (category='{spec.category}').")

    def unregister_tool_spec(self, name: str) -> None:
        """
        Unregisters a ToolSpec and its synonyms from the registry.
        """
        t_clean = str(name).lower().strip()
        spec = self._tool_specs.pop(t_clean, None)
        if spec:
            self._driver_registry.pop(t_clean, None)
            if spec.synonyms:
                for syn in spec.synonyms:
                    self._tool_specs.pop(syn.lower().strip(), None)
                    self._driver_registry.pop(syn.lower().strip(), None)
            self._planner_embedding_cache.clear()
            logger.info(f"Unregistered ToolSpec: '{spec.name}'.")

    def get_tool_spec(self, name: str) -> Optional[ToolSpec]:
        """
        Retrieves ToolSpec by name or synonym.
        """
        return self._tool_specs.get(str(name).lower().strip())

    def list_tool_specs(self) -> List[ToolSpec]:
        """
        Lists all unique registered ToolSpecs.
        """
        seen = set()
        specs = []
        for s in self._tool_specs.values():
            if s.name not in seen:
                seen.add(s.name)
                specs.append(s)
        return specs

    def compile_v3_from_v2(
        self,
        *,
        tool_name: str,
        intent: Any,
        dataset_id: str,
        arguments: Dict[str, Any],
        analysis_mode: AnalysisMode,
        dataset_fingerprint: Optional[str] = None,
        operation_id: str = "op_1",
    ) -> AnalysisSpecV3:
        """Compile a validated registry capability into the V3 envelope.

        The registry supplies capability semantics; callers must still provide
        the request's explicit analysis mode.  This prevents a planner or
        migration shim from silently converting retrospective analysis into a
        prospective model (or vice versa).
        """
        spec = self.get_tool_spec(tool_name)
        if spec is None:
            raise ValueError(f"Unknown tool: '{tool_name}'")
        if not spec.requires_dataset:
            raise ValueError(
                f"Tool '{spec.name}' is non-analytical and cannot compile to an executable dataset plan."
            )
        if analysis_mode not in spec.analysis_modes:
            allowed = [mode.value for mode in spec.analysis_modes]
            raise ValueError(
                f"Tool '{spec.name}' does not support analysis mode '{analysis_mode.value}'. "
                f"Allowed modes: {allowed}."
            )
        validated = spec.input_model.model_validate(arguments)
        compiled = AnalysisSpecV3.from_v2(
            intent,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            objective=spec.objective,
            analysis_mode=analysis_mode,
            tool=spec.name,
            arguments=validated.model_dump(mode="json"),
            operation_id=operation_id,
        )
        unsupported_artifacts = set(compiled.requested_artifacts) - set(spec.supported_artifacts)
        if unsupported_artifacts:
            values = sorted(artifact.value for artifact in unsupported_artifacts)
            raise ValueError(
                f"Tool '{spec.name}' cannot produce requested artifacts: {values}."
            )
        allowed_charts = set(spec.supported_chart_types or [])
        for visualization in compiled.visualizations:
            if visualization.chart_type not in allowed_charts:
                raise ValueError(
                    f"Tool '{spec.name}' cannot produce chart '{visualization.chart_type}'. "
                    f"Allowed charts: {sorted(allowed_charts)}."
                )
        return compiled

    def capability_manifest(self, canonical_only: bool = True) -> List[Dict[str, Any]]:
        """Return deterministic public metadata for prompts, audits and tests."""
        specs = self.list_tool_specs()
        if canonical_only and self._canonical_tool_names:
            specs = [s for s in specs if s.name.lower().strip() in self._canonical_tool_names]
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "objective": spec.objective.value,
                "analysis_modes": [mode.value for mode in spec.analysis_modes],
                "requires_dataset": spec.requires_dataset,
                "task_family": getattr(spec.task_family, "value", spec.task_family),
                "result_contract": spec.result_contract,
                "supported_artifacts": [artifact.value for artifact in spec.supported_artifacts],
                "supported_chart_types": list(spec.supported_chart_types or []),
                "side_effect_level": spec.side_effect_level,
                "input_schema": spec.input_model.model_json_schema(),
            }
            for spec in sorted(specs, key=lambda item: item.name)
        ]

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> BaseModel:
        """
        Validates arguments against the tool's strict Pydantic input model.
        """
        spec = self.get_tool_spec(tool_name)
        if not spec:
            raise ValueError(f"Unknown tool: '{tool_name}'")
        return spec.input_model.model_validate(arguments)

    def get_driver(self, tool_name: str) -> Optional[Any]:
        """
        Retrieves the execution driver/compiler for a tool name or intent key.
        """
        t_clean = str(tool_name).lower().strip()
        if t_clean in self._tool_specs and self._tool_specs[t_clean].compiler_or_driver:
            return self._tool_specs[t_clean].compiler_or_driver
        return self._driver_registry.get(t_clean)

    def get_driver_for_intent(self, intent_key: str) -> Optional[Any]:
        """
        Retrieves the registered driver instance for an intent key.
        """
        return self.get_driver(intent_key)

    def register_driver(self, intent_or_tool: Any, driver_instance: Any) -> None:
        """
        Registers an execution driver instance for a given intent or tool name.
        """
        key = intent_or_tool.value if hasattr(intent_or_tool, "value") else str(intent_or_tool).strip()
        self._driver_registry[key] = driver_instance
        logger.info(f"Registered driver '{driver_instance.__class__.__name__}' for intent '{key}'.")

    def register_tool(
        self,
        tool_name: str,
        schema_signature: str,
        intent_type: Any,
        synonyms: Optional[List[str]] = None,
        description: str = "",
        driver_instance: Optional[Any] = None
    ) -> None:
        """
        Registers a dynamic analytical or conversational tool into the single source of truth registry.
        """
        t_clean = tool_name.lower().strip()
        sig_str = f"{schema_signature} {description}".strip() if description else schema_signature
        self._tool_definitions[t_clean] = sig_str
        self._tool_intent_map[t_clean] = intent_type
        if synonyms:
            for syn in synonyms:
                self._tool_intent_map[syn.lower().strip()] = intent_type
        if driver_instance is not None:
            self.register_driver(intent_type, driver_instance)
            self.register_driver(t_clean, driver_instance)

    def get_planner_tool_definitions(self) -> Dict[str, str]:
        res = dict(self._tool_definitions)
        for s in self.list_tool_specs():
            res[s.name] = f"{s.description}"
        return res

    def select_planner_tools(
        self,
        query: str,
        limit: Optional[int] = None,
        has_dataset: Optional[bool] = None,
    ) -> List[ToolSpec]:
        """Return a semantically relevant, registry-derived planning tool set.

        This is retrieval over live ToolSpec descriptions and metadata, not a
        keyword routing table. It keeps a small local model within context
        while preserving registry-declared clarification/chat options.
        """
        from modules.capability_retriever import CapabilityRetriever

        retriever = CapabilityRetriever(self)
        return retriever.retrieve(
            query=query,
            limit=limit,
            has_dataset=has_dataset,
            embedding_cache=self._planner_embedding_cache,
        )

    def expand_tools_for_requested_visualizations(
        self,
        selected_specs: List[ToolSpec],
        command: Any,
    ) -> List[ToolSpec]:
        """Add live chart-capable ToolSpecs after a truthful chart rejection.

        This is repair-time capability recall, not an execution fallback: the
        model still receives the schemas and must choose/validate a new tool.
        It prevents an initial retrieval miss from trapping the planner inside
        a table-only capability after the user explicitly requested a chart.
        """
        raw_calls = command.get("commands") if isinstance(command, dict) else None
        calls = raw_calls if isinstance(raw_calls, list) else [command]
        requested: set[str] = set()
        for call in calls:
            if not isinstance(call, dict):
                continue
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            raw_chart = args.get("chart_type", args.get("chart"))
            if raw_chart is not None:
                value = str(getattr(raw_chart, "value", raw_chart)).strip().lower()
                if value and value not in {"none", "null"}:
                    requested.add(value)
        if not requested:
            return selected_specs

        expanded = list(selected_specs)
        known_names = {spec.name for spec in expanded}
        for spec in self.list_tool_specs():
            supported = spec.supported_chart_types
            if not supported:
                continue
            supported_names = {
                str(getattr(value, "value", value)).strip().lower()
                for value in supported
            }
            if requested.intersection(supported_names) and spec.name not in known_names:
                expanded.append(spec)
                known_names.add(spec.name)
        return expanded

    def get_planner_tools_prompt(self, specs: Optional[List[ToolSpec]] = None) -> str:
        lines = []
        for s in specs if specs is not None else self.list_tool_specs():
            lines.append(f"- {s.name}: {s.description}")
        if specs is None:
            selected_names = {s.name for s in self.list_tool_specs()}
            for t_name, sig in self._tool_definitions.items():
                if t_name not in selected_names:
                    lines.append(f"- {t_name}: {sig}")
        return "\n".join(lines)

    def get_openai_tool_definitions(self, specs: Optional[List[ToolSpec]] = None) -> List[Dict[str, Any]]:
        """
        Generates OpenAI-compatible tool calling schemas for Ollama.
        """
        tools = []
        for s in specs if specs is not None else self.list_tool_specs():
            tools.append({
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": self._planner_arguments_schema(s)
                }
            })
        return tools

    @staticmethod
    def _inline_local_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Inline a Pydantic schema's local ``#/$defs/...`` references.

        A ToolSpec schema is normally a complete JSON Schema document, so its
        local references are valid at that document's root.  When it becomes
        the nested ``arguments`` schema of DataPilot's discriminated command
        schema, the root changes and Ollama resolves ``#/$defs/...`` against
        the *outer* document.  The result is a 400 before the model sees the
        query.  Inlining keeps the typed schema and the registry dispatch
        intact while making the combined schema self-contained for Ollama.
        """
        definitions = deepcopy(schema.get("$defs", {}))

        def resolve(value: Any, stack: tuple[str, ...] = ()) -> Any:
            if isinstance(value, list):
                return [resolve(item, stack) for item in value]
            if not isinstance(value, dict):
                return value

            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                definition_name = ref.removeprefix("#/$defs/")
                if definition_name not in definitions:
                    raise RuntimeError(
                        f"CAPABILITY_SCHEMA_INVALID: Missing local definition '{definition_name}'."
                    )
                if definition_name in stack:
                    raise RuntimeError(
                        f"CAPABILITY_SCHEMA_INVALID: Recursive local definition '{definition_name}' is unsupported."
                    )
                resolved_definition = resolve(deepcopy(definitions[definition_name]), stack + (definition_name,))
                sibling_fields = {key: item for key, item in value.items() if key != "$ref"}
                if sibling_fields:
                    if not isinstance(resolved_definition, dict):
                        raise RuntimeError(
                            f"CAPABILITY_SCHEMA_INVALID: Cannot merge fields into local definition '{definition_name}'."
                        )
                    resolved_definition.update(resolve(sibling_fields, stack))
                return resolved_definition

            return {
                key: resolve(item, stack)
                for key, item in value.items()
                if key != "$defs"
            }

        return resolve(deepcopy(schema))

    @classmethod
    def _planner_argument_schemas(cls, spec: ToolSpec) -> List[Dict[str, Any]]:
        models = spec.planner_input_models or [spec.planner_input_model or spec.input_model]
        return [cls._inline_local_schema_refs(model.model_json_schema()) for model in models]

    @classmethod
    def _planner_arguments_schema(cls, spec: ToolSpec) -> Dict[str, Any]:
        schemas = cls._planner_argument_schemas(spec)
        return schemas[0] if len(schemas) == 1 else {"oneOf": schemas}

    def get_dynamic_command_json_schema(
        self,
        specs: Optional[List[ToolSpec]] = None,
        *,
        allow_command_sequence: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates dynamic JSON schema for Ollama structured output based on live registered ToolSpecs.
        Fails fast if no ToolSpecs are registered (zero fallback).
        """
        specs = specs if specs is not None else self.list_tool_specs()
        if not specs:
            raise RuntimeError("CAPABILITY_REGISTRY_UNAVAILABLE: No ToolSpecs are registered in CapabilityRegistry.")

        tool_call_variants = []
        for spec in specs:
            for typed_arguments_schema in self._planner_argument_schemas(spec):
                tool_call_variants.append({
                    "type": "object",
                    "properties": {
                        "tool": {"const": spec.name},
                        "arguments": typed_arguments_schema,
                    },
                    "required": ["tool", "arguments"],
                    "additionalProperties": False,
                })

        # The planner normally emits one canonical operation.  A command array
        # is enabled only for callers that explicitly support a composed
        # workflow.  This avoids asking a small local model to plan unrelated
        # parallel analyses for a single analytical request.
        if not allow_command_sequence:
            if len(tool_call_variants) == 1:
                return tool_call_variants[0]
            return {"oneOf": tool_call_variants}

        multi_command_variant = {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"oneOf": tool_call_variants},
                },
            },
            "required": ["commands"],
            "additionalProperties": False,
        }

        return {"oneOf": tool_call_variants + [multi_command_variant]}

    def get_tool_intent_map(self) -> Dict[str, Any]:
        res = dict(self._tool_intent_map)
        for s in self.list_tool_specs():
            from backend.ds_engine.intent_schema import IntentType
            cat_upper = s.category.upper() if s.category else ""
            if not hasattr(IntentType, cat_upper):
                raise ValueError(f"REGISTRY_CONTRACT_INVALID: ToolSpec '{s.name}' has invalid category '{s.category}'. Must match a valid IntentType member.")
            cat_intent = getattr(IntentType, cat_upper)
            res[s.name] = cat_intent
            if s.synonyms:
                for syn in s.synonyms:
                    res[syn] = cat_intent
        return res

    def register(self, capability: Capability) -> None:
        if capability.name in self._registry:
            logger.error(f"Failed to register capability: '{capability.name}' already exists.")
            raise ValueError(f"Capability with name '{capability.name}' is already registered.")
        
        self._registry[capability.name] = capability
        logger.info(f"Successfully registered capability: '{capability.name}' in category '{capability.category}'.")

    def find_matching_capability(self, intent: str, tags: Any = None, df: Any = None) -> Optional[Capability]:
        if tags is not None and not isinstance(tags, (list, set, tuple)):
            if hasattr(tags, 'columns'):
                df = tags
                tags = None
            else:
                tags = [str(tags)]

        tags_to_match = set(tags) if tags is not None and isinstance(tags, (list, set, tuple)) else set()
        
        best_match = None
        max_score = -1

        for cap in self._registry.values():
            score = 0
            if intent in cap.intent_tags or intent == cap.name:
                score += 10
            
            common_tags = tags_to_match.intersection(set(cap.intent_tags))
            score += len(common_tags)

            if score > max_score and score > 0:
                best_match = cap
                max_score = score

        return best_match

    def list_all_capabilities(self) -> List[Capability]:
        return list(self._registry.values())

    def get_capabilities_by_category(self, category: str) -> List[Capability]:
        return [cap for cap in self._registry.values() if cap.category == category]

    def execute_capability(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._registry:
            logger.error(f"Cannot execute capability: '{name}' not found.")
            raise KeyError(f"Capability '{name}' not found in registry.")

        capability = self._registry[name]
        logger.info(f"Executing capability: '{name}'")
        
        try:
            result = capability.handler_func(*args, **kwargs)
            logger.info(f"Successfully executed capability: '{name}'")
            return result
        except Exception as e:
            logger.error(f"Error executing capability '{name}': {str(e)}", exc_info=True)
            raise


# Global Singleton Instance
capability_registry = CapabilityRegistry()
capability_registry._bootstrap_canonical_tools()

def register_default_capabilities(registry: CapabilityRegistry) -> None:
    registry._bootstrap_canonical_tools()

def get_canonical_tool_definitions() -> Dict[str, str]:
    """
    Returns authoritative dictionary of tools and schema signatures directly from capability_registry.
    """
    return capability_registry.get_planner_tool_definitions()
