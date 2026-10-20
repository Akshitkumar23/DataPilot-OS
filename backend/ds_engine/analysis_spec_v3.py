"""Versioned, provider-neutral analytical contract for DataPilot.

``AnalysisSpecV3`` is the boundary between probabilistic planning and
deterministic execution.  It intentionally contains no query-text heuristics,
dataset-specific names, or inferred defaults.  A planner/compiler must provide
every scientific decision explicitly, or emit a clarification/unsupported/
unsafe decision instead of an executable plan.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.ds_engine.analysis_decision import AnalysisDecision, DecisionType
from backend.ds_engine.intent_schema import AnalysisIntentV2, FilterSpec


class AnalysisObjective(str, Enum):
    DESCRIBE = "describe"
    COMPARE = "compare"
    INFER = "infer"
    PREDICT = "predict"
    FORECAST = "forecast"
    TRANSFORM = "transform"
    EXPLAIN = "explain"
    CONVERSE = "converse"


class AnalysisMode(str, Enum):
    DESCRIPTIVE = "descriptive"
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"


class RequestedArtifact(str, Enum):
    TABLE = "table"
    CHART = "chart"
    METRICS = "metrics"
    MODEL = "model"
    TRANSFORMED_DATA = "transformed_data"
    WRITTEN_FINDING = "written_finding"
    CODE = "code"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetBindingV3(StrictModel):
    dataset_id: str = Field(min_length=1)
    fingerprint: Optional[str] = Field(
        default=None,
        description="Stable content/schema fingerprint when available.",
    )


class ColumnRolesV3(StrictModel):
    targets: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_duplicate_roles(self):
        for field_name in ("targets", "features", "dimensions", "metrics", "entities"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain only non-empty column names.")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicate column names.")
        return self


class TimeAxisV3(StrictModel):
    column: str = Field(min_length=1)
    sequence_column: Optional[str] = None
    entity_column: Optional[str] = None
    frequency: Optional[str] = None
    grain: Optional[str] = None
    horizon: Optional[int] = Field(default=None, gt=0)
    timezone: Optional[str] = None


class VisualizationRequestV3(StrictModel):
    chart_type: str = Field(min_length=1)
    x: Optional[str] = None
    y: List[str] = Field(default_factory=list)
    facet: Optional[str] = None
    color: Optional[str] = None


class DeclaredAssumptionV3(StrictModel):
    statement: str = Field(min_length=1)
    evidence: str = Field(
        min_length=1,
        description="User, dataset-profile, policy, or registry evidence for this assumption.",
    )


class AnalysisOperationV3(StrictModel):
    operation_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: Dict[str, Any]
    depends_on: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_self_dependency(self):
        if self.operation_id in self.depends_on:
            raise ValueError(f"Operation '{self.operation_id}' cannot depend on itself.")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError(f"Operation '{self.operation_id}' contains duplicate dependencies.")
        return self


class ClarificationRequestV3(StrictModel):
    question: str = Field(min_length=1)
    missing_fields: List[str] = Field(min_length=1)
    candidates: Dict[str, List[str]] = Field(default_factory=dict)


class AnalysisSpecV3(StrictModel):
    """Single authoritative semantic plan contract.

    Execution is allowed only for a fully grounded DAG.  Non-executable
    decisions are mutually exclusive with operations, preventing the historical
    mixed "clarify and execute" failure mode.
    """

    schema_version: Literal["3.0"] = "3.0"
    decision: AnalysisDecision
    dataset: Optional[DatasetBindingV3]
    objective: Optional[AnalysisObjective]
    analysis_mode: Optional[AnalysisMode]
    operations: List[AnalysisOperationV3] = Field(default_factory=list)
    columns: ColumnRolesV3 = Field(default_factory=ColumnRolesV3)
    filters: List[FilterSpec] = Field(default_factory=list)
    time_axis: Optional[TimeAxisV3] = None
    requested_artifacts: List[RequestedArtifact] = Field(default_factory=list)
    visualizations: List[VisualizationRequestV3] = Field(default_factory=list)
    assumptions: List[DeclaredAssumptionV3] = Field(default_factory=list)
    clarification: Optional[ClarificationRequestV3] = None
    decision_reason: Optional[str] = None
    raw_query: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision_and_dag(self):
        if self.decision == AnalysisDecision.EXECUTE:
            missing = [
                name
                for name, value in (
                    ("dataset", self.dataset),
                    ("objective", self.objective),
                    ("analysis_mode", self.analysis_mode),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"Executable plans require: {', '.join(missing)}.")
            if not self.operations:
                raise ValueError("Executable plans require at least one operation.")
            if self.clarification is not None:
                raise ValueError("Executable plans cannot also request clarification.")
        else:
            if self.operations:
                raise ValueError("Non-executable decisions cannot contain operations.")
            if self.decision == AnalysisDecision.CLARIFY:
                if self.clarification is None:
                    raise ValueError("Clarify decisions require a clarification request.")
            elif not self.decision_reason or not self.decision_reason.strip():
                raise ValueError("Unsupported and unsafe decisions require a reason.")

        operation_ids = [operation.operation_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("Operation IDs must be unique.")
        known = set(operation_ids)
        for operation in self.operations:
            unknown = set(operation.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Operation '{operation.operation_id}' has unknown dependencies: {sorted(unknown)}."
                )

        visiting: set[str] = set()
        visited: set[str] = set()
        dependencies = {op.operation_id: op.depends_on for op in self.operations}

        def visit(operation_id: str) -> None:
            if operation_id in visiting:
                raise ValueError("Operation dependency graph contains a cycle.")
            if operation_id in visited:
                return
            visiting.add(operation_id)
            for dependency in dependencies.get(operation_id, []):
                visit(dependency)
            visiting.remove(operation_id)
            visited.add(operation_id)

        for operation_id in operation_ids:
            visit(operation_id)
        return self

    @classmethod
    def from_v2(
        cls,
        intent: AnalysisIntentV2,
        *,
        dataset_id: str,
        objective: AnalysisObjective,
        analysis_mode: AnalysisMode,
        tool: str,
        arguments: Dict[str, Any],
        operation_id: str = "op_1",
        dataset_fingerprint: Optional[str] = None,
    ) -> "AnalysisSpecV3":
        """Losslessly envelope a V2 intent without guessing missing semantics.

        Objective, mode, tool identity and validated arguments are deliberately
        required from the registry/compiler.  The migration layer never derives
        them from raw text or task-family keywords.
        """
        visualizations: List[VisualizationRequestV3] = []
        if intent.visualization and intent.visualization.type != "none":
            raw_y = intent.visualization.y
            y = raw_y if isinstance(raw_y, list) else ([raw_y] if raw_y else [])
            visualizations.append(
                VisualizationRequestV3(
                    chart_type=intent.visualization.type,
                    x=intent.visualization.x,
                    y=y,
                    facet=intent.visualization.facet,
                    color=intent.visualization.color,
                )
            )

        time_axis = None
        time_column = intent.time_column or (
            intent.time_context.time_column if intent.time_context else None
        )
        if time_column:
            context = intent.time_context
            time_axis = TimeAxisV3(
                column=time_column,
                sequence_column=intent.sequence_column or (context.sequence_column if context else None),
                entity_column=context.entity_id if context else None,
                frequency=context.frequency if context else None,
                horizon=intent.horizon or (context.horizon if context else None),
            )

        artifact_map = {
            "table": RequestedArtifact.TABLE,
            "chart": RequestedArtifact.CHART,
            "metrics": RequestedArtifact.METRICS,
            "model": RequestedArtifact.MODEL,
            "transformed_data": RequestedArtifact.TRANSFORMED_DATA,
            "written_finding": RequestedArtifact.WRITTEN_FINDING,
            "code": RequestedArtifact.CODE,
        }
        unknown_artifacts = sorted(set(intent.required_artifacts) - set(artifact_map))
        if unknown_artifacts:
            raise ValueError(f"V2 plan contains unsupported artifact values: {unknown_artifacts}.")

        return cls(
            decision=AnalysisDecision.EXECUTE,
            dataset=DatasetBindingV3(dataset_id=dataset_id, fingerprint=dataset_fingerprint),
            objective=objective,
            analysis_mode=analysis_mode,
            operations=[
                AnalysisOperationV3(
                    operation_id=operation_id,
                    tool=tool,
                    arguments=arguments,
                )
            ],
            columns=ColumnRolesV3(
                targets=[intent.target_column] if intent.target_column else [],
                features=[column for column in intent.subjects if column != intent.target_column],
                dimensions=list(intent.grouping),
                metrics=list(intent.subjects),
                entities=(
                    [intent.time_context.entity_id]
                    if intent.time_context and intent.time_context.entity_id
                    else []
                ),
            ),
            filters=list(intent.conditions),
            time_axis=time_axis,
            requested_artifacts=[artifact_map[value] for value in intent.required_artifacts],
            visualizations=visualizations,
            assumptions=[
                DeclaredAssumptionV3(statement=value, evidence="v2_contract")
                for value in intent.assumptions
            ],
            raw_query=intent.raw_query or intent.prompt_summary or "V2 migration",
        )

