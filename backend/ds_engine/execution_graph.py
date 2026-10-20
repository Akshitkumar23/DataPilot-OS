"""Deterministic compilation of validated ToolSpec calls into an execution DAG."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    step_index: int = Field(ge=1)
    tool: str
    arguments: Dict[str, Any]
    depends_on: List[str] = Field(default_factory=list)
    input_contract: str
    output_contract: str
    task_family: str


class ExecutionGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_version: str = "1.0"
    nodes: List[ExecutionGraphNode] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [node.operation_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Execution graph operation IDs must be unique.")
        known: set[str] = set()
        for node in self.nodes:
            unknown = set(node.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Node '{node.operation_id}' depends on unavailable prior nodes: {sorted(unknown)}."
                )
            known.add(node.operation_id)
        return self


class ExecutionGraphCompiler:
    """Compile the planner's ordered calls without inventing transformations."""

    @classmethod
    def compile(cls, commands: List[Dict[str, Any]], registry) -> ExecutionGraph:
        if not isinstance(commands, list) or not 1 <= len(commands) <= 8:
            raise ValueError("INVALID_PLAN_SEQUENCE: A plan must contain between 1 and 8 steps.")

        nodes: List[ExecutionGraphNode] = []
        previous = None
        for index, command in enumerate(commands, start=1):
            if not isinstance(command, dict) or set(command) != {"tool", "arguments"}:
                raise ValueError(
                    "INVALID_PLAN_SEQUENCE: Each step requires exactly tool and arguments."
                )
            tool_name = command["tool"]
            spec = registry.get_tool_spec(tool_name)
            if spec is None:
                raise ValueError(f"UNSUPPORTED_OPERATION: Unknown tool: {tool_name}")
            if not spec.requires_dataset:
                raise ValueError(
                    "INVALID_PLAN_SEQUENCE: Conversational/clarification tools cannot be execution nodes."
                )
            # A model-training node is permitted only as the terminal consumer
            # of a dataframe produced by earlier read-only transformations.
            # It must never be an intermediate node: no downstream step can
            # safely assume a reusable fitted model or mutate source data.
            if spec.side_effect_level != "read_only" and index != len(commands):
                raise ValueError(
                    f"INVALID_PLAN_SEQUENCE: Tool '{spec.name}' has side effect level "
                    f"'{spec.side_effect_level}' and must be the final plan node."
                )
            if previous is not None and previous.output_contract != "dataframe":
                raise ValueError(
                    f"INVALID_PLAN_SEQUENCE: Node '{previous.operation_id}' produces "
                    f"'{previous.output_contract}', not the dataframe required by the next node."
                )
            if previous is not None:
                allowed = set(spec.allowed_predecessors or [])
                if previous.tool not in allowed and previous.task_family not in allowed:
                    raise ValueError(
                        f"INVALID_PLAN_SEQUENCE: Tool '{spec.name}' does not allow predecessor "
                        f"'{previous.tool}'."
                    )

            validated = registry.validate_tool_call(spec.name, command["arguments"])
            operation_id = f"op_{index}"
            node = ExecutionGraphNode(
                operation_id=operation_id,
                step_index=index,
                tool=spec.name,
                arguments=validated.model_dump(mode="json"),
                depends_on=[previous.operation_id] if previous else [],
                input_contract="dataframe",
                output_contract=spec.result_contract,
                task_family=getattr(spec.task_family, "value", str(spec.task_family)),
            )
            nodes.append(node)
            previous = node

        return ExecutionGraph(nodes=nodes)
