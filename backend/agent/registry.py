"""
DataPilot OS — Agent Tool & Capability Registry.
Provides dynamic capability registration, schema discovery, tool dispatching,
and declarative policy-based approval evaluation for the Local Workspace Agent.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from backend.agent.permissions import AgentPolicy
from backend.agent.tools import ToolResult

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    READ_ONLY = "READ_ONLY"
    WORKSPACE_WRITE = "WORKSPACE_WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXECUTE_COMMAND = "EXECUTE_COMMAND"
    VERIFICATION = "VERIFICATION"


@dataclass
class AgentTool:
    name: str
    description: str
    signature: str
    category: ToolCategory
    handler: Callable[..., ToolResult]
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    prompt_note: Optional[str] = None

    def requires_approval(self, policy: AgentPolicy) -> bool:
        """Evaluates whether this tool requires explicit human approval under the given policy."""
        if policy == AgentPolicy.REVIEW:
            return self.category in (
                ToolCategory.WORKSPACE_WRITE,
                ToolCategory.DESTRUCTIVE,
                ToolCategory.EXECUTE_COMMAND,
            )
        elif policy == AgentPolicy.AUTO_EDIT:
            return self.category in (
                ToolCategory.DESTRUCTIVE,
                ToolCategory.EXECUTE_COMMAND,
            )
        elif policy == AgentPolicy.AUTONOMOUS:
            return self.category == ToolCategory.DESTRUCTIVE
        return True


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        """Registers a tool in the capability registry."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered agent tool: {tool.name} (category: {tool.category})")

    def unregister(self, name: str) -> bool:
        """Unregisters a tool (primarily for dynamic test fixtures)."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def is_valid_tool(self, name: str) -> bool:
        return name in self._tools

    def get_valid_tool_names(self) -> Set[str]:
        return set(self._tools.keys())

    def list_tools(self) -> List[AgentTool]:
        return list(self._tools.values())

    def format_system_prompt_tools(self) -> str:
        """Generates the formatted tool specification list for the planner's system prompt."""
        lines = []
        for tool in self._tools.values():
            entry = f"- {tool.signature}"
            if tool.prompt_note:
                entry += f"  # {tool.prompt_note}"
            lines.append(entry)
        return "\n".join(lines)

    def evaluate_step_approval(self, tool_name: str, policy: AgentPolicy) -> bool:
        """Determines if a proposed tool call requires approval under the given policy."""
        tool = self._tools.get(tool_name)
        if not tool:
            return True  # Fail-closed for unknown tools
        return tool.requires_approval(policy)

    def dispatch(self, tool_name: str, args: Dict[str, Any], task_id: str) -> ToolResult:
        """Dispatches execution to the registered tool handler."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Unrecognized tool '{tool_name}'",
                error_code="UNKNOWN_TOOL",
            )

        kwargs = dict(args)
        # Check if the tool handler expects task_id
        try:
            sig = inspect.signature(tool.handler)
            if "task_id" in sig.parameters and "task_id" not in kwargs:
                kwargs["task_id"] = task_id
        except Exception:
            pass

        return tool.handler(**kwargs)


# Global singleton registry
agent_tool_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    signature: str,
    category: ToolCategory,
    parameters_schema: Optional[Dict[str, Any]] = None,
    prompt_note: Optional[str] = None,
):
    """Decorator to register a tool into the global capability registry."""
    def decorator(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        tool = AgentTool(
            name=name,
            description=description,
            signature=signature,
            category=category,
            handler=fn,
            parameters_schema=parameters_schema or {},
            prompt_note=prompt_note,
        )
        agent_tool_registry.register(tool)
        return fn
    return decorator


def _initialize_default_tools():
    """Initializes and registers all standard DataPilot OS Local Workspace Agent tools."""
    from backend.agent import tools

    agent_tool_registry.register(
        AgentTool(
            name="list_directory",
            description="Lists files and folders within an authorized path with bounded traversal.",
            signature="list_directory(path: str, recursive: bool = False, max_depth: int = 2)",
            category=ToolCategory.READ_ONLY,
            handler=tools.list_directory,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="get_file_metadata",
            description="Retrieves metadata, size, line count, permissions, and SHA-256 for a file.",
            signature="get_file_metadata(path: str)",
            category=ToolCategory.READ_ONLY,
            handler=tools.get_file_metadata,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="read_file",
            description="Reads text content from an authorized file with optional line bounds.",
            signature="read_file(path: str, line_start: int = None, line_end: int = None)",
            category=ToolCategory.READ_ONLY,
            handler=tools.read_file,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="search_files",
            description="Searches for files matching a glob pattern within the workspace.",
            signature="search_files(workspace_path: str, pattern: str)",
            category=ToolCategory.READ_ONLY,
            handler=tools.search_files,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="search_file_contents",
            description="Searches for text matches across files within the workspace.",
            signature="search_file_contents(workspace_path: str, query: str)",
            category=ToolCategory.READ_ONLY,
            handler=tools.search_file_contents,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="create_directory",
            description="Creates a new directory within the authorized workspace.",
            signature="create_directory(path: str)",
            category=ToolCategory.WORKSPACE_WRITE,
            handler=tools.create_directory,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="create_file",
            description="Creates a new file with the specified content.",
            signature="create_file(path: str, content: str, overwrite: bool = False)",
            category=ToolCategory.WORKSPACE_WRITE,
            handler=tools.create_file,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="apply_file_patch",
            description="Atomically updates an existing file with optimistic concurrency validation.",
            signature="apply_file_patch(path: str, new_content: str)",
            category=ToolCategory.WORKSPACE_WRITE,
            handler=tools.apply_file_patch,
            prompt_note="expected_hash is optional and should be omitted unless an exact SHA-256 hex string is known",
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="move_file",
            description="Moves or renames a file within the authorized workspace.",
            signature="move_file(source_path: str, target_path: str)",
            category=ToolCategory.WORKSPACE_WRITE,
            handler=tools.move_file,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="delete_file",
            description="Safely deletes an authorized file.",
            signature="delete_file(path: str, explicit_confirmed: bool = False)",
            category=ToolCategory.DESTRUCTIVE,
            handler=tools.delete_file,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="execute_command",
            description="Executes a command safely under the Windows subprocess runner with process tree termination.",
            signature="execute_command(command: str, cwd: str = None, timeout_seconds: int = 30)",
            category=ToolCategory.EXECUTE_COMMAND,
            handler=tools.execute_command,
            prompt_note="Use for running build/test tools, not for reading or listing files.",
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="get_diff",
            description="Computes a unified diff between current file contents and proposed content.",
            signature="get_diff(path: str, new_content: str)",
            category=ToolCategory.READ_ONLY,
            handler=tools.get_diff,
        )
    )

    agent_tool_registry.register(
        AgentTool(
            name="run_verification",
            description="Executes post-task verification checks against the workspace.",
            signature="run_verification(task_id: str, checks: list)",
            category=ToolCategory.VERIFICATION,
            handler=tools.run_verification,
        )
    )


# Automatically initialize default standard tools on module load
_initialize_default_tools()
