"""Registry-driven semantic tool retrieval contracts."""

from modules.capability_registry import capability_registry


def test_event_query_retrieves_event_tool_and_registry_safeguards():
    tools = capability_registry.select_planner_tools(
        "Find the variance pattern before an equipment failure and draw a box plot."
    )
    names = {tool.name for tool in tools}

    assert "event_window_analysis" in names
    assert "eda_data" not in names
    assert {tool.name for tool in capability_registry.list_tool_specs() if tool.planner_always_available} <= names


def test_specific_event_query_retrieves_a_compact_candidate_set_with_event_tool_first():
    """Retrieval supplies compact recall; the constrained planner makes the final selection."""
    tools = capability_registry.select_planner_tools(
        "Machine failure hone se pehle vibration_x_g, vibration_y_g, aur "
        "vibration_z_g me variance spike ka pattern dikhta hai? Box plot draw karein.",
        has_dataset=True,
    )
    executable = [tool.name for tool in tools if not tool.planner_always_available]
    assert executable[0] == "event_window_analysis"
    assert len(executable) <= 4
    assert "clarification_needed" in {tool.name for tool in tools}


def test_irrelevant_query_does_not_receive_arbitrary_executable_tools():
    tools = capability_registry.select_planner_tools("hello there", has_dataset=False)
    assert tools
    assert all(tool.planner_always_available for tool in tools)


def test_supervised_classification_ranks_training_first_in_compact_candidates(monkeypatch):
    """Capability retrieval preserves recall; schema validation selects the final tool."""
    monkeypatch.setenv("OLLAMA_TOOL_RETRIEVER", "lexical")
    tools = capability_registry.select_planner_tools(
        "Train a Random Forest classifier for a multi-class target and plot a confusion matrix.",
        has_dataset=True,
    )
    names = {tool.name for tool in tools}

    executable = [tool.name for tool in tools if not tool.planner_always_available]
    assert executable[0] == "train_model"
    assert len(executable) <= 4
