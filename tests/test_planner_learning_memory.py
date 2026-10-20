from __future__ import annotations

import json

from modules.planner_learning_memory import PlannerLearningMemory
from scripts.build_planner_pattern_memory import compile_records
from training.generate_planner_review_queue import normalise_candidate, valid_candidate


class _EmbeddingService:
    def embed_texts(self, texts, model):
        # Deterministic test embeddings: query/pattern matching is intentionally
        # semantic-only in production; this fake avoids a live Ollama dependency.
        return [[float(len(text)), float(text.count("average") + text.count("mean")), 1.0] for text in texts]


def test_compiled_memory_redacts_columns_and_excludes_executable_arguments(monkeypatch):
    records = [{
        "case_id": "case-1",
        "label_status": "human-reviewed",
        "expected_decision": "tool",
        "expected_command": {"tool": "group_data", "arguments": {"group_by": ["region"], "metric": "sales"}},
        "query": "Show average sales by region.",
        "required_columns": ["sales", "region"],
    }]
    compiled = compile_records(records)
    assert compiled[0]["tool"] == "group_data"
    assert "sales" not in compiled[0]["query_pattern"].lower()
    assert "region" not in compiled[0]["query_pattern"].lower()
    assert "arguments" not in compiled[0]


def test_memory_only_returns_allowed_reviewed_tools(tmp_path, monkeypatch):
    memory_path = tmp_path / "memory.jsonl"
    memory_path.write_text("\n".join([
        json.dumps({"review_status": "human-reviewed", "case_id": "a", "tool": "group_data", "query_pattern": "Average a metric by a dimension", "argument_fields": ["metric"]}),
        json.dumps({"review_status": "unreviewed", "case_id": "b", "tool": "train_model", "query_pattern": "Train a model", "argument_fields": ["target"]}),
    ]) + "\n", encoding="utf-8")
    import modules.planner_learning_memory as memory_module
    monkeypatch.setattr(memory_module, "ollama_service", _EmbeddingService())
    memory = PlannerLearningMemory(memory_path)
    guidance = memory.render_guidance("Calculate the average metric by category", {"group_data"})
    assert "group_data" in guidance
    assert "train_model" not in guidance


def test_review_queue_normalises_json_envelopes_before_validation():
    query = normalise_candidate('{"request":"Show average sales by region.","column_names":["sales","region"]}')
    assert query == "Show average sales by region."
    assert valid_candidate(query, ["sales", "region"], "Original different query")
