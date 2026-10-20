"""Reviewed, local planner-pattern memory.

This is deliberately *not* a query-to-tool fallback or a source of executable
arguments.  It retrieves only human-reviewed, redacted planning patterns with
the local embedding model, then gives the semantic planner compact guidance.
The live ToolSpec schema and the active dataset remain the sole authorities for
tool selection, exact columns, values, and execution.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable

from backend.intelligence.ollama_service import ollama_service

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = ROOT / "training" / "planner_pattern_memory.jsonl"
APPROVED_LABELS = frozenset({"seed-reviewed", "human-reviewed"})


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(a * a for a in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


class PlannerLearningMemory:
    """Local semantic retrieval over reviewed, non-executable plan patterns."""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH, embedding_model: str | None = None) -> None:
        self.path = path
        self.embedding_model = embedding_model or os.getenv("OLLAMA_PLANNER_MEMORY_EMBEDDING_MODEL", "nomic-embed-text")
        self._records: list[dict[str, Any]] | None = None
        self._vectors: list[list[float]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            self._records = records
            return records
        try:
            for line_number, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
                if record.get("review_status") not in APPROVED_LABELS:
                    continue
                if not isinstance(record.get("tool"), str) or not isinstance(record.get("query_pattern"), str):
                    raise ValueError("tool and query_pattern are required")
                records.append(record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # A malformed optional memory must not weaken the canonical planner.
            logger.warning("PLANNER_MEMORY_DISABLED | path=%s | error=%s", self.path, exc)
            records = []
        self._records = records
        return records

    @staticmethod
    def _document(record: dict[str, Any]) -> str:
        fields = record.get("argument_fields") or []
        return "\n".join((
            "Reviewed planning pattern.",
            f"Natural-language pattern: {record['query_pattern']}",
            f"Registered tool: {record['tool']}",
            "Argument roles: " + ", ".join(str(field) for field in fields),
        ))

    def retrieve(self, query: str, allowed_tools: Iterable[str], top_k: int = 2) -> list[dict[str, Any]]:
        """Return semantic matches only; embedding failure returns no guidance.

        There is intentionally no lexical or tool-name fallback here.  Memory is
        optional guidance, never a routing mechanism, so an unavailable local
        embedding service must leave the strict planner unchanged.
        """
        if os.getenv('OLLAMA_PLANNER_MEMORY_ENABLED', '1').lower() in {'0', 'false', 'no'}:
            return []
        if not query.strip() or top_k < 1:
            return []
        allowed = {str(tool) for tool in allowed_tools}
        records = [record for record in self._load() if record["tool"] in allowed]
        if not records:
            return []
        documents = [self._document(record) for record in records]
        query_vectors = ollama_service.embed_texts([query], self.embedding_model)
        if query_vectors is None:
            return []
        if self._vectors is None or getattr(self, '_vector_documents', None) != documents:
            vectors = ollama_service.embed_texts(documents, self.embedding_model)
            if vectors is None:
                return []
            self._vectors = vectors
            self._vector_documents = list(documents)
        ranked = sorted(
            (
                (_cosine_similarity(query_vectors[0], vector), record)
                for record, vector in zip(records, self._vectors)
            ),
            key=lambda item: (-item[0], item[1].get("case_id", "")),
        )
        return [record for score, record in ranked[:top_k] if score > 0]

    def render_guidance(self, query: str, allowed_tools: Iterable[str]) -> str:
        matches = self.retrieve(query, allowed_tools)
        if not matches:
            return ""
        patterns: list[str] = []
        for record in matches:
            fields = ", ".join(record.get("argument_fields") or []) or "(no arguments)"
            patterns.append(
                f"- Pattern: {record['query_pattern']}\n"
                f"  Tool shape: {record['tool']} with argument roles: {fields}"
            )
        return "\n".join(patterns)


planner_learning_memory = PlannerLearningMemory()
