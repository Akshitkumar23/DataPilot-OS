"""
DataPilot OS — Capability Retriever.
Performs deterministic, schema-aware capability retrieval over live ToolSpec definitions.
Zero hardcoded query-to-tool dictionaries.
Default retrieval: Lexical TF-IDF / BM25 with zero external startup dependencies.
Optional: Embedding hybrid reranker when local embedding model is configured.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class CapabilityRetriever:
    """
    Retrieves a compact, schema-aware subset of live ToolSpecs for the semantic planner.
    Ensures small local LLMs only see relevant tools within strict context constraints.
    """

    def __init__(self, registry: Any = None):
        self._registry = registry
        self._embedding_cache: Dict[Tuple[str, Tuple[str, ...]], List[List[float]]] = {}

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from modules.capability_registry import capability_registry
            return capability_registry
        return self._registry

    @classmethod
    def build_spec_document(cls, spec: Any) -> str:
        """
        Build searchable capability evidence directly from the live ToolSpec.
        Retrieval inspects typed arguments, task families, descriptions,
        synonyms, and supported physical chart outputs.
        """
        schema = spec.input_model.model_json_schema()
        field_evidence: List[str] = []
        for field_name, definition in schema.get("properties", {}).items():
            parts = [field_name]
            if isinstance(definition, dict):
                for key in ("title", "description"):
                    if definition.get(key):
                        parts.append(str(definition[key]))
                enum_values = definition.get("enum")
                if isinstance(enum_values, list):
                    parts.extend(str(value) for value in enum_values)
            field_evidence.append(" ".join(parts))

        family = getattr(spec.task_family, "value", spec.task_family) or ""
        return "\n".join((
            f"tool: {spec.name}",
            f"category: {spec.category}",
            f"task family: {family}",
            f"description: {spec.description}",
            f"registered synonyms: {' '.join(spec.synonyms)}",
            f"typed inputs: {' | '.join(field_evidence)}",
        ))

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        has_dataset: Optional[bool] = None,
        embedding_cache: Optional[Dict[Any, Any]] = None,
    ) -> List[Any]:
        """
        Retrieve semantically relevant ToolSpecs for query.
        Guarantees always-available tools (clarification_needed, etc.) are included.
        """
        specs = [
            spec for spec in self.registry.list_tool_specs()
            if has_dataset is not False or not spec.requires_dataset
        ]
        if not specs:
            raise RuntimeError("CAPABILITY_REGISTRY_UNAVAILABLE: No ToolSpecs are registered in CapabilityRegistry.")

        max_tools = limit if limit is not None else int(os.getenv("OLLAMA_PLANNER_TOOL_LIMIT", "4"))
        if max_tools < 1:
            raise ValueError("OLLAMA_PLANNER_TOOL_LIMIT must be at least 1.")

        documents = [self.build_spec_document(spec) for spec in specs]
        query_text = str(query or "").strip()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        lexical_vectors = vectorizer.fit_transform(documents + [query_text])
        lexical_query_vector = lexical_vectors[-1]
        lexical_scores = (lexical_vectors[:-1] @ lexical_query_vector.T).toarray().ravel().tolist()

        retriever_type = os.getenv("OLLAMA_TOOL_RETRIEVER", "lexical").strip().lower()
        scores: List[float] = []

        cache = embedding_cache if embedding_cache is not None else self._embedding_cache

        if retriever_type == "semantic":
            try:
                from backend.intelligence.ollama_service import ollama_service

                embedding_model = os.getenv("OLLAMA_TOOL_EMBEDDING_MODEL", "nomic-embed-text")
                cache_key = (embedding_model, tuple(documents))
                tool_vectors = cache.get(cache_key)
                query_vectors = ollama_service.embed_texts([query_text], embedding_model)
                if tool_vectors is None and query_vectors is not None:
                    vectors = ollama_service.embed_texts(documents, embedding_model)
                    if vectors is not None:
                        cache[cache_key] = vectors
                        tool_vectors = vectors
                if tool_vectors is not None and query_vectors is not None:
                    query_vector = query_vectors[0]

                    def cosine_similarity(left: List[float], right: List[float]) -> float:
                        num = sum(a * b for a, b in zip(left, right))
                        l_norm = sum(a * a for a in left) ** 0.5
                        r_norm = sum(b * b for b in right) ** 0.5
                        return num / (l_norm * r_norm) if l_norm and r_norm else 0.0

                    semantic_scores = [cosine_similarity(vec, query_vector) for vec in tool_vectors]
                    lexical_peak = max(lexical_scores, default=0.0)
                    semantic_floor = min(semantic_scores, default=0.0)
                    semantic_range = max(semantic_scores, default=0.0) - semantic_floor
                    scores = [
                        0.65 * (lexical / lexical_peak if lexical_peak else 0.0)
                        + 0.35 * ((semantic - semantic_floor) / semantic_range if semantic_range else 0.0)
                        for lexical, semantic in zip(lexical_scores, semantic_scores)
                    ]
            except Exception as e:
                logger.warning("SEMANTIC_RETRIEVAL_FAILED | Falling back to lexical: %s", e)
                scores = []

        if not scores:
            scores = lexical_scores

        ranked_pairs = [
            (score, spec) for score, spec in sorted(
                zip(scores, specs),
                key=lambda item: (-item[0], item[1].name),
            )
            if score > 0
        ]
        ranked = [spec for _, spec in ranked_pairs[:max_tools]]
        always_available = [
            spec for spec in specs
            if spec.planner_always_available
            and not (has_dataset is True and spec.planner_without_dataset_only)
        ]
        if has_dataset is False:
            always_available.sort(key=lambda s: 0 if s.name == "general_chat" else 1)

        selected: List[Any] = []
        for spec in ranked + always_available:
            if spec.name not in {existing.name for existing in selected}:
                selected.append(spec)
        if has_dataset is False:
            selected.sort(key=lambda s: 0 if s.name == "general_chat" else 1)
        return selected


capability_retriever = CapabilityRetriever()
