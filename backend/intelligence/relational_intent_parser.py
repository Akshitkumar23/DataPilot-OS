"""
DataPilot OS — Phase 7 / FW-04 Relational Intent Parser.
Extracts multi-table intent, candidate relations, join keys, dimensions, and measures.
Ollama is the MANDATORY semantic interface for natural-language relational requests.
No regex, token-matching, or keyword fallback path exists.
If Ollama is unavailable, fails, or violates schema, an explicit structured error is returned.
DataPilot Kernel strictly grounds and validates against RelationCatalog and actual data.
Zero domain or dataset hardcoding.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, ValidationError
import json
import logging

from backend.intelligence.ollama_service import ollama_service, OllamaRequest
from backend.sql_engine.relation_catalog import relation_catalog
from backend.core.error_states import ErrorState

logger = logging.getLogger(__name__)


class RelationalIntent(BaseModel):
    """Structured interpretation of a natural language multi-table request."""
    intent: Literal["RELATIONAL_ENRICHMENT", "CROSS_TABLE_AGGREGATION", "RELATIONAL_QUERY"] = "RELATIONAL_QUERY"
    candidate_relations: List[str]
    join_keys_explicit: List[Dict[str, str]] = Field(default_factory=list)
    join_type: Literal["INNER", "LEFT", "RIGHT", "FULL", "CROSS"]
    requested_dimensions: List[str] = Field(default_factory=list)
    requested_measures: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None



class RelationalIntentResult(BaseModel):
    """Result envelope for relational intent parsing."""
    success: bool
    intent: Optional[RelationalIntent] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    source: str = "ollama_semantic"
    details: Optional[Dict[str, Any]] = None


class RelationalIntentParser:
    """
    Translates natural-language multi-table requests into structured RelationalIntent.
    Strict Ollama boundary: Ollama is mandatory for semantic parsing; no heuristic/regex fallback.
    Kernel owns all schema grounding, candidate evaluation, safety, and execution.
    """

    RELATIONAL_SCHEMA = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["RELATIONAL_ENRICHMENT", "CROSS_TABLE_AGGREGATION", "RELATIONAL_QUERY"]
            },
            "candidate_relations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "join_keys_explicit": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "left_table": {"type": "string"},
                        "left_key": {"type": "string"},
                        "right_table": {"type": "string"},
                        "right_key": {"type": "string"}
                    },
                    "required": ["left_key", "right_key"]
                }
            },
            "join_type": {
                "type": "string",
                "enum": ["INNER", "LEFT", "RIGHT", "FULL", "CROSS"]
            },
            "requested_dimensions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "requested_measures": {
                "type": "array",
                "items": {"type": "string"}
            },
            "needs_clarification": {"type": "boolean"},
            "clarification_reason": {"type": "string"}
        },
        "required": ["intent", "candidate_relations", "join_type", "needs_clarification"]
    }

    SYSTEM_PROMPT = """You are the DataPilot OS Relational Semantic Intent Extractor.
Your job is ONLY to extract semantic relational intents from user requests into JSON.
DO NOT invent tables or columns.
Extract candidate table names and column concepts explicitly mentioned by the user.
If the user explicitly states join keys (e.g. 'orders.customer_id = customers.id'), include them.
If the relationship or keys are not explicitly specified, leave join_keys_explicit empty.
DO NOT generate SQL. DO NOT execute SQL. DO NOT invent relationships.
The deterministic kernel will independently ground and validate all relations and columns against the real catalog."""

    @classmethod
    def parse_intent_result(cls, user_query: str) -> RelationalIntentResult:
        """
        Parses a natural-language relational query using the mandatory Ollama semantic layer.
        Returns a structured RelationalIntentResult with explicit failure states if Ollama is unavailable,
        times out, returns invalid JSON, or violates schema.
        ZERO semantic fallback to regex or token matching.
        """
        if not user_query or not user_query.strip():
            return RelationalIntentResult(
                success=False,
                error_code="INVALID_QUERY",
                error_message="User query string cannot be empty."
            )

        # 1. Verify Ollama server availability
        try:
            status = ollama_service.detect_server()
            if not status.is_running:
                return RelationalIntentResult(
                    success=False,
                    error_code=ErrorState.OLLAMA_UNAVAILABLE.value,
                    error_message=(
                        "The Ollama semantic service is unavailable. "
                        "Natural-language relational requests cannot be interpreted until Ollama is available."
                    ),
                    details={"ollama_url": status.ollama_url, "error": status.error}
                )
        except Exception as e:
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_UNAVAILABLE.value,
                error_message=f"Failed to check Ollama service status: {str(e)}"
            )

        # 2. Call Ollama structured request
        available_tables = [r.table_name for r in relation_catalog.list_relations()]
        user_prompt = f"Available tables in catalog: {', '.join(available_tables)}\nUser request: \"{user_query.strip()}\""
        req = OllamaRequest(
            prompt=user_prompt,
            system_prompt=cls.SYSTEM_PROMPT,
            response_schema=cls.RELATIONAL_SCHEMA,
            temperature=0.0
        )

        try:
            res = ollama_service.structured_request(req)
        except Exception as e:
            err_str = str(e).lower()
            if "timeout" in err_str:
                return RelationalIntentResult(
                    success=False,
                    error_code=ErrorState.OLLAMA_TIMEOUT.value,
                    error_message="Ollama timed out while interpreting relational intent."
                )
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_UNAVAILABLE.value,
                error_message=f"Error communicating with Ollama: {str(e)}"
            )

        if not res or not res.success:
            err_code = res.error_code if res and res.error_code else ErrorState.OLLAMA_INVALID_RESPONSE.value
            err_msg = res.error_message if res and res.error_message else "Ollama failed to return a successful response."
            if "timeout" in err_msg.lower():
                err_code = ErrorState.OLLAMA_TIMEOUT.value
            return RelationalIntentResult(
                success=False,
                error_code=err_code,
                error_message=err_msg
            )

        # 3. Parse JSON response
        if not res.content:
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_INVALID_RESPONSE.value,
                error_message="Ollama returned an empty response body."
            )

        try:
            raw_data = json.loads(res.content)
            if not isinstance(raw_data, dict):
                return RelationalIntentResult(
                    success=False,
                    error_code=ErrorState.OLLAMA_INVALID_RESPONSE.value,
                    error_message=f"Ollama returned non-dictionary JSON: {type(raw_data).__name__}"
                )
        except Exception as e:
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_INVALID_RESPONSE.value,
                error_message=f"Ollama response could not be parsed as JSON: {str(e)}",
                details={"raw_content": res.content[:500]}
            )

        # 4. Check for prohibited executable SQL output in semantic response
        raw_content_upper = res.content.upper()
        if any(raw_content_upper.strip().startswith(prefix) for prefix in ["SELECT ", "WITH ", "INSERT ", "CREATE "]):
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_SCHEMA_VALIDATION_ERROR.value,
                error_message="Ollama emitted raw executable SQL instead of structured RelationalIntent."
            )

        # 5. Normalize join_keys_explicit if keys contain table qualifications
        if "join_keys_explicit" in raw_data and isinstance(raw_data["join_keys_explicit"], list):
            norm_keys = []
            for k in raw_data["join_keys_explicit"]:
                if not isinstance(k, dict):
                    continue
                lk = str(k.get("left_key", ""))
                rk = str(k.get("right_key", ""))
                lt = k.get("left_table")
                rt = k.get("right_table")
                if "." in lk and not lt:
                    lt, lk = lk.split(".", 1)
                if "." in rk and not rt:
                    rt, rk = rk.split(".", 1)
                norm_keys.append({
                    "left_table": lt or "",
                    "left_key": lk,
                    "right_table": rt or "",
                    "right_key": rk
                })
            raw_data["join_keys_explicit"] = norm_keys

        # 6. Validate against Pydantic schema
        try:
            intent = RelationalIntent(**raw_data)
        except ValidationError as e:
            return RelationalIntentResult(
                success=False,
                error_code=ErrorState.OLLAMA_SCHEMA_VALIDATION_ERROR.value,
                error_message=f"Ollama response failed schema validation: {str(e)}",
                details={"errors": e.errors()}
            )

        return RelationalIntentResult(
            success=True,
            intent=intent,
            source="ollama_semantic"
        )

    @classmethod
    def parse_intent(cls, user_query: str) -> RelationalIntent:
        """
        Parses a user query using Ollama as the mandatory semantic layer.
        Raises RuntimeError or returns RelationalIntent.
        If Ollama fails or is unavailable, raises RuntimeError with the explicit error state.
        ZERO heuristic fallback.
        """
        res = cls.parse_intent_result(user_query)
        if not res.success:
            raise RuntimeError(f"{res.error_code}: {res.error_message}")
        return res.intent
