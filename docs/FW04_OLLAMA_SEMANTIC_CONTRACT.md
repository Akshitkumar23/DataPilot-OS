# DataPilot OS — FW-04 Ollama Semantic Contract
## Governance, Boundaries, and Integration Protocol

**Capability**: FW-04 Relational Intent Parsing  
**Date**: October 2026  
**Status**: Governed, Production-Hardened, Zero-Fallback  

---

### 1. Mandatory Semantic Interface Declaration

> **Architectural Invariant**:
> **Ollama is the mandatory semantic interpretation layer for natural-language relational requests. If Ollama is unavailable or its structured output cannot be validated, execution stops with an explicit semantic-service error. No alternate natural-language parser is used.**

DataPilot OS enforces ONE authoritative semantic path:

```
User Natural Language Request
             ↓
           Ollama
             ↓
Structured RelationalIntent JSON
             ↓
Pydantic Schema Validation
             ↓
Deterministic DataPilot Kernel
             ↓
       RelationCatalog
             ↓
    RelationshipResolver
             ↓
         JoinPlanner
             ↓
           DuckDB
             ↓
         Provenance
```

There is **zero heuristic, token-matching, regex, or keyword fallback path**. If Ollama is unavailable, times out, returns invalid JSON, or violates the schema, the request fails fast with an explicit structured failure code.

---

### 2. Division of Responsibility

| Capability Domain | Ollama Semantic Layer | Deterministic DataPilot Kernel |
|---|:---:|:---:|
| Extract mentioned table candidates | **YES** | Validates against `RelationCatalog` |
| Extract explicit join keys | **YES** | Validates column existence & type compatibility |
| Extract requested join type | **YES** | Verifies ANSI SQL legality |
| Identify need for clarification | **YES** | Evaluates candidate ambiguity |
| Infer foreign keys from data | **NO** | **YES** (uniqueness $\ge 0.999$, overlap $\ge 0.70$) |
| Calculate cardinality & row expansion | **NO** | **YES** (1:1, 1:N, N:1, N:N; fan-out $>1.5\times$ guard) |
| Reject generic column names | **NO** | **YES** (`status`, `name`, `type`, `date`) |
| Generate executable SQL | **NO** | **YES** (`JoinPlanner.compile_sql()`) |
| Execute SQL | **NO** | **YES** (`DuckDBManager`) |
| Record lineage & provenance | **NO** | **YES** (`ProvenanceStore.record_multi_source_event()`) |

---

### 3. Strict Structured Schema Contract (`RelationalIntent`)

Ollama output is strictly bound to the `RELATIONAL_SCHEMA` and validated into `RelationalIntent`:

```json
{
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
```

---

### 4. Explicit Failure States

The system implements distinct structured error states:

1. `OLLAMA_UNAVAILABLE`: Ollama daemon is offline or unreachable.
   - Message: *"The Ollama semantic service is unavailable. Natural-language relational requests cannot be interpreted until Ollama is available."*
2. `OLLAMA_TIMEOUT`: Ollama did not respond within configured timeout.
3. `OLLAMA_INVALID_RESPONSE`: Ollama emitted invalid JSON or empty body.
4. `OLLAMA_SCHEMA_VALIDATION_ERROR`: Ollama emitted JSON violating the Pydantic contract (or emitted raw SQL text).
5. `RELATION_NOT_FOUND`: Candidate relation hallucinated by LLM or absent from `RelationCatalog`.
6. `COLUMN_NOT_FOUND` / `UNKNOWN_COLUMN`: Explicit key column absent from referenced table.
7. `INCOMPATIBLE_JOIN_TYPES`: Incompatible physical data types (e.g., numeric ID to string).
8. `RELATIONSHIP_NOT_VERIFIED`: Inferred key fails uniqueness or inclusion ratio thresholds.
9. `JOIN_REQUIRES_CLARIFICATION`: Ambiguous candidates or missing keys require analyst intervention.
