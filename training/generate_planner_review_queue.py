"""Create an offline, review-required paraphrase curriculum with local Ollama.

Each candidate inherits a seed's already validated ToolSpec label, but is marked
``synthetic-unreviewed``.  It is never consumed by runtime retrieval or adapter
training until a human reviewer promotes it.  No customer CSV values are sent
outside the local Ollama service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.intelligence.ollama_service import OllamaRequest, ollama_service
from scripts.validate_planner_corpus import DEFAULT_CORPUS, load_jsonl

DEFAULT_OUTPUT = ROOT / "training" / "review_queue" / "planner_paraphrase_candidates.jsonl"

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paraphrases": {"type": "array", "items": {"type": "string"}, "minItems": 1}
    },
    "required": ["paraphrases"],
    "additionalProperties": False,
}


def normalise_candidate(text: Any) -> str:
    """Accept a model's accidental JSON envelope only by extracting ``request``.

    The review queue must contain a human-readable user query, never model
    protocol JSON serialized as a string.
    """
    if not isinstance(text, str):
        return ""
    candidate = text.strip()
    try:
        envelope = json.loads(candidate)
    except json.JSONDecodeError:
        return candidate
    if isinstance(envelope, dict) and isinstance(envelope.get("request"), str):
        return envelope["request"].strip()
    return ""


def valid_candidate(text: Any, required_columns: list[str], original: str) -> bool:
    if not isinstance(text, str) or not text.strip() or text.strip() == original.strip():
        return False
    lowered = text.lower()
    # Exact field names are essential evidence for this application's strict
    # runtime column grounding.  This validation is corpus hygiene, not routing.
    return all(column.lower() in lowered for column in required_columns)


def generate_for_record(record: dict[str, Any], variants: int, model: str | None) -> list[str]:
    query = record["query"]
    columns = record["required_columns"]
    prompt = (
        "Create distinct natural-language paraphrases for the following data-science request. "
        "Preserve exactly the same analytical intent, all explicit column names, filters, values, "
        "calculation, grouping, time conditions, and requested chart. Do not add requirements or change scope.\n\n"
        f"SOURCE REQUEST: {query}\n"
        f"COLUMN NAMES THAT MUST APPEAR EXACTLY: {json.dumps(columns)}\n"
        f"Return exactly {variants} paraphrases as JSON."
    )
    response = ollama_service.structured_request(OllamaRequest(
        prompt=prompt,
        system_prompt=(
            "You create training candidates for a strict data-science planner. "
            "Do not solve the request and do not output tool calls. Preserve semantics exactly."
        ),
        response_schema=RESPONSE_SCHEMA,
        temperature=0.7,
        top_p=0.9,
        max_tokens=max(256, variants * 96),
        model=model,
        think=False,
    ))
    if not response.success or not isinstance(response.parsed_json, dict):
        return []
    raw = response.parsed_json.get("paraphrases", [])
    accepted: list[str] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        canonical = normalise_candidate(item)
        key = canonical.casefold()
        if key and key not in seen and valid_candidate(canonical, columns, query):
            accepted.append(canonical)
            seen.add(key)
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants-per-seed", type=int, default=8)
    parser.add_argument("--model", default=None, help="Optional explicit local Ollama model; defaults to DataPilot's active planner.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of reviewed train seeds to process.")
    parser.add_argument("--confirm-local-generation", action="store_true", help="Required because this invokes local Ollama and writes a review queue.")
    args = parser.parse_args()
    if not args.confirm_local_generation:
        raise RuntimeError("Pass --confirm-local-generation after confirming the input corpus contains no customer data.")
    if args.variants_per_seed < 1 or args.variants_per_seed > 20:
        raise ValueError("--variants-per-seed must be between 1 and 20.")
    records = [record for record in load_jsonl(args.corpus) if record.get("split") == "train" and record.get("label_status") in {"seed-reviewed", "human-reviewed"}]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("No reviewed train seeds were available.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {line.get("candidate_id") for line in load_jsonl(args.output)} if args.output.exists() else set()
    created: list[dict[str, Any]] = []
    for record in records:
        for index, query in enumerate(generate_for_record(record, args.variants_per_seed, args.model), start=1):
            candidate_id = f"{record['case_id']}-paraphrase-{index:02d}"
            if candidate_id in existing:
                continue
            created.append({
                "schema_version": "planner-review-queue-v1",
                "candidate_id": candidate_id,
                "source_case_id": record["case_id"],
                "query": query,
                "required_columns": record["required_columns"],
                "expected_decision": record["expected_decision"],
                "expected_command": record.get("expected_command"),
                "review_status": "synthetic-unreviewed",
                "reviewer": None,
                "review_note": None,
            })
    if created:
        with args.output.open("a", encoding="utf-8") as handle:
            for candidate in created:
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(json.dumps({"source_seeds": len(records), "new_candidates": len(created), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
