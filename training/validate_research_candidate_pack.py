"""Validate an externally generated DataPilot research candidate pack.

This is an intake gate, not a human-review or training importer.  It verifies
the immutable structural claims made by the research report and writes an
audit receipt.  Candidate records remain pending after a successful run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


LANGUAGE_FILES = {
    "hindi_candidates.jsonl": ("hi", 150),
    "hinglish_candidates.jsonl": ("hinglish", 150),
}
COMMON_FIELDS = {
    "record_id", "provenance_type", "language", "source_ids",
    "query_template", "task_family", "expected_behavior",
    "required_arguments", "forbidden_assumptions", "expected_artifacts",
    "review_status", "reviewer_notes",
}
WORKFLOW_FIELDS = {
    "workflow_id", "user_request", "dataset_schema_placeholders",
    "ordered_steps", "dependency_edges", "required_tool_families",
    "expected_artifacts", "clarification_conditions", "unsupported_reason",
    "safety_leakage_notes",
}
EXPECTED_PROVENANCE = "AI-assisted-research-candidate"
ALLOWED_BEHAVIOURS = {"executable", "clarification_required", "unsupported"}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: every JSONL row must be an object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_common(record: dict, identity: str, errors: list[str]) -> None:
    missing = sorted(COMMON_FIELDS - record.keys())
    if missing:
        errors.append(f"{identity}: missing fields {missing}")
    if record.get("provenance_type") != EXPECTED_PROVENANCE:
        errors.append(f"{identity}: provenance_type must be {EXPECTED_PROVENANCE!r}")
    if record.get("review_status") != "pending":
        errors.append(f"{identity}: review_status must remain 'pending'")
    if record.get("reviewer_notes") not in ("", None):
        errors.append(f"{identity}: reviewer_notes must be empty before review")
    if record.get("expected_behavior") not in ALLOWED_BEHAVIOURS:
        errors.append(f"{identity}: unsupported expected_behavior")
    if not isinstance(record.get("source_ids"), list) or not record.get("source_ids"):
        errors.append(f"{identity}: source_ids must be a non-empty list")
    if not isinstance(record.get("query_template"), str) or not record.get("query_template", "").strip():
        errors.append(f"{identity}: query_template must be non-empty")


def _step_id(step: dict) -> str | None:
    for key in ("step_id", "id", "name"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _edge_endpoints(edge: dict) -> tuple[str | None, str | None]:
    source = edge.get("from") or edge.get("source") or edge.get("upstream_step_id")
    target = edge.get("to") or edge.get("target") or edge.get("downstream_step_id")
    return source, target


def validate_pack(pack_dir: Path) -> dict:
    required = set(LANGUAGE_FILES) | {"multistep_candidates.jsonl", "source_manifest.json"}
    missing_files = sorted(name for name in required if not (pack_dir / name).is_file())
    if missing_files:
        raise ValueError(f"RESEARCH_PACK_INCOMPLETE: missing {missing_files}")

    errors: list[str] = []
    all_ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    behaviour_counts: dict[str, Counter[str]] = {}
    referenced_source_ids: set[str] = set()

    for filename, (language, expected_count) in LANGUAGE_FILES.items():
        rows = _read_jsonl(pack_dir / filename)
        if len(rows) < expected_count:
            errors.append(f"{filename}: expected at least {expected_count} rows, found {len(rows)}")
        counts: Counter[str] = Counter()
        seen_queries: set[str] = set()
        for index, row in enumerate(rows, 1):
            identity = str(row.get("record_id") or f"{filename}:{index}")
            _validate_common(row, identity, errors)
            referenced_source_ids.update(value for value in row.get("source_ids", []) if isinstance(value, str))
            if row.get("language") != language:
                errors.append(f"{identity}: language must be {language!r}")
            if identity in all_ids:
                errors.append(f"{identity}: duplicate record/workflow identifier")
            all_ids.add(identity)
            normalized_query = " ".join(str(row.get("query_template", "")).split()).casefold()
            if normalized_query in seen_queries:
                errors.append(f"{identity}: duplicate normalized query_template in {filename}")
            seen_queries.add(normalized_query)
            counts[str(row.get("expected_behavior"))] += 1
            language_counts[language] += 1
        behaviour_counts[language] = counts
        if counts["clarification_required"] != 30 or counts["unsupported"] != 30:
            errors.append(
                f"{filename}: expected 30 clarification_required and 30 unsupported; found {dict(counts)}"
            )

    workflows = _read_jsonl(pack_dir / "multistep_candidates.jsonl")
    if len(workflows) < 210:
        errors.append(f"multistep_candidates.jsonl: expected at least 210 rows, found {len(workflows)}")
    workflow_languages: Counter[str] = Counter()
    seen_workflow_queries: set[tuple[str, str]] = set()
    for index, row in enumerate(workflows, 1):
        identity = str(row.get("workflow_id") or row.get("record_id") or f"workflow:{index}")
        _validate_common(row, identity, errors)
        referenced_source_ids.update(value for value in row.get("source_ids", []) if isinstance(value, str))
        missing = sorted(WORKFLOW_FIELDS - row.keys())
        if missing:
            errors.append(f"{identity}: missing workflow fields {missing}")
        if identity in all_ids:
            errors.append(f"{identity}: duplicate record/workflow identifier")
        all_ids.add(identity)
        language = str(row.get("language"))
        workflow_languages[language] += 1
        workflow_query = " ".join(str(row.get("user_request", "")).split()).casefold()
        workflow_query_key = (language, workflow_query)
        if workflow_query_key in seen_workflow_queries:
            errors.append(f"{identity}: duplicate normalized user_request within language")
        seen_workflow_queries.add(workflow_query_key)
        steps = row.get("ordered_steps")
        edges = row.get("dependency_edges")
        if not isinstance(steps, list) or len(steps) < 2 or not all(isinstance(s, dict) for s in steps):
            errors.append(f"{identity}: ordered_steps must contain at least two objects")
            continue
        step_ids = [_step_id(step) for step in steps]
        if any(step_id is None for step_id in step_ids) or len(set(step_ids)) != len(step_ids):
            errors.append(f"{identity}: steps require unique step_id/id/name values")
        if not isinstance(edges, list) or not edges:
            errors.append(f"{identity}: dependency_edges must be non-empty")
            continue
        for edge_index, edge in enumerate(edges, 1):
            if not isinstance(edge, dict):
                errors.append(f"{identity}: dependency edge {edge_index} must be an object")
                continue
            source, target = _edge_endpoints(edge)
            if source not in step_ids or target not in step_ids or source == target:
                errors.append(f"{identity}: dependency edge {edge_index} has invalid endpoints")
            artefact = edge.get("artifact") or edge.get("artifact_id") or edge.get("output")
            if not isinstance(artefact, str) or not artefact.strip():
                errors.append(f"{identity}: dependency edge {edge_index} needs an artifact identifier")
            elif source in step_ids and target in step_ids:
                producer = steps[step_ids.index(source)]
                consumer = steps[step_ids.index(target)]
                if artefact not in producer.get("produces", []):
                    errors.append(f"{identity}: edge artifact {artefact!r} is not produced by {source}")
                if artefact not in consumer.get("consumes", []):
                    errors.append(f"{identity}: edge artifact {artefact!r} is not consumed by {target}")
    for language in ("en", "hi", "hinglish"):
        if workflow_languages[language] < 70:
            errors.append(f"multistep_candidates.jsonl: expected at least 70 {language} workflows, found {workflow_languages[language]}")

    manifest_raw = json.loads((pack_dir / "source_manifest.json").read_text(encoding="utf-8"))
    sources = manifest_raw.get("sources") if isinstance(manifest_raw, dict) else manifest_raw
    if not isinstance(sources, list) or len(sources) < 33:
        errors.append("source_manifest.json: expected at least 33 source records")
        sources = sources if isinstance(sources, list) else []
    source_ids: set[str] = set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            errors.append(f"source_manifest.json:{index}: source must be an object")
            continue
        source_id = source.get("source_id") or source.get("id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in source_ids:
            errors.append(f"source_manifest.json:{index}: source id is missing or duplicated")
        else:
            source_ids.add(source_id)
        url = source.get("url") or source.get("location")
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append(f"source_manifest.json:{index}: public HTTP(S) URL is required")
        if not (source.get("licence_note") or source.get("license_note") or source.get("terms_note")):
            errors.append(f"source_manifest.json:{index}: licence/terms note is required")
        if not (source.get("exactly_what_was_used") or source.get("exact_usage")):
            errors.append(f"source_manifest.json:{index}: exactly_what_was_used is required")

    unknown_sources = sorted(referenced_source_ids - source_ids)
    if unknown_sources:
        errors.append(f"candidate source_ids missing from manifest: {unknown_sources[:20]}")

    if errors:
        raise ValueError("RESEARCH_PACK_INVALID:\n- " + "\n- ".join(errors[:100]))

    files = sorted(required)
    return {
        "schema_version": "datapilot-research-pack-intake-v1",
        "status": "structurally_valid_pending_human_review",
        "training_authorized": False,
        "human_reviewed_records": 0,
        "record_counts": {
            "hindi": language_counts["hi"],
            "hinglish": language_counts["hinglish"],
            "multistep": len(workflows),
            "total_candidates": language_counts.total() + len(workflows),
        },
        "behaviour_counts": {key: dict(value) for key, value in behaviour_counts.items()},
        "workflow_language_counts": dict(workflow_languages),
        "source_count": len(sources),
        "file_sha256": {name: _sha256(pack_dir / name) for name in files},
        "next_gate": "cryptographically bound record-level human semantic/language review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = validate_pack(args.pack_dir)
    if args.receipt:
        if args.receipt.exists():
            raise FileExistsError(f"Refusing to overwrite intake receipt: {args.receipt}")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
