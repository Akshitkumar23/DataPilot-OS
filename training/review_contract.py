"""Cryptographic content binding for DataPilot's offline human-review workflow.

The hashes in this module prove that a review row still refers to the exact
query and reference plan that was presented to the reviewer.  They do not prove
the reviewer's identity; organizations that need identity assurance should
store the generated workspace in an authenticated review system or sign the
manifest externally.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REVIEW_SCHEMA_VERSION = "datapilot-human-review-v2"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def reviewable_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return only immutable, reviewer-visible label content."""
    return {
        "schema_version": record.get("schema_version"),
        "case_id": record.get("case_id"),
        "dataset_id": record.get("dataset_id"),
        "query": record.get("query"),
        "required_columns": record.get("required_columns", []),
        "expected_decision": record.get("expected_decision"),
        "expected_command": record.get("expected_command"),
        "expected_commands": record.get("expected_commands"),
    }


def record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(reviewable_record(record)).encode("utf-8"))


def batch_content_sha256(rows: Iterable[dict[str, Any]]) -> str:
    immutable = [
        {
            "case_id": row["case_id"],
            "record_sha256": row["record_sha256"],
            "corpus_sha256": row["corpus_sha256"],
            "review_schema_version": row["review_schema_version"],
            "review_batch_id": row["review_batch_id"],
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json(immutable).encode("utf-8"))
