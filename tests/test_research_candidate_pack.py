import json

import pytest

from training.validate_research_candidate_pack import validate_pack


def _candidate(record_id, language, behaviour="executable"):
    return {
        "record_id": record_id,
        "provenance_type": "AI-assisted-research-candidate",
        "language": language,
        "source_ids": ["s1"],
        "query_template": f"<metric> ka analysis karo; candidate {record_id}",
        "task_family": "grouped_metrics",
        "expected_behavior": behaviour,
        "required_arguments": ["metric"],
        "forbidden_assumptions": [],
        "expected_artifacts": ["table"],
        "review_status": "pending",
        "reviewer_notes": "",
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _pack(tmp_path):
    for filename, language in (("hindi_candidates.jsonl", "hi"), ("hinglish_candidates.jsonl", "hinglish")):
        rows = []
        for index in range(150):
            behaviour = "clarification_required" if index < 30 else "unsupported" if index < 60 else "executable"
            rows.append(_candidate(f"{language}-{index}", language, behaviour))
        _write_jsonl(tmp_path / filename, rows)
    workflows = []
    languages = ["en"] * 70 + ["hi"] * 70 + ["hinglish"] * 70
    for index, language in enumerate(languages):
        row = _candidate(f"wf-record-{index}", language)
        row.update({
            "workflow_id": f"wf-{index}",
            "user_request": f"filter then aggregate workflow {index}",
            "dataset_schema_placeholders": ["metric"],
            "ordered_steps": [
                {"step_id": "s1", "produces": ["filtered_rows"], "consumes": ["source_table"]},
                {"step_id": "s2", "produces": ["result"], "consumes": ["filtered_rows"]},
            ],
            "dependency_edges": [{"from": "s1", "to": "s2", "artifact_id": "filtered_rows"}],
            "required_tool_families": ["filter", "aggregate"],
            "clarification_conditions": [],
            "unsupported_reason": "",
            "safety_leakage_notes": [],
        })
        workflows.append(row)
    _write_jsonl(tmp_path / "multistep_candidates.jsonl", workflows)
    sources = [{
        "source_id": f"s{index}", "url": f"https://example.org/{index}",
        "licence_note": "public documentation; verify redistribution separately",
        "exactly_what_was_used": "terminology and method semantics",
    } for index in range(1, 34)]
    (tmp_path / "source_manifest.json").write_text(json.dumps({"sources": sources}), encoding="utf-8")


def test_valid_pack_is_still_not_training_authorized(tmp_path):
    _pack(tmp_path)
    receipt = validate_pack(tmp_path)
    assert receipt["record_counts"]["total_candidates"] == 510
    assert receipt["training_authorized"] is False
    assert receipt["human_reviewed_records"] == 0


def test_valid_superset_is_accepted(tmp_path):
    _pack(tmp_path)
    for filename, language in (("hindi_candidates.jsonl", "hi"), ("hinglish_candidates.jsonl", "hinglish")):
        path = tmp_path / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(_candidate(f"{language}-extra", language))
        _write_jsonl(path, rows)
    receipt = validate_pack(tmp_path)
    assert receipt["record_counts"]["total_candidates"] == 512


def test_missing_appendix_fails_closed(tmp_path):
    _pack(tmp_path)
    (tmp_path / "hindi_candidates.jsonl").unlink()
    with pytest.raises(ValueError, match="RESEARCH_PACK_INCOMPLETE"):
        validate_pack(tmp_path)


def test_workflow_without_dependency_artifact_is_rejected(tmp_path):
    _pack(tmp_path)
    path = tmp_path / "multistep_candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["dependency_edges"][0].pop("artifact_id")
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="artifact identifier"):
        validate_pack(tmp_path)
