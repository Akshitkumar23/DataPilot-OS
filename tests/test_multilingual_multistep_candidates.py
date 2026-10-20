import json

from training.build_multilingual_multistep_candidates import build


def _record(case_id, dataset_id):
    commands = [
        {"tool": "filter_data", "arguments": {"column": "age", "operator": ">", "value": 30}},
        {"tool": "aggregate_data", "arguments": {"column": "balance", "aggregation": "mean"}},
    ]
    return {
        "schema_version": "planner-sft-v2", "case_id": case_id, "split": "train",
        "dataset_id": dataset_id, "query": "source", "required_columns": ["age", "balance"],
        "expected_decision": "tool", "expected_commands": commands,
        "task_family_id": case_id, "messages": [], "label_status": "contract-verified",
        "verification": {"kind": "independent-numeric-plan", "query_semantics_verified": False,
                         "source_sha256": "source", "result_sha256": case_id},
    }


def test_multilingual_candidates_preserve_plan_and_do_not_claim_human_review(tmp_path):
    source = tmp_path / "pilot.jsonl"
    records = [_record("one", "uci_bank_marketing"), _record("two", "uci_adult_income")]
    source.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    output = tmp_path / "out"

    report = build(source, output, count_per_language=2)
    candidates = [json.loads(line) for line in (output / "multilingual_multistep_candidates.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["candidate_count"] == 4
    assert report["language_counts"] == {"hi": 2, "hinglish": 2}
    assert all(len(item["expected_commands"]) == 2 for item in candidates)
    assert all(item["human_reviewed"] is False for item in candidates)
    assert all(item["verification"]["language_reviewed"] is False for item in candidates)
    assert all("`age`" in item["query"] and "`balance`" in item["query"] for item in candidates)
