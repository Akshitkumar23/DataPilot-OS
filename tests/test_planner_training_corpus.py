import json
from pathlib import Path

from scripts.validate_planner_corpus import load_jsonl, validate_records


ROOT = Path(__file__).resolve().parents[1]


def test_seed_corpus_is_grounded_in_live_registry_and_real_benchmark_schemas():
    records = load_jsonl(ROOT / "training" / "planner_sft_seed.jsonl")
    provenance = json.loads((ROOT / "data" / "real_benchmarks" / "provenance.json").read_text(encoding="utf-8"))

    validate_records(records, provenance)

    assert {record["dataset_id"] for record in records} == {
        "uci_bank_marketing",
        "uci_bike_sharing_hourly",
        "uci_ai4i_predictive_maintenance",
    }
    assert any(record["expected_decision"] == "clarification" for record in records)
    assert {record["split"] for record in records} == {"train", "validation", "test"}
