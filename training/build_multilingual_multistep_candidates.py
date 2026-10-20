"""Build review-only Hindi/Hinglish variants of numerically verified plans.

This script does not claim human authorship or semantic approval. It converts a
balanced sample of already SQL-oracle-verified dependent workflows into a
language-review queue while preserving the exact command plan, dataset source
hash and numeric result hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry


AGG_HI = {
    "sum": "योग", "mean": "औसत", "min": "न्यूनतम", "max": "अधिकतम",
    "median": "माध्यिका", "count": "गैर-रिक्त मानों की संख्या",
}
AGG_HINGLISH = {
    "sum": "sum", "mean": "average", "min": "minimum", "max": "maximum",
    "median": "median", "count": "non-null count",
}
OP_HI = {">": "से अधिक", "<": "से कम", ">=": "से कम नहीं", "<=": "से अधिक नहीं"}
OP_HINGLISH = {">": "se zyada", "<": "se kam", ">=": "ya usse zyada", "<=": "ya usse kam"}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _balanced(records: list[dict], count: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if len(record.get("expected_commands", [])) > 1:
            buckets[record["dataset_id"]].append(record)
    for values in buckets.values():
        values.sort(key=lambda item: item["case_id"])
    selected: list[dict] = []
    while len(selected) < count and any(buckets.values()):
        for dataset_id in sorted(buckets):
            if buckets[dataset_id] and len(selected) < count:
                selected.append(buckets[dataset_id].pop(0))
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} dependent workflows available; requested {count}")
    return selected


def _render(record: dict, language: str) -> str:
    commands = record["expected_commands"]
    if len(commands) != 2 or commands[0]["tool"] != "filter_data" or commands[1]["tool"] != "aggregate_data":
        raise ValueError(f"Unsupported multilingual renderer plan: {record['case_id']}")
    filt, aggregate = commands[0]["arguments"], commands[1]["arguments"]
    filter_col, op, value = filt["column"], filt["operator"], filt["value"]
    metric = aggregate.get("column") or aggregate.get("metric")
    aggregation = aggregate["aggregation"]
    if language == "hi":
        return (
            f"पहले केवल वे पंक्तियाँ रखें जिनमें `{filter_col}` {value} {OP_HI[op]} है। "
            f"फिर उसी छनित परिणाम पर `{metric}` का {AGG_HI[aggregation]} निकालें। "
            "परिणाम तालिका में दें; कोई चार्ट न बनाएँ।"
        )
    return (
        f"Pehle sirf woh rows rakho jahan `{filter_col}` {value} {OP_HINGLISH[op]} hai. "
        f"Phir usi filtered result par `{metric}` ka {AGG_HINGLISH[aggregation]} nikalo. "
        "Output table mein do; chart mat banao."
    )


def build(source: Path, output: Path, count_per_language: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_records = _load(source)
    selected = _balanced(source_records, count_per_language)
    registry = CapabilityRegistry(auto_bootstrap=True)
    candidates: list[dict] = []
    inherited_evidence: list[dict] = []
    for record in selected:
        for language in ("hi", "hinglish"):
            query = _render(record, language)
            commands = deepcopy(record["expected_commands"])
            for command in commands:
                registry.validate_tool_call(command["tool"], command["arguments"])
            candidate = deepcopy(record)
            candidate.update({
                "case_id": f"{record['case_id']}-{language}",
                "split": "review",
                "query": query,
                "language": language,
                "language_provenance": "ai-assisted-official-terminology-informed",
                "label_status": "candidate-numeric-verified",
                "curation_status": "human-language-and-semantic-review-required",
                "human_reviewed": False,
                "review_required": True,
                "source_case_id": record["case_id"],
                "source_ids": ["cstt_ai_ml_glossary", "rajbhasha_technical_hindi_guidance"],
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": json.dumps({"mode": "tool", "commands": commands}, ensure_ascii=False)},
                ],
            })
            candidate["verification"] = {
                **record["verification"],
                "query_semantics_verified": False,
                "language_reviewed": False,
                "inherited_numeric_evidence_from": record["case_id"],
            }
            candidates.append(candidate)
            inherited_evidence.append({
                "case_id": candidate["case_id"],
                "passed": True,
                "result_sha256": record["verification"]["result_sha256"],
                "source_case_id": record["case_id"],
                "evidence_scope": "identical typed command plan and source data; language semantics not verified",
            })
    output.mkdir(parents=True)
    corpus = output / "multilingual_multistep_candidates.jsonl"
    corpus.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in candidates) + "\n", encoding="utf-8")
    (output / "numeric_evidence.json").write_text(
        json.dumps(inherited_evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = {
        "schema_version": "datapilot-language-source-manifest-v1",
        "candidate_authorship": "AI-assisted; terminology-informed; not copied examples and not human-reviewed",
        "sources": [
            {
                "id": "cstt_ai_ml_glossary",
                "publisher": "Commission for Scientific and Technical Terminology, Ministry of Education, Government of India",
                "title": "Artificial Intelligence-Machine Learning Glossary (English-Hindi)",
                "url": "https://www.shabd.education.gov.in/lexicon.jsp?lexicon=cstt_fund_AIML_EngHin_glossary",
                "use": "Hindi technical terminology reference only",
            },
            {
                "id": "rajbhasha_technical_hindi_guidance",
                "publisher": "Department of Official Language, Ministry of Home Affairs, Government of India",
                "title": "Rajbhasha Bharati — technical Hindi standardisation discussion",
                "url": "https://rajbhasha.gov.in/sites/default/files/Rajbhasha%20Bharti%20for%20Net%2026-6-25.pdf",
                "use": "Evidence that technical Hindi usage is variable; motivates human naturalness review",
            },
        ],
    }
    (output / "language_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    report = {
        "schema_version": "datapilot-multilingual-multistep-candidates-v1",
        "source_corpus": str(source.resolve()),
        "source_record_count": len(source_records),
        "candidate_count": len(candidates),
        "language_counts": dict(Counter(item["language"] for item in candidates)),
        "genuine_dependent_multistep_count": len(candidates),
        "distinct_source_workflows": len(selected),
        "dataset_counts": dict(Counter(item["dataset_id"] for item in candidates)),
        "corpus_sha256": digest,
        "human_reviewed": 0,
        "training_ready": False,
        "limitations": [
            "Numeric evidence is inherited from the identical source plan; it does not validate language meaning.",
            "Every Hindi/Hinglish query requires record-level human naturalness and semantic review.",
            "Coverage is limited to dependent filter_data -> aggregate_data workflows.",
        ],
    }
    (output / "candidate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-language", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.count_per_language), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
