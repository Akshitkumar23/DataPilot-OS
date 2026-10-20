"""Production-grade planner-corpus gates beyond schema validity and row count."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from training.audit_corpus_quality import audit
from training.corpus_contract import reference_commands, reference_family


def assess_release_corpus(
    records: list[dict[str, Any]], *,
    min_multistep: int = 100,
    min_hindi: int = 100,
    min_hinglish: int = 100,
    min_test_families: int = 20,
) -> dict[str, Any]:
    quality = audit(records, min_test_families=min_test_families)
    registry = CapabilityRegistry(auto_bootstrap=True)
    executable_tools = {
        spec.name for spec in registry.list_tool_specs()
        if spec.requires_dataset and spec.category not in {'general_chat', 'clarification', 'clarify'}
    }
    labels = Counter(record.get('label_status') for record in records)
    languages = Counter(record.get('language') for record in records)
    human_language_provenance = Counter(
        record.get('language') for record in records
        if record.get('language_provenance') == 'human-authored'
    )
    multi_step = sum(
        record.get('expected_decision') == 'tool' and len(reference_commands(record)) > 1
        for record in records
    )
    split_tools = {}
    for split in ('train', 'validation', 'test'):
        split_tools[split] = {
            command['tool']
            for record in records if record.get('split') == split
            for command in reference_commands(record)
        }
    blockers = list(quality['blockers'])
    if labels.get('human-reviewed', 0) != len(records):
        blockers.append(
            f"Every production record must be human-reviewed; {labels.get('human-reviewed', 0)}/{len(records)} qualify"
        )
    if multi_step < min_multistep:
        blockers.append(f'multi-step examples {multi_step} < required {min_multistep}')
    if human_language_provenance.get('hi', 0) < min_hindi:
        blockers.append(f'human-authored Hindi examples {human_language_provenance.get("hi", 0)} < required {min_hindi}')
    if human_language_provenance.get('hinglish', 0) < min_hinglish:
        blockers.append(f'human-authored Hinglish examples {human_language_provenance.get("hinglish", 0)} < required {min_hinglish}')
    for split in ('validation', 'test'):
        missing = sorted(executable_tools - split_tools[split])
        if missing:
            blockers.append(f'{split} missing live tools: ' + ', '.join(missing))
    if quality['splits']['test']['unique_dataset_plans'] < min_test_families:
        blockers.append(
            f"independent test families {quality['splits']['test']['unique_dataset_plans']} < required {min_test_families}"
        )
    return {
        'schema_version': 'datapilot-release-corpus-gate-v1',
        'ready': not blockers,
        'record_count': len(records),
        'label_counts': dict(labels),
        'language_counts': dict(languages),
        'human_authored_language_counts': dict(human_language_provenance),
        'multi_step_count': multi_step,
        'live_executable_tools': sorted(executable_tools),
        'split_tool_coverage': {key: sorted(value) for key, value in split_tools.items()},
        'quality': quality,
        'blockers': blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--enforce', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.corpus.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = assess_release_corpus(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.enforce and not report['ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
