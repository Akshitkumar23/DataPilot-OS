"""Evaluate a base planner and optional PEFT adapter on a held-out corpus.

This is deliberately an offline HF/PEFT evaluation tool.  It does not contact
Ollama, change DataPilot's active planner, or promote an adapter.  It measures
what the planner actually emits: JSON parsing, live ToolSpec validity, exact
tool-plan match, and schema-grounded required columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.capability_registry import CapabilityRegistry
from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, load_jsonl, validate_records
from training.planner_prompt import build_prompt, fingerprint, PROMPT_VERSION
from training.corpus_contract import reference_commands, reference_family
from modules.planner_prompt_contract import RUNTIME_PROMPT_VERSION as PROMPT_VERSION
from training.evaluation_checkpoint import write_checkpoint, require_context_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "training" / "planner_sft_contract_verified.jsonl")
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument('--base-revision', default='main')
    parser.add_argument("--candidate-adapter", type=Path, help="PEFT adapter directory. Omit to evaluate the base model only.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument('--max-context-tokens', type=int, default=16384)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit; never use for release promotion.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--pilot-policy', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume only an identical partial evaluation.')
    return parser.parse_args()


def extract_columns(value: Any) -> set[str]:
    """Collect strings from the emitted command without schema-name heuristics."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return set().union(*(extract_columns(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(extract_columns(item) for item in value.values())) if value else set()
    return set()


def parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        decoded = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg}"
    if not isinstance(decoded, dict):
        return None, "response_not_object"
    return decoded, None


def evaluation_messages(record: dict[str, Any], provenance=None, registry=None) -> list[dict[str, Any]]:
    """Use the same complete prompt as SFT, excluding the reference answer."""
    if provenance is not None and registry is not None:
        return build_prompt(record, provenance, registry)
    messages = record["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Evaluation record must end in a reference assistant answer")
    prompt = messages[:-1]
    if not any(message.get("role") == "user" for message in prompt):
        raise ValueError("Evaluation prompt is missing its user query")
    return prompt


def compare_record(record: dict[str, Any], response: dict[str, Any], registry: CapabilityRegistry) -> dict[str, Any]:
    # The runtime wire protocol has no mode field. Retain legacy report input
    # compatibility, but compare live single/sequence calls with the same grader.
    if 'mode' not in response:
        if set(response) == {'tool', 'arguments'}:
            spec = next((s for s in registry.list_tool_specs() if s.name == response['tool']), None)
            if spec and getattr(spec, 'category', None) == 'clarification':
                response = {'mode': 'clarify', 'clarification': response['arguments'].get('reason')}
            else:
                response = {'mode': 'tool', 'commands': [response]}
        elif set(response) == {'commands'}:
            response = {'mode': 'tool', 'commands': response['commands']}
    expected_decision = record["expected_decision"]
    if expected_decision == "clarification":
        passed = response.get("mode") == "clarify" and isinstance(response.get("clarification"), str) and bool(response['clarification'].strip()) and not response.get('commands')
        return {"passed": passed, "tool_match": passed, "schema_valid": passed, "columns_grounded": passed}

    commands = response.get("commands")
    expected_commands = reference_commands(record)
    if response.get("mode") != "tool" or not isinstance(commands, list) or len(commands) != len(expected_commands):
        return {"passed": False, "tool_match": False, "schema_valid": False, "columns_grounded": False, "reason": "command_count_mismatch"}
    if len(expected_commands) > 1:
        steps = [compare_record({'expected_decision': 'tool', 'expected_command': expected,
                                 'required_columns': []}, {'mode': 'tool', 'commands': [observed]}, registry)
                 for expected, observed in zip(expected_commands, commands)]
        coverage = set(record.get('required_columns', [])) <= extract_columns(commands)
        result = {key: all(step.get(key, False) for step in steps)
                  for key in ('passed', 'tool_match', 'schema_valid', 'columns_grounded', 'exact_plan', 'raw_exact_plan')}
        result['passed'] = result['passed'] and coverage
        result['columns_grounded'] = result['columns_grounded'] and coverage
        result.update(steps=steps, required_column_coverage=coverage)
        return result
    observed = commands[0]
    expected = expected_commands[0]
    if not isinstance(observed, dict) or not isinstance(observed.get("tool"), str) or not isinstance(observed.get("arguments"), dict):
        return {"passed": False, "tool_match": False, "schema_valid": False, "columns_grounded": False, "reason": "malformed_tool_call"}
    tool_match = observed["tool"] == expected["tool"]
    try:
        normalized_observed = registry.validate_tool_call(observed["tool"], observed["arguments"]).model_dump(mode='json')
        normalized_expected = registry.validate_tool_call(expected["tool"], expected["arguments"]).model_dump(mode='json')
        schema_valid = True
    except Exception as exc:
        schema_valid = False
        validation_error = str(exc)
    observed_columns = extract_columns(observed["arguments"])
    required = set(record.get("required_columns", []))
    required_column_coverage = required <= observed_columns
    exact_plan = bool(tool_match and schema_valid and normalized_observed == normalized_expected)
    # A string appearing as a literal filter value is not proof of a column binding.
    # Exact validated reference agreement establishes bindings for this labelled case.
    columns_grounded = bool(required_column_coverage and exact_plan)
    result = {
        "passed": bool(tool_match and schema_valid and columns_grounded and exact_plan),
        "tool_match": tool_match,
        "schema_valid": schema_valid,
        "columns_grounded": columns_grounded,
        "exact_plan": exact_plan,
        "raw_exact_plan": observed == expected,
        "required_column_coverage": required_column_coverage,
        "missing_required_columns": sorted(required - observed_columns),
    }
    if not schema_valid:
        result["validation_error"] = validation_error
    return result


def main() -> int:
    args = parse_args()
    checkpoint = args.output.with_suffix('.partial.json')
    if args.output.exists() or (checkpoint.exists() and not args.resume):
        raise FileExistsError('Use a new evaluation output path; existing evidence will not be overwritten.')
    records = load_jsonl(args.corpus)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    validate_records(records, provenance)
    pilot = None
    if args.pilot_policy:
        from training.pilot_policy import validate_pilot
        pilot = validate_pilot(records, args.corpus, args.pilot_policy)
    records = [record for record in records if record["split"] == args.split]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError(f"No records in split {args.split!r}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.candidate_adapter) if args.candidate_adapter else args.base_model, **({} if args.candidate_adapter else {'revision': args.base_revision}), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model_name = args.base_model
    if args.candidate_adapter:
        model = PeftModel.from_pretrained(model, str(args.candidate_adapter))
        model_name = f"{args.base_model}+{args.candidate_adapter.name}"
    model.eval()
    registry = CapabilityRegistry(auto_bootstrap=True)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    from training.prepare_runtime_corpus import RuntimeCorpusBuilder
    builder = RuntimeCorpusBuilder(provenance, registry)
    identity = {'model': model_name, 'base_model': args.base_model,
                'base_revision': args.base_revision,
                'prompt_version': PROMPT_VERSION, 'corpus_sha256': fingerprint(records),
                'split': args.split, 'smoke_only': args.limit is not None}
    from training.runtime_identity import runtime_identity
    identity['runtime_contract'] = runtime_identity()
    identity['execution_grader_version'] = 'sql-output-v1'
    identity['generation_config'] = {'max_new_tokens': args.max_new_tokens, 'max_context_tokens': args.max_context_tokens}
    if args.candidate_adapter:
        import hashlib
        identity['adapter_files'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.candidate_adapter.iterdir())
            if path.suffix in {'.safetensors', '.bin', '.json'} and path.is_file()
        }
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        for key, value in identity.items():
            if saved.get(key) != value:
                raise ValueError(f'EVALUATION_RESUME_IDENTITY_MISMATCH: {key}')
        rows = saved.get('results', [])
        if [r['case_id'] for r in rows] != [r['case_id'] for r in records[:len(rows)]]:
            raise ValueError('EVALUATION_RESUME_CASE_ORDER_MISMATCH')
        for row, record in zip(rows, records):
            if row.get('prompt_sha256') != fingerprint(builder.inputs(record)):
                raise ValueError('EVALUATION_RESUME_PROMPT_MISMATCH')
            counters['total'] += 1
            for metric in ('passed', 'tool_match', 'schema_valid', 'columns_grounded'):
                counters[metric] += int(bool(row.get(metric)))
    write_checkpoint(checkpoint, identity, rows, len(records))

    for record in records[len(rows):]:
        messages = builder.inputs(record)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        require_context_budget(inputs['input_ids'].shape[1], args.max_new_tokens,
                               min(args.max_context_tokens, model.config.max_position_embeddings))
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        started = perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latency_ms = (perf_counter() - started) * 1000
        completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response, parse_error = parse_response(completion)
        scored = {"passed": False, "tool_match": False, "schema_valid": False, "columns_grounded": False}
        if response is not None:
            scored = compare_record(record, response, registry)
        else:
            scored["reason"] = parse_error
        from training.execution_grader import grade_execution
        scored.update(grade_execution(record, response, builder.frame(record)) if response is not None else
                      {'execution_verified': False, 'execution_reason': parse_error})
        counters["total"] += 1
        for metric in ("passed", "tool_match", "schema_valid", "columns_grounded"):
            counters[metric] += int(bool(scored.get(metric)))
        rows.append({"case_id": record["case_id"], "dataset_id": record["dataset_id"],
                     "expected_command": record.get('expected_command'),
                     "expected_commands": reference_commands(record),
                     "prompt_sha256": fingerprint(messages),
                     "latency_ms": latency_ms, "completion": completion, **scored})
        write_checkpoint(checkpoint, identity, rows, len(records))
        print(f"Evaluated {len(rows)}/{len(records)}: {record['case_id']} passed={scored['passed']}", flush=True)

    total = counters["total"]
    rates = {key + "_rate": counters[key] / total for key in ("passed", "tool_match", "schema_valid", "columns_grounded")}
    payload = {"model": model_name, "base_model": args.base_model, "adapter": str(args.candidate_adapter) if args.candidate_adapter else None, "split": args.split, "case_count": total, "metrics": {**dict(counters), **rates}, "results": rows, "status": "evaluation_only_unpromoted"}
    payload.update(prompt_version=PROMPT_VERSION, corpus_sha256=fingerprint(records), smoke_only=args.limit is not None)
    payload['runtime_contract'] = identity['runtime_contract']
    payload['base_revision'] = args.base_revision
    payload['generation_config'] = identity['generation_config']
    payload['adapter_files'] = identity.get('adapter_files', {})
    payload['research_scope'] = pilot['allowed_tools'] if pilot else 'full-registry'
    payload['deployment_allowed'] = False
    payload['required_test_families'] = sorted({s.name for s in registry.list_tool_specs()
                                              if s.category not in {'general_chat', 'clarify', 'clarification'}} | {'clarification', 'multi_step'})
    payload['per_tool'] = {}
    payload['per_language'] = {}
    payload['execution_grader_version'] = identity['execution_grader_version']
    checked = [r for r in rows if r.get('execution_verified') is not None]
    payload['execution_metrics'] = {
        'evaluated': len(checked), 'not_evaluated': len(rows) - len(checked),
        'passed': sum(r.get('execution_verified') is True for r in checked),
        'pass_rate': sum(r.get('execution_verified') is True for r in checked) / len(checked) if checked else None,
    }
    for row, record in zip(rows, records):
        tool = reference_family(record)
        counts = payload['per_tool'].setdefault(tool, {'total': 0, 'passed': 0})
        counts['total'] += 1
        counts['passed'] += int(row['passed'])
        language = payload['per_language'].setdefault(record.get('language', 'unspecified'),
                    {'total': 0, 'plan_passed': 0, 'execution_passed': 0, 'execution_not_evaluated': 0})
        language['total'] += 1
        language['plan_passed'] += int(row['passed'])
        language['execution_passed'] += int(row.get('execution_verified') is True)
        language['execution_not_evaluated'] += int(row.get('execution_verified') is None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": payload["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
