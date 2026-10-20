"""Durable evaluation evidence. Partial reports never qualify for promotion."""
import json
from pathlib import Path


def write_checkpoint(path, identity, rows, expected_count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**identity, 'results': rows, 'case_count': len(rows),
               'expected_case_count': expected_count,
               'status': 'evaluation_incomplete', 'deployment_allowed': False}
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def require_context_budget(input_tokens, max_new_tokens, context_limit):
    if max_new_tokens < 1 or context_limit < 1:
        raise ValueError('INVALID_GENERATION_BUDGET')
    if input_tokens + max_new_tokens > context_limit:
        raise ValueError(f'PROMPT_TOO_LONG: {input_tokens} input + {max_new_tokens} output tokens exceed {context_limit}; no truncation applied')
