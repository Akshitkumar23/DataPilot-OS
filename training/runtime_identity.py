"""Fingerprint planner dependencies without recording credentials or data values."""
import hashlib
import os
from pathlib import Path

from training.planner_prompt import fingerprint

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'modules/planner_prompt_contract.py', 'modules/capability_registry.py',
    'modules/planner_learning_memory.py', 'backend/ds_engine/intent_schema.py',
    'training/prepare_runtime_corpus.py', 'training/planner_pattern_memory.jsonl',
    'modules/query_orchestrator.py', 'modules/semantic_planner.py',
    'backend/ds_engine/command_resolver.py', 'backend/ds_engine/semantic_feasibility.py',
    'backend/ds_engine/typed_operators.py', 'backend/ds_engine/result_validator.py',
    'training/reference_execution.py',
    'training/execution_grader.py',
)
SETTINGS = {
    'OLLAMA_PLANNER_MEMORY_ENABLED': '1',
    'OLLAMA_TOOL_RETRIEVER': 'semantic', 'OLLAMA_PLANNER_TOOL_LIMIT': '5',
    'OLLAMA_TOOL_EMBEDDING_MODEL': 'nomic-embed-text',
    'OLLAMA_PLANNER_MEMORY_EMBEDDING_MODEL': 'nomic-embed-text',
}


def runtime_identity(root=ROOT):
    root = Path(root)
    content = {'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         if (root / name).is_file() else None for name in FILES},
               'settings': {name: os.getenv(name, default) for name, default in SETTINGS.items()}}
    return {**content, 'sha256': fingerprint(content),
            'limit': 'Per-case prompt hashes additionally detect retrieval-output differences.'}
