"""Add a verified local preflight receipt and create a new, integrity-checked ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.verify_bundle_integrity import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_receipt(bundle, receipt, config):
    """Reject stale or inherited receipts even when Python assertions are disabled."""
    from training.runtime_identity import FILES
    if receipt.get('refreshed_runtime_identity_only'):
        raise ValueError('Fresh preflight required: rewritten runtime identity is not evidence')
    if not receipt.get('tokenizer_checked') or receipt.get('ready_for_gpu_smoke') is not True or receipt.get('failures'):
        raise ValueError('Successful full tokenizer preflight required')
    if not receipt.get('records') or receipt.get('schema_valid') != receipt['records']:
        raise ValueError('Incomplete schema verification')
    for key in ('base_model', 'base_revision'):
        if receipt.get(key) != config.get(key):
            raise ValueError('Preflight model identity mismatch: ' + key)
    for key, name in (('corpus_file_sha256', 'training/planner_sft_seed.jsonl'),
                      ('policy_file_sha256', 'training/pilot-policy.json')):
        if receipt.get(key) != digest(bundle / name):
            raise ValueError('Preflight input changed: ' + name)
    contract = receipt.get('runtime_contract', {})
    if set(contract.get('files', {})) != set(FILES):
        raise ValueError('Preflight runtime file inventory differs')
    if contract.get('settings', {}).get('OLLAMA_TOOL_RETRIEVER') != config.get('retriever'):
        raise ValueError('Preflight retriever differs from training configuration')
    memory_enabled = contract.get('settings', {}).get('OLLAMA_PLANNER_MEMORY_ENABLED', '1').lower() not in {'0', 'false', 'no'}
    if memory_enabled != bool(config.get('planner_memory_enabled', True)):
        raise ValueError('Preflight planner-memory setting differs from training configuration')
    for name, sha in contract['files'].items():
        path = bundle / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError('Changed runtime source: ' + name)


def finalize(bundle, preflight):
    bundle = bundle.resolve()
    archive = bundle.parent / (bundle.name + '.zip')
    if archive.exists() or (bundle / 'LOCAL_PREFLIGHT.json').exists():
        raise FileExistsError('Final delivery already exists; do not overwrite it.')
    verify(bundle)
    receipt = json.loads(preflight.read_text(encoding='utf-8'))
    config = json.loads((bundle / 'RUN_CONFIG.json').read_text(encoding='utf-8'))
    validate_receipt(bundle, receipt, config)
    shutil.copy2(preflight, bundle / 'LOCAL_PREFLIGHT.json')
    start = bundle / 'START_HERE.md'
    start.write_text(
        '# DataPilot Kaggle upload instructions\n\n'
        '1. Upload this ZIP as a **Private Kaggle Dataset**. Do not upload your project or customer CSVs.\n'
        '2. Create a fresh Kaggle notebook by importing `DataPilot_Kaggle_QLoRA.ipynb` from this folder.\n'
        '3. Attach the uploaded Dataset using Add Input. Enable GPU and Internet.\n'
        '4. Run All. The notebook verifies hashes, dependencies, full prompts and GPU smoke before long training.\n'
        '5. Keep outputs if interrupted. Matching saved checkpoints can resume; do not delete prior adapters.\n'
        '6. Download `datapilot-training-results.zip`. Do not activate the adapter before reviewing evaluation.\n\n'
        f"Local checks: {receipt['schema_valid']}/{receipt['records']} current runtime targets valid; "
        f"maximum observed full example {receipt['max_observed_tokens']} tokens; configured budget {receipt['max_seq_length']}.\n\n"
        'Local tokenizer preflight passed. Actual CUDA training has **not** run locally. The first Kaggle smoke must pass.\n\n'
        'This is a filtering/aggregation/multistep/chart research curriculum, not a universal data-science model. '
        'The full-registry release gate intentionally stays blocked. Review provenance and remaining limits in '
        '`training/KAGGLE_CURRICULUM_DECISIONS.md`.\n', encoding='utf-8')
    manifest_path = bundle / 'BUNDLE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = [p for p in bundle.rglob('*') if p.is_file() and p != manifest_path
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    manifest['files'] = {p.relative_to(bundle).as_posix(): digest(p) for p in files}
    manifest['local_tokenizer_preflight'] = 'passed_gpu_unverified'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    count = verify(bundle)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for p in [*files, manifest_path]:
            out.write(p, p.relative_to(bundle.parent))
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        for name, sha in manifest['files'].items():
            assert hashlib.sha256(check.read(bundle.name + '/' + name)).hexdigest() == sha
    print(json.dumps({'archive': str(archive), 'sha256': digest(archive), 'bytes': archive.stat().st_size,
                      'verified_files': count, 'gpu_training_completed': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    a = p.parse_args()
    finalize(a.bundle, a.preflight)
