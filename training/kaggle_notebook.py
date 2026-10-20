"""Fresh-process, failure-stopping Kaggle workflow for the numerical curriculum."""
import textwrap


def notebook(bundle_name):
    cells = []
    def add(kind, source):
        cells.append({'cell_type': kind, 'id': f'datapilot-{len(cells)}', 'metadata': {},
                      'source': textwrap.dedent(source).strip().splitlines(keepends=True),
                      **({'outputs': [], 'execution_count': None} if kind == 'code' else {})})
    add('markdown', '''
    # DataPilot — multilingual numerical planner training
    Train and evaluate a local planner on documented public datasets. The curriculum covers
    filtering, grouped/scalar aggregation, dependent filter→aggregate workflows, and bar/line
    charts. English, Hindi and Hinglish requests are programmatically/AI authored; numerical
    evidence and separate human-reviewed archetypes are included with their actual provenance.
    This experiment does not certify all data-science capabilities or deploy an Ollama model.

    Attach the ZIP as a **Private Dataset**, enable **GPU** and **Internet**, then **Run All**.
    Each failing subprocess stops the notebook. Keep all output/checkpoints if an error occurs.
    ''')
    add('code', f'''
    import os, sys, subprocess, json, shutil, shlex, hashlib
    from pathlib import Path
    BUNDLE_NAME = {bundle_name!r}
    def run(command):
        args = shlex.split(command) if isinstance(command, str) else list(command)
        if args[0] == 'python': args[0] = sys.executable
        subprocess.run(args, check=True)
    matches = []
    for path in Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json'):
        if json.loads(path.read_text()).get('bundle') == BUNDLE_NAME:
            matches.append(path.parent)
    assert len(matches) == 1, f'Expected one matching bundle, found {{len(matches)}}'
    workspace = Path('/kaggle/working') / BUNDLE_NAME
    if not workspace.exists(): shutil.copytree(matches[0], workspace)
    os.chdir(workspace)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Explicit offline retrieval configuration; saved in every runtime identity.
    # No Ollama service or embeddings are assumed to exist inside Kaggle.
    os.environ['OLLAMA_TOOL_RETRIEVER'] = 'lexical'
    os.environ['OLLAMA_PLANNER_MEMORY_ENABLED'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['MPLBACKEND'] = 'Agg'
    run('python training/verify_bundle_integrity.py .')
    run('python training/audit_corpus_quality.py --corpus training/planner_sft_seed.jsonl --output /kaggle/working/quality.json --enforce')
    print('Workspace:', workspace)
    ''')
    add('code', '''
    # Each dependency check runs in a fresh interpreter, avoiding stale kernel imports.
    run('nvidia-smi')
    run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'training/requirements-train.txt'])
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])
    run('python training/check_peft_environment.py')
    run('python scripts/validate_planner_corpus.py')
    run('python training/train_planner_adapter.py --dry-run --corpus training/planner_sft_seed.jsonl --pilot-policy training/pilot-policy.json')
    # Resolve once and reuse the immutable base checkpoint for smoke, SFT and evaluation.
    config = json.loads(Path('RUN_CONFIG.json').read_text())
    model, revision = config['base_model'], config['base_revision']
    from training.runtime_identity import runtime_identity
    from training.planner_prompt import fingerprint
    runtime = runtime_identity()
    corpus_hash = hashlib.sha256(Path('training/planner_sft_seed.jsonl').read_bytes()).hexdigest()
    policy_hash = hashlib.sha256(Path('training/pilot-policy.json').read_bytes()).hexdigest()
    print('Pinned base:', model, revision)
    ''')
    add('code', '''
    preflight = Path('/kaggle/working/token-preflight.json')
    if not preflight.exists():
        run(['python', 'training/kaggle_preflight.py', '--output', str(preflight), '--base-model', model, '--base-revision', revision])
    report = json.loads(preflight.read_text())
    assert report['ready_for_gpu_smoke'] and report['base_revision'] == revision
    assert report['corpus_file_sha256'] == corpus_hash and report['policy_file_sha256'] == policy_hash, 'Stale preflight; preserve old output and use a fresh run'
    assert report['runtime_contract'] == runtime, 'Preflight runtime differs'
    seq, new_tokens = report['max_seq_length'], report['max_new_tokens']
    print('Actual full prompt ceiling:', seq, 'Completion allowance:', new_tokens)
    common = ['--corpus', 'training/planner_sft_seed.jsonl', '--pilot-policy', 'training/pilot-policy.json',
              '--base-model', model, '--base-revision', revision]
    train_flags = ['--max-seq-length', str(seq), '--batch-size', '1', '--allow-contract-verified', '--confirm-gpu-training']
    smoke = Path('/kaggle/working/planner-smoke')
    if not (smoke / 'run_manifest.json').exists():
        run(['python', 'training/train_planner_adapter.py', *common, *train_flags,
             '--output-dir', str(smoke), '--max-steps', '2', '--gradient-accumulation-steps', '1'])
    receipt = json.loads((smoke / 'run_manifest.json').read_text())
    assert receipt['status'] == 'smoke_only' and receipt['base_revision'] == revision
    assert receipt['max_seq_length'] == seq
    assert receipt['corpus_sha256'] == corpus_hash and receipt['runtime_contract'] == runtime, 'Stale smoke result'
    print('GPU smoke passed:', receipt['training_loss'], receipt['peak_cuda_memory_gib'], 'GiB')
    ''')
    add('code', '''
    # Training runs in its own process; GPU memory is released before evaluation.
    adapter = Path('/kaggle/working/planner-qlora')
    if not (adapter / 'run_manifest.json').exists():
        run(['python', 'training/train_planner_adapter.py', *common, *train_flags,
             '--output-dir', str(adapter), '--epochs', '2', '--gradient-accumulation-steps', '16', '--resume'])
    trained = json.loads((adapter / 'run_manifest.json').read_text())
    assert trained['status'] == 'trained_unpromoted' and trained['base_revision'] == revision
    assert trained['corpus_sha256'] == json.loads(Path('training/pilot-policy.json').read_text())['corpus_sha256']
    assert trained['runtime_contract'] == runtime and trained['max_seq_length'] == seq, 'Stale trained adapter'
    print('Adapter saved:', adapter)
    ''')
    add('code', '''
    eval_flags = ['--split', 'test', '--max-context-tokens', str(seq), '--max-new-tokens', str(new_tokens)]
    test_records = [json.loads(line) for line in Path('training/planner_sft_seed.jsonl').read_text().splitlines() if line.strip()]
    test_records = [row for row in test_records if row['split'] == 'test']
    for name, extra in [('base', []), ('adapter', ['--candidate-adapter', str(adapter)])]:
        result = Path('/kaggle/working') / f'{name}-evaluation.json'
        if not result.exists():
            run(['python', 'training/evaluate_planner_adapter.py', *common, *eval_flags, *extra,
                 '--resume', '--output', str(result)])
        payload = json.loads(result.read_text())
        assert payload['status'] == 'evaluation_only_unpromoted' and payload['base_revision'] == revision
        assert payload['corpus_sha256'] == fingerprint(test_records) and payload['runtime_contract'] == runtime, 'Stale evaluation'
        assert payload['generation_config'] == {'max_new_tokens': new_tokens, 'max_context_tokens': seq}
        if name == 'adapter':
            current_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(adapter.iterdir()) if p.is_file() and p.suffix in {'.safetensors', '.bin', '.json'}}
            assert payload['adapter_files'] == current_files, 'Evaluation belongs to another adapter'
        print(name, payload['metrics'])
    run('python training/assess_adapter_release.py --baseline /kaggle/working/base-evaluation.json --candidate /kaggle/working/adapter-evaluation.json --output /kaggle/working/release-decision.json --report-only')
    print(json.loads(Path('/kaggle/working/release-decision.json').read_text()))
    ''')
    add('code', '''
    # Save/download these outputs; neither export nor activation happens here.
    deliverables = Path('/kaggle/working/datapilot-training-results')
    deliverables.mkdir(exist_ok=True)
    for filename in ['token-preflight.json', 'base-evaluation.json', 'adapter-evaluation.json', 'release-decision.json', 'quality.json']:
        shutil.copy2(Path('/kaggle/working') / filename, deliverables / filename)
    shutil.copytree(adapter, deliverables / 'adapter', dirs_exist_ok=True)
    shutil.copy2('RUN_CONFIG.json', deliverables / 'RUN_CONFIG.json')
    archive = shutil.make_archive('/kaggle/working/datapilot-training-results', 'zip', deliverables)
    print('Download:', archive)
    ''')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': cells}
