"""Build a private, real-public-data-only DataPilot QLoRA bundle for Kaggle.

The bundle deliberately excludes Electron builds, virtual environments, local
Ollama models, customer datasets, and synthetic-only source datasets. The
executable pilot uses explicit contract-verified labels. Supplemental unapproved
review candidates are evidence only, never silently added to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.planner_training_readiness import APPROVED_LABELS, CONTRACT_VERIFIED_LABEL, assess, load_data_kinds
from scripts.validate_planner_corpus import DEFAULT_PROVENANCE, load_jsonl

BUNDLE_NAME = "DataPilot-Kaggle-Professional-V5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_source_tree(source: Path, destination: Path) -> None:
    """Copy Python source only; no cache, environments, builds, or secrets."""
    def source_only(directory, names):
        return [name for name in names if name.startswith('.') or name == '__pycache__'
                or (Path(directory, name).is_file() and Path(name).suffix != '.py')]
    shutil.copytree(
        source,
        destination,
        ignore=source_only,
    )


def filtered_records(corpus_path: Path) -> list[dict[str, Any]]:
    kinds = load_data_kinds(DEFAULT_PROVENANCE)
    return [
        record for record in load_jsonl(corpus_path)
        if kinds.get(record.get("dataset_id")) == "observed_public"
    ]


def notebook_payload(bundle_name: str, pilot: bool = False) -> dict[str, Any]:
    if pilot:
        from training.kaggle_notebook import notebook
        return notebook(bundle_name)
    cells = [
        ("markdown", [
            "# DataPilot — real-public-data-only QLoRA workspace\n",
            "This notebook trains only an **unpromoted research adapter** from real-public-data, live-ToolSpec contract-verified labels. "
            "Do not upload customer CSVs, secrets, local Ollama models, or synthetic-only data.\n",
        ]),
        ("code", [
            "import subprocess, sys, shlex\n",
            "def run(command):\n",
            "    parts = shlex.split(command)\n",
            "    if parts[0] == 'python': parts[0] = sys.executable\n",
            "    elif parts[0] == 'pip': parts = [sys.executable, '-m', 'pip', *parts[1:]]\n",
            "    subprocess.run(parts, check=True)\n",
        ]),
        ("code", [
            "# Kaggle expands uploaded ZIPs into the attached input directory.\n",
            "from pathlib import Path\n",
            "import shutil, os\n",
            f"BUNDLE_NAME = '{bundle_name}'\n",
            "matches = [p.parent for p in Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json') if p.parent.name == BUNDLE_NAME]\n",
            "assert len(matches) == 1, f'Attach exactly one DataPilot input bundle; found {len(matches)}.'\n",
            "workspace = Path('/kaggle/working') / BUNDLE_NAME\n",
            "if not workspace.exists(): shutil.copytree(matches[0], workspace)\n",
            "os.chdir(workspace)\n",
            "run('python training/verify_bundle_integrity.py .')\n",
            "print('Training workspace:', workspace)\n",
            "# CPU-only gate runs before installing packages or loading model weights.\n",
            "run('python training/audit_corpus_quality.py --corpus training/planner_sft_seed.jsonl --output /kaggle/working/corpus-quality.json --enforce')\n",
        ]),
        ("code", [
            "run('nvidia-smi')\n",
            "import torch\n",
            "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'\n",
            "print(torch.cuda.get_device_name(0))\n",
            "!pip install -q -r training/requirements-train.txt\n",
            "# Kaggle images may carry an incompatible torchao/Transformers 5 stack.\n",
            "run('pip uninstall -y torchao')\n",
            "import transformers\n",
            "assert int(transformers.__version__.split('.')[0]) < 5, f'Unsupported Transformers version: {transformers.__version__}'\n",
            "print('Training stack:', transformers.__version__)\n",
            "!python scripts/validate_planner_corpus.py\n",
            "!python scripts/planner_training_readiness.py --real-only --allow-contract-verified --enforce\n",
            "# Execute one representative real-data plan per ToolSpec before training.\n",
            "!python training/verify_execution_corpus.py --corpus training/planner_sft_seed.jsonl --split validation --samples-per-tool 1 --output /kaggle/working/execution-evidence.json --enforce\n",
        ]),
        ("markdown", [
            "## Research training gate\n",
            "A green gate authorizes only an unpromoted research adapter. Contract verification checks the live tool schema, not scientific correctness or human acceptance. "
            "Production still requires human-reviewed labels and the release checklist.\n",
        ]),
        ("code", [
            "# This creates an unpromoted research adapter only.\n",
            "!python training/train_planner_adapter.py --base-model Qwen/Qwen2.5-3B-Instruct --output-dir /kaggle/working/planner-qlora --epochs 3 --max-seq-length 16384 --batch-size 1 --gradient-accumulation-steps 16 --allow-contract-verified --confirm-gpu-training\n",
        ]),
        ("code", [
            "# Measure the unchanged base model on the held-out datasets.\n",
            "!python training/evaluate_planner_adapter.py --corpus training/planner_sft_seed.jsonl --split test --base-model Qwen/Qwen2.5-3B-Instruct --output /kaggle/working/base-evaluation.json\n",
            "# Measure the trained adapter on exactly the same cases.\n",
            "!python training/evaluate_planner_adapter.py --corpus training/planner_sft_seed.jsonl --split test --base-model Qwen/Qwen2.5-3B-Instruct --candidate-adapter /kaggle/working/planner-qlora --output /kaggle/working/adapter-evaluation.json\n",
            "!python training/assess_adapter_release.py --baseline /kaggle/working/base-evaluation.json --candidate /kaggle/working/adapter-evaluation.json --output /kaggle/working/release-decision.json --report-only\n",
        ]),
        ("markdown", [
            "## Export is a separate human gate\n",
            "Only if the release-decision says eligible and you have reviewed the failures may you run the export command below. It prepares merged Hugging Face weights and conversion instructions; it does not import or activate an Ollama model.\n",
        ]),
        ("markdown", [
            "Run All deliberately does not export. After manual review, create a separate code cell with:\n\n",
            "```python\n",
            "run('python training/export_adapter_for_ollama.py --base-model Qwen/Qwen2.5-3B-Instruct --adapter /kaggle/working/planner-qlora --release-decision /kaggle/working/release-decision.json --output-hf /kaggle/working/planner-merged-hf')\n",
            "```\n",
        ]),
    ]
    if pilot:
        flag = ' --pilot-policy training/pilot-policy.json'
        for _, source in cells:
            for i, line in enumerate(source):
                if 'scripts/planner_training_readiness.py' in line:
                    source[i] = "!python training/train_planner_adapter.py --dry-run --corpus training/planner_sft_seed.jsonl --pilot-policy training/pilot-policy.json\n"
                elif line.startswith('!python training/train_planner_adapter.py') or line.startswith('!python training/evaluate_planner_adapter.py'):
                    source[i] = line.rstrip() + flag + '\n'
                    if 'train_planner_adapter.py' in line:
                        source[i] = source[i].replace('--max-seq-length 16384', '--max-seq-length 4096')
        cells[0][1].append('Scope: filtering, aggregation, grouping and sequential combinations only. Questions are programmatically authored on observed public data, not human-reviewed.\n')
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": cell_type,
                "metadata": {},
                # IPython ! commands do not propagate failure to Run All.
                # Make dependency/validation/training failures stop the job.
                "source": [f"run({line[1:].strip()!r})\n" if line.startswith("!") else line for line in source],
                "id": f"datapilot-{index}",
                **({"outputs": [], "execution_count": None} if cell_type == 'code' else {}),
            }
            for index, (cell_type, source) in enumerate(cells)
        ],
    }


def readme() -> str:
    return """# DataPilot Kaggle training workspace

This folder is a **private Kaggle upload bundle** for one purpose: build and
evaluate a QLoRA adapter for DataPilot's local planner using real, documented
public data and live-ToolSpec contract-verified labels.

**Current status: curation/research only, not ready for another full training run.**
Read `CORPUS_QUALITY_REPORT.json` and `TRAINING_STATUS.md` first. Existing record
counts include paraphrase expansions; they do not establish independent task
diversity. The new quality gate may reject a corpus accepted by the older gate.

## What is included

- Six documented public UCI datasets: Bank Marketing, Bike Sharing,
  Appliances Energy, Wine Quality, Online Retail, and Adult Income.
- Source URLs, licences, SHA-256 hashes, schemas, and row counts.
- A 1,400-record real-public-data corpus: 1,100 training records, 100
  validation records, and 200 dataset-held-out test records. Every executable
  label validates against the live ToolSpec registry. It intentionally excludes
  AI4I because UCI describes AI4I as synthetic reference data.
- The live ToolSpec registry, corpus validator, readiness gate, QLoRA trainer,
  execution verifier, held-out base-versus-adapter evaluator, release-decision
  gate, and this Kaggle notebook.

## What is intentionally excluded

- Customer/company CSVs, user prompts containing personal data, API keys,
  local Ollama models, Electron builds, Python virtual environments, and
  synthetic-only source data.

## Before training

1. Upload this ZIP to a **Private** Kaggle Dataset.
2. Open `DataPilot_Kaggle_QLoRA.ipynb` in a Kaggle Notebook and enable GPU.
3. Run the validator and readiness report.
4. Run the research gate using `--allow-contract-verified`; it proves only
   data provenance, schema validity, coverage, and split hygiene.  It does
   **not** convert labels into human-reviewed labels.
5. Keep validation/test datasets disjoint from training datasets.
6. Train only after both the readiness gate and `audit_corpus_quality.py --enforce`
   succeed. Measure complete prompt token lengths and do a short GPU memory
   smoke test before committing to the full run; 16,384 is a context ceiling,
   not a verified T4 performance guarantee.
7. Run the execution verifier, then the held-out baseline and adapter
   evaluations.  Export is allowed only if `assess_adapter_release.py` emits
   an eligible manual-review decision.

The included adapter remains unpromoted after training. Human review, a
successful converted-model benchmark, and the release checklist are mandatory
before deployment to Ollama.

See `CURATION_WORKFLOW.md` for the exact review workflow.
"""


def curation_workflow() -> str:
    return """# Real-data corpus curation workflow

Every record must be traceable to a documented public dataset and represent a
realistic analyst request.  Do not call a model-produced answer “reviewed”.

1. Choose a documented dataset schema and a real analyst-style question.
2. Decide whether it is executable, genuinely ambiguous, or unsupported.
3. If executable, author one exact ToolSpec command using the live registry.
4. Run `scripts/validate_planner_corpus.py`.
5. Execute the command against the source dataset and inspect the result,
   chart, and evidence manifest.
6. A reviewer records their identity and rationale, then marks the label
   `human-reviewed`.
7. Assign splits by **dataset**, not random paraphrase: unseen datasets must
   remain in validation/test.
8. Run `scripts/planner_training_readiness.py --real-only --enforce`.

When a desktop query fails, classify it before adding a label:

- planner selected a wrong existing capability → reviewed learning example;
- capability is absent → engineering backlog, not a training label;
- schema makes request impossible → clarification example;
- execution/UI defect → regression test, not a planner label.
"""


def build(
    output: Path,
    pilot_dir: Path | None = None,
    review_workspace: Path | None = None,
    supplemental_workspaces: list[Path] | None = None,
    defer_archive: bool = False,
) -> tuple[Path, Path]:
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle folder: {output}")
    output.mkdir(parents=True)
    corpus_path = pilot_dir / 'pilot.jsonl' if pilot_dir else ROOT / "training" / "planner_sft_contract_verified.jsonl"
    records = filtered_records(corpus_path)
    kinds = load_data_kinds(DEFAULT_PROVENANCE)
    allowed_ids = {record["dataset_id"] for record in records}
    source_manifest = json.loads((ROOT / "data" / "benchmark_sources.json").read_text(encoding="utf-8"))
    allowed_specs = [item for item in source_manifest["datasets"] if item.get("data_kind") == "observed_public"]
    allowed_ids.update(item["id"] for item in allowed_specs)

    # Source needed by the offline validators and trainer.
    copy_source_tree(ROOT / "modules", output / "modules")
    copy_source_tree(ROOT / "backend", output / "backend")
    (output / "scripts").mkdir()
    for name in ("validate_planner_corpus.py", "planner_training_readiness.py"):
        shutil.copy2(ROOT / "scripts" / name, output / "scripts" / name)
    (output / "training").mkdir()
    for name in (
        "train_planner_adapter.py", "build_contract_verified_corpus.py", "verify_execution_corpus.py", "evaluate_planner_adapter.py", "check_peft_environment.py",
        "assess_adapter_release.py", "export_adapter_for_ollama.py", "requirements-train.txt", "MODEL_RELEASE_CHECKLIST.md",
        "planner_prompt.py", "audit_corpus_quality.py", "prepare_runtime_corpus.py",
        "verify_bundle_integrity.py",
        "verify_numeric_oracles.py", "prepare_corpus_review.py",
        "corpus_contract.py", "reference_execution.py", "execution_grader.py",
        "pilot_policy.py", "evaluation_checkpoint.py", "runtime_identity.py",
        "prepare_corpus_review.py", "import_human_reviews.py", "ai_semantic_review.py",
        "review_contract.py", "release_corpus_gate.py", "build_family_disjoint_corpus.py",
        "professional_capability_matrix.py",
        "build_multilingual_multistep_candidates.py",
        "build_deep_research_candidate_pack.py",
        "validate_research_candidate_pack.py",
        "prepare_research_pack_review.py",
        "kaggle_preflight.py", "kaggle_notebook.py", "build_kaggle_curriculum.py", "KAGGLE_CURRICULUM_DECISIONS.md",
    ):
        shutil.copy2(ROOT / "training" / name, output / "training" / name)
    (output / "training" / "planner_sft_seed.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    if pilot_dir:
        from training.pilot_policy import validate_pilot
        policy = validate_pilot(load_jsonl(corpus_path), corpus_path, pilot_dir / 'pilot-policy.json')
        policy['corpus_sha256'] = sha256(output / 'training' / 'planner_sft_seed.jsonl')
        write_json(output / 'training' / 'pilot-policy.json', policy)
        shutil.copy2(pilot_dir / 'numeric-evidence.json', output / 'training' / 'numeric-evidence.json')
        write_json(output / 'RUN_CONFIG.json', {
            'base_model': 'Qwen/Qwen2.5-3B-Instruct',
            'base_revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
            'epochs': 2, 'retriever': 'lexical', 'planner_memory_enabled': False,
            'deployment_allowed': False,
            'research_scope': policy['allowed_tools'],
        })

    data_dir = output / "data"
    benchmark_dir = data_dir / "real_benchmarks"
    benchmark_dir.mkdir(parents=True)
    filtered_provenance = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
    filtered_provenance["datasets"] = [item for item in filtered_provenance["datasets"] if item.get("id") in allowed_ids and kinds.get(item.get("id")) == "observed_public"]
    write_json(data_dir / "benchmark_sources.json", {"schema_version": source_manifest["schema_version"], "purpose": source_manifest["purpose"], "datasets": allowed_specs})
    filtered_provenance["manifest"] = "data/benchmark_sources.json"
    write_json(benchmark_dir / "provenance.json", filtered_provenance)
    for item in filtered_provenance["datasets"]:
        source = ROOT / Path(str(item["file"]).replace("\\", "/"))
        shutil.copy2(source, benchmark_dir / source.name)

    (output / "README_START_HERE.md").write_text(readme(), encoding="utf-8")
    shutil.copy2(ROOT / 'docs' / 'KAGGLE_TRAINING_RESEARCH_AUDIT.md', output / 'RESEARCH_AND_LIMITATIONS.md')
    shutil.copy2(ROOT / 'docs' / 'LABEL_REVIEW_STRATEGY.md', output / 'LABEL_REVIEW_STRATEGY.md')
    shutil.copy2(ROOT / 'docs' / 'KAGGLE_V3_HANDOFF.md', output / 'V3_START_HERE.md')
    shutil.copy2(ROOT / 'docs' / 'DATAPILOT_TRAINING_EVAL_DEEP_RESEARCH.md', output / 'TRAINING_EVAL_RESEARCH.md')
    shutil.copy2(ROOT / 'docs' / 'DATAPILOT_PROFESSIONAL_DS_CAPABILITY_HARDENING.md', output / 'PROFESSIONAL_DS_CAPABILITY_HARDENING.md')
    shutil.copy2(ROOT / 'docs' / 'PRETRAINING_ENGINEERING_RESEARCH_20260916.md', output / 'PRETRAINING_ENGINEERING_RESEARCH.md')
    shutil.copy2(ROOT / 'docs' / 'CHATGPT_DEEP_RESEARCH_CORPUS_PROMPT.md', output / 'CHATGPT_DEEP_RESEARCH_CORPUS_PROMPT.md')
    shutil.copy2(ROOT / 'docs' / 'datapilot_professional_capability_matrix.json', output / 'PROFESSIONAL_CAPABILITY_MATRIX.json')
    (output / "CURATION_WORKFLOW.md").write_text(curation_workflow(), encoding="utf-8")
    write_json(output / "DataPilot_Kaggle_QLoRA.ipynb", notebook_payload(output.name, pilot=pilot_dir is not None))
    if pilot_dir:
        (output / 'README_START_HERE.md').write_text(
            '# Numerically verified DataPilot research pilot\n\n'
            'This package trains only filtering, aggregation, grouping and sequential combinations. '
            'It does not train the entire DataPilot capability registry.\n\n'
            f"{policy['split_counts']} reference plans independently checked against SQL. "
            'Questions are programmatically/AI authored on six observed public datasets. '
            'Separate reviewed archetypes are supporting evidence; they are not counted as approval of these derived examples. '
            'Numeric correctness is not proof of natural-language understanding.\n\n'
            'Upload privately, attach the bundle and open DataPilot_Kaggle_QLoRA.ipynb. '
            'Enable Internet for dependencies/model download and a CUDA GPU for training. '
            'Use a fresh session; preserve all prior adapters. No automatic export or deployment.\n\n'
            'Read training/pilot-policy.json and numeric-evidence.json. '
            'INITIAL_READINESS_REPORT.json is the FULL-registry gate and is expected to remain blocked; '
            'the explicit --pilot-policy gate authorizes only this scoped research experiment. '
            'Run All checks full token lengths and runs the longest training samples through a two-step GPU smoke test. '
            'It then trains, compares base and adapter, and packages downloadable results.\n', encoding='utf-8')

    report = assess(
        records, provenance_path=benchmark_dir / "provenance.json",
        min_train=1000, min_validation=100, min_test=200,
        min_train_per_tool=25, min_clarification_train=100,
        min_distinct_train_datasets=4, min_distinct_test_datasets=2,
        real_only=True, require_disjoint_test_datasets=True,
        accepted_labels=APPROVED_LABELS | {CONTRACT_VERIFIED_LABEL},
    )
    write_json(output / "INITIAL_READINESS_REPORT.json", report)
    from training.audit_corpus_quality import audit
    quality = audit(records)
    write_json(output / 'CORPUS_QUALITY_REPORT.json', quality)
    (output / 'TRAINING_STATUS.md').write_text(
        '# Research bundle status\n\n'
        'This is not a production-ready training corpus. Read CORPUS_QUALITY_REPORT.json.\n\n'
        + '\n'.join('- ' + item for item in quality['blockers'])
        + '\n\nNo new training or export is authorized by record count alone. '
        'Full-schema prompts need a measured GPU memory smoke test before a long run.\n', encoding='utf-8')
    if review_workspace is not None:
        resolved_review = review_workspace.resolve()
        review_root = (ROOT / "training" / "review_queue").resolve()
        if review_root not in resolved_review.parents:
            raise ValueError("Review workspace must be inside training/review_queue")
        required_review_files = {"review_manifest.json", "variant_review_template.csv", "review_tasks.json", "README.md"}
        missing_review_files = sorted(name for name in required_review_files if not (resolved_review / name).is_file())
        if missing_review_files:
            raise FileNotFoundError(f"Incomplete review workspace: {missing_review_files}")
        shutil.copytree(resolved_review, output / "human_review_workspace")
    for workspace in supplemental_workspaces or []:
        resolved_workspace = workspace.resolve()
        training_root = (ROOT / "training").resolve()
        if training_root != resolved_workspace and training_root not in resolved_workspace.parents:
            raise ValueError("Supplemental workspace must be inside the project's training directory")
        if not resolved_workspace.is_dir() or not any(resolved_workspace.iterdir()):
            raise FileNotFoundError(f"Missing or empty supplemental workspace: {workspace}")
        destination = output / "supplemental_workspaces" / resolved_workspace.name
        if destination.exists():
            raise ValueError(f"Duplicate supplemental workspace name: {resolved_workspace.name}")
        shutil.copytree(resolved_workspace, destination)
    if supplemental_workspaces:
        status_path = output / 'TRAINING_STATUS.md'
        status_path.write_text(
            status_path.read_text(encoding='utf-8')
            + '\n## Supplemental verified/review workspaces\n\n'
            + '\n'.join(
                f"- `supplemental_workspaces/{workspace.resolve().name}`"
                for workspace in supplemental_workspaces
            )
            + '\n\nThese workspaces add evidence and review candidates; they do not silently promote records into the training corpus.\n',
            encoding='utf-8',
        )
    if pilot_dir:
        (output / 'TRAINING_STATUS.md').write_text(
            '# Scoped research pilot\n\nLocal corpus and numerical gates passed. '
            'GPU environment and training are not yet verified. Full-registry/production readiness is NOT granted. '
            'Use the explicit pilot notebook, not commands from historical V3 notes.\n', encoding='utf-8')
    files = [path for path in output.rglob("*") if path.is_file()]
    write_json(output / "BUNDLE_MANIFEST.json", {
        "bundle": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "real_public_dataset_ids": sorted(item["id"] for item in filtered_provenance["datasets"]),
        "contract_verified_label_count": len(records),
        "files": {str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in files},
    })
    archive = output.parent / f"{output.name}.zip"
    if defer_archive:
        return output, archive
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent))
    return output, archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / BUNDLE_NAME)
    parser.add_argument('--pilot-dir', type=Path)
    parser.add_argument('--review-workspace', type=Path)
    parser.add_argument('--supplemental-workspace', type=Path, action='append', default=[])
    parser.add_argument('--defer-archive', action='store_true', help='Run local tokenizer preflight before final packaging.')
    args = parser.parse_args()
    folder, archive = build(
        args.output, args.pilot_dir, args.review_workspace, args.supplemental_workspace, args.defer_archive
    )
    print(json.dumps({"folder": str(folder), "archive": str(archive)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
