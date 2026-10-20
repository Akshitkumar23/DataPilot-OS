"""One-time, auditable QLoRA training entry point for the DataPilot planner.

This script is intentionally not imported by the desktop product.  It can only
run after a caller explicitly confirms GPU training, and it refuses a CPU or
non-CUDA host rather than silently producing an impractically slow run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_planner_corpus import DEFAULT_CORPUS, DEFAULT_PROVENANCE, load_jsonl, validate_records
from scripts.planner_training_readiness import APPROVED_LABELS, CONTRACT_VERIFIED_LABEL, assess
from training.planner_prompt import build_prompt, check_token_budget, PROMPT_VERSION
from training.audit_corpus_quality import audit
from training.release_corpus_gate import assess_release_corpus
from modules.planner_prompt_contract import RUNTIME_PROMPT_VERSION as PROMPT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training" / "artifacts" / "planner-qlora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument('--max-steps', type=int, default=-1, help='Positive value runs a smoke test; never release evidence.')
    parser.add_argument('--base-revision', default='main')
    parser.add_argument('--resume', action='store_true', help='Resume a matching checkpoint in output-dir.')
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=12.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate corpus and print the intended run; do not import GPU libraries or train.")
    parser.add_argument('--pilot-policy', type=Path, help='Explicit limited-scope numerical research pilot, never production approval.')
    parser.add_argument("--confirm-gpu-training", action="store_true", help="Required acknowledgement before weights can be changed.")
    parser.add_argument(
        "--allow-contract-verified",
        action="store_true",
        help="Permit contract-verified labels for a research-only, unpromoted adapter. Human-reviewed labels remain required for production promotion.",
    )
    return parser.parse_args()


def load_and_validate(corpus_path: Path, provenance_path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(corpus_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_records(records, provenance)
    return records


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_dataset(messages: list[list[dict[str, Any]]]) -> dict[str, list]:
    """Keep the native model template; supervise the final assistant response."""
    if any(not turns or turns[-1].get("role") != "assistant" or not turns[-1].get("content") for turns in messages):
        raise ValueError("Each training example requires a non-empty final assistant response.")
    return {"prompt": [turns[:-1] for turns in messages], "completion": [turns[-1:] for turns in messages]}


def preflight_cuda(min_memory_gib: float) -> tuple[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training dependencies are missing. Install training/requirements-train.txt in a dedicated GPU environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Refusing CPU fine-tuning; use an approved CUDA training host.")
    device = torch.cuda.get_device_properties(0)
    memory_gib = device.total_memory / (1024 ** 3)
    if memory_gib < min_memory_gib:
        raise RuntimeError(f"GPU '{device.name}' has {memory_gib:.1f} GiB VRAM; this run requires at least {min_memory_gib:.1f} GiB.")
    return device.name, memory_gib


def main() -> int:
    args = parse_args()
    # This entry point is single-GPU. Avoid Trainer's implicit DataParallel
    # replicating a quantized PEFT model when Kaggle exposes two T4 devices.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.epochs <= 0 or args.max_seq_length < 256 or args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("Training hyperparameters must be positive; max-seq-length must be at least 256.")
    records = load_and_validate(args.corpus, args.provenance)
    split_counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")}
    print(json.dumps({"corpus_records": len(records), "splits": split_counts, "base_model": args.base_model}, indent=2))
    pilot = None
    if args.pilot_policy:
        from training.pilot_policy import validate_pilot
        pilot = validate_pilot(records, args.corpus, args.pilot_policy)
    if args.dry_run:
        quality = audit(records)
        print(json.dumps(quality, indent=2))
        print("DRY RUN: no model weights, adapters, or runtime configuration were changed.")
        return 0 if quality['ready_for_new_training'] else 2
    if not args.confirm_gpu_training:
        raise RuntimeError("Pass --confirm-gpu-training only after human review of corpus, licences, and held-out evaluation plan.")
    readiness = assess(
        records,
        provenance_path=args.provenance,
        min_train=1000,
        min_validation=100,
        min_test=200,
        min_train_per_tool=25,
        min_clarification_train=100,
        min_distinct_train_datasets=4,
        min_distinct_test_datasets=2,
        real_only=True,
        require_disjoint_test_datasets=True,
        accepted_labels=APPROVED_LABELS | ({CONTRACT_VERIFIED_LABEL} if args.allow_contract_verified else set()),
    )
    if not pilot and not readiness["ready"]:
        raise RuntimeError("Corpus is not training-ready:\n- " + "\n- ".join(readiness["blockers"]))
    quality = audit(records)
    if not quality['ready_for_new_training']:
        raise RuntimeError('CORPUS_QUALITY_FAILED: ' + '; '.join(quality['blockers']))
    if not pilot and not args.allow_contract_verified:
        release_gate = assess_release_corpus(records)
        if not release_gate['ready']:
            raise RuntimeError('PRODUCTION_CORPUS_GATE_FAILED:\n- ' + '\n- '.join(release_gate['blockers']))

    device_name, memory_gib = preflight_cuda(args.min_gpu_memory_gib)
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from modules.capability_registry import CapabilityRegistry
    registry = CapabilityRegistry(auto_bootstrap=True)
    # Pilot scope limits reference labels, never the runtime prompt/registry.
    provenance = json.loads(args.provenance.read_text(encoding='utf-8'))
    from training.prepare_runtime_corpus import RuntimeCorpusBuilder
    builder = RuntimeCorpusBuilder(provenance, registry)
    prepared = []
    prepared_evidence = []
    for record in records:
        if record['split'] in {'train', 'validation'}:
            example = builder.example(record)
            prepared_evidence.append({key: example[key] for key in
                ('case_id', 'split', 'source_record_sha256', 'prompt_sha256', 'response_schema_sha256')})
            prepared.append((record['split'], example['prompt'] + example['completion']))
    train_records = [messages for split, messages in prepared if split == 'train']
    validation_records = [messages for split, messages in prepared if split == 'validation']
    from training.runtime_identity import runtime_identity
    contract_identity = runtime_identity()
    input_contract = {
        'runtime_contract': contract_identity, 'cases': prepared_evidence,
        'base_model': args.base_model, 'base_revision': args.base_revision,
        'max_seq_length': args.max_seq_length, 'seed': args.seed,
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'max_steps': args.max_steps,
        'status': 'inputs_prepared_not_trained'}
    resume_checkpoint = None
    if args.output_dir.exists():
        if not args.resume or (args.output_dir / 'run_manifest.json').exists():
            raise FileExistsError('Use a fresh output directory or --resume for an interrupted matching run.')
        previous = json.loads((args.output_dir / 'input_contract.json').read_text(encoding='utf-8'))
        if previous != input_contract:
            raise ValueError('TRAINING_RESUME_CONTRACT_MISMATCH')
        from transformers.trainer_utils import get_last_checkpoint
        resume_checkpoint = get_last_checkpoint(str(args.output_dir))
        if not resume_checkpoint:
            raise ValueError('No saved checkpoint found; preserve this failed-run directory and choose a new output-dir.')
    else:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / 'input_contract.json').write_text(json.dumps(input_contract, indent=2), encoding='utf-8')
    import torch
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for _, messages in prepared:
        check_token_budget(tokenizer, messages, args.max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, revision=args.base_revision, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    if args.max_steps > 0:
        # Exercise the longest full prompt during the GPU smoke test.
        train_records.sort(key=lambda turns: len(tokenizer.apply_chat_template(turns, tokenize=True)), reverse=True)
    peft_config = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    run_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=2e-4,
        max_length=args.max_seq_length,
        # Prompt/completion masking works with the native Qwen2.5 template.
        completion_only_loss=True,
        eos_token="<|im_end|>",
        packing=False,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        seed=args.seed,
        eval_strategy="no" if args.max_steps > 0 else "steps",
        eval_steps=50,
        save_strategy="no" if args.max_steps > 0 else "steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=args.max_steps <= 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=run_args,
        train_dataset=Dataset.from_dict(completion_dataset(train_records)),
        eval_dataset=Dataset.from_dict(completion_dataset(validation_records)),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if args.max_steps > 0:
        # Sequential sampling ensures the first smoke batches include the worst case.
        from torch.utils.data import SequentialSampler
        trainer._get_train_sampler = lambda *a, **kw: SequentialSampler(trainer.train_dataset)
    # T4 mixed precision needs FP32 optimizer parameters, even if PEFT/TRL
    # initialized adapters from a base configuration that advertises BF16.
    for parameter in trainer.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    print("Trainable parameter dtypes:", sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}), flush=True)
    print(f"Training on {device_name} ({memory_gib:.1f} GiB VRAM).")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    import math
    if not math.isfinite(train_result.training_loss):
        raise RuntimeError('NONFINITE_TRAINING_LOSS: preserve logs and diagnose before rerunning.')
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30).splitlines()
    except Exception:
        pip_freeze = []
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "research_scope": pilot['allowed_tools'] if pilot else 'full-registry',
        "deployment_allowed": False,
        "prompt_version": PROMPT_VERSION,
        "runtime_contract": contract_identity,
        "corpus": str(args.corpus),
        "split_counts": split_counts,
        "epochs": args.epochs,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "completion_only_loss": True,
        "packing": False,
        "training_script_sha256": sha256_file(Path(__file__)),
        "gpu": device_name,
        "gpu_memory_gib": memory_gib,
        "status": "smoke_only" if args.max_steps > 0 else "trained_unpromoted",
        "max_steps": args.max_steps,
        "training_loss": train_result.training_loss,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "label_tier": "contract-verified-allowed" if args.allow_contract_verified else "human-reviewed-only",
        "corpus_sha256": sha256_file(args.corpus),
        "provenance_sha256": sha256_file(args.provenance),
        "python_version": sys.version,
        "pip_freeze": pip_freeze,
        "promotion_requirement": "Run fixed held-out planner benchmark and obtain human approval before deployment. Contract-verified labels alone never qualify an adapter for production.",
    }, indent=2), encoding="utf-8")
    print('SMOKE COMPLETE' if args.max_steps > 0 else 'TRAINING COMPLETE: adapter awaits held-out evaluation.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
