# Planner training and evaluation corpus

This folder is for **offline model development only**.  It is never imported
by the desktop application, never contributes runtime routing rules, and never
allows a model to modify its own implementation.

`planner_sft_seed.jsonl` contains a small, schema-valid seed set grounded in
the exact columns of the versioned real benchmark datasets in
`data/real_benchmarks/`.  It teaches the desired output *shape*: a user query
must become either one strict ToolSpec command or an explicit clarification.
It is not enough data to train or promote an LLM.

## Curation gate

After editing the corpus, run:

```powershell
.venv\Scripts\python.exe scripts\validate_planner_corpus.py
```

The gate checks all dataset IDs and required columns against
`data/real_benchmarks/provenance.json`, validates each tool call against the
live `CapabilityRegistry`, and rejects malformed JSONL.  A passing gate only
means the labels are mechanically valid; a data scientist must still review
the analytical intent and safety implications.

## Correct training process

1. Grow this into at least 1,000 human-reviewed examples spanning every
   supported ToolSpec, paraphrases, multi-turn clarifications, invalid column
   names, ambiguous requests, and negative/safety cases.
2. Keep a fixed, untouched test set by dataset and query family.  Do not use
   it in training or prompt tuning.
3. Run `modules/planner_benchmark.py` against the active baseline and the
   candidate model.  Promotion remains a human decision gated by plan accuracy,
   column grounding, safety behaviour, and latency—not a self-training loop.
4. Fine-tune only in a separately approved GPU environment using public or
   properly anonymized training data.  Do not upload customer datasets or
   prompts.  Ship the reviewed adapter/model for local inference afterward.
5. Re-run the full regression suite and package smoke test before release.

## Real-public-data-only Kaggle workspace

Run `scripts/build_kaggle_training_bundle.py` to create a private upload ZIP on
the Desktop.  It packages only documented observed-public UCI data, copies a
real-only seed corpus, and excludes the AI4I synthetic reference dataset,
unreviewed queue candidates, local models, customer CSVs, Electron builds, and
credentials.  The generated readiness gate requires 1,000 reviewed train,
100 validation, and 200 test examples; at least 25 reviewed training examples
for each live analytical ToolSpec; 100 clarification examples; and disjoint
training/test datasets.  These thresholds are deliberately stricter than a
single successful demo.

The current laptop has an integrated AMD GPU and is suitable for local Ollama
inference, local semantic retrieval, corpus generation, and evaluation work,
but not a practical CUDA/QLoRA training host.  This separation preserves local
customer-data privacy while avoiding a fake "trained" model built from a few
CSV files.

## CPU-capable learning now

DataPilot now compiles reviewed labels into
`planner_pattern_memory.jsonl`. The desktop planner retrieves only redacted,
approved intent-to-tool *shapes* through local `nomic-embed-text` embeddings.
This is CPU-friendly continual learning: it improves interpretation of similar
future requests without letting past labels inject columns, filter values, or
commands into a new dataset.

To create more review candidates locally, use
`generate_planner_review_queue.py`. Synthetic candidates are never considered
training labels until a human marks them `human-reviewed`; this protects the
system from learning its own planner mistakes.

## One-time adapter training

`train_planner_adapter.py` is intentionally separate from the desktop runtime.
First use its safe, no-download/no-training preflight:

```powershell
.venv\Scripts\python.exe training\train_planner_adapter.py --dry-run
```

It only permits actual training after all of these conditions are true:

- an approved CUDA GPU host has the isolated dependencies from
  `training/requirements-train.txt`;
- the corpus has at least 1,000 reviewed training records plus non-empty
  validation and test splits;
- the caller explicitly supplies `--confirm-gpu-training`.

The resulting adapter is written as `trained_unpromoted`.  Follow
`MODEL_RELEASE_CHECKLIST.md` before converting or deploying it with Ollama.
