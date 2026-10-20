# Kaggle curriculum delivery — 2026-09-14

## What this experiment can establish

The base model already knows language. This SFT experiment teaches the exact current
DataPilot planning protocol: schema-grounded columns, filter predicates, aggregation,
dependent filter-to-aggregation calls, and explicit bar/line output contracts.
It cannot create operators that the engine does not implement, replace statistical
judgement, or prove that arbitrary future requests will succeed.

The supported numerical pilot is intentionally distinct from the full-registry release
gate. The notebook may train an unpromoted research adapter, but must not mark it
production-ready or activate it in Ollama. The complete registry remains available in
the runtime prompt; answers are not fed into retrieval or used to restrict the grammar.

## Evidence, not inflated labels

- Source CSVs are documented observed public datasets, with source and licence metadata.
- New English/Hindi/Hinglish requests are programmatic/AI-authored, not real customer logs.
- User-supplied human attestations apply to 480 approved abstract archetypes. The 150
  needs-changes records remain separate. Approval does not transfer to bound derivatives.
- Every executable reference is compared with independent DuckDB SQL on the actual CSV.
  Result hashes, source hashes, command hashes and execution manifests accompany each case.
- Translations and chart variants inherit the original task-family split. Test datasets
  do not occur in training or validation. More paraphrases do not mean more independent plans.

## Previous failure modes addressed

1. Trainer/evaluator no longer immediately reject the explicit pilot policy.
2. Full current runtime prompts replace incompatible scoped-registry prompt wrappers.
3. Tokenizer measures every full prompt/completion before allocating training weights.
   No truncation of schema, columns or answers is accepted.
4. A separate two-step GPU smoke exercises the longest training examples. Long training
   begins only after a finite-loss smoke; GPU memory is measured, not assumed.
5. Training/evaluation use fresh Python processes, a pinned base revision and matching
   Transformers/TRL/PEFT packages. Optional torchao is not used for NF4 training.
6. Partial evaluation is checkpointed per example. Resume verifies case order, prompt,
   runtime configuration, base revision and adapter identity. Completed stale results
   are rejected, never silently reused.
7. Adapter and base use the same held-out test cases. Numeric oracle verification is
   separate from model exact-plan scoring; exact-plan scoring may reject equivalent plans.
8. No automatic export, activation, production promotion, or old-adapter deletion.

## What still requires Kaggle

CUDA package import, actual longest-input memory fit, finite forward/backward loss,
full training, held-out base/adapter generation and saved output verification. CPU-local
preflight cannot certify these GPU operations. A Kaggle interruption before the first
checkpoint requires a new output directory; partial adapters must be preserved.

## What is outside this curriculum

Supervised ML, leakage-aware prediction design, forecasting, event windows, causal
inference, all clarification/failure families, and arbitrary multi-step programs are
not trained by this numerical pilot. Existing reviewed templates for those families
remain useful curation inputs, not certified executable training labels. After this
experiment, add independently executed families and unseen-query evaluations before
claiming broader coverage. Human review cannot be manufactured by an AI process.

## References used for the implementation

- [TRL 0.22.2 SFT Trainer](https://huggingface.co/docs/trl/v0.22.2/en/sft_trainer): native
  conversational prompt/completion datasets, completion-only loss, explicit length handling.
- [PEFT quantization guide](https://huggingface.co/docs/peft/main/developer_guides/quantization):
  quantized-base adapter training rather than updating quantized base weights directly.
- [PEFT 0.17.1 backend checks](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/import_utils.py):
  optional torchao compatibility must not break a bitsandbytes workflow.
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct): immutable checkpoint
  revision is stored in RUN_CONFIG.json for the tokenizer, trainer and both evaluations.

Pinned versions make the experiment reproducible; they are not a claim that the packages
are the newest, or that this exact stack has already run on the user's Kaggle GPU.
