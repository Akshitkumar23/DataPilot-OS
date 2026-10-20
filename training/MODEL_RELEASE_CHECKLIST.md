# Planner adapter release checklist

An adapter is **not deployable** merely because its loss falls.  Before a human
promotes it to DataPilot, record all of the following against an untouched test
set:

1. Base-versus-adapter held-out report: exact ToolSpec-plan match, strict
   argument-schema validity, and no schema/tool-selection regression.
2. Exact column-role grounding against each test dataset schema.
3. Clarification precision: under-specified and unsupported requests must ask,
   not invent arguments.
4. Typed execution and result-manifest validation, including charts, filters,
   time ordering, and leakage interdicts.
5. Latency and repair-rate comparison with the current production planner.
6. A regression check across every previously fixed incident.
7. Conversion verification: merged Hugging Face model → GGUF → Ollama candidate
   must pass the same held-out benchmark before it is even considered.
8. Reproducibility manifest: corpus SHA-256, dataset provenance hashes, base
   model revision, adapter hyperparameters, evaluator version, and Git commit.

Promotion requires human approval after all gates pass.  Never auto-promote
from a successful run, feedback count, or training-loss metric.
