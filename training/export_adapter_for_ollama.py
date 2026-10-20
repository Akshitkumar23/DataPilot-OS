"""Merge a reviewed PEFT adapter and prepare, but never import, an Ollama model.

Run this only after ``assess_adapter_release.py`` reports eligibility.  The
script merges an adapter into its exact base model in Hugging Face format and
writes reproducible GGUF/Ollama instructions.  It intentionally does not run
``ollama create`` because a human must inspect the converted model first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--release-decision", type=Path, required=True)
    parser.add_argument("--output-hf", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads(args.release_decision.read_text(encoding="utf-8"))
    if not decision.get("eligible_for_manual_export_review"):
        raise RuntimeError("Release decision is not eligible; refusing to merge or prepare an Ollama export.")
    if not (args.adapter / "adapter_config.json").exists():
        raise FileNotFoundError("adapter_config.json is missing; provide a PEFT adapter directory.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args.output_hf.mkdir(parents=True, exist_ok=True)
    # Merge in FP16 rather than the training-time 4-bit representation.  The
    # resulting standard HF folder can be converted and independently checked.
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    merged.save_pretrained(str(args.output_hf), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model, use_fast=True).save_pretrained(str(args.output_hf))
    (args.output_hf / "OLLAMA_CONVERSION.md").write_text(
        """# Manual Ollama conversion gate

This directory contains merged Hugging Face weights, not an Ollama model.

1. Use a pinned, current `llama.cpp` checkout to convert this directory to a
   high-fidelity GGUF (`convert_hf_to_gguf.py`).
2. Quantize the GGUF with `llama-quantize` (for example Q4_K_M after measuring
   accuracy/latency).
3. Run the held-out planner benchmark again against the converted model.
4. Only after human approval, create a local Ollama model with this Modelfile:

```
FROM ./datapilot-planner-q4_k_m.gguf
PARAMETER temperature 0
PARAMETER top_p 0.9
```

5. Set `OLLAMA_PLANNER_MODEL` explicitly to that new model name.  Do not
   overwrite the existing planner model.
""",
        encoding="utf-8",
    )
    print(json.dumps({"merged_hf": str(args.output_hf), "status": "prepared_not_imported"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
