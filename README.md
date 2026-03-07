# PAGENT

Patch AGENT (PAGENT) is a type-aware patch rewriting pipeline built for SWE-bench case studies.
It augments model-generated patches with static analysis and LLM-guided type inference so that
Type & Data-Shape ("Class 1") failures can be repaired automatically and evaluated consistently.

- **Core strengths**: unified patch parsing, configurable type inference (PAGENT w/o Type Inference /
  PAGENT w/o Code Analysis / Full PAGENT), optional SWE-bench orchestration, and experiment utilities
  for profiling/ablation analysis.
- **Primary use case**: take a failing SWE-bench prediction, rerun it through Full PAGENT, and collect
  the improved patch plus telemetry that explains which variables/types were rewritten.


---

## Prerequisites

* Python 3.10+
* Git (PAGENT clones SWE-bench repos at specific commits)
* Access to the HuggingFace `princeton-nlp/SWE-bench_Lite` dataset (set `HF_HOME` / token as needed)
* Mypy executable on PATH (used by the static inference pass)
* LLM credentials for the provider configured in `llm_provider.py`

### Python environment
```bash
python -m venv .venv
.venv\Scripts\activate                  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt          # if provided
# otherwise install the observed runtime deps:
pip install datasets tqdm pydantic mypy networkx rich
```
Add additional libraries if `ImportError`s appear; `core.py` and `utils/*.py` print clear names when
requirements are missing.

---

## Input assets

1. **Patch cases** — `model_failed_cases/*.txt` contain per-instance descriptions, gold diffs, and
   model-generated patches. Filenames follow `{instance_id}_{Model_Name}.txt`.
2. **SWE-bench repos** — automatically cloned under `pagent_class1_work/repos/{instance_id}`; ensure
   you have network access and GitHub bandwidth.

---

## Running PAGENT

### Single patch (manual invocation)
Use this when you already have a local repo checkout and a diff to improve:
```bash
python core.py \
  --codebase /path/to/repo \
  --patch /tmp/model.patch \
  --output /tmp/improved.patch \
  --inference-mode hybrid            # Full PAGENT (default)
  # use --inference-mode static (PAGENT w/o Type Inference)
  # or --inference-mode llm (PAGENT w/o Code Analysis)
```
Optional: `--variable my_var` seeds the analyzer with an extra symbol; `--extra-context-file` attaches
sideband reasoning text (only meaningful in hybrid mode).

Results printed to stdout include:
* `PAGENT_TYPES_JSON` — type inference payload (variables, sources, confidences)
* `PAGENT_METRICS_JSON` — timing + LLM call statistics

---

## Ablation modes
`type_inference_engine.py` exposes `InferenceMode` with three options designed for the paper’s ablation
study. Any entry point that runs `core.py` accepts `--inference-mode` (flag names shown in parentheses):

| Configuration | CLI flag | Description |
| --- | --- | --- |
| PAGENT w/o Type Inference | `static` | Disables type inference; relies only on AST/CFG heuristics and mypy insights. |
| PAGENT w/o Code Analysis | `llm` | Disables static analysis and relies on LLM reasoning over extracted code slices. |
| Full PAGENT | `hybrid` | Combines static analysis with LLM validation and arbitration. |

To run an ablation sweep, execute the same script three times while switching the flag and collect the
resulting improved patches / metrics. Example:
```bash
for mode in static llm hybrid; do
  python run_pagent_on_class1.py --inference-mode "$mode" --limit 50 --work-dir ./runs/$mode
done
```

`config.json` holds additional experiment blocks (`b2_ablation`, `b3_profiling`, etc.) if you need to
automate sweeps programmatically.

---

