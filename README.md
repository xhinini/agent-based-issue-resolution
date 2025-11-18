# PAGENT

Patch AGENT (PAGENT) is a type-aware patch rewriting pipeline built for SWE-bench case studies.
It augments model-generated patches with static analysis and LLM-guided type inference so that
Type & Data-Shape ("Class 1") failures can be repaired automatically and evaluated consistently.

- **Core strengths**: unified patch parsing, configurable type inference (PAGENT w/o Type Inference /
  PAGENT w/o Code Analysis / Full PAGENT), optional SWE-bench orchestration, and experiment utilities
  for profiling/ablation analysis.
- **Primary use case**: take a failing SWE-bench prediction, rerun it through Full PAGENT, and collect
  the improved patch plus telemetry that explains which variables/types were rewritten.

**Author**: Haoran Xue (hrx00@yorku.ca, [https://tracyxue.web.app/](https://tracyxue.web.app/))

**Paper**: [Type-Guided Patch Agents for SWE-bench (ArXiv: 2506.17772)](https://arxiv.org/abs/2506.17772)

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `core.py` | End-to-end writer: parses a single patch, runs type inference, rewrites, and emits metrics. |
| `type_inference_engine.py` | Implements static / LLM / hybrid inference plus context extraction. |
| `run_pagent_on_class1.py` | Batch driver that reprocesses every Class 1 failure in `taxonomy_results_voted.csv`. |
| `swebench_integration.py` | Glue code for SWE-bench runners (prepare repo, call `core.py`, return improved patch). |
| `model_failed_cases/` | Input TXT files containing descriptions and model-generated diffs. |
| `experiment_results/`, `pagent_class1_work/` | Default destinations for intermediate / final artifacts. |
| `utils/` | Helper scripts (resolved instance counts, timing stats, taxonomy sub-categories, etc.). |
| `config.json` | Central experiment knobs (LLM models, ablation selections, profiling targets). |

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

1. **Taxonomy CSV** — `taxonomy_results_voted.csv` (or another voted taxonomy) includes
   `instance_id`, `model_name`, and `class` labels.
2. **Patch cases** — `model_failed_cases/*.txt` contain per-instance descriptions, gold diffs, and
   model-generated patches. Filenames follow `{instance_id}_{Model_Name}.txt`.
3. **SWE-bench repos** — automatically cloned under `pagent_class1_work/repos/{instance_id}`; ensure
   you have network access and GitHub bandwidth.

---

## Running PAGENT

### 1. Single patch (manual invocation)
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

### 2. Reprocess every Class 1 failure
```bash
python run_pagent_on_class1.py \
  --taxonomy-csv taxonomy_results_voted.csv \
  --patches-dir ./model_failed_cases \
  --work-dir ./pagent_class1_work \
  --inference-mode hybrid \
  --dataset-name princeton-nlp/SWE-bench_Lite \
  --limit 25 --resume                 # optional knobs
```
This script:
1. Filters Class 1 entries from the taxonomy CSV
2. Parses each TXT case file to recover the model diff
3. Clones the repo at the SWE-bench base commit
4. Runs PAGENT and stores artifacts under `work-dir/pagent_outputs/`
5. Logs per-case status into `work-dir/processing_results.jsonl`

Produced files per entry:
* `{instance_id}_{Model}_improved.patch`
* `{instance_id}_{Model}_type_inference.txt` (if type payload present)
* JSON log row with metrics / errors (`status` = `success`, `clone_failed`, `pagent_failed`, etc.)

### 3. SWE-bench pipeline integration
`python swebench_integration.py` exposes a `PAGENTPipeline` class that SWE-bench agents call to
rewrite patches just-in-time. Instantiate it with the desired inference mode and pass the
`SWEBenchInstance`, repo path, and diff string; it returns the improved patch or `None`.

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

## Utilities & analysis
`utils/README_SCRIPTS.md` documents three helper programs:
1. `extract_resolved_instances.py` — gather unique successes across ablation folders
2. `calculate_timing_stats.py` — compute median/mean/P90 per pipeline phase from JSONL logs
3. `classify_subcategories.py` — enrich taxonomy rows with finer-grained Class 1 subcategories

Each script reads/writes project-root CSV/TXT files, so run them from the repo root.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `clone_failed` rows in `processing_results.jsonl` | Check GitHub connectivity, ensure the repo exists, and confirm the SWE-bench dataset item exposes `repo` + `base_commit`. Delete the partial repo under `work-dir/repos/instance` and rerun. |
| `file_not_found` / `parse_failed` | Verify the TXT file naming (`instance_model.txt`) and that the `### Model Generated Patch` section contains a unified diff. |
| `mypy` or `datasets` not found | Ensure your virtualenv is active and the dependencies listed above are installed. |
| LLM request failures | Confirm API keys in `llm_provider.py`, reduce batch size using `--limit`, or increase retry settings in `config.json`. |
| HuggingFace dataset errors | Log in via `huggingface-cli login` and set `HF_HOME` to a writable path if running inside a sandbox. |

---

## Support
If you extend PAGENT (new datasets, inference backends, patch repair heuristics), prefer adding new
scripts beside the existing ones and update this README plus `config.json` so other contributors can
reproduce your results.

---

## Results Summary

### Table 1. Distribution of failure patterns across agents

| ↓ Failure Patterns / Agents → | Agentless (GPT-4o) | Aider (GPT-4o & Claude-3) | AppMap Naive (GPT-4o) | ACR (GPT-4o) | Moatless 1 (GPT-4o) | Moatless 2 (Claude-3.5) | SWEA (Claude-3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Type & data structure | 34 | 21 | 31 | 24 | 35 | 37 | 35 |
| Contract/Architecture | 47 | 59 | 44 | 60 | 51 | 41 | 39 |
| Inadequate error handling | 25 | 21 | 19 | 20 | 20 | 26 | 26 |
| Framework Utilities | 5 | 11 | 18 | 7 | 6 | 9 | 7 |
| Version Compatibility | 3 | 2 | 2 | 3 | 2 | 1 | 7 |

### Table 4. Performance improvement with PAGENT integration

| Issue Resolution Agents | Original | Post-PAGENT | Rate Impr. | Rel. Gain |
| --- | --- | --- | --- | --- |
| Aider (GPT-4o & Claude 3 Opus) | 80/300 (26.67%) | 85/300 (28.33%) | +5 | +6.22% |
| Agentless (GPT-4o) | 82/300 (27.33%) | 97/300 (32.33%) | +15 | +18.29% |
| AutoCodeRover (GPT-4o) | 94/300 (31.33%) | 103/300 (34.33%) | +9 | +9.58% |
| Moatless Tool 1 | 35/300 (11.67%) | 46/300 (15.33%) | +11 | +31.36% |
| Moatless Tool 2 | 41/300 (13.67%) | 55/300 (18.33%) | +14 | +34.09% |
| SWE-Agent (Claude 3 Opus) | 35/300 (11.67%) | 44/300 (14.67%) | +9 | +25.71% |
| AppMap Naive (GPT-4o) | 44/300 (14.67%) | 56/300 (18.67%) | +12 | +27.27% |

### Table 5. Ablation results for PAGENT and its two variants

| Configuration | Resolved Cases | Resolution Rate |
| --- | --- | --- |
| PAGENT w/o Type Inference | 0 / 217 | 0.00% |
| PAGENT w/o Code Analysis | 1 / 217 | 0.46% |
| Full PAGENT | 75 / 217 | 34.56% |

### Table 6. Summary timing statistics

| Component | Median | Mean | P90 | Min | Max | Std. Dev. |
| --- | --- | --- | --- | --- | --- | --- |
| Static Analysis | 0.864 | 0.873 | 1.032 | 0.100 | 1.216 | 0.131 |
| Type Inference (LLM) | 4.256 | 5.990 | 11.418 | 0.200 | 42.573 | 5.522 |
| Patch Rewriting (LLM) | 3.888 | 4.373 | 6.872 | 1.827 | 11.819 | 1.900 |
