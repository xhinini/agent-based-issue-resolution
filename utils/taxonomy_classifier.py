import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
import re

# Ensure project root is on sys.path so we can import llm_provider when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_provider import (
    responses_parse,
    MODEL_GPT4O_MINI,
)

TAXONOMY_TEXT = """# Taxonomy 
1. **Type & Data-Shape/Schema Mismanagement**
   Local (or cross-lib) issues with representation/layout of data: wrong dtype/shape/nullability/index/order, parsing/encoding (e.g., dates/timezones), serialization/deserialization, axis misalignment, or schema/key/column mismatch between libraries/components.
   **Include** when the fix adds/adjusts **casts/coercions**, **validators/normalizers**, **schema declarations/adapters**, **NA handling (dropna/fillna)**, **index/sort alignment**, or **I/O parsing options**; **prefer #1** even if the misrepresentation **triggered** a protocol error.
   **Exclude** only when interaction rules are changed **independent** of data normalization.

2. **Contract/Architecture Violation**
   Breaks protocols between components (APIs, call order, return contracts), lifecycle/state machines, inheritance/dispatch, or config-driven behavior.
   **Include** when the fix rewires interfaces, adjusts call sequencing/state transitions, changes override points, or modifies configuration semantics **not primarily caused** by data representation.
   **Exclude** if the root cause is mis-typed/mis-shaped/mis-aligned data (→ #1).

3. **Fault & Edge-Condition Handling**
   Missing/incorrect guards, boundary checks, or exception flow; logic works on the common path but fails at true edges.
   **Include** when the fix adds boundary math, pre/post-conditions, or corrects exception propagation **unrelated** to data normalization.
   **Exclude** if the guard solely compensates for wrong dtype/shape/schema/NA alignment (→ #1).

4. **Framework/Abstraction Bypass**
   Hand-rolled logic that ignores existing utilities, helpers, or configuration knobs.
   **Include** when the fix replaces bespoke code with the framework’s abstraction **without changing contracts**.
   **Exclude** if the core issue is type/shape/schema/versioning (→ #1 or #5).

5. **Version/Compatibility Drift**
   Behavioral or API changes across library/runtime versions (deprecations, semantic shifts, optional deps).
   **Include** when the fix gates by version, swaps APIs due to deprecation, or adjusts for semantic changes.
   **Exclude** if version is incidental and the failure stems from data representation (→ #1).

6. **Algorithmic Inefficiency**
   Semantics are correct but too slow/memory-heavy (unnecessary passes, wrong data structure, bad complexity).
   **Include** when the fix improves performance **with identical observable behavior**.
   **Exclude** if correctness was wrong (→ other classes).

# Brief classification guideline

* Choose exactly one label using this order of tests:

  1. Does it fix **data representation/shape/schema/NA/index/parsing** (including cross-lib alignment)? → **#1**
  2. Else, does the patch change **how components interact** (protocols, lifecycle, inheritance, config semantics)? → **#2**
  3. Else, is it about **guards/exceptions/boundaries**? → **#3**
  4. Else, does it **replace custom code with a framework/helper/config**? → **#4**
  5. Else, is it **version-specific** (gates, deprecations, semantic shifts)? → **#5**
  6. Else, is it **performance-only** with same semantics? → **#6**

* Tie-break priority (when two seem plausible): **#1 > #2 > #3 > #4 > #5 > #6**.
  *If a protocol error is a downstream effect of bad dtype/shape/schema, classify as **#1**.*

* Quick cues:
  • Adds/edits `astype/convert/reshape/validate/normalize`, `dropna/fillna`, index/sort alignment, schema/adapter, parse options → **#1**
  • Rewrites call graph, changes state machine/overrides, or config semantics (not driven by data normalization) → **#2**
  • Adds `if` boundary checks, range math, or `try…except` for true edges → **#3**
  • Swaps bespoke logic to built-ins/flags/hooks only → **#4**
  • Adds version gates/alt paths for libs/runtimes → **#5**
  • Vectorizes/caches/restructures for speed with identical outputs → **#6**

Please provide analysis in the following aspects:
    1. Test Analysis:
    - Analyze test failures and their root causes
    - Identify which parts of the code are being tested
    - Compare test behavior between gold and model patches

    2. Patch Comparison:
    - Analyze syntactic and semantic differences between patches
    - Identify key changes in each patch
    - Evaluate if the model patch addresses the core issue

    3. Problem Classification:
    - Categorize the bug type (e.g., logic error, API misuse)
    - Assess required domain knowledge
    - Identify relevant dependencies and context and really understand the issue

    4. Model Performance Analysis:
    - Analyze why the model patch failed
    - Identify any patterns in the model's approach
    - Assess if the model understood the core problem

    5. Repair Strategy Analysis:
    - Compare strategies used in gold vs model patch
    - Identify missing knowledge or context
    - List required reasoning steps for correct solution

    Please be specific and provide concrete examples from the code where relevant comprehensively. You should examine the context very carefully and find out the root causes logically.
"""

# Optional model fallbacks (when a provider/free tier is unavailable)
FALLBACK_MODELS: Dict[str, str] = {
    "minimax/minimax-m2:free": "minimax/minimax-m2",
}

PROMPT_TEMPLATE = (
    "You are an expert software analysis assistant.\n"
    "Given the full case text below, identify what went wrong in the MODEL-generated patch as compared to the GOLD patch.\n"
    "Assumptions: GOLD patch is the correct resolution; MODEL patch is the one authored by the named model.\n\n"
    "Follow the taxonomy strictly and pick exactly one class (1-6), then explain concisely.\n\n"
    "TAXONOMY AND RULES:\n{taxonomy}\n\n"
    "CASE CONTEXT (instance_id={instance_id}, model_name={model_name}):\n"
    "----- BEGIN CASE TEXT -----\n{case_text}\n----- END CASE TEXT -----\n\n"
    "Output MUST follow the required schema. If uncertain, choose the single most defensible class per the tie-break rules."
)


class TaxonomyResponse(BaseModel):
    label: int = Field(description="One of 1..6, the chosen taxonomy class id")
    label_name: str = Field(description="Human-readable name for the chosen class")
    explanation: str = Field(description="Short explanation justifying the label; <= 120 words")


@dataclass
class WorkItem:
    instance_id: str
    model_name: str
    file_path: Path


def load_presence_csv(path: Path) -> Dict[str, List[str]]:
    """Return mapping instance_id -> list of model_failed_cases filenames from CSV.
    Expects columns: instance_id, model_failed_cases_files (semicolon-separated list).
    """
    mapping: Dict[str, List[str]] = {}
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get('instance_id', '').strip()
            files = row.get('model_failed_cases_files', '') or ''
            names = [s for s in (s.strip() for s in files.split(';')) if s]
            mapping[iid] = names
    return mapping


def build_worklist(presence_csv: Path, cases_dir: Path) -> List[WorkItem]:
    mapping = load_presence_csv(presence_csv)
    work: List[WorkItem] = []
    for iid, names in mapping.items():
        for name in names:
            fp = cases_dir / name
            if not fp.exists():
                continue
            # Derive model name from filename: <instance_id>_<model_name>.txt
            base = fp.stem  # without .txt
            prefix = iid + "_"
            model_name_raw = base[len(prefix):] if base.startswith(prefix) else base
            # Normalize: underscores -> spaces for readability
            model_name = model_name_raw.replace('_', ' ').strip()
            work.append(WorkItem(instance_id=iid, model_name=model_name, file_path=fp))
    return work

def _extract_relevant_content(case_text: str, max_chars: int, use_filter: bool = True) -> str:
    """Filter out noisy environment/setup logs while preserving:
    - Head metadata (Instance ID, Model, ISSUE, DESCRIPTION)
    - GOLD/MODEL/Proposed patch sections
    - Code fences (```...```), unified diffs (diff --git, @@, +++/---, +/- lines)
    - Test invocation and outputs, failures, tracebacks (pytest lines, FAILED/ERROR, E   , Traceback, assert)
    Never blindly truncate important segments; apply max_chars only after filtering.
    """
    if not use_filter:
        return case_text[:max_chars]

    lines = case_text.splitlines()

    # Patterns to always include (tests/failures)
    test_keep = re.compile(r"(pytest\s|-rA|=+\s*test session|FAILED|ERROR|E\s{2,}|Traceback|assert|short test summary info|collected\s+\d+|platform\s|plugins:)",
                           re.IGNORECASE)
    # Diff line recognition
    def is_diff_line(s: str) -> bool:
        return (s.startswith('diff --git') or s.startswith('index ') or s.startswith('--- ') or s.startswith('+++ ') or s.startswith('@@ ') or
                (len(s) > 0 and s[0] in '+- '))

    # Noisy env/setup logs to drop unless inside a kept block
    noise = [
        r"^\+?\s*source\s+/opt/miniconda3/",
        r"^\+?\s*conda\s+activate",
        r"^\+?\s*export\s+",
        r"^\+?\s*PS1=",
        r"^\+?\s*PATH=",
        r"^\+?\s*CONDA_",
        r"^\+?\s*pip\s+install",
        r"Requirement already satisfied:",
        r"^\s*(writing|reading)\s+manifest",
        r"^\s*running\s+",
        r"^\s*gcc\s+-",
        r"^\s*Installing\s+",
        r"^\s*Internet access disabled",
        r"^\s*Full Python Version:",
        r"^\s*Executable:\s",
    ]
    noise_re = re.compile("|".join(noise))

    included: List[str] = []
    in_code_fence = False
    fence_delim = None
    in_diff = False
    # Add a small header (non-noise)
    header_budget = 200
    header_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code fences
        if line.strip().startswith("```"):
            delim = line.strip()[:3]
            if not in_code_fence:
                in_code_fence = True
                fence_delim = delim
                included.append(line)
            else:
                included.append(line)
                in_code_fence = False
                fence_delim = None
            i += 1
            continue

        if in_code_fence:
            included.append(line)
            i += 1
            continue

        # Diff blocks
        if line.startswith('diff --git'):
            in_diff = True
            included.append(line)
            i += 1
            continue
        if in_diff:
            if is_diff_line(line) or line.strip() == '' or line.startswith('Checking patch') or line.startswith('Applied patch'):
                included.append(line)
                i += 1
                continue
            else:
                in_diff = False
                # fallthrough to regular processing

        # Test lines
        if test_keep.search(line):
            included.append(line)
            i += 1
            continue

        # GOLD/MODEL/Proposed sections
        if re.search(r"###\s*(Gold|GOLD|Model|Proposed).*", line):
            # Include this heading and the next ~80 lines or until next ###
            included.append(line)
            j = i + 1
            lines_taken = 0
            while j < len(lines) and lines_taken < 80:
                if lines[j].startswith('### '):
                    break
                included.append(lines[j])
                lines_taken += 1
                j += 1
            i = j
            continue

        # Brief header metadata
        if header_count < header_budget and not noise_re.search(line):
            if any(k in line for k in ["Instance ID", "# Instance ID", "Model:", "ISSUE", "DESCRIPTION", "Original Case Description"]):
                included.append(line)
                header_count += 1
                i += 1
                continue

        # Skip noisy lines
        if noise_re.search(line):
            i += 1
            continue

        i += 1

    # Join and apply final cap
    text = "\n".join(included)
    if len(text) > max_chars:
        # Keep the start and the last chunk; ensure diffs/tests kept at end are preserved
        head = text[: max_chars // 2]
        tail = text[- max_chars // 2 :]
        text = head + "\n\n...\n\n" + tail
    return text


def classify_case(instance_id: str, model_name: str, case_text: str, model_slug: str = MODEL_GPT4O_MINI, max_chars: int = 60000, retries: int = 2, use_filter: bool = True) -> TaxonomyResponse:
    candidates: List[str] = [model_slug]
    if model_slug in FALLBACK_MODELS:
        candidates.append(FALLBACK_MODELS[model_slug])
    last_err: Optional[Exception] = None
    initial_max = max_chars
    for mslug in candidates:
        # Fresh context budget per candidate
        max_chars = initial_max
        truncated = _extract_relevant_content(case_text, max_chars=max_chars, use_filter=use_filter)
        for attempt in range(retries + 1):
            try:
                prompt = PROMPT_TEMPLATE.format(
                    taxonomy=TAXONOMY_TEXT,
                    instance_id=instance_id,
                    model_name=model_name,
                    case_text=truncated,
                )
                messages = [
                    {"role": "system", "content": "You classify software repair patches using a strict taxonomy with structured outputs. Return strictly JSON matching the schema."},
                    {"role": "user", "content": prompt},
                ]
                parsed = responses_parse(
                    model=mslug,
                    input=messages,
                    text_format=TaxonomyResponse,
                    max_output_tokens=600,
                )
                if not getattr(parsed, 'output_parsed', None):
                    raise ValueError("No parsed output returned from LLM")
                return parsed.output_parsed
            except Exception as e:
                last_err = e
                if attempt < retries:
                    # Halve the max_chars progressively (bounded as configured earlier)
                    max_chars = max(60000, max_chars // 2)
                    truncated = _extract_relevant_content(case_text, max_chars=max_chars, use_filter=use_filter)
                    continue
                # Exhausted attempts for this candidate; try next candidate (fallback)
                break
    raise last_err or RuntimeError("Unknown LLM error")


def summarize(results: List[Tuple[str, str, TaxonomyResponse]]) -> Tuple[Dict[int, int], Dict[str, Dict[int, int]]]:
    overall: Dict[int, int] = {}
    by_model: Dict[str, Dict[int, int]] = {}
    for iid, model_name, resp in results:
        overall[resp.label] = overall.get(resp.label, 0) + 1
        by_model.setdefault(model_name, {})
        by_model[model_name][resp.label] = by_model[model_name].get(resp.label, 0) + 1
    return overall, by_model


def write_results_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        headers = list(rows[0].keys())
    else:
        headers = ["instance_id", "model_name", "class", "explanation"]
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_distributions(out_dir: Path, overall: Dict[int, int], by_model: Dict[str, Dict[int, int]], suffix: str = "") -> None:
    # Overall
    overall_rows = [{"class": str(k), "count": str(v)} for k, v in sorted(overall.items())]
    overall_name = f"taxonomy_distribution_overall{suffix}.csv"
    write_results_csv(out_dir / overall_name, overall_rows)
    # By model (one row per model with columns class_1..class_6)
    model_headers = ["model_name"] + [f"class_{i}" for i in range(1, 7)]
    model_rows: List[Dict[str, str]] = []
    for model_name, counts in sorted(by_model.items()):
        row = {"model_name": model_name}
        for i in range(1, 7):
            row[f"class_{i}"] = str(counts.get(i, 0))
        model_rows.append(row)
    # Write custom since we want fixed headers
    by_model_name = f"taxonomy_distribution_by_model{suffix}.csv"
    out = out_dir / by_model_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=model_headers)
        w.writeheader()
        for r in model_rows:
            w.writerow(r)


def _sanitize_model_for_filename(model_slug: str) -> str:
    return model_slug.replace('/', '_').replace(':', '_').replace(' ', '_')


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _is_error_row(row: Dict[str, str]) -> bool:
    cls = (row.get('class') or '').strip()
    exp = (row.get('explanation') or '').strip().lower()
    if not cls:
        return True
    if exp.startswith('"error') or 'validation error' in exp or 'maximum context length' in exp:
        return True
    return False


def _distributions_from_rows(rows: List[Dict[str, str]]) -> Tuple[Dict[int, int], Dict[str, Dict[int, int]]]:
    overall: Dict[int, int] = {}
    by_model: Dict[str, Dict[int, int]] = {}
    for r in rows:
        cls = (r.get('class') or '').strip()
        if not cls.isdigit():
            continue
        label = int(cls)
        model_name = r.get('model_name', '')
        overall[label] = overall.get(label, 0) + 1
        by_model.setdefault(model_name, {})
        by_model[model_name][label] = by_model[model_name].get(label, 0) + 1
    return overall, by_model


def main():
    parser = argparse.ArgumentParser(description="Classify taxonomy for model_failed_cases using LLM structured outputs")
    parser.add_argument('--instances-csv', default='instance_presence.csv', help='CSV produced earlier with model_failed_cases_files column')
    parser.add_argument('--cases-dir', default='model_failed_cases', help='Directory with *.txt case files')
    parser.add_argument('--out', default='taxonomy_results.csv', help='Output CSV for classifications')
    parser.add_argument('--dry-run', action='store_true', help='List planned items and exit without LLM calls')
    parser.add_argument('--limit', type=int, default=0, help='Optional limit of cases for a quick test')
    parser.add_argument('--max-chars', type=int, default=10000, help='Max characters from case text to include in prompt (auto-truncated)')
    parser.add_argument('--retry', type=int, default=2, help='Retries per item with progressively reduced context')
    parser.add_argument('--model', default=MODEL_GPT4O_MINI, help='LLM model slug to use (single-model mode)')
    parser.add_argument('--models', default='', help='Comma-separated list of model slugs to run sequentially (multi-model mode)')
    parser.add_argument('--rerun-errors', action='store_true', help='Rerun only rows with errors and update CSVs in-place')
    parser.add_argument('--no-filter', action='store_true', help='Disable noise filtering; use raw case text (still bounded by max-chars)')
    parser.add_argument('--verbose-errors', action='store_true', help='Print errors immediately when a row fails')
    args = parser.parse_args()

    presence_csv = Path(args.instances_csv)
    cases_dir = Path(args.cases_dir)
    out_csv = Path(args.out)

    work = build_worklist(presence_csv, cases_dir)
    if args.limit and args.limit > 0:
        work = work[: args.limit]

    if args.dry_run:
        print(f"Dry-run: {len(work)} items ready")
        preview = work[: min(10, len(work))]
        for w in preview:
            print(f"- {w.instance_id} | {w.model_name} | {w.file_path}")
        return

    # Multi-model mode
    models_arg = [m.strip() for m in args.models.split(',') if m.strip()]
    if models_arg:
        if args.rerun_errors:
            # Update only error rows in per-model CSVs (derived from --out pattern)
            for mslug in models_arg:
                suffix = "_" + _sanitize_model_for_filename(mslug)
                per_model_out = out_csv.parent / f"{out_csv.stem}{suffix}{out_csv.suffix}"
                if not per_model_out.exists():
                    print(f"Skip: {per_model_out} not found for rerun-errors")
                    continue
                rows = _read_csv_rows(per_model_out)
                # Build key set to rerun
                error_keys = {(r.get('instance_id',''), r.get('model_name','')) for r in rows if _is_error_row(r)}
                if not error_keys:
                    print(f"No errors to rerun in {per_model_out}")
                    continue
                # Build worklist and filter to keys
                full_work = build_worklist(presence_csv, cases_dir)
                work = [w for w in full_work if (w.instance_id, w.model_name) in error_keys]
                print(f"Rerunning {len(work)} error rows in {per_model_out}")
                # Run and update
                updates: Dict[Tuple[str,str], Dict[str,str]] = {}
                for item in work:
                    try:
                        text = item.file_path.read_text(encoding='utf-8', errors='ignore')
                        resp = classify_case(item.instance_id, item.model_name, text, model_slug=mslug, max_chars=args.max_chars, retries=args.retry, use_filter=(not args.no_filter))
                        updates[(item.instance_id, item.model_name)] = {
                            'class': str(resp.label),
                            'explanation': f'"{resp.explanation.replace("\n", " ").strip()}"',
                        }
                    except Exception as e:
                        if args.verbose_errors:
                            print(f"[ERROR] {mslug} | {item.instance_id} | {item.model_name}: {e}")
                        updates[(item.instance_id, item.model_name)] = {
                            'class': '',
                            'explanation': f'"ERROR: {str(e)}"',
                        }
                # Apply updates to rows
                for r in rows:
                    key = (r.get('instance_id',''), r.get('model_name',''))
                    if key in updates:
                        r['class'] = updates[key]['class']
                        r['explanation'] = updates[key]['explanation']
                write_results_csv(per_model_out, rows)
                overall, by_model = _distributions_from_rows(rows)
                write_distributions(per_model_out.parent, overall, by_model, suffix=suffix)
                print(f"Updated: {per_model_out}")
            return

        for mslug in models_arg:
            print(f"\n=== Running taxonomy classification with model: {mslug} ===")
            results_rows: List[Dict[str, str]] = []
            collected: List[Tuple[str, str, TaxonomyResponse]] = []
            for item in work:
                try:
                    text = item.file_path.read_text(encoding='utf-8', errors='ignore')
                    resp = classify_case(item.instance_id, item.model_name, text, model_slug=mslug, max_chars=args.max_chars, retries=args.retry, use_filter=(not args.no_filter))
                    results_rows.append({
                        'instance_id': item.instance_id,
                        'model_name': item.model_name,
                        'class': str(resp.label),
                        'explanation': f'"{resp.explanation.replace("\n", " ").strip()}"',
                    })
                    collected.append((item.instance_id, item.model_name, resp))
                except Exception as e:
                    if args.verbose_errors:
                        print(f"[ERROR] {mslug} | {item.instance_id} | {item.model_name}: {e}")
                    results_rows.append({
                        'instance_id': item.instance_id,
                        'model_name': item.model_name,
                        'class': '',
                        'explanation': f'"ERROR: {str(e)}"',
                    })

            suffix = "_" + _sanitize_model_for_filename(mslug)
            per_model_out = out_csv.parent / f"{out_csv.stem}{suffix}{out_csv.suffix}"
            write_results_csv(per_model_out, results_rows)
            overall, by_model = summarize(collected)
            write_distributions(per_model_out.parent, overall, by_model, suffix=suffix)
            print(f"Wrote classifications to: {per_model_out}")
            print(f"Distributions saved with suffix '{suffix}' in {per_model_out.parent}")
        return

    # Single-model mode (default)
    if args.rerun_errors:
        target = out_csv
        if not target.exists():
            print(f"Skip: {target} not found for rerun-errors")
            return
        rows = _read_csv_rows(target)
        error_keys = {(r.get('instance_id',''), r.get('model_name','')) for r in rows if _is_error_row(r)}
        if not error_keys:
            print(f"No errors to rerun in {target}")
            return
        full_work = build_worklist(presence_csv, cases_dir)
        work = [w for w in full_work if (w.instance_id, w.model_name) in error_keys]
        print(f"Rerunning {len(work)} error rows in {target}")
        updates: Dict[Tuple[str,str], Dict[str,str]] = {}
        for item in work:
            try:
                text = item.file_path.read_text(encoding='utf-8', errors='ignore')
                resp = classify_case(item.instance_id, item.model_name, text, model_slug=args.model, max_chars=args.max_chars, retries=args.retry, use_filter=(not args.no_filter))
                updates[(item.instance_id, item.model_name)] = {
                    'class': str(resp.label),
                    'explanation': f'"{resp.explanation.replace("\n", " ").strip()}"',
                }
            except Exception as e:
                if args.verbose_errors:
                    print(f"[ERROR] {args.model} | {item.instance_id} | {item.model_name}: {e}")
                updates[(item.instance_id, item.model_name)] = {
                    'class': '',
                    'explanation': f'"ERROR: {str(e)}"',
                }
        for r in rows:
            key = (r.get('instance_id',''), r.get('model_name',''))
            if key in updates:
                r['class'] = updates[key]['class']
                r['explanation'] = updates[key]['explanation']
        write_results_csv(target, rows)
        overall, by_model = _distributions_from_rows(rows)
        write_distributions(target.parent, overall, by_model)
        print(f"Updated: {target}")
        return

    results_rows: List[Dict[str, str]] = []
    collected: List[Tuple[str, str, TaxonomyResponse]] = []
    for item in work:
        try:
            text = item.file_path.read_text(encoding='utf-8', errors='ignore')
            resp = classify_case(item.instance_id, item.model_name, text, model_slug=args.model, max_chars=args.max_chars, retries=args.retry, use_filter=(not args.no_filter))
            results_rows.append({
                'instance_id': item.instance_id,
                'model_name': item.model_name,
                'class': str(resp.label),
                'explanation': f'"{resp.explanation.replace("\n", " ").strip()}"',
            })
            collected.append((item.instance_id, item.model_name, resp))
        except Exception as e:
            if args.verbose_errors:
                print(f"[ERROR] {args.model} | {item.instance_id} | {item.model_name}: {e}")
            results_rows.append({
                'instance_id': item.instance_id,
                'model_name': item.model_name,
                'class': '',
                'explanation': f'"ERROR: {str(e)}"',
            })

    write_results_csv(out_csv, results_rows)
    overall, by_model = summarize(collected)
    write_distributions(out_csv.parent, overall, by_model)
    print(f"Wrote classifications to: {out_csv}")
    print(f"Overall distribution and per-model breakdown saved next to the CSV.")


if __name__ == '__main__':
    main()
