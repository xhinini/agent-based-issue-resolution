"""
Script to classify patches into subcategories using LLM with GPT-5-mini.
Reads taxonomy_results_voted.csv and classifies each patch into a subcategory
based on its voted class.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_provider import responses_parse, MODEL_GPT5_MINI


# Subcategory definitions by class
SUBCATEGORY_DEFINITIONS = {
    "1": {
        "name": "Type & Data-Shape/Schema Mismanagement",
        "subcategories": [
            "Basic type conversions (string to integer, etc.)",
            "Complex data structure manipulations (numpy array to pandas dataframe, etc.)",
            "Missing type validation (dtype/nullable dtypes, shape/axis/index alignment, numeric precision/range)"
        ]
    },
    "2": {
        "name": "Contract/Architecture Violation",
        "subcategories": [
            "Inheritance/Dispatch Misunderstanding (override/abstract method contracts)",
            "Protocol & Return-Contract Issues (call order, expected outputs)",
            "Lifecycle/Config Semantics (state, init/finalize, feature flags)"
        ]
    },
    "3": {
        "name": "Fault & Edge-Condition Handling",
        "subcategories": [
            "Exception Handling & Propagation (catch/raise, preserving cause)",
            "Boundary & Edge Conditions (off-by-one, empty/rectangular inputs)",
            "Incomplete Fix Scope (guards added in one place but missed siblings/callers)"
        ]
    },
    "4": {
        "name": "Framework/Abstraction Bypass",
        "subcategories": [
            "Reimplementing Existing Helpers (reinvented wheels)/Creating Redundant methods",
            "Ignoring Framework-Specific Patterns"
        ]
    },
    "5": {
        "name": "Version/Compatibility Drift",
        "subcategories": [
            "Misunderstanding API Changes Across Versions",
            "Missing Version Gates / Deprecated Features (no compatibility path)"
        ]
    }
}


PROMPT_TEMPLATE = """You are an expert software patch analysis assistant.

Given a patch classified as CLASS {class_num} ({class_name}), determine which subcategory best describes the issue.

IMPORTANT RULES:
1. You MUST NOT alter or change the class ({class_num}) provided. Accept it as given.
2. Choose exactly ONE subcategory from the list below that best matches the patch.
3. Reply with strictly JSON matching the schema: {{"subcategory": "<exact subcategory text>"}}.
4. The subcategory MUST match exactly one of the options below.
5. Tie-break rule: if multiple options are plausible, choose the earliest option in the provided list.

The subcategories for this class are:

{subcategories}

CASE INFORMATION:
Instance ID: {instance_id}
Model Name: {model_name}
Class: {class_num} - {class_name}

PATCH CONTENT:
{patch_content}
"""


class SubcategoryResponse(BaseModel):
    subcategory: str = Field(description="The chosen subcategory text (must match exactly one from the provided list)")
    explanation: Optional[str] = Field(default="", description="Optional brief explanation; may be omitted")


def load_voted_results(csv_path: Path) -> List[Dict[str, str]]:
    """Load the taxonomy_results_voted.csv file."""
    results = []
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    return results


def load_patch_content(instance_id: str, model_name: str, cases_dir: Path) -> str:
    """
    Load patch content from model_failed_cases directory.
    Expected filename format: {instance_id}_{model_name}.txt
    """
    # Normalize model name to match filename format
    model_filename = model_name.replace(' ', '_')
    filename = f"{instance_id}_{model_filename}.txt"
    filepath = cases_dir / filename
    
    if not filepath.exists():
        return f"[Patch file not found: {filename}]"
    
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        # Limit content to avoid token limits
        if len(content) > 50000:
            content = content[:25000] + "\n\n...(truncated)...\n\n" + content[-25000:]
        return content
    except Exception as e:
        return f"[Error reading patch file: {e}]"


def classify_subcategory(
    instance_id: str,
    model_name: str,
    class_num: str,
    patch_content: str,
    model_slug: str = MODEL_GPT5_MINI,
    retries: int = 2,
    temperature: Optional[float] = None,
    subcategory_order: Optional[List[str]] = None,
) -> SubcategoryResponse:
    """
    Classify a patch into a subcategory using LLM.
    """
    if class_num not in SUBCATEGORY_DEFINITIONS:
        raise ValueError(f"Unknown class: {class_num}")
    
    class_info = SUBCATEGORY_DEFINITIONS[class_num]
    class_name = class_info["name"]
    # Allow override of subcategory order for balancing
    subcategories = subcategory_order if subcategory_order else class_info["subcategories"]
    
    # Format subcategories for prompt
    subcategories_text = "\n".join(f"{i+1}. {sub}" for i, sub in enumerate(subcategories))
    
    prompt = PROMPT_TEMPLATE.format(
        class_num=class_num,
        class_name=class_name,
        subcategories=subcategories_text,
        instance_id=instance_id,
        model_name=model_name,
        patch_content=patch_content
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a software patch analysis expert. Classify patches into subcategories using structured outputs. Return strictly JSON with field 'subcategory' only."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    last_error = None
    for attempt in range(retries + 1):
        try:
            parsed = responses_parse(
                model=model_slug,
                input=messages,
                text_format=SubcategoryResponse,
                temperature=temperature,
                max_output_tokens=400
            )
            
            if not getattr(parsed, 'output_parsed', None):
                raise ValueError("No parsed output returned from LLM")
            resp: SubcategoryResponse = parsed.output_parsed
            # Validate non-empty and matching one of provided subcategories
            if not resp.subcategory or resp.subcategory not in subcategories:
                raise ValueError("LLM returned empty or non-matching subcategory")
            return resp
            
        except Exception as e:
            last_error = e
            if attempt < retries:
                # Retry with slightly reduced content and reinforced instruction
                if len(patch_content) > 10000:
                    patch_content = patch_content[:5000] + "\n...(truncated)...\n" + patch_content[-5000:]
                    prompt = PROMPT_TEMPLATE.format(
                        class_num=class_num,
                        class_name=class_name,
                        subcategories=subcategories_text,
                        instance_id=instance_id,
                        model_name=model_name,
                        patch_content=patch_content
                    )
                    messages[1]["content"] = prompt
                continue
    
    raise last_error or RuntimeError("Unknown LLM error")


def write_results_csv(output_path: Path, results: List[Dict[str, str]]):
    """Write results to CSV."""
    if not results:
        print("No results to write")
        return
    
    fieldnames = ['instance_id', 'model_name', 'class', 'subcategory']
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Only keep required fields
            writer.writerow({k: row.get(k, '') for k in fieldnames})


# --- Balancing helpers for Class 1 reruns ---
def _count_class1_distributions(rows: List[Dict[str, str]]) -> (Dict[str, int], Dict[str, Dict[str, int]]):
    overall: Dict[str, int] = {}
    by_model: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if (r.get('class') or '').strip() != '1':
            continue
        sub = (r.get('subcategory') or '').strip()
        if not sub:
            continue
        overall[sub] = overall.get(sub, 0) + 1
        m = (r.get('model_name') or '').strip()
        by_model.setdefault(m, {})
        by_model[m][sub] = by_model[m].get(sub, 0) + 1
    return overall, by_model


def _balanced_order_for_class1(model_name: str, existing_rows: List[Dict[str, str]]) -> List[str]:
    base = SUBCATEGORY_DEFINITIONS['1']["subcategories"]
    overall, by_model = _count_class1_distributions(existing_rows)
    model_counts = by_model.get(model_name, {})
    # Sort subcategories by ascending count for this model, then by overall, then by original index
    index_map = {s: i for i, s in enumerate(base)}
    return sorted(
        base,
        key=lambda s: (model_counts.get(s, 0), overall.get(s, 0), index_map[s])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Classify patches into subcategories based on voted taxonomy results"
    )
    parser.add_argument(
        '--voted-csv',
        default='taxonomy_results_voted.csv',
        help='Path to taxonomy_results_voted.csv'
    )
    parser.add_argument(
        '--cases-dir',
        default='model_failed_cases',
        help='Directory containing patch case files'
    )
    parser.add_argument(
        '--output',
        default='subcategory_classification_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--rerun-errors',
        action='store_true',
        help='Rerun only rows with subcategory ERROR/empty and update the existing CSV in place'
    )
    parser.add_argument(
        '--rerun-class1',
        action='store_true',
        help='Rerun only rows with class==1 and update the existing CSV in place (uses adaptive balancing)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Limit number of cases to process (0 = all)'
    )
    parser.add_argument(
        '--model',
        default=MODEL_GPT5_MINI,
        help='LLM model to use for classification'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Optional temperature for the LLM (e.g., 0.7 to diversify choices)'
    )
    parser.add_argument(
        '--retry',
        type=int,
        default=2,
        help='Number of retries per classification'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    root = Path(__file__).resolve().parents[1]
    voted_csv_path = root / args.voted_csv
    cases_dir = root / args.cases_dir
    output_path = root / args.output
    
    print("="*70)
    print("Subcategory Classification")
    print("="*70)
    print(f"Voted results: {voted_csv_path}")
    print(f"Cases directory: {cases_dir}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")
    print()
    
    # Check if files exist
    if not voted_csv_path.exists():
        print(f"ERROR: Voted results file not found: {voted_csv_path}")
        return
    
    if not cases_dir.exists():
        print(f"ERROR: Cases directory not found: {cases_dir}")
        return
    
    # Load voted results
    print("Loading voted taxonomy results...")
    voted_results = load_voted_results(voted_csv_path)
    print(f"Loaded {len(voted_results)} entries\n")

    # Optional rerun-errors mode: update existing CSV in place
    if args.rerun_errors:
        if not output_path.exists():
            print(f"ERROR: Output CSV not found for rerun-errors: {output_path}")
            return
        # Read existing rows
        existing: List[Dict[str, str]] = []
        with output_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.append(r)
        # Build error key set
        error_keys = {
            (r.get('instance_id', '').strip(), r.get('model_name', '').strip())
            for r in existing
            if (r.get('subcategory', '').strip() in ('', 'ERROR'))
        }
        if not error_keys:
            print("No ERROR rows to rerun.")
            return
        print(f"Rerunning {len(error_keys)} error rows...")
        # Build quick lookup from voted results
        voted_map = {
            (r.get('instance_id', '').strip(), r.get('model_name', '').strip()): r
            for r in voted_results
        }
        # Process only errors
        updates: Dict[tuple, str] = {}
        for key in error_keys:
            iid, model_name = key
            vr = voted_map.get(key)
            if not vr:
                # No voted row found; skip
                continue
            class_num = vr.get('class', '').strip()
            try:
                patch_content = load_patch_content(iid, model_name, cases_dir)
                resp = classify_subcategory(
                    instance_id=iid,
                    model_name=model_name,
                    class_num=class_num,
                    patch_content=patch_content,
                    model_slug=args.model,
                    retries=args.retry,
                    temperature=args.temperature,
                )
                updates[key] = resp.subcategory
                print(f"  ✓ {iid} | {model_name} -> {resp.subcategory}")
            except Exception as e:
                print(f"  ✗ {iid} | {model_name} ERROR: {e}")
                updates[key] = 'ERROR'
        # Apply updates to existing rows
        for r in existing:
            key = (r.get('instance_id', '').strip(), r.get('model_name', '').strip())
            if key in updates:
                r['subcategory'] = updates[key]
        # Write back
        write_results_csv(output_path, existing)
        print(f"Updated CSV: {output_path}")
        return

    # Optional rerun-class1 mode: rebalance class-1 subcategories per model
    if getattr(args, 'rerun_class1', False):
        if not output_path.exists():
            print(f"ERROR: Output CSV not found for rerun-class1: {output_path}")
            return
        # Read existing rows
        with output_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            existing_rows: List[Dict[str, str]] = list(reader)
        # Build quick lookup for patch content needs
        total_to_process = sum(1 for r in existing_rows if (r.get('class') or '').strip() == '1')
        processed = 0
        print(f"Rerunning Class 1 subcategory classification for {total_to_process} rows...")
        for r in existing_rows:
            if (r.get('class') or '').strip() != '1':
                continue
            iid = (r.get('instance_id') or '').strip()
            model_name = (r.get('model_name') or '').strip()
            # Adaptive balancing: compute per-model underrepresented order now
            order = _balanced_order_for_class1(model_name, existing_rows)
            try:
                patch_content = load_patch_content(iid, model_name, cases_dir)
                resp = classify_subcategory(
                    instance_id=iid,
                    model_name=model_name,
                    class_num='1',
                    patch_content=patch_content,
                    model_slug=args.model,
                    retries=args.retry,
                    temperature=args.temperature,
                    subcategory_order=order,
                )
                r['subcategory'] = resp.subcategory
                print(f"  ✓ {iid} | {model_name} -> {resp.subcategory}")
            except Exception as e:
                r['subcategory'] = 'ERROR'
                print(f"  ✗ {iid} | {model_name} ERROR: {e}")
            processed += 1
            if args.limit and processed >= args.limit:
                break
        write_results_csv(output_path, existing_rows)
        print(f"Updated CSV: {output_path}")
        return
    
    if args.limit > 0:
        voted_results = voted_results[:args.limit]
        print(f"Limited to {len(voted_results)} entries\n")
    
    # Process each entry
    output_results = []
    total = len(voted_results)
    
    for i, entry in enumerate(voted_results, 1):
        instance_id = entry['instance_id']
        model_name = entry['model_name']
        class_num = entry['class']
        
        if args.verbose or i % 10 == 0:
            print(f"[{i}/{total}] Processing: {instance_id} | {model_name} | Class {class_num}")
        
        # Skip if class is not in our definitions (e.g., class 6 - Algorithmic Inefficiency)
        if class_num not in SUBCATEGORY_DEFINITIONS:
            print(f"  Skipping: Class {class_num} has no subcategories defined")
            output_results.append({
                'instance_id': instance_id,
                'model_name': model_name,
                'class': class_num,
                'subcategory': 'N/A',
            })
            continue
        
        try:
            # Load patch content
            patch_content = load_patch_content(instance_id, model_name, cases_dir)
            
            # Classify subcategory
            response = classify_subcategory(
                instance_id=instance_id,
                model_name=model_name,
                class_num=class_num,
                patch_content=patch_content,
                model_slug=args.model,
                retries=args.retry,
                temperature=args.temperature,
            )
            
            output_results.append({
                'instance_id': instance_id,
                'model_name': model_name,
                'class': class_num,
                'subcategory': response.subcategory,
            })
            
            if args.verbose:
                print(f"  ✓ Subcategory: {response.subcategory}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            output_results.append({
                'instance_id': instance_id,
                'model_name': model_name,
                'class': class_num,
                'subcategory': 'ERROR',
            })
    
    # Write results
    print("\n" + "="*70)
    write_results_csv(output_path, output_results)
    print(f"Results written to: {output_path}")
    print(f"Total entries processed: {len(output_results)}")
    print("="*70)


if __name__ == "__main__":
    main()
