import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Set


def load_model_resolved_map(ablations_dir: Path) -> Dict[str, Set[str]]:
    model_to_resolved: Dict[str, Set[str]] = {}
    for fp in ablations_dir.glob("*.json"):
        try:
            with fp.open('r', encoding='utf-8') as f:
                data = json.load(f)
            resolved_ids = data.get('resolved_ids', []) or []
            # Model name is the human-readable prefix before .pagent_pagent_eval
            stem = fp.stem  # e.g., "Agentless GPT 4o.pagent_pagent_eval_Agentless_GPT_4o"
            model_name = stem.split('.pagent_pagent_eval', 1)[0]
            model_to_resolved[model_name] = set(resolved_ids)
        except Exception:
            continue
    return model_to_resolved


def add_resolved_column(input_csv: Path, ablations_dir: Path, output_csv: Path) -> None:
    model_map = load_model_resolved_map(ablations_dir)

    with input_csv.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if 'resolved' not in fieldnames:
        fieldnames.append('resolved')

    for r in rows:
        iid = (r.get('instance_id') or '').strip()
        model = (r.get('model_name') or '').strip()
        resolved_set = model_map.get(model)
        if resolved_set is None:
            r['resolved'] = ''  # unknown model mapping
        else:
            r['resolved'] = 'true' if iid in resolved_set else 'false'

    with output_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description='Add a resolved column to subcategory classification CSV based on Ablations/Pagent resolved_ids')
    parser.add_argument('--input', default='subcategory_classification_results.csv', help='Input CSV path')
    parser.add_argument('--ablations-dir', default=str(Path('Ablations') / 'Pagent'), help='Directory with eval JSON files')
    parser.add_argument('--output', default='subcategory_classification_results_with_resolved.csv', help='Output CSV path (ignored if --inplace)')
    parser.add_argument('--inplace', action='store_true', help='Update the input CSV in place')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_csv = (root / args.input).resolve()
    ablations_dir = (root / args.ablations_dir).resolve()
    output_csv = input_csv if args.inplace else (root / args.output).resolve()

    if not input_csv.exists():
        print(f"ERROR: Input CSV not found: {input_csv}")
        return
    if not ablations_dir.exists():
        print(f"ERROR: Ablations directory not found: {ablations_dir}")
        return

    add_resolved_column(input_csv, ablations_dir, output_csv)
    print(f"Wrote updated CSV: {output_csv}")


if __name__ == '__main__':
    main()
