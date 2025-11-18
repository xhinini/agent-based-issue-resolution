import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple


def read_instance_ids(path: Path) -> List[str]:
    ids: List[str] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def find_matches(directory: Path, instance_id: str, exts: Tuple[str, ...], json_analysis_only: bool = False) -> List[Path]:
    """
    Find files in 'directory' matching the given instance_id.
    - For txt: files starting with '<instance_id>_' and ending with '.txt'.
    - For json: by default any json starting with '<instance_id>_', but if json_analysis_only=True,
      require filenames ending with '_analysis.json'.
    Search is non-recursive (top-level only), as per provided examples.
    """
    if not directory.exists():
        return []

    matches: List[Path] = []

    # Build regex pattern per extension
    for p in directory.iterdir():
        if not p.is_file():
            continue
        name = p.name
        lower = name.lower()
        if any(lower.endswith(ext) for ext in exts):
            if lower.endswith('.txt'):
                # Example: astropy__astropy-6938_AppMap_Navie_GPT_4o.txt
                # Accept <id>.txt too (rare), but primarily <id>_*.txt
                if name.startswith(instance_id + '_') or name == f"{instance_id}.txt":
                    matches.append(p)
            elif lower.endswith('.json'):
                if name.startswith(instance_id + '_') or name == f"{instance_id}.json":
                    if not json_analysis_only or lower.endswith('_analysis.json'):
                        matches.append(p)
    return matches


def main():
    parser = argparse.ArgumentParser(description="Generate CSV of instance presence across folders")
    parser.add_argument('--instances', default='114_instance_id.txt', help='Path to instance id list (one per line)')
    parser.add_argument('--cases-dir', default='model_failed_cases', help='Directory containing .txt case files')
    parser.add_argument('--analysis-dir', default='patch_analysis_results', help='Directory containing .json analysis files')
    parser.add_argument('--out', default='instance_presence.csv', help='Output CSV path')
    parser.add_argument('--json-analysis-only', action='store_true', help='Only count JSON files ending with _analysis.json')
    args = parser.parse_args()

    instances_path = Path(args.instances)
    cases_dir = Path(args.cases_dir)
    analysis_dir = Path(args.analysis_dir)
    out_path = Path(args.out)

    instance_ids = read_instance_ids(instances_path)

    rows: List[Dict[str, str]] = []

    for iid in instance_ids:
        txt_matches = find_matches(cases_dir, iid, exts=(".txt",))
        json_matches = find_matches(analysis_dir, iid, exts=(".json",), json_analysis_only=args.json_analysis_only)

        rows.append({
            'instance_id': iid,
            'model_failed_cases_count': str(len(txt_matches)),
            'patch_analysis_results_count': str(len(json_matches)),
            'has_7_txt': 'TRUE' if len(txt_matches) >= 7 else 'FALSE',
            'has_7_json': 'TRUE' if len(json_matches) >= 7 else 'FALSE',
            'model_failed_cases_files': ';'.join(sorted([m.name for m in txt_matches])),
            'patch_analysis_results_files': ';'.join(sorted([m.name for m in json_matches])),
        })

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'instance_id','model_failed_cases_count','patch_analysis_results_count','has_7_txt','has_7_json','model_failed_cases_files','patch_analysis_results_files'
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"CSV written to: {out_path}")


if __name__ == '__main__':
    main()
