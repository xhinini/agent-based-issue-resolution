import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
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


def _distributions(rows: List[Dict[str, str]]) -> Tuple[Dict[int, int], Dict[str, Dict[int, int]]]:
    overall: Dict[int, int] = {}
    by_model: Dict[str, Dict[int, int]] = {}
    for r in rows:
        c = (r.get('class') or '').strip()
        if not c.isdigit():
            continue
        label = int(c)
        m = r.get('model_name', '')
        overall[label] = overall.get(label, 0) + 1
        by_model.setdefault(m, {})
        by_model[m][label] = by_model[m].get(label, 0) + 1
    return overall, by_model


def _write_distributions(out_dir: Path, out_stem: str, overall: Dict[int, int], by_model: Dict[str, Dict[int, int]]) -> None:
    overall_rows = [{"class": str(k), "count": str(v)} for k, v in sorted(overall.items())]
    _write_csv(out_dir / f"{out_stem}_distribution_overall.csv", overall_rows)
    headers = ["model_name"] + [f"class_{i}" for i in range(1, 7)]
    model_rows: List[Dict[str, str]] = []
    for name, counts in sorted(by_model.items()):
        row = {"model_name": name}
        for i in range(1, 7):
            row[f"class_{i}"] = str(counts.get(i, 0))
        model_rows.append(row)
    out = out_dir / f"{out_stem}_distribution_by_model.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in model_rows:
            w.writerow(r)


def _derive_name_from_path(p: Path) -> str:
    stem = p.stem
    pref = "taxonomy_results_"
    if stem.startswith(pref):
        return stem[len(pref):]
    return stem


def aggregate(inputs: List[Path], names: List[str], weights: List[float], model_priority: List[str]) -> List[Dict[str, str]]:
    data: Dict[Tuple[str, str], Dict[str, str]] = {}
    votes: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path, name in zip(inputs, names):
        rows = _read_csv(path)
        for r in rows:
            iid = (r.get('instance_id') or '').strip()
            mname = (r.get('model_name') or '').strip()
            if not iid or not mname:
                continue
            key = (iid, mname)
            data.setdefault(key, {"instance_id": iid, "model_name": mname})
            votes.setdefault(key, {})
            c = (r.get('class') or '').strip()
            if c.isdigit():
                votes[key][name] = c
            else:
                votes[key][name] = ""
    out_rows: List[Dict[str, str]] = []
    name_to_w = {n: float(w) for n, w in zip(names, weights)}
    priority = list(model_priority) if model_priority else names
    for key, base in data.items():
        vmap = votes.get(key, {})
        weight_by_label: Dict[str, float] = {}
        for n, lab in vmap.items():
            if lab and lab.isdigit():
                weight_by_label[lab] = weight_by_label.get(lab, 0.0) + name_to_w.get(n, 1.0)
        chosen = ""
        diag = {n: (vmap.get(n) or "") for n in names}
        if weight_by_label:
            maxw = max(weight_by_label.values())
            top = [lab for lab, w in weight_by_label.items() if w == maxw]
            if len(top) == 1:
                chosen = top[0]
            else:
                picked = ""
                for n in priority:
                    lab = vmap.get(n)
                    if lab in top:
                        picked = lab
                        break
                chosen = picked or sorted(top)[0]
        expl = f"VOTE: {json.dumps(diag, ensure_ascii=False)}"
        row = {"instance_id": base["instance_id"], "model_name": base["model_name"], "class": chosen, "explanation": f'"{expl}"'}
        out_rows.append(row)
    return out_rows


def main():
    p = argparse.ArgumentParser(description="Aggregate taxonomy CSVs by majority voting with tie-breakers")
    p.add_argument('--inputs', required=True, help='Comma-separated list of input taxonomy_results_*.csv files')
    p.add_argument('--names', default='', help='Comma-separated short names matching inputs; defaults from filenames')
    p.add_argument('--weights', default='', help='Comma-separated floats matching names; default 1 per input')
    p.add_argument('--model-priority', default='', help='Comma-separated names to resolve ties; default = names order')
    p.add_argument('--out', default='taxonomy_results_voted.csv', help='Output CSV path for final voted classifications')
    args = p.parse_args()

    input_paths = [Path(s.strip()) for s in args.inputs.split(',') if s.strip()]
    if not input_paths:
        raise SystemExit('No inputs provided')
    if args.names.strip():
        names = [s.strip() for s in args.names.split(',') if s.strip()]
    else:
        names = [_derive_name_from_path(p) for p in input_paths]
    if len(names) != len(input_paths):
        raise SystemExit('names count must match inputs count')
    if args.weights.strip():
        weights = [float(s.strip()) for s in args.weights.split(',') if s.strip()]
        if len(weights) != len(names):
            raise SystemExit('weights count must match names count')
    else:
        weights = [1.0 for _ in names]
    if args.model_priority.strip():
        model_priority = [s.strip() for s in args.model_priority.split(',') if s.strip()]
    else:
        model_priority = names

    out_rows = aggregate(input_paths, names, weights, model_priority)
    out_path = Path(args.out)
    _write_csv(out_path, out_rows)
    overall, by_model = _distributions(out_rows)
    _write_distributions(out_path.parent, out_path.stem, overall, by_model)
    print(f"Wrote voted classifications to: {out_path}")


if __name__ == '__main__':
    main()
