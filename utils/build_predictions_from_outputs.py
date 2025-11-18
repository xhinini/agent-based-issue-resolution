import argparse
import json
from pathlib import Path
from typing import List
import re


def collect_improved_patches(outputs_dir: Path) -> List[dict]:
    rows = []
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    # Pattern: <instance_id>_<model_name>_improved.patch
    for p in sorted(outputs_dir.glob("*_improved.patch")):
        name = p.name
        base = name[:-len("_improved.patch")] if name.endswith("_improved.patch") else p.stem
        # Split from the rightmost underscore to separate model name
        if "_" not in base:
            # Skip unexpected file
            continue
        maybe_instance_id, model_name = base.rsplit("_", 1)
        instance_id = maybe_instance_id
        patch_text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not patch_text:
            continue
        rows.append({
            "instance_id": instance_id,
            "model_patch": patch_text,
            "model_name_or_path": model_name,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build SWE-bench predictions.jsonl from PAGENT outputs")
    parser.add_argument("--outputs-dir", default="./pagent_class1_work/pagent_outputs", help="Directory containing *_improved.patch files")
    parser.add_argument("--out", default="./pagent_class1_work/predictions.jsonl", help="Path to write predictions.jsonl (ignored if --split-by-model)")
    parser.add_argument("--split-by-model", action="store_true", help="Write one predictions file per model into --out-dir")
    parser.add_argument("--out-dir", default="./pagent_class1_work/predictions_by_model", help="Directory for per-model predictions when --split-by-model is set")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    rows = collect_improved_patches(outputs_dir)

    if args.split_by_model:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # group by model
        by_model = {}
        for r in rows:
            model = r.get("model_name_or_path", "unknown")
            by_model.setdefault(model, []).append(r)
        count_files = 0
        total = 0
        for model, items in by_model.items():
            # normalize model to safe filename
            norm = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-") or "unknown"
            out_path = out_dir / f"{norm}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"Wrote {len(items)} predictions to {out_path}")
            total += len(items)
            count_files += 1
        print(f"Wrote {total} predictions across {count_files} model files in {out_dir}")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            print(f"No improved patches found in {outputs_dir}")
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
