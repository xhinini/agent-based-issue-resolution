"""
Run PAGENT on All Class 1 Model-Generated Patches

This script:
1. Reads taxonomy_results_voted.csv to identify all Class 1 patches
2. Extracts problem description, gold patch, and model patch from txt files
3. Clones repos from SWE-bench dataset at correct commits
4. Runs PAGENT type inference and patch rewriting
5. Saves improved patches and results

Usage:
    python run_pagent_on_class1.py \
        --taxonomy-csv taxonomy_results_voted.csv \
        --patches-dir ./model_failed_cases \
        --work-dir ./pagent_class1_work \
        --inference-mode hybrid \
        --dataset-name princeton-nlp/SWE-bench_Lite
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
import subprocess
import shutil
from tqdm import tqdm


@dataclass
class Class1Entry:
    """Represents a Class 1 taxonomy entry."""
    instance_id: str
    model_name: str
    
    def __repr__(self):
        return f"Class1Entry({self.instance_id}, {self.model_name})"


@dataclass
class PatchContent:
    """Extracted content from a patch txt file."""
    instance_id: str
    model_name: str
    problem_description: str
    model_patch: str
    
    
class TaxonomyReader:
    """Reads taxonomy_results_voted.csv and filters Class 1 entries."""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise ValueError(f"Taxonomy CSV not found: {csv_path}")
    
    def get_class1_entries(self) -> List[Class1Entry]:
        """Extract all Class 1 entries."""
        class1_entries = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['class'] == '1':
                    class1_entries.append(Class1Entry(
                        instance_id=row['instance_id'],
                        model_name=row['model_name']
                    ))
        
        print(f"Found {len(class1_entries)} Class 1 entries")
        return class1_entries


class PatchFileParser:
    """Parses txt files in model_failed_cases to extract patches."""
    
    def __init__(self, patches_dir: str):
        self.patches_dir = Path(patches_dir)
        if not self.patches_dir.exists():
            raise ValueError(f"Patches directory not found: {patches_dir}")
    
    def normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name to match file naming convention.
        
        Examples:
        - "Agentless GPT 4o" -> "Agentless_GPT_4o"
        - "Moatless Tools Claude 3 5 Sonnet" -> "Moatless_Tools_Claude_3_5_Sonnet"
        """
        return model_name.replace(' ', '_')
    
    def find_patch_file(self, instance_id: str, model_name: str) -> Optional[Path]:
        """Find the txt file for given instance_id and model_name."""
        # Normalize model name
        normalized_model = self.normalize_model_name(model_name)
        
        # Expected filename format: instance_id_model_name.txt
        expected_filename = f"{instance_id}_{normalized_model}.txt"
        expected_path = self.patches_dir / expected_filename
        
        if expected_path.exists():
            return expected_path
        
        # Fallback: search for files containing both instance_id and model parts
        pattern = f"{instance_id}_*{normalized_model}*.txt"
        matches = list(self.patches_dir.glob(pattern))
        if matches:
            return matches[0]
        
        # Try case-insensitive search
        for file_path in self.patches_dir.glob(f"{instance_id}_*.txt"):
            if normalized_model.lower() in file_path.stem.lower():
                return file_path
        
        return None
    
    def extract_section(self, content: str, start_marker: str, end_marker: Optional[str] = None) -> str:
        """Extract content between markers."""
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return ""
        
        start_idx += len(start_marker)
        
        if end_marker:
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                return content[start_idx:].strip()
            return content[start_idx:end_idx].strip()
        
        return content[start_idx:].strip()
    
    def extract_diff_block(self, content: str, start_pattern: str) -> str:
        """
        Extract a unified diff block starting from a pattern.
        Stops at the next major section marker or end of content.
        """
        lines = content.splitlines()
        diff_lines = []
        in_diff = False
        
        for line in lines:
            if start_pattern in line:
                in_diff = True
                continue
            
            if in_diff:
                # Stop at next major section
                if line.startswith('###') and not line.strip().startswith('diff'):
                    break
                
                # Include diff lines
                if line.startswith('diff --git') or line.startswith('---') or \
                   line.startswith('+++') or line.startswith('@@') or \
                   line.startswith('+') or line.startswith('-') or line.startswith(' '):
                    diff_lines.append(line)
                elif diff_lines and line.strip() == '':
                    # Allow blank lines within diff
                    diff_lines.append(line)
                elif diff_lines:
                    # Non-diff line after we started collecting, might be end
                    # Check if it's really the end or just context
                    if not any(line.strip().startswith(marker) for marker in ['diff', '---', '+++', '@@', '+', '-', ' ', '']):
                        break
        
        return '\n'.join(diff_lines).strip()
    
    def parse_patch_file(self, file_path: Path) -> Optional[PatchContent]:
        """Parse a patch txt file to extract all components."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract instance_id and model_name from header
            instance_id_match = re.search(r'# Instance ID: (.+)', content)
            model_name_match = re.search(r'# Model: (.+)', content)
            
            if not instance_id_match or not model_name_match:
                print(f"Warning: Could not extract metadata from {file_path.name}")
                return None
            
            instance_id = instance_id_match.group(1).strip()
            model_name = model_name_match.group(1).strip()
            
            # Extract problem description (between ### DESCRIPTION and ### GOLD_PATCH)
            desc_start = content.find('### DESCRIPTION')
            gold_start = content.find('### GOLD_PATCH')
            
            if desc_start != -1 and gold_start != -1:
                problem_description = content[desc_start + len('### DESCRIPTION'):gold_start].strip()
            else:
                problem_description = ""
            
            # Extract model patch
            model_patch = self.extract_diff_block(content, '### Model Generated Patch')
            
            # Fallback: if model patch not found, try other markers
            if not model_patch:
                model_patch = self.extract_diff_block(content, '### MODEL_OUTPUTS')
            
            if not model_patch:
                # Try to find any diff after the problem description
                lines = content.splitlines()
                in_model_section = False
                diff_lines = []
                for line in lines:
                    if '### Model Generated Patch' in line or '### MODEL' in line:
                        in_model_section = True
                        continue
                    if in_model_section and (line.startswith('diff ') or line.startswith('--- ') or line.startswith('+++ ')):
                        diff_lines.append(line)
                    elif in_model_section and diff_lines and line.startswith('###'):
                        break
                    elif in_model_section and diff_lines:
                        diff_lines.append(line)
                
                if diff_lines:
                    model_patch = '\n'.join(diff_lines).strip()
            
            return PatchContent(
                instance_id=instance_id,
                model_name=model_name,
                problem_description=problem_description,
                model_patch=model_patch
            )
            
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
            return None


class SWEBenchRepoManager:
    """Manages cloning repos from SWE-bench dataset."""
    
    def __init__(self, work_dir: str, dataset_name: str = "princeton-nlp/SWE-bench_Lite"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir = self.work_dir / "repos"
        self.repos_dir.mkdir(exist_ok=True)
        
        print(f"Loading SWE-bench dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split='test')
        self.instances_by_id = {item['instance_id']: item for item in self.dataset}
        print(f"Loaded {len(self.instances_by_id)} instances")
    
    def get_instance_info(self, instance_id: str) -> Optional[Dict]:
        """Get instance metadata from dataset."""
        return self.instances_by_id.get(instance_id)
    
    def get_repo_path(self, instance_id: str) -> Path:
        """Get local path for a repo."""
        safe_id = instance_id.replace('/', '_').replace('__', '_')
        return self.repos_dir / safe_id
    
    def clone_repo(self, instance_id: str) -> Optional[Path]:
        """Clone repo at correct commit for instance."""
        instance_info = self.get_instance_info(instance_id)
        if not instance_info:
            print(f"Warning: Instance {instance_id} not found in dataset")
            return None
        
        repo_path = self.get_repo_path(instance_id)
        
        # Check if already cloned at correct commit
        if repo_path.exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_commit = result.stdout.strip()
                if current_commit == instance_info['base_commit']:
                    return repo_path
                else:
                    shutil.rmtree(repo_path)
            except:
                shutil.rmtree(repo_path)
        
        # Clone repo
        git_url = f"https://github.com/{instance_info['repo']}.git"
        
        try:
            subprocess.run(
                ["git", "clone", "--quiet", git_url, str(repo_path)],
                check=True,
                capture_output=True
            )
            
            # Checkout specific commit
            subprocess.run(
                ["git", "checkout", "--quiet", instance_info['base_commit']],
                cwd=str(repo_path),
                check=True,
                capture_output=True
            )
            
            return repo_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repo for {instance_id}: {e}")
            return None


class PAGENTRunner:
    """Runs PAGENT on patches."""
    
    def __init__(self, inference_mode: str = "hybrid", core_script: str = "core.py"):
        self.inference_mode = inference_mode
        self.core_script = Path(core_script)
        if not self.core_script.exists():
            raise ValueError(f"Pagent Core file not found at: {core_script}")
    
    def run_pagent(self, repo_path: Path, model_patch: str, instance_id: str, output_dir: Path, *, extra_context: Optional[str] = None) -> Tuple[Optional[str], Optional[dict], Optional[dict]]:
        """Run PAGENT on a model patch and return (improved_patch, metrics_dict, types_dict)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write model patch to temp file
        patch_file = output_dir / f"{instance_id}_model.patch"
        with open(patch_file, 'w', encoding='utf-8') as f:
            f.write(model_patch)
        
        # Output file for improved patch
        improved_file = output_dir / f"{instance_id}_improved.patch"
        
        try:
            cmd = [
                "python", str(self.core_script),
                "--codebase", str(repo_path),
                "--patch", str(patch_file),
                "--output", str(improved_file),
                "--inference-mode", self.inference_mode
            ]
            # Pass extra context only in hybrid mode
            if self.inference_mode == "hybrid" and extra_context:
                extra_file = output_dir / f"{instance_id}_extra_context.txt"
                with open(extra_file, 'w', encoding='utf-8') as ef:
                    ef.write(extra_context)
                cmd += ["--extra-context-file", str(extra_file)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout
            )
            # Try to parse metrics and types from stdout
            metrics: Optional[dict] = None
            types_info: Optional[dict] = None
            try:
                if result.stdout:
                    for line in result.stdout.splitlines():
                        if line.startswith("PAGENT_METRICS_JSON:"):
                            payload = line.split("PAGENT_METRICS_JSON:", 1)[1].strip()
                            metrics = json.loads(payload)
                        elif line.startswith("PAGENT_TYPES_JSON:"):
                            payload = line.split("PAGENT_TYPES_JSON:", 1)[1].strip()
                            types_info = json.loads(payload)
            except Exception:
                metrics = metrics or None
                types_info = types_info or None

            if result.returncode == 0 and improved_file.exists():
                with open(improved_file, 'r', encoding='utf-8') as f:
                    return f.read(), metrics, types_info
            else:
                print(f"PAGENT failed for {instance_id}: {result.stderr[:200]}")
                return None, metrics, types_info
                
        except subprocess.TimeoutExpired:
            print(f"PAGENT timed out for {instance_id}")
            return None, None, None
        except Exception as e:
            print(f"Error running PAGENT for {instance_id}: {e}")
            return None, None, None


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run PAGENT on all Class 1 patches")
    parser.add_argument("--taxonomy-csv", default="taxonomy_results_voted.csv",
                        help="Path to taxonomy results CSV")
    parser.add_argument("--patches-dir", default="./model_failed_cases",
                        help="Directory containing model patch txt files")
    parser.add_argument("--work-dir", default="./pagent_class1_work",
                        help="Working directory for processing")
    parser.add_argument("--inference-mode", choices=["static", "llm", "hybrid"], default="hybrid",
                        help="PAGENT inference mode")
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite",
                        help="SWE-bench dataset name")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of patches to process (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing progress")
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directories
    pagent_outputs_dir = work_dir / "pagent_outputs"
    pagent_outputs_dir.mkdir(exist_ok=True)
    
    results_file = work_dir / "processing_results.jsonl"
    processed_file = work_dir / "processed_entries.txt"
    
    # Load already processed entries if resuming
    processed_entries = set()
    if args.resume and processed_file.exists():
        with open(processed_file, 'r') as f:
            processed_entries = set(line.strip() for line in f)
        print(f"Resuming: {len(processed_entries)} entries already processed")
    
    # Step 1: Read taxonomy and get Class 1 entries
    print("\n=== Step 1: Reading Taxonomy ===")
    taxonomy_reader = TaxonomyReader(args.taxonomy_csv)
    class1_entries = taxonomy_reader.get_class1_entries()
    
    if args.limit:
        class1_entries = class1_entries[:args.limit]
        print(f"Limited to {len(class1_entries)} entries for testing")
    
    # Step 2: Initialize components
    print("\n=== Step 2: Initializing Components ===")
    patch_parser = PatchFileParser(args.patches_dir)
    repo_manager = SWEBenchRepoManager(args.work_dir, args.dataset_name)
    pagent_runner = PAGENTRunner(args.inference_mode)
    
    # Step 3: Process each Class 1 entry
    print(f"\n=== Step 3: Processing {len(class1_entries)} Class 1 Entries ===")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with open(results_file, 'a', encoding='utf-8') as results_f:
        for entry in tqdm(class1_entries, desc="Processing patches"):
            entry_key = f"{entry.instance_id}_{entry.model_name}"
            
            # Skip if already processed
            if entry_key in processed_entries:
                skipped_count += 1
                continue
            
            result = {
                "instance_id": entry.instance_id,
                "model_name": entry.model_name,
                "status": "unknown"
            }
            
            try:
                # Find and parse patch file
                patch_file = patch_parser.find_patch_file(entry.instance_id, entry.model_name)
                if not patch_file:
                    print(f"\nWarning: Patch file not found for {entry_key}")
                    result["status"] = "file_not_found"
                    failed_count += 1
                    results_f.write(json.dumps(result) + '\n')
                    continue
                
                patch_content = patch_parser.parse_patch_file(patch_file)
                if not patch_content or not patch_content.model_patch:
                    print(f"\nWarning: Could not extract model patch from {patch_file.name}")
                    result["status"] = "parse_failed"
                    failed_count += 1
                    results_f.write(json.dumps(result) + '\n')
                    continue
                
                # Clone repo
                repo_path = repo_manager.clone_repo(entry.instance_id)
                if not repo_path:
                    result["status"] = "clone_failed"
                    failed_count += 1
                    results_f.write(json.dumps(result) + '\n')
                    continue
                
                # Run PAGENT
                improved_patch, metrics, types_info = pagent_runner.run_pagent(
                    repo_path,
                    patch_content.model_patch,
                    entry_key,
                    pagent_outputs_dir
                )
                
                # Save type inference summary file if available
                types_summary_file: Optional[Path] = None
                if types_info:
                    types_summary_file = pagent_outputs_dir / f"{entry_key}_type_inference.txt"
                    try:
                        with open(types_summary_file, 'w', encoding='utf-8') as tf:
                            tf.write(f"Type Inference Summary\n")
                            tf.write(f"Instance: {entry.instance_id}\n")
                            tf.write(f"Model: {entry.model_name}\n")
                            tf.write(f"Mode: {types_info.get('mode','')}\n")
                            vars_an = types_info.get('variables_analyzed') or []
                            tf.write(f"Variables analyzed ({len(vars_an)}): {', '.join(vars_an)}\n\n")
                            tf.write("Inferences:\n")
                            for item in types_info.get('inferences', []):
                                name = item.get('name','')
                                tp = item.get('inferred_type','')
                                conf = item.get('confidence', 0)
                                src = item.get('source','')
                                loc = item.get('location')
                                loc_str = f" @ {loc[0]}:{loc[1]}" if isinstance(loc, list) and len(loc)==2 else ""
                                tf.write(f"- {name}: {tp} (confidence: {conf:.2f}, source: {src}){loc_str}\n")
                    except Exception:
                        types_summary_file = None

                if improved_patch:
                    result["status"] = "success"
                    result["improved_patch_length"] = len(improved_patch)
                    if metrics is not None:
                        result["metrics"] = metrics
                    if types_summary_file is not None:
                        result["types_summary_file"] = str(types_summary_file)
                    success_count += 1
                else:
                    result["status"] = "pagent_failed"
                    if metrics is not None:
                        result["metrics"] = metrics
                    if types_summary_file is not None:
                        result["types_summary_file"] = str(types_summary_file)
                    failed_count += 1
                
                results_f.write(json.dumps(result) + '\n')
                results_f.flush()
                
                # Mark as processed
                with open(processed_file, 'a') as pf:
                    pf.write(entry_key + '\n')
                
            except Exception as e:
                print(f"\nError processing {entry_key}: {e}")
                result["status"] = "error"
                result["error"] = str(e)
                failed_count += 1
                results_f.write(json.dumps(result) + '\n')
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total entries: {len(class1_entries)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"\nResults saved to: {results_file}")
    print(f"Improved patches saved to: {pagent_outputs_dir}")


if __name__ == "__main__":
    main()
