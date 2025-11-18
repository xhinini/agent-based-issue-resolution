"""
SWE-bench Integration for PAGENT

This module integrates PAGENT with SWE-bench evaluation framework.
It manages:
1. Loading SWE-bench dataset instances
2. Cloning repos at correct commits
3. Running PAGENT type inference and patch rewriting
4. Formatting predictions for SWE-bench evaluation
5. Batch processing multiple instances

Architecture:
- PAGENT runs LOCALLY with full type inference capabilities
- SWE-bench Docker is used ONLY for evaluation (test execution)
- Clean separation: patch generation (local) vs. evaluation (docker)

Usage:
    python swebench_integration.py \
        --instance-ids django__django-12345,astropy__astropy-6789 \
        --model-patches-dir ./model_failed_cases \
        --work-dir ./swebench_work \
        --inference-mode hybrid \
        --output-predictions ./predictions.json
"""

import os
import json
import subprocess
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datasets import load_dataset
import re


@dataclass
class SWEBenchInstance:
    """Represents a single SWE-bench task instance."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # Gold patch (for reference only, not used in solving)
    test_patch: str
    version: str
    
    
class SWEBenchDatasetManager:
    """
    Manages loading and querying SWE-bench datasets.
    Supports: SWE-bench, SWE-bench_Lite, SWE-bench_Verified
    """
    
    def __init__(self, dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"):
        """
        Initialize dataset manager.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split (usually 'test')
        """
        print(f"Loading dataset: {dataset_name} (split: {split})")
        self.dataset = load_dataset(dataset_name, split=split)
        self.instances_by_id = {item['instance_id']: item for item in self.dataset}
        print(f"Loaded {len(self.instances_by_id)} instances")
        
    def get_instance(self, instance_id: str) -> Optional[SWEBenchInstance]:
        """Get instance by ID."""
        item = self.instances_by_id.get(instance_id)
        if not item:
            return None
            
        return SWEBenchInstance(
            instance_id=item['instance_id'],
            repo=item['repo'],
            base_commit=item['base_commit'],
            problem_statement=item['problem_statement'],
            patch=item.get('patch', ''),
            test_patch=item.get('test_patch', ''),
            version=item.get('version', '')
        )
    
    def get_instances(self, instance_ids: List[str]) -> List[SWEBenchInstance]:
        """Get multiple instances by IDs."""
        instances = []
        for iid in instance_ids:
            instance = self.get_instance(iid)
            if instance:
                instances.append(instance)
            else:
                print(f"Warning: Instance {iid} not found in dataset")
        return instances
    
    def get_all_instance_ids(self) -> List[str]:
        """Get all instance IDs in the dataset."""
        return list(self.instances_by_id.keys())


class RepoManager:
    """
    Manages local repository clones at specific commits.
    Handles cloning, checkout, and cleanup.
    """
    
    def __init__(self, work_dir: str):
        """
        Initialize repo manager.
        
        Args:
            work_dir: Working directory for repo clones
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir = self.work_dir / "repos"
        self.repos_dir.mkdir(exist_ok=True)
        
    def get_repo_path(self, instance: SWEBenchInstance) -> Path:
        """Get local path for a repo instance."""
        # Use instance_id to create unique directory
        safe_id = instance.instance_id.replace('/', '_').replace('__', '_')
        return self.repos_dir / safe_id
    
    def prepare_repo(self, instance: SWEBenchInstance) -> Path:
        """
        Clone and checkout repo at the correct commit.
        
        Args:
            instance: SWE-bench instance
            
        Returns:
            Path to prepared repo directory
        """
        repo_path = self.get_repo_path(instance)
        
        # If repo already exists and is at correct commit, reuse it
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
                if current_commit == instance.base_commit:
                    print(f"Reusing existing repo at {repo_path} (commit: {instance.base_commit[:8]})")
                    return repo_path
                else:
                    print(f"Repo exists but at wrong commit. Re-preparing...")
                    shutil.rmtree(repo_path)
            except:
                print(f"Error checking existing repo. Re-cloning...")
                shutil.rmtree(repo_path)
        
        # Clone repo
        print(f"Cloning {instance.repo} to {repo_path}")
        git_url = f"https://github.com/{instance.repo}.git"
        
        try:
            subprocess.run(
                ["git", "clone", git_url, str(repo_path)],
                check=True,
                capture_output=True
            )
            
            # Checkout specific commit
            print(f"Checking out commit {instance.base_commit[:8]}")
            subprocess.run(
                ["git", "checkout", instance.base_commit],
                cwd=str(repo_path),
                check=True,
                capture_output=True
            )
            
            print(f"Repo prepared successfully at {repo_path}")
            return repo_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error preparing repo: {e}")
            raise
    
    def cleanup_repo(self, instance: SWEBenchInstance):
        """Clean up a specific repo."""
        repo_path = self.get_repo_path(instance)
        if repo_path.exists():
            shutil.rmtree(repo_path)
            print(f"Cleaned up repo: {repo_path}")
    
    def cleanup_all(self):
        """Clean up all cloned repos."""
        if self.repos_dir.exists():
            shutil.rmtree(self.repos_dir)
            self.repos_dir.mkdir(exist_ok=True)
            print("Cleaned up all repos")


class ModelPatchLoader:
    """
    Loads model-generated patches from local directory.
    Expects patches in the format used by model_failed_cases directory.
    """
    
    def __init__(self, patches_dir: str):
        """
        Initialize patch loader.
        
        Args:
            patches_dir: Directory containing model-generated patches
        """
        self.patches_dir = Path(patches_dir)
        if not self.patches_dir.exists():
            raise ValueError(f"Patches directory does not exist: {patches_dir}")
    
    def find_patch_file(self, instance_id: str, model_name: Optional[str] = None) -> Optional[Path]:
        """
        Find patch file for an instance.
        
        Args:
            instance_id: SWE-bench instance ID
            model_name: Optional model name to filter by
            
        Returns:
            Path to patch file or None
        """
        # Try exact match first
        pattern = f"{instance_id}_*.txt" if not model_name else f"{instance_id}_{model_name}.txt"
        matches = list(self.patches_dir.glob(pattern))
        
        if matches:
            return matches[0]
        
        # Try relaxed match (instance_id might have underscores replaced)
        relaxed_id = instance_id.replace('__', '_').replace('/', '_')
        pattern = f"*{relaxed_id}*.txt"
        matches = list(self.patches_dir.glob(pattern))
        
        if matches:
            return matches[0]
        
        return None
    
    def load_patch(self, instance_id: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Load patch content for an instance.
        
        Args:
            instance_id: SWE-bench instance ID
            model_name: Optional model name to filter by
            
        Returns:
            Patch content as string or None
        """
        patch_file = self.find_patch_file(instance_id, model_name)
        if not patch_file:
            return None
            
        try:
            with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract patch from content if it contains other information
            # Look for unified diff markers
            if 'diff --git' in content or '---' in content:
                return self._extract_patch_from_content(content)
            
            return content
            
        except Exception as e:
            print(f"Error loading patch from {patch_file}: {e}")
            return None
    
    def _extract_patch_from_content(self, content: str) -> str:
        """Extract actual patch from content that may include logs/output."""
        lines = content.splitlines()
        patch_lines = []
        in_patch = False
        
        for line in lines:
            # Start of patch
            if line.startswith('diff --git') or (line.startswith('---') and '/' in line):
                in_patch = True
            
            if in_patch:
                patch_lines.append(line)
        
        if patch_lines:
            return '\n'.join(patch_lines)
        
        # If no clear patch markers, return original content
        return content


class PAGENTPipeline:
    """
    Orchestrates PAGENT execution for patch improvement.
    Runs type inference, patch rewriting, and validation.
    """
    
    def __init__(self, inference_mode: str = "hybrid", core_script: str = "core.py"):
        """
        Initialize PAGENT pipeline.
        
        Args:
            inference_mode: Type inference mode (static/llm/hybrid)
            core_script: Path to core.py
        """
        self.inference_mode = inference_mode
        self.core_script = Path(core_script)
        if not self.core_script.exists():
            raise ValueError(f"core.py not found at: {core_script}")
    
    def run_pagent(self,
                   instance: SWEBenchInstance,
                   repo_path: Path,
                   original_patch: str,
                   output_dir: Path) -> Optional[str]:
        """
        Run PAGENT on a patch.
        
        Args:
            instance: SWE-bench instance
            repo_path: Path to prepared repo
            original_patch: Original model-generated patch
            output_dir: Directory for outputs
            
        Returns:
            Improved patch content or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write original patch to temp file
        original_patch_file = output_dir / f"{instance.instance_id}_original.patch"
        with open(original_patch_file, 'w', encoding='utf-8') as f:
            f.write(original_patch)
        
        # Output file for improved patch
        improved_patch_file = output_dir / f"{instance.instance_id}_improved.patch"
        
        # Run PAGENT
        print(f"Running PAGENT on {instance.instance_id} (mode: {self.inference_mode})")
        
        try:
            cmd = [
                "python", str(self.core_script),
                "--codebase", str(repo_path),
                "--patch", str(original_patch_file),
                "--output", str(improved_patch_file),
                "--inference-mode", self.inference_mode
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0 and improved_patch_file.exists():
                with open(improved_patch_file, 'r', encoding='utf-8') as f:
                    improved_patch = f.read()
                print(f"PAGENT succeeded for {instance.instance_id}")
                return improved_patch
            else:
                print(f"PAGENT failed for {instance.instance_id}")
                print(f"STDOUT: {result.stdout[:500]}")
                print(f"STDERR: {result.stderr[:500]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"PAGENT timed out for {instance.instance_id}")
            return None
        except Exception as e:
            print(f"Error running PAGENT for {instance.instance_id}: {e}")
            return None


class SWEBenchEvaluator:
    """
    Formats predictions and runs SWE-bench evaluation.
    """
    
    def __init__(self, swebench_dir: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            swebench_dir: Path to SWE-bench repo (cloned locally)
        """
        self.swebench_dir = Path(swebench_dir) if swebench_dir else None
        
    def format_predictions(self,
                          results: List[Dict[str, Any]],
                          model_name: str = "PAGENT") -> List[Dict[str, str]]:
        """
        Format predictions for SWE-bench evaluation.
        
        Args:
            results: List of dicts with instance_id and patch
            model_name: Model name for identification
            
        Returns:
            List of predictions in SWE-bench format
        """
        predictions = []
        for result in results:
            predictions.append({
                "instance_id": result["instance_id"],
                "model_patch": result["patch"],
                "model_name_or_path": model_name
            })
        return predictions
    
    def save_predictions(self, predictions: List[Dict[str, str]], output_file: Path):
        """Save predictions to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved {len(predictions)} predictions to {output_file}")
    
    def run_evaluation(self,
                       predictions_file: Path,
                       dataset_name: str = "princeton-nlp/SWE-bench_Lite",
                       max_workers: int = 4,
                       run_id: str = "pagent_eval") -> Optional[str]:
        """
        Run SWE-bench evaluation.
        
        Args:
            predictions_file: Path to predictions JSON
            dataset_name: Dataset to evaluate on
            max_workers: Number of parallel workers
            run_id: Run identifier
            
        Returns:
            Path to evaluation results or None
        """
        if not self.swebench_dir:
            print("SWE-bench directory not set. Skipping evaluation.")
            print(f"To evaluate, run:")
            print(f"  python -m swebench.harness.run_evaluation \\")
            print(f"    --dataset_name {dataset_name} \\")
            print(f"    --predictions_path {predictions_file} \\")
            print(f"    --max_workers {max_workers} \\")
            print(f"    --run_id {run_id}")
            return None
        
        print(f"Running SWE-bench evaluation (run_id: {run_id})")
        
        try:
            cmd = [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", dataset_name,
                "--predictions_path", str(predictions_file),
                "--max_workers", str(max_workers),
                "--run_id", run_id
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.swebench_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("Evaluation completed successfully")
                print(result.stdout)
                return f"./logs/run_evaluation/{run_id}"
            else:
                print(f"Evaluation failed")
                print(result.stderr)
                return None
                
        except Exception as e:
            print(f"Error running evaluation: {e}")
            return None


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class PAGENTSWEBenchPipeline:
    """
    Main orchestrator for PAGENT + SWE-bench integration.
    """
    
    def __init__(self,
                 work_dir: str,
                 model_patches_dir: str,
                 inference_mode: str = "hybrid",
                 dataset_name: str = "princeton-nlp/SWE-bench_Lite",
                 swebench_dir: Optional[str] = None):
        """
        Initialize pipeline.
        
        Args:
            work_dir: Working directory
            model_patches_dir: Directory with model-generated patches
            inference_mode: PAGENT inference mode
            dataset_name: SWE-bench dataset name
            swebench_dir: Path to SWE-bench repo for evaluation
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_manager = SWEBenchDatasetManager(dataset_name)
        self.repo_manager = RepoManager(work_dir)
        self.patch_loader = ModelPatchLoader(model_patches_dir)
        self.pagent = PAGENTPipeline(inference_mode)
        self.evaluator = SWEBenchEvaluator(swebench_dir)
        
        self.outputs_dir = self.work_dir / "pagent_outputs"
        self.outputs_dir.mkdir(exist_ok=True)
    
    def process_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Process a single instance.
        
        Args:
            instance_id: SWE-bench instance ID
            
        Returns:
            Dict with instance_id and improved patch, or None
        """
        print(f"\n{'='*60}")
        print(f"Processing: {instance_id}")
        print(f"{'='*60}")
        
        # Get instance from dataset
        instance = self.dataset_manager.get_instance(instance_id)
        if not instance:
            print(f"Instance not found in dataset: {instance_id}")
            return None
        
        # Load model-generated patch
        original_patch = self.patch_loader.load_patch(instance_id)
        if not original_patch:
            print(f"No model patch found for: {instance_id}")
            return None
        
        print(f"Loaded model patch ({len(original_patch)} chars)")
        
        # Prepare repo
        try:
            repo_path = self.repo_manager.prepare_repo(instance)
        except Exception as e:
            print(f"Failed to prepare repo: {e}")
            return None
        
        # Run PAGENT
        improved_patch = self.pagent.run_pagent(
            instance, repo_path, original_patch, self.outputs_dir
        )
        
        if improved_patch:
            return {
                "instance_id": instance_id,
                "patch": improved_patch,
                "original_patch": original_patch
            }
        else:
            # Fall back to original patch if PAGENT fails
            print(f"Using original patch as fallback for {instance_id}")
            return {
                "instance_id": instance_id,
                "patch": original_patch,
                "original_patch": original_patch
            }
    
    def process_batch(self, instance_ids: List[str], model_name: str = "PAGENT") -> Path:
        """
        Process multiple instances and generate predictions file.
        
        Args:
            instance_ids: List of instance IDs to process
            model_name: Model name for predictions
            
        Returns:
            Path to predictions file
        """
        results = []
        
        for instance_id in instance_ids:
            result = self.process_instance(instance_id)
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Processed {len(results)}/{len(instance_ids)} instances successfully")
        print(f"{'='*60}")
        
        # Format and save predictions
        predictions = self.evaluator.format_predictions(results, model_name)
        predictions_file = self.work_dir / "predictions.json"
        self.evaluator.save_predictions(predictions, predictions_file)
        
        return predictions_file
    
    def run_evaluation(self, predictions_file: Path, run_id: str = "pagent_eval"):
        """Run SWE-bench evaluation on predictions."""
        return self.evaluator.run_evaluation(predictions_file, run_id=run_id)


def main():
    parser = argparse.ArgumentParser(description="PAGENT + SWE-bench Integration")
    parser.add_argument("--instance-ids", required=True,
                        help="Comma-separated list of instance IDs to process")
    parser.add_argument("--model-patches-dir", required=True,
                        help="Directory containing model-generated patches")
    parser.add_argument("--work-dir", default="./swebench_work",
                        help="Working directory for repos and outputs")
    parser.add_argument("--inference-mode", choices=["static", "llm", "hybrid"],
                        default="hybrid", help="PAGENT inference mode")
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite",
                        help="SWE-bench dataset name")
    parser.add_argument("--swebench-dir", default=None,
                        help="Path to SWE-bench repo for evaluation")
    parser.add_argument("--model-name", default="PAGENT",
                        help="Model name for predictions")
    parser.add_argument("--run-evaluation", action="store_true",
                        help="Run SWE-bench evaluation after processing")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up repos after processing")
    
    args = parser.parse_args()
    
    # Parse instance IDs
    instance_ids = [iid.strip() for iid in args.instance_ids.split(',')]
    
    # Initialize pipeline
    pipeline = PAGENTSWEBenchPipeline(
        work_dir=args.work_dir,
        model_patches_dir=args.model_patches_dir,
        inference_mode=args.inference_mode,
        dataset_name=args.dataset_name,
        swebench_dir=args.swebench_dir
    )
    
    # Process instances
    predictions_file = pipeline.process_batch(instance_ids, args.model_name)
    
    # Run evaluation if requested
    if args.run_evaluation:
        pipeline.run_evaluation(predictions_file)
    
    # Cleanup if requested
    if args.cleanup:
        pipeline.repo_manager.cleanup_all()
    
    print("\nDone!")
    print(f"Predictions saved to: {predictions_file}")


if __name__ == "__main__":
    main()
