"""
Extract Instance IDs from Model Failed Cases Directory

This utility scans the model_failed_cases directory and extracts
unique instance IDs that can be used with swebench_integration.py

Usage:
    python utils/extract_instance_ids.py --patches-dir ./model_failed_cases
    
Output:
    - Prints comma-separated instance IDs
    - Optionally saves to file
"""

import argparse
import re
from pathlib import Path
from collections import Counter


def extract_instance_id_from_filename(filename: str) -> str:
    """
    Extract instance_id from filename.
    
    Patterns:
    - django__django-12345_ModelName.txt -> django__django-12345
    - astropy__astropy-6789.txt -> astropy__astropy-6789
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: owner__repo-number
    pattern = r'([a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+-\d+)'
    match = re.search(pattern, name)
    
    if match:
        return match.group(1)
    
    return None


def extract_instance_ids(patches_dir: str) -> list:
    """
    Extract all instance IDs from patches directory.
    
    Args:
        patches_dir: Directory containing patch files
        
    Returns:
        List of unique instance IDs
    """
    patches_path = Path(patches_dir)
    
    if not patches_path.exists():
        raise ValueError(f"Directory not found: {patches_dir}")
    
    instance_ids = set()
    file_count = 0
    
    for file_path in patches_path.glob("*.txt"):
        file_count += 1
        instance_id = extract_instance_id_from_filename(file_path.name)
        if instance_id:
            instance_ids.add(instance_id)
    
    print(f"Scanned {file_count} files")
    print(f"Found {len(instance_ids)} unique instance IDs")
    
    return sorted(list(instance_ids))


def count_instances_by_repo(instance_ids: list) -> dict:
    """Count instances by repository."""
    repo_counts = Counter()
    
    for iid in instance_ids:
        # Extract repo from instance_id (owner__repo-number)
        repo = iid.rsplit('-', 1)[0]  # Remove number
        repo_counts[repo] += 1
    
    return dict(repo_counts)


def main():
    parser = argparse.ArgumentParser(description="Extract instance IDs from patches directory")
    parser.add_argument("--patches-dir", required=True,
                        help="Directory containing model-generated patches")
    parser.add_argument("--output-file", default=None,
                        help="Optional: Save instance IDs to file (one per line)")
    parser.add_argument("--format", choices=["comma", "newline", "json"], default="comma",
                        help="Output format")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics by repository")
    
    args = parser.parse_args()
    
    # Extract instance IDs
    instance_ids = extract_instance_ids(args.patches_dir)
    
    if not instance_ids:
        print("No instance IDs found!")
        return
    
    # Show statistics
    if args.stats:
        print("\nStatistics by Repository:")
        print("-" * 40)
        repo_counts = count_instances_by_repo(instance_ids)
        for repo, count in sorted(repo_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {repo}: {count}")
        print("-" * 40)
    
    # Format output
    if args.format == "comma":
        output = ",".join(instance_ids)
    elif args.format == "newline":
        output = "\n".join(instance_ids)
    elif args.format == "json":
        import json
        output = json.dumps(instance_ids, indent=2)
    
    # Print to console
    print(f"\n{len(instance_ids)} Instance IDs:")
    print("=" * 60)
    print(output)
    print("=" * 60)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output)
        print(f"\nSaved to: {args.output_file}")
    
    # Print usage example
    print("\nUsage Example:")
    print("-" * 60)
    if args.format == "comma":
        # Take first 3 for example
        example_ids = ",".join(instance_ids[:min(3, len(instance_ids))])
        print(f"python swebench_integration.py \\")
        print(f"  --instance-ids {example_ids} \\")
        print(f"  --model-patches-dir {args.patches_dir} \\")
        print(f"  --work-dir ./swebench_work \\")
        print(f"  --inference-mode hybrid")
    print("-" * 60)


if __name__ == "__main__":
    main()
