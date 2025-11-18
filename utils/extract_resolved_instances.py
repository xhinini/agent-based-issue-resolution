"""
Script to extract unique resolved instance IDs from all JSON evaluation files 
in the Ablations/Pagent folder.
"""

import json
from pathlib import Path
from typing import Set


def extract_resolved_instances(ablations_dir: Path) -> tuple[Set[str], int]:
    """
    Extract unique resolved instance IDs from all JSON files in the given directory.
    
    Args:
        ablations_dir: Path to the Ablations/Pagent directory
        
    Returns:
        Tuple of (set of unique resolved instance IDs, count)
    """
    all_resolved_ids: Set[str] = set()
    
    # Find all JSON files in the directory
    json_files = list(ablations_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {ablations_dir}")
        return set(), 0
    
    print(f"Found {len(json_files)} JSON files to process:")
    
    for json_file in json_files:
        print(f"  Processing: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            resolved_ids = data.get('resolved_ids', [])
            print(f"    Found {len(resolved_ids)} resolved instances")
            
            all_resolved_ids.update(resolved_ids)
            
        except Exception as e:
            print(f"    Error processing {json_file.name}: {e}")
            continue
    
    return all_resolved_ids, len(all_resolved_ids)


def main():
    # Set up paths
    root = Path(__file__).resolve().parents[1]
    ablations_dir = root / "Ablations" / "Pagent"
    output_file = root / "resolved_instances_unique.txt"
    
    print("="*60)
    print("Extracting Unique Resolved Instance IDs")
    print("="*60)
    print(f"\nSource directory: {ablations_dir}")
    print(f"Output file: {output_file}\n")
    
    # Extract resolved instances
    resolved_ids, count = extract_resolved_instances(ablations_dir)
    
    # Sort for consistent output
    sorted_ids = sorted(resolved_ids)
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Unique Resolved Instance IDs\n")
        f.write(f"="*60 + "\n")
        f.write(f"Total Count: {count}\n")
        f.write(f"="*60 + "\n\n")
        
        for instance_id in sorted_ids:
            f.write(f"{instance_id}\n")
    
    print("="*60)
    print(f"Results written to: {output_file}")
    print(f"Total unique resolved instances: {count}")
    print("="*60)
    
    # Also print the list
    print("\nUnique Resolved Instance IDs:")
    for instance_id in sorted_ids:
        print(f"  - {instance_id}")


if __name__ == "__main__":
    main()
