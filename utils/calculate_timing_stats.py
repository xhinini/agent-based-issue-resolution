"""
Script to calculate aggregate statistics for timing metrics from processing_results.jsonl.
Reports median, mean, P90, min, max, and standard deviation for:
- Static analysis time
- Type inference (LLM) time
- Patch rewriting (LLM) time
"""

import json
import csv
from pathlib import Path
from typing import List, Dict
import statistics


def load_processing_results(jsonl_path: Path) -> List[Dict]:
    """Load all entries from the JSONL file."""
    results = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
                    continue
    
    return results


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate aggregate statistics for a list of values."""
    if not values:
        return {
            'median': 0,
            'mean': 0,
            'p90': 0,
            'min': 0,
            'max': 0,
            'std_dev': 0,
            'count': 0
        }
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    p90_index = int(n * 0.9)
    
    return {
        'median': statistics.median(values),
        'mean': statistics.mean(values),
        'p90': sorted_values[min(p90_index, n-1)],
        'min': min(values),
        'max': max(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
        'count': len(values)
    }


def extract_timing_metrics(results: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract timing metrics from processing results.
    
    Returns dictionary with keys:
    - static_seconds: Time for static analysis
    - llm_inference_seconds: Time for type inference (LLM calls)
    - rewriter_seconds: Time for patch rewriting (LLM)
    """
    metrics = {
        'static_seconds': [],
        'llm_inference_seconds': [],
        'rewriter_seconds': []
    }
    
    for entry in results:
        # Only process successful entries
        if entry.get('status') != 'success':
            continue
            
        metrics_data = entry.get('metrics', {})
        
        # Static analysis time
        static_time = metrics_data.get('static_seconds')
        if static_time is not None:
            metrics['static_seconds'].append(static_time)
        
        # LLM inference time (type inference)
        llm_inference_time = metrics_data.get('llm_inference_seconds')
        if llm_inference_time is not None:
            metrics['llm_inference_seconds'].append(llm_inference_time)
        
        # Rewriter time (patch rewriting)
        rewriter_time = metrics_data.get('rewriter_seconds')
        if rewriter_time is not None:
            metrics['rewriter_seconds'].append(rewriter_time)
    
    return metrics


def write_statistics_csv(output_path: Path, stats_by_component: Dict[str, Dict[str, float]]):
    """Write statistics to a CSV file."""
    
    # Prepare rows for CSV
    rows = []
    for component, stats in stats_by_component.items():
        row = {
            'component': component,
            'median': f"{stats['median']:.3f}",
            'mean': f"{stats['mean']:.3f}",
            'p90': f"{stats['p90']:.3f}",
            'min': f"{stats['min']:.3f}",
            'max': f"{stats['max']:.3f}",
            'std_dev': f"{stats['std_dev']:.3f}",
            'count': stats['count']
        }
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['component', 'median', 'mean', 'p90', 'min', 'max', 'std_dev', 'count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    # Set up paths
    root = Path(__file__).resolve().parents[1]
    jsonl_path = root / "pagent_runs" / "pagent" / "processing_results.jsonl"
    output_csv = root / "timing_statistics.csv"
    
    print("="*70)
    print("Calculating Timing Statistics from Processing Results")
    print("="*70)
    print(f"\nSource file: {jsonl_path}")
    print(f"Output file: {output_csv}\n")
    
    # Check if source file exists
    if not jsonl_path.exists():
        print(f"ERROR: Source file not found: {jsonl_path}")
        return
    
    # Load results
    print("Loading processing results...")
    results = load_processing_results(jsonl_path)
    print(f"Loaded {len(results)} entries\n")
    
    # Extract timing metrics
    print("Extracting timing metrics...")
    metrics = extract_timing_metrics(results)
    
    print(f"  Static analysis: {len(metrics['static_seconds'])} measurements")
    print(f"  Type inference (LLM): {len(metrics['llm_inference_seconds'])} measurements")
    print(f"  Patch rewriting (LLM): {len(metrics['rewriter_seconds'])} measurements")
    print()
    
    # Calculate statistics for each component
    print("Calculating statistics...")
    stats_by_component = {}
    
    component_names = {
        'static_seconds': 'Static Analysis',
        'llm_inference_seconds': 'Type Inference (LLM)',
        'rewriter_seconds': 'Patch Rewriting (LLM)'
    }
    
    for key, name in component_names.items():
        stats = calculate_statistics(metrics[key])
        stats_by_component[name] = stats
        
        print(f"\n{name}:")
        print(f"  Median:   {stats['median']:.3f}s")
        print(f"  Mean:     {stats['mean']:.3f}s")
        print(f"  P90:      {stats['p90']:.3f}s")
        print(f"  Min:      {stats['min']:.3f}s")
        print(f"  Max:      {stats['max']:.3f}s")
        print(f"  Std Dev:  {stats['std_dev']:.3f}s")
        print(f"  Count:    {stats['count']}")
    
    # Write to CSV
    print("\n" + "="*70)
    write_statistics_csv(output_csv, stats_by_component)
    print(f"Statistics written to: {output_csv}")
    print("="*70)


if __name__ == "__main__":
    main()
