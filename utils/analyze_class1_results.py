"""
Analyze PAGENT Class 1 Processing Results

This utility analyzes the results from run_pagent_on_class1.py:
- Success/failure statistics
- Processing time analysis
- Error categorization
- Comparison across inference modes
- Prepares data for visualization

Usage:
    python utils/analyze_class1_results.py \
        --results-dir ./pagent_class1_work \
        --output-report ./class1_analysis.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import csv


def load_results(results_file: Path) -> List[Dict]:
    """Load processing results from JSONL file."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_status_distribution(results: List[Dict]) -> Dict:
    """Analyze distribution of processing statuses."""
    status_counter = Counter(r['status'] for r in results)
    total = len(results)
    
    return {
        'total': total,
        'counts': dict(status_counter),
        'percentages': {status: (count / total * 100) for status, count in status_counter.items()}
    }


def analyze_by_repository(results: List[Dict]) -> Dict:
    """Analyze success rates by repository."""
    repo_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'total': 0})
    
    for result in results:
        # Extract repo from instance_id (format: repo__package-number)
        instance_id = result['instance_id']
        repo = instance_id.split('-')[0]  # e.g., django__django
        
        repo_stats[repo]['total'] += 1
        if result['status'] == 'success':
            repo_stats[repo]['success'] += 1
        else:
            repo_stats[repo]['failed'] += 1
    
    # Calculate success rates
    for repo in repo_stats:
        total = repo_stats[repo]['total']
        success = repo_stats[repo]['success']
        repo_stats[repo]['success_rate'] = (success / total * 100) if total > 0 else 0
    
    # Sort by success rate
    sorted_repos = sorted(
        repo_stats.items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )
    
    return dict(sorted_repos)


def analyze_by_model(results: List[Dict]) -> Dict:
    """Analyze success rates by model."""
    model_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'total': 0})
    
    for result in results:
        model = result['model_name']
        
        model_stats[model]['total'] += 1
        if result['status'] == 'success':
            model_stats[model]['success'] += 1
        else:
            model_stats[model]['failed'] += 1
    
    # Calculate success rates
    for model in model_stats:
        total = model_stats[model]['total']
        success = model_stats[model]['success']
        model_stats[model]['success_rate'] = (success / total * 100) if total > 0 else 0
    
    # Sort by success rate
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )
    
    return dict(sorted_models)


def analyze_failure_reasons(results: List[Dict]) -> Dict:
    """Categorize failure reasons."""
    failures = [r for r in results if r['status'] != 'success']
    
    failure_types = Counter(r['status'] for r in failures)
    
    # Extract error messages for 'error' status
    error_messages = []
    for r in failures:
        if r['status'] == 'error' and 'error' in r:
            error_messages.append(r['error'])
    
    return {
        'total_failures': len(failures),
        'failure_types': dict(failure_types),
        'sample_errors': error_messages[:5]  # First 5 errors
    }


def analyze_patch_sizes(results_dir: Path) -> Dict:
    """Analyze sizes of improved patches."""
    outputs_dir = results_dir / "pagent_outputs"
    
    if not outputs_dir.exists():
        return {}
    
    patch_sizes = []
    for patch_file in outputs_dir.glob("*_improved.patch"):
        try:
            size = patch_file.stat().st_size
            patch_sizes.append(size)
        except:
            continue
    
    if not patch_sizes:
        return {}
    
    return {
        'count': len(patch_sizes),
        'min_bytes': min(patch_sizes),
        'max_bytes': max(patch_sizes),
        'avg_bytes': sum(patch_sizes) / len(patch_sizes),
        'median_bytes': sorted(patch_sizes)[len(patch_sizes) // 2]
    }


def compare_modes(results_dirs: Dict[str, Path]) -> Dict:
    """Compare results across different inference modes."""
    comparison = {}
    
    for mode, results_dir in results_dirs.items():
        results_file = results_dir / "processing_results.jsonl"
        if not results_file.exists():
            continue
        
        results = load_results(results_file)
        status_dist = analyze_status_distribution(results)
        
        comparison[mode] = {
            'total': status_dist['total'],
            'success': status_dist['counts'].get('success', 0),
            'success_rate': status_dist['percentages'].get('success', 0)
        }
    
    return comparison


def generate_report(results_dir: Path, mode_name: str = None) -> Dict:
    """Generate comprehensive analysis report."""
    results_file = results_dir / "processing_results.jsonl"
    
    if not results_file.exists():
        return {'error': f'Results file not found: {results_file}'}
    
    results = load_results(results_file)
    
    report = {
        'mode': mode_name or 'unknown',
        'results_dir': str(results_dir),
        'status_distribution': analyze_status_distribution(results),
        'by_repository': analyze_by_repository(results),
        'by_model': analyze_by_model(results),
        'failure_analysis': analyze_failure_reasons(results),
        'patch_sizes': analyze_patch_sizes(results_dir)
    }
    
    return report


def print_summary(report: Dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print(f"PAGENT CLASS 1 PROCESSING REPORT - {report.get('mode', 'Unknown').upper()}")
    print("="*60)
    
    # Status distribution
    status_dist = report['status_distribution']
    print(f"\n📊 Overall Statistics:")
    print(f"  Total processed: {status_dist['total']}")
    print(f"\n  Status breakdown:")
    for status, count in status_dist['counts'].items():
        pct = status_dist['percentages'][status]
        print(f"    {status:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Success rate highlight
    success_rate = status_dist['percentages'].get('success', 0)
    print(f"\n  ✨ Success rate: {success_rate:.1f}%")
    
    # Top repositories
    print(f"\n🏆 Top 5 Repositories by Success Rate:")
    repo_stats = list(report['by_repository'].items())[:5]
    for repo, stats in repo_stats:
        print(f"  {repo:30s}: {stats['success']:3d}/{stats['total']:3d} ({stats['success_rate']:5.1f}%)")
    
    # Top models
    print(f"\n🤖 Top 5 Models by Success Rate:")
    model_stats = list(report['by_model'].items())[:5]
    for model, stats in model_stats:
        print(f"  {model:30s}: {stats['success']:3d}/{stats['total']:3d} ({stats['success_rate']:5.1f}%)")
    
    # Failure analysis
    failure_analysis = report['failure_analysis']
    if failure_analysis['total_failures'] > 0:
        print(f"\n❌ Failure Analysis:")
        print(f"  Total failures: {failure_analysis['total_failures']}")
        print(f"\n  Failure types:")
        for failure_type, count in failure_analysis['failure_types'].items():
            print(f"    {failure_type:15s}: {count:4d}")
    
    # Patch sizes
    patch_sizes = report.get('patch_sizes', {})
    if patch_sizes:
        print(f"\n📝 Improved Patch Statistics:")
        print(f"  Count: {patch_sizes['count']}")
        print(f"  Average size: {patch_sizes['avg_bytes']:.0f} bytes")
        print(f"  Median size: {patch_sizes['median_bytes']:.0f} bytes")
        print(f"  Range: {patch_sizes['min_bytes']}-{patch_sizes['max_bytes']} bytes")
    
    print("\n" + "="*60)


def export_for_visualization(report: Dict, output_dir: Path):
    """Export data in formats suitable for visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export status distribution as CSV
    status_csv = output_dir / "status_distribution.csv"
    with open(status_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Status', 'Count', 'Percentage'])
        for status, count in report['status_distribution']['counts'].items():
            pct = report['status_distribution']['percentages'][status]
            writer.writerow([status, count, f"{pct:.2f}"])
    
    # Export repository stats as CSV
    repo_csv = output_dir / "repository_stats.csv"
    with open(repo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Repository', 'Total', 'Success', 'Failed', 'Success Rate'])
        for repo, stats in report['by_repository'].items():
            writer.writerow([
                repo,
                stats['total'],
                stats['success'],
                stats['failed'],
                f"{stats['success_rate']:.2f}"
            ])
    
    # Export model stats as CSV
    model_csv = output_dir / "model_stats.csv"
    with open(model_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Total', 'Success', 'Failed', 'Success Rate'])
        for model, stats in report['by_model'].items():
            writer.writerow([
                model,
                stats['total'],
                stats['success'],
                stats['failed'],
                f"{stats['success_rate']:.2f}"
            ])
    
    print(f"\n📁 Visualization data exported to: {output_dir}")
    print(f"  - {status_csv.name}")
    print(f"  - {repo_csv.name}")
    print(f"  - {model_csv.name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PAGENT Class 1 processing results")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing processing_results.jsonl")
    parser.add_argument("--mode-name", default=None,
                        help="Name of inference mode (static/llm/hybrid)")
    parser.add_argument("--output-report", default=None,
                        help="Path to save JSON report")
    parser.add_argument("--export-viz", default=None,
                        help="Directory to export visualization data")
    parser.add_argument("--compare-modes", nargs='+', default=None,
                        help="Compare multiple result directories (format: mode1:dir1 mode2:dir2)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.compare_modes:
        # Parse mode:dir pairs
        mode_dirs = {}
        for pair in args.compare_modes:
            mode, dir_path = pair.split(':')
            mode_dirs[mode] = Path(dir_path)
        
        comparison = compare_modes(mode_dirs)
        
        print("\n" + "="*60)
        print("MODE COMPARISON")
        print("="*60)
        print(f"\n{'Mode':<15} {'Total':<10} {'Success':<10} {'Success Rate':<15}")
        print("-" * 60)
        for mode, stats in comparison.items():
            print(f"{mode:<15} {stats['total']:<10} {stats['success']:<10} {stats['success_rate']:<14.1f}%")
        print("="*60)
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison saved to: {args.output_report}")
    
    else:
        # Generate report for single mode
        report = generate_report(results_dir, args.mode_name)
        
        if 'error' in report:
            print(f"Error: {report['error']}")
            return
        
        # Print summary
        print_summary(report)
        
        # Save JSON report
        if args.output_report:
            output_path = Path(args.output_report)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Full report saved to: {output_path}")
        
        # Export visualization data
        if args.export_viz:
            export_for_visualization(report, Path(args.export_viz))


if __name__ == "__main__":
    main()
