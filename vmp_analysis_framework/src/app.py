#!/usr/bin/env python3
"""
VMP Analysis Framework - Full Data Processing
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import DataLoader, DataValidator
from analysis.metrics import VMPMetricsAnalyzer
from analysis.performance import PerformanceAnalyzer
from analysis.security import SecurityAnalyzer
from analysis.statistics import StatisticalAnalyzer
from visualization.plots import PlotGenerator
from visualization.plots import HeatmapGenerator
from reporting.generator import ReportGenerator
from utils.parallel import ParallelProcessor


def analyze_single_entry(entry_data):
    """Analyze a single VMP transformation entry"""
    print("\n=== Single Entry Analysis ===")
    print(f"Function: {entry_data['function']}")
    print(f"Original assembly length: {len(entry_data['original_assembly'])} chars")
    print(f"VMP assembly length: {len(entry_data['vmp_assembly'])} chars")
    print(f"Bytecode size: {entry_data['bytecode_size']} bytes")

    # Analyze metrics
    analyzer = VMPMetricsAnalyzer()
    metrics = analyzer.analyze_entry(entry_data)

    print("\nKey Metrics:")
    # Skip vm_handler_count and instruction_diversity
    skip_metrics = {'vm_handler_count', 'instruction_diversity'}
    
    for metric, value in metrics['metrics'].items():
        if metric not in skip_metrics and isinstance(value, (int, float)):
            print(f"  {metric}: {value:.2f}")

    return metrics


def run_full_analysis(jsonl_file_path):
    """Run full analysis on complete dataset"""
    print(f"\n=== Running Full Analysis ===")
    print(f"Input file: {jsonl_file_path}")

    # Load all data
    print("\n1. Loading data...")
    loader = DataLoader({'batch_size': 1000})  # Larger batch size for full processing

    # Load all entries
    data = []
    total_lines = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                continue
            
            # Progress indicator
            if total_lines % 10000 == 0:
                print(f"  Loaded {total_lines} lines...")

    print(f"Total lines processed: {total_lines}")
    print(f"Successfully loaded: {len(data)} entries")

    # Validate data
    print("\n2. Validating data...")
    validator = DataValidator()
    clean_data = validator.validate_and_clean(data)
    print(f"Valid entries after cleaning: {len(clean_data)}")

    # Analyze metrics
    print("\n3. Analyzing VMP metrics...")
    metrics_analyzer = VMPMetricsAnalyzer()
    metrics_results = []
    
    # Process in batches for better performance
    batch_size = 1000
    total_batches = (len(clean_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(clean_data))
        batch = clean_data[start_idx:end_idx]
        
        for entry in batch:
            result = metrics_analyzer.analyze_entry(entry)
            if result:
                metrics_results.append(result)
        
        print(f"  Processed batch {batch_idx + 1}/{total_batches} ({end_idx}/{len(clean_data)} entries)")

    print(f"Successfully analyzed {len(metrics_results)} entries")

    # Compute comprehensive statistics
    print("\n4. Computing statistics...")
    if metrics_results:
        # Metrics to analyze (excluding vm_handler_count and instruction_diversity)
        metrics_to_analyze = [
            'code_expansion_rate',
            'control_flow_complexity',
            'obfuscation_strength',
            'anti_debug_features'
        ]
        
        stats = {}
        for metric in metrics_to_analyze:
            values = [r['metrics'][metric] for r in metrics_results 
                     if metric in r['metrics']]
            if values:
                stats[metric] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
                print(f"\n{metric}:")
                print(f"  Count: {stats[metric]['count']}")
                print(f"  Average: {stats[metric]['average']:.2f}")
                print(f"  Min: {stats[metric]['min']:.2f}")
                print(f"  Max: {stats[metric]['max']:.2f}")

        # Category distribution
        categories = {}
        for entry in clean_data:
            cat = entry.get('function_category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("\n\nFunction categories distribution:")
        total_categorized = sum(categories.values())
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_categorized) * 100
            print(f"  {cat}: {count} ({percentage:.1f}%)")

    return clean_data, metrics_results, stats


def generate_full_report(clean_data, metrics_results, stats):
    """Generate comprehensive report"""
    print("\n5. Generating comprehensive report...")

    # Create outputs directory
    os.makedirs('outputs/reports', exist_ok=True)

    # Create detailed report
    report_path = 'outputs/reports/full_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=== VMP Analysis Full Report ===\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total samples in dataset: {len(clean_data)}\n")
        f.write(f"Successfully analyzed: {len(metrics_results)} samples\n")
        f.write(f"Analysis success rate: {(len(metrics_results)/len(clean_data)*100):.1f}%\n\n")

        if metrics_results:
            # Detailed statistics (excluding vm_handler_count and instruction_diversity)
            f.write("Detailed Statistics:\n")
            f.write("-" * 30 + "\n\n")

            metrics_to_report = [
                'code_expansion_rate',
                'control_flow_complexity',
                'obfuscation_strength',
                'anti_debug_features'
            ]

            for metric in metrics_to_report:
                values = [r['metrics'][metric] for r in metrics_results
                          if metric in r['metrics']]
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    
                    # Calculate standard deviation
                    variance = sum((x - avg_val) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5

                    f.write(f"{metric}:\n")
                    f.write(f"  Sample count: {len(values)}\n")
                    f.write(f"  Average: {avg_val:.2f}\n")
                    f.write(f"  Std Dev: {std_dev:.2f}\n")
                    f.write(f"  Min: {min_val:.2f}\n")
                    f.write(f"  Max: {max_val:.2f}\n\n")

            # Category analysis
            f.write("\nFunction Category Analysis:\n")
            f.write("-" * 30 + "\n")
            
            categories = {}
            category_metrics = {}
            
            for i, entry in enumerate(clean_data):
                cat = entry.get('function_category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
                
                # Collect metrics by category
                if i < len(metrics_results):
                    if cat not in category_metrics:
                        category_metrics[cat] = []
                    category_metrics[cat].append(metrics_results[i])
            
            total_functions = sum(categories.values())
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_functions) * 100
                f.write(f"\n{cat.upper()}:\n")
                f.write(f"  Count: {count} ({percentage:.1f}%)\n")
                
                # Category-specific metrics
                if cat in category_metrics and category_metrics[cat]:
                    cat_metrics = category_metrics[cat]
                    avg_expansion = sum(m['metrics']['code_expansion_rate'] for m in cat_metrics) / len(cat_metrics)
                    avg_complexity = sum(m['metrics']['control_flow_complexity'] for m in cat_metrics) / len(cat_metrics)
                    f.write(f"  Avg expansion rate: {avg_expansion:.2f}x\n")
                    f.write(f"  Avg complexity: {avg_complexity:.2f}\n")

            # Top protected functions
            f.write("\n\nMost Protected Functions (by code expansion):\n")
            f.write("-" * 30 + "\n")
            sorted_by_expansion = sorted(metrics_results,
                                         key=lambda x: x['metrics']['code_expansion_rate'],
                                         reverse=True)

            for i, result in enumerate(sorted_by_expansion[:20]):  # Top 20
                f.write(f"{i + 1:2d}. {result['function'][:50]:<50} - {result['metrics']['code_expansion_rate']:>6.2f}x expansion\n")

            # Most complex functions
            f.write("\n\nMost Complex Functions (by control flow):\n")
            f.write("-" * 30 + "\n")
            sorted_by_complexity = sorted(metrics_results,
                                        key=lambda x: x['metrics']['control_flow_complexity'],
                                        reverse=True)

            for i, result in enumerate(sorted_by_complexity[:20]):  # Top 20
                f.write(f"{i + 1:2d}. {result['function'][:50]:<50} - complexity: {result['metrics']['control_flow_complexity']:>6.2f}\n")

            # Summary
            f.write("\n\nSummary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"The analysis processed {len(clean_data)} VMP-protected functions.\n")
            f.write(f"Average code expansion rate: {stats.get('code_expansion_rate', {}).get('average', 0):.2f}x\n")
            f.write(f"Average control flow complexity: {stats.get('control_flow_complexity', {}).get('average', 0):.2f}\n")
            f.write(f"Functions with highest protection levels show expansion rates up to {stats.get('code_expansion_rate', {}).get('max', 0):.2f}x\n")

    print(f"Report saved to: {report_path}")
    
    # Also create a summary JSON file
    json_report_path = 'outputs/reports/analysis_summary.json'
    summary = {
        'total_samples': len(clean_data),
        'analyzed_samples': len(metrics_results),
        'statistics': stats,
        'top_protected_functions': [
            {
                'function': r['function'],
                'expansion_rate': r['metrics']['code_expansion_rate'],
                'complexity': r['metrics']['control_flow_complexity']
            }
            for r in sorted_by_expansion[:10]
        ]
    }
    
    with open(json_report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"JSON summary saved to: {json_report_path}")


def main():
    """Main analysis function"""
    # File path
    jsonl_file = "/data/jiacheng/dylan/l4av/l4av/vmp_transformer/examples/mibench_all_transformations_20250616_021303_version2.jsonl"

    # Check if file exists
    if not os.path.exists(jsonl_file):
        print(f"Error: File not found - {jsonl_file}")
        print("\nPlease ensure the file path is correct.")
        return

    print("=== VMP Analysis Framework ===")
    print(f"Processing file: {jsonl_file}")
    
    # Get file size
    file_size = os.path.getsize(jsonl_file) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")

    # Run full analysis
    try:
        clean_data, metrics_results, stats = run_full_analysis(jsonl_file)
        
        # Generate comprehensive report
        generate_full_report(clean_data, metrics_results, stats)
        
        print("\n=== Analysis Complete ===")
        print("Results saved in outputs/reports/")
        print("\nGenerated files:")
        print("  - full_analysis_report.txt (detailed text report)")
        print("  - analysis_summary.json (structured summary)")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("Please check the input file format and try again.")
        raise


if __name__ == '__main__':
    main()