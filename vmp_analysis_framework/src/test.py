#!/usr/bin/env python3
"""
Example usage of VMP Analysis Framework
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
    for metric, value in metrics['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.2f}")

    return metrics


def run_minimal_analysis(jsonl_file_path):
    """Run a minimal analysis on a subset of data"""
    print(f"\n=== Running Minimal Analysis ===")
    print(f"Input file: {jsonl_file_path}")

    # Load a small subset of data
    print("\n1. Loading data...")
    loader = DataLoader({'batch_size': 100})

    # Load only first 100 entries for demo
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Limit to 100 entries for demo
                break
            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(data)} entries")

    # Validate data
    print("\n2. Validating data...")
    validator = DataValidator()
    clean_data = validator.validate_and_clean(data)
    print(f"Valid entries: {len(clean_data)}")

    # Analyze metrics
    print("\n3. Analyzing VMP metrics...")
    metrics_analyzer = VMPMetricsAnalyzer()
    metrics_results = []

    for entry in clean_data[:10]:  # Analyze first 10 for speed
        result = metrics_analyzer.analyze_entry(entry)
        if result:
            metrics_results.append(result)

    print(f"Analyzed {len(metrics_results)} entries")

    # Basic statistics
    print("\n4. Computing statistics...")
    if metrics_results:
        # Calculate average metrics
        avg_expansion = sum(r['metrics']['code_expansion_rate'] for r in metrics_results) / len(metrics_results)
        avg_complexity = sum(r['metrics']['control_flow_complexity'] for r in metrics_results) / len(metrics_results)

        print(f"\nAverage code expansion rate: {avg_expansion:.2f}x")
        print(f"Average control flow complexity: {avg_complexity:.2f}")

        # Category distribution
        categories = {}
        for entry in clean_data:
            cat = entry.get('function_category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("\nFunction categories:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

    return clean_data, metrics_results


def generate_sample_report(clean_data, metrics_results):
    """Generate a simple report"""
    print("\n5. Generating report...")

    # Create outputs directory
    os.makedirs('outputs/reports', exist_ok=True)

    # Create simple text report
    report_path = 'outputs/reports/sample_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=== VMP Analysis Sample Report ===\n\n")
        f.write(f"Total samples analyzed: {len(clean_data)}\n")
        f.write(f"Metrics computed for: {len(metrics_results)} samples\n\n")

        if metrics_results:
            # Summary statistics
            f.write("Summary Statistics:\n")

            metrics_to_summarize = [
                'code_expansion_rate',
                'instruction_diversity',
                'control_flow_complexity',
                'obfuscation_strength',
                'anti_debug_features',
                'vm_handler_count'
            ]

            for metric in metrics_to_summarize:
                values = [r['metrics'][metric] for r in metrics_results
                          if metric in r['metrics']]
                if values:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)

                    f.write(f"\n{metric}:\n")
                    f.write(f"  Average: {avg_val:.2f}\n")
                    f.write(f"  Min: {min_val:.2f}\n")
                    f.write(f"  Max: {max_val:.2f}\n")

            # Top protected functions
            f.write("\n\nTop Protected Functions (by code expansion):\n")
            sorted_by_expansion = sorted(metrics_results,
                                         key=lambda x: x['metrics']['code_expansion_rate'],
                                         reverse=True)

            for i, result in enumerate(sorted_by_expansion[:5]):
                f.write(f"{i + 1}. {result['function']} - {result['metrics']['code_expansion_rate']:.2f}x expansion\n")

    print(f"Report saved to: {report_path}")


def main():
    """Main example function"""
    # File path from the user's request
    jsonl_file = "/data/jiacheng/dylan/l4av/l4av/vmp_transformer/examples/mibench_all_transformations_20250616_021303_version2.jsonl"

    # Check if file exists
    if not os.path.exists(jsonl_file):
        print(f"Error: File not found - {jsonl_file}")
        print("\nPlease update the file path in the script.")
        return

    # Analyze the first entry from the provided data
    print("=== Analyzing Sample Entry ===")
    sample_entry = {
        "line": 1,
        "function": "mv88fx_snd_readl",
        "original_assembly": """下面的汇编指令对应的C源代码是什么:

<mv88fx_snd_readl>:
  endbr64
  push   %rbp
  mov    %rsp,%rbp
  sub    $0x20,%rsp
  mov    %rdi,-0x18(%rbp)
  mov    %esi,-0x1c(%rbp)
  mov    -0x1c(%rbp),%eax
  sub    $0xf000000,%eax
  mov    %eax,%edi
  call   <readl@plt>
  mov    %eax,-0x4(%rbp)
  mov    -0x4(%rbp),%eax
  leave
  ret""",
        "vmp_assembly": "; VMP Protected Assembly (x86_64)\n; Generated by Advanced VMP Transformer\n; Warning: This code is protected against debugging and tampering\n; Original function: mv88fx_snd_readl\n\nsection .data\n    vmp_code_seg db 56, 91, 48, 33, 42, 146, 109, 110, 33, 185, 243, 131, 231, 103, 59, 169, 150, 58, 1, 2, 6, 163, 32, 241, 14, 191, 75, 69, 201, 25, 1, 2, 7, 1, 2, 6, 16, 131, 188, 55, 21, 196, 78, 60, 90, 14, 1, 2, 7, 1, 2, 7, 8, 1, 32, 0, 0, 0, 0, 0, 0, 0, 184, 226, 120, 238, 105, 236, 198, 178, 128, 1, 2, 5, 1, 2, 6, 8, 1, 232, 255, 255, 255, 255, 255, 255, 255, 42, 194, 188, 66, 1, 217, 76, 152, 50, 1, 2, 20, 1, 2, 6, 8, 1, 228, 255, 255, 255, 255, 255, 255, 255, 190, 57, 197, 190, 79, 49, 122, 207, 203, 1, 2, 16, 1, 2, 6, 8, 1, 228, 255, 255, 255, 255, 255, 255, 255, 71, 150, 16, 19, 208, 167, 24, 60, 152, 14, 1, 2, 16, 1, 2, 16, 8, 1, 0, 0, 0, 15, 0, 0, 0, 0, 195, 116, 64, 181, 185, 46, 102, 166, 80, 1, 2, 16, 1, 2, 21, 172, 245, 145, 204, 7, 19, 230, 165, 7, 5, 3, 114, 101, 97, 100, 108, 173, 100, 230, 219, 15, 247, 183, 74, 87, 1, 2, 16, 1, 2, 6, 8, 1, 252, 255, 255, 255, 255, 255, 255, 255, 93, 219, 17, 26, 74, 126, 183, 133, 37, 1, 2, 16, 1, 2, 6, 8, 1, 252, 255, 255, 255, 255, 255, 255, 255, 215, 205, 227, 53, 48, 237, 146, 148, 246, 1, 2, 6, 1, 2, 7, 173, 18, 183, 26, 235, 145, 47, 129, 103, 1, 2, 0 ; 271 bytes total",
        "bytecode_size": 271,
        "function_category": "audio"
    }

    # Analyze the sample entry
    analyze_single_entry(sample_entry)

    # Try to run analysis on the actual file if it exists
    try:
        clean_data, metrics_results = run_minimal_analysis(jsonl_file)
        generate_sample_report(clean_data, metrics_results)
    except FileNotFoundError:
        print(f"\nSkipping full analysis - file not found: {jsonl_file}")

    print("\n=== Analysis Complete ===")
    print("\nTo run full analysis on your data:")
    print("1. Ensure the JSONL file path is correct")
    print("2. Run: python main.py")
    print("3. Results will be saved in the outputs/ directory")


if __name__ == '__main__':
    main()