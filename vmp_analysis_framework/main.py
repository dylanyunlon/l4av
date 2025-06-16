#!/usr/bin/env python3
"""
VMP Analysis Framework - Main Entry Point
Analyzes Virtual Machine Protection transformations
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import yaml
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.loader import DataLoader
from data.validator import DataValidator
from analysis.metrics import VMPMetricsAnalyzer
from analysis.performance import PerformanceAnalyzer
from analysis.security import SecurityAnalyzer
from analysis.statistics import StatisticalAnalyzer
from visualization.plots import PlotGenerator
from visualization.heatmaps import HeatmapGenerator
from reporting.generator import ReportGenerator
from utils.parallel import ParallelProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vmp_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VMPAnalysisFramework:
    """Main framework for VMP analysis"""

    def __init__(self, config_path='config.yaml'):
        """Initialize the framework with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            # Create default config
            config = {
                'data': {
                    'input_file': r'C:\flythings\img_output\l4vmp\test\l4av\vmp_transformer\examples\mibench_all_transformations_20250616_021303.jsonl\mibench_all_transformations_20250616_021303.jsonl',
                    'batch_size': 10000,
                    'num_workers': 8
                },
                'analysis': {
                    'metrics': ['code_expansion', 'instruction_diversity', 'control_flow_complexity',
                                'anti_debug_features', 'register_usage', 'jump_density'],
                    'enable_performance': True,
                    'enable_security': True,
                    'enable_statistics': True
                },
                'visualization': {
                    'enable_plots': True,
                    'enable_heatmaps': True,
                    'dpi': 300,
                    'figure_format': 'png'
                },
                'reporting': {
                    'generate_latex': True,
                    'generate_html': True,
                    'generate_pdf': True
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        return config

    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['outputs', 'outputs/figures', 'outputs/reports', 'outputs/results']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        start_time = time.time()
        logger.info("Starting VMP Analysis Framework")

        try:
            # 1. Load and validate data
            logger.info("Loading data...")
            loader = DataLoader(self.config['data'])
            data = loader.load_jsonl(self.config['data']['input_file'])

            logger.info("Validating data...")
            validator = DataValidator()
            clean_data = validator.validate_and_clean(data)
            logger.info(f"Loaded {len(clean_data)} valid records")

            # 2. Run analyses in parallel
            logger.info("Running VMP metrics analysis...")
            metrics_analyzer = VMPMetricsAnalyzer()
            metrics_results = ParallelProcessor.process_in_parallel(
                clean_data,
                metrics_analyzer.analyze_entry,
                self.config['data']['num_workers']
            )

            # 3. Performance analysis
            if self.config['analysis']['enable_performance']:
                logger.info("Running performance analysis...")
                perf_analyzer = PerformanceAnalyzer()
                perf_results = perf_analyzer.analyze(clean_data, metrics_results)

            # 4. Security analysis
            if self.config['analysis']['enable_security']:
                logger.info("Running security analysis...")
                sec_analyzer = SecurityAnalyzer()
                sec_results = sec_analyzer.analyze(clean_data)

            # 5. Statistical analysis
            if self.config['analysis']['enable_statistics']:
                logger.info("Running statistical analysis...")
                stat_analyzer = StatisticalAnalyzer()
                stat_results = stat_analyzer.analyze(metrics_results)

            # 6. Generate visualizations
            if self.config['visualization']['enable_plots']:
                logger.info("Generating plots...")
                plot_gen = PlotGenerator(self.config['visualization'])
                plot_gen.generate_all_plots(metrics_results, stat_results)

            if self.config['visualization']['enable_heatmaps']:
                logger.info("Generating heatmaps...")
                heatmap_gen = HeatmapGenerator(self.config['visualization'])
                heatmap_gen.generate_protection_heatmap(metrics_results)

            # 7. Generate reports
            logger.info("Generating reports...")
            report_gen = ReportGenerator(self.config['reporting'])
            report_gen.generate_comprehensive_report({
                'metrics': metrics_results,
                'performance': perf_results if self.config['analysis']['enable_performance'] else None,
                'security': sec_results if self.config['analysis']['enable_security'] else None,
                'statistics': stat_results if self.config['analysis']['enable_statistics'] else None
            })

            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VMP Analysis Framework')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input', type=str,
                        help='Override input file path')
    parser.add_argument('--workers', type=int,
                        help='Override number of workers')

    args = parser.parse_args()

    framework = VMPAnalysisFramework(args.config)

    # Override config if command line arguments provided
    if args.input:
        framework.config['data']['input_file'] = args.input
    if args.workers:
        framework.config['data']['num_workers'] = args.workers

    framework.run_analysis()


if __name__ == '__main__':
    main()