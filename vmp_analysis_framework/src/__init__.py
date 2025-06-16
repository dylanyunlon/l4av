# src/__init__.py
"""VMP Analysis Framework - Main package"""

__version__ = "1.0.0"
__author__ = "VMP Analysis Team"


# src/data/__init__.py
from .loader import DataLoader
from .preprocessor import DataPreprocessor

__all__ = ['DataLoader', 'DataPreprocessor']


# src/analysis/__init__.py
from .base_analyzer import BaseAnalyzer
from .code_metrics import CodeMetricsAnalyzer
from .performance import PerformanceAnalyzer
from .security import SecurityAnalyzer

__all__ = [
    'BaseAnalyzer',
    'CodeMetricsAnalyzer',
    'PerformanceAnalyzer',
    'SecurityAnalyzer'
]


# src/metrics/__init__.py
"""Metrics calculation modules"""

# This package can be extended with specific metric calculators
# Currently metrics are calculated within the analyzer modules


# src/statistics/__init__.py
from .descriptive import DescriptiveStatistics
from .correlation import CorrelationAnalysis
from .clustering import ClusteringAnalysis
from .anomaly import AnomalyDetector

__all__ = [
    'DescriptiveStatistics',
    'CorrelationAnalysis',
    'ClusteringAnalysis',
    'AnomalyDetector'
]


# src/visualization/__init__.py
from .plotter import VisualizationManager
from .heatmaps import HeatmapGenerator
from .distributions import DistributionPlotter

__all__ = [
    'VisualizationManager',
    'HeatmapGenerator',
    'DistributionPlotter'
]


# src/reporting/__init__.py
from .report_generator import ReportGenerator
from .latex_exporter import LaTeXExporter

__all__ = ['ReportGenerator', 'LaTeXExporter']