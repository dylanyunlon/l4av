"""
Report Generation Module for VMP Analysis
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive analysis reports"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.output_dir = Path('outputs/reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comprehensive_report(self, results: dict[str, Any]) -> None:
        """Generate comprehensive analysis report in multiple formats"""
        logger.info("Generating comprehensive report")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate different report formats
        if self.config.get('generate_latex', True):
            self._generate_latex_report(results, timestamp)

        if self.config.get('generate_html', True):
            self._generate_html_report(results, timestamp)

        if self.config.get('generate_pdf', False):
            self._generate_pdf_report(results, timestamp)

        # Always generate JSON summary
        self._generate_json_summary(results, timestamp)

    def _generate_latex_report(self, results: dict[str, Any], timestamp: str) -> None:
        """Generate LaTeX report"""
        latex_template = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}

\geometry{margin=1in}

\title{VMP Transformation Analysis Report}
\author{VMP Analysis Framework}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\newpage

\section{Executive Summary}

This report presents a comprehensive analysis of Virtual Machine Protection (VMP) transformations 
performed on {{ total_samples }} code samples across {{ num_categories }} function categories.

\subsection{Key Findings}

\begin{itemize}
    \item Average code expansion rate: {{ avg_expansion_rate }}x
    \item Average obfuscation strength: {{ avg_obfuscation_strength }}
    \item Total unique security mechanisms detected: {{ unique_security_mechanisms }}
    \item Functions with extreme protection (>95th percentile): {{ extreme_protection_count }}
\end{itemize}

\section{Metrics Analysis}

\subsection{Code Expansion Analysis}

{{ code_expansion_table }}

\subsection{Protection Effectiveness Metrics}

{{ protection_metrics_table }}

\section{Performance Impact Analysis}

\subsection{Estimated Performance Overhead}

{{ performance_overhead_table }}

\subsection{Memory Impact}

The VMP transformations show significant memory overhead:
\begin{itemize}
    \item Average memory overhead: {{ avg_memory_overhead }}x
    \item Maximum memory overhead observed: {{ max_memory_overhead }}x
\end{itemize}

\section{Security Analysis}

\subsection{Security Mechanisms Distribution}

{{ security_mechanisms_table }}

\subsection{Vulnerability Assessment}

{{ vulnerability_table }}

\section{Statistical Analysis}

\subsection{Correlation Analysis}

Strong correlations (|r| > 0.7) were found between:
{{ correlation_findings }}

\subsection{Clustering Results}

{{ clustering_summary }}

\section{Category-Specific Analysis}

{{ category_analysis_table }}

\section{Anomaly Detection}

\subsection{Outliers Detected}

{{ outlier_summary }}

\subsection{Failed Protections}

{{ failed_protections_summary }}

\section{Conclusions}

{{ conclusions }}

\section{Recommendations}

{{ recommendations }}

\appendix
\section{Methodology}

{{ methodology }}

\end{document}
"""

        # Prepare template data
        template_data = self._prepare_latex_data(results)

        # Render template
        template = Template(latex_template)
        latex_content = template.render(**template_data)

        # Save LaTeX file
        output_path = self.output_dir / f"vmp_analysis_report_{timestamp}.tex"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        logger.info(f"LaTeX report saved to {output_path}")

    def _prepare_latex_data(self, results: dict[str, Any]) -> dict[str, Any]:
        """Prepare data for LaTeX template"""
        data = {
            'total_samples': 0,
            'num_categories': 0,
            'avg_expansion_rate': 'N/A',
            'avg_obfuscation_strength': 'N/A',
            'unique_security_mechanisms': 0,
            'extreme_protection_count': 0,
            'avg_memory_overhead': 'N/A',
            'max_memory_overhead': 'N/A'
        }

        # Extract metrics results
        if 'metrics' in results and results['metrics']:
            metrics_df = pd.DataFrame([r['metrics'] for r in results['metrics'] if r])
            data['total_samples'] = len(metrics_df)

            if 'code_expansion_rate' in metrics_df.columns:
                data['avg_expansion_rate'] = f"{metrics_df['code_expansion_rate'].mean():.2f}"

            if 'obfuscation_strength' in metrics_df.columns:
                data['avg_obfuscation_strength'] = f"{metrics_df['obfuscation_strength'].mean():.2f}"

            # Count categories
            categories = set(r['function_category'] for r in results['metrics'] if r)
            data['num_categories'] = len(categories)

        # Extract security results
        if 'security' in results and results['security']:
            security_data = results['security']
            if 'individual_results' in security_data:
                all_mechanisms = []
                for r in security_data['individual_results']:
                    all_mechanisms.extend(r['security']['security_mechanisms'])
                data['unique_security_mechanisms'] = len(set(all_mechanisms))

        # Create tables
        data['code_expansion_table'] = self._create_latex_table_expansion(results)
        data['protection_metrics_table'] = self._create_latex_table_protection(results)
        data['performance_overhead_table'] = self._create_latex_table_performance(results)
        data['security_mechanisms_table'] = self._create_latex_table_security(results)
        data['vulnerability_table'] = self._create_latex_table_vulnerabilities(results)
        data['category_analysis_table'] = self._create_latex_table_categories(results)

        # Analysis summaries
        data['correlation_findings'] = self._summarize_correlations(results)
        data['clustering_summary'] = self._summarize_clustering(results)
        data['outlier_summary'] = self._summarize_outliers(results)
        data['failed_protections_summary'] = self._summarize_failed_protections(results)

        # Conclusions and recommendations
        data['conclusions'] = self._generate_conclusions(results)
        data['recommendations'] = self._generate_recommendations(results)
        data['methodology'] = self._describe_methodology()

        return data

    def _create_latex_table_expansion(self, results: dict[str, Any]) -> str:
        """Create LaTeX table for code expansion metrics"""
        if 'statistics' not in results or not results['statistics']:
            return "No code expansion data available."

        stats = results['statistics'].get('descriptive_stats', {})
        if 'code_expansion_rate' not in stats:
            return "No code expansion statistics available."

        expansion_stats = stats['code_expansion_rate']

        table = r"""
\begin{table}[H]
\centering
\caption{Code Expansion Rate Statistics}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
"""

        table += f"Mean & {expansion_stats['mean']:.2f}x \\\\\n"
        table += f"Median & {expansion_stats['median']:.2f}x \\\\\n"
        table += f"Std. Deviation & {expansion_stats['std']:.2f} \\\\\n"
        table += f"Minimum & {expansion_stats['min']:.2f}x \\\\\n"
        table += f"Maximum & {expansion_stats['max']:.2f}x \\\\\n"
        table += f"25th Percentile & {expansion_stats['q1']:.2f}x \\\\\n"
        table += f"75th Percentile & {expansion_stats['q3']:.2f}x \\\\\n"

        table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return table

    def _create_latex_table_protection(self, results: dict[str, Any]) -> str:
        """Create LaTeX table for protection metrics"""
        if 'statistics' not in results or not results['statistics']:
            return "No protection metrics data available."

        stats = results['statistics'].get('descriptive_stats', {})

        metrics_to_show = [
            ('obfuscation_strength', 'Obfuscation Strength'),
            ('control_flow_complexity', 'Control Flow Complexity'),
            ('instruction_diversity', 'Instruction Diversity'),
            ('anti_debug_features', 'Anti-Debug Features'),
            ('vm_handler_count', 'VM Handler Count')
        ]

        table = r"""
\begin{table}[H]
\centering
\caption{Protection Effectiveness Metrics}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Median} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
"""

        for metric_key, metric_name in metrics_to_show:
            if metric_key in stats:
                s = stats[metric_key]
                table += f"{metric_name} & {s['mean']:.2f} & {s['median']:.2f} & "
                table += f"{s['std']:.2f} & {s['min']:.2f} & {s['max']:.2f} \\\\\n"

        table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return table

    def _generate_html_report(self, results: dict[str, Any], timestamp: str) -> None:
        """Generate HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VMP Analysis Report - {{ timestamp }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }

        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1, h2, h3 {
            color: #2c3e50;
        }

        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }

        .metric-card {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #34495e;
            color: white;
            font-weight: bold;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .success {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .chart-container {
            margin: 20px 0;
            text-align: center;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .summary-item {
            text-align: center;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
        }

        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }

        .toc a {
            color: #3498db;
            text-decoration: none;
        }

        .toc a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>VMP Transformation Analysis Report</h1>
        <p><strong>Generated:</strong> {{ generation_date }}</p>

        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#summary">Executive Summary</a></li>
                <li><a href="#metrics">Metrics Analysis</a></li>
                <li><a href="#performance">Performance Impact</a></li>
                <li><a href="#security">Security Analysis</a></li>
                <li><a href="#statistics">Statistical Analysis</a></li>
                <li><a href="#conclusions">Conclusions</a></li>
            </ul>
        </div>

        <h2 id="summary">Executive Summary</h2>

        <div class="summary-grid">
            <div class="summary-item">
                <h3>Total Samples</h3>
                <div class="metric-value">{{ total_samples }}</div>
            </div>
            <div class="summary-item">
                <h3>Categories</h3>
                <div class="metric-value">{{ num_categories }}</div>
            </div>
            <div class="summary-item">
                <h3>Avg Expansion</h3>
                <div class="metric-value">{{ avg_expansion_rate }}x</div>
            </div>
            <div class="summary-item">
                <h3>Security Mechanisms</h3>
                <div class="metric-value">{{ unique_security_mechanisms }}</div>
            </div>
        </div>

        {{ content_sections }}

        <h2 id="conclusions">Conclusions</h2>
        <div class="metric-card">
            {{ conclusions }}
        </div>

        <h2>Recommendations</h2>
        <div class="success">
            {{ recommendations }}
        </div>

        <footer>
            <p style="text-align: center; color: #7f8c8d; margin-top: 50px;">
                Generated by VMP Analysis Framework - {{ timestamp }}
            </p>
        </footer>
    </div>
</body>
</html>
"""

        # Prepare template data
        template_data = self._prepare_html_data(results)
        template_data['timestamp'] = timestamp
        template_data['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)

        # Save HTML file
        output_path = self.output_dir / f"vmp_analysis_report_{timestamp}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")

    def _prepare_html_data(self, results: dict[str, Any]) -> dict[str, Any]:
        """Prepare data for HTML template"""
        data = self._prepare_latex_data(results)  # Reuse base data

        # Generate HTML-specific content sections
        content_sections = []

        # Metrics section
        if 'metrics' in results:
            content_sections.append(self._generate_html_metrics_section(results))

        # Performance section
        if 'performance' in results:
            content_sections.append(self._generate_html_performance_section(results))

        # Security section
        if 'security' in results:
            content_sections.append(self._generate_html_security_section(results))

        # Statistics section
        if 'statistics' in results:
            content_sections.append(self._generate_html_statistics_section(results))

        data['content_sections'] = '\n'.join(content_sections)

        return data

    def _generate_json_summary(self, results: dict[str, Any], timestamp: str) -> None:
        """Generate JSON summary of results"""
        summary = {
            'timestamp': timestamp,
            'generation_date': datetime.now().isoformat(),
            'summary_statistics': {},
            'key_findings': [],
            'recommendations': []
        }

        # Extract summary statistics
        if 'statistics' in results and 'descriptive_stats' in results['statistics']:
            stats = results['statistics']['descriptive_stats']

            for metric, values in stats.items():
                if isinstance(values, dict) and 'mean' in values:
                    summary['summary_statistics'][metric] = {
                        'mean': values['mean'],
                        'median': values['median'],
                        'std': values['std'],
                        'min': values['min'],
                        'max': values['max']
                    }

        # Extract key findings
        summary['key_findings'] = self._extract_key_findings(results)

        # Generate recommendations
        summary['recommendations'] = self._extract_recommendations(results)

        # Add metadata
        summary['metadata'] = {
            'total_samples_analyzed': len(results.get('metrics', [])),
            'analysis_modules_run': list(results.keys()),
            'framework_version': '1.0.0'
        }

        # Save JSON file
        output_path = self.output_dir / f"vmp_analysis_summary_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"JSON summary saved to {output_path}")

        def _extract_recommendations(self, results: dict[str, Any]) -> List[str]:
            """Generate recommendations based on analysis results"""
            recommendations = []

            # Performance-based recommendations
            if 'performance' in results and 'aggregated_stats' in results['performance']:
                perf_stats = results['performance']['aggregated_stats']
                if 'overall' in perf_stats:
                    overall = perf_stats['overall']
                    if 'estimated_execution_overhead' in overall:
                        avg_overhead = overall['estimated_execution_overhead']['mean']
                        if avg_overhead > 5:
                            recommendations.append(
                                "Consider optimizing VM dispatch mechanisms to reduce execution overhead"
                            )

            # Security-based recommendations
            if 'security' in results and 'aggregated_stats' in results['security']:
                sec_stats = results['security']['aggregated_stats']
                for category, stats in sec_stats.items():
                    if 'anti_debug_score' in stats and stats['anti_debug_score']['mean'] < 0.3:
                        recommendations.append(
                            f"Enhance anti-debugging protection for {category} functions"
                        )

            # Statistical-based recommendations
            if 'statistics' in results and 'clustering_results' in results['statistics']:
                clustering = results['statistics']['clustering_results']
                if 'cluster_profiles' in clustering:
                    weak_clusters = [c for c in clustering['cluster_profiles']
                                     if c['avg_metrics'].get('obfuscation_strength', 0) < 0.3]
                    if weak_clusters:
                        recommendations.append(
                            f"Strengthen protection for {sum(c['size'] for c in weak_clusters)} "
                            f"functions in weak protection clusters"
                        )

            return recommendations

        def _generate_conclusions(self, results: dict[str, Any]) -> str:
            """Generate conclusions from analysis"""
            conclusions = []

            # Overall effectiveness
            conclusions.append(
                "The VMP transformation system demonstrates varying levels of effectiveness "
                "across different function categories."
            )

            # Category-specific conclusions
            if 'statistics' in results and 'category_comparison' in results['statistics']:
                comparisons = results['statistics']['category_comparison']
                significant_metrics = [m for m, data in comparisons.items()
                                       if data.get('significant', False)]
                if significant_metrics:
                    conclusions.append(
                        f"Significant differences were found in {len(significant_metrics)} "
                        f"metrics across function categories, suggesting category-specific "
                        f"protection strategies may be beneficial."
                    )

            # Performance impact
            if 'performance' in results:
                conclusions.append(
                    "Performance analysis indicates substantial overhead in protected code, "
                    "with memory usage and execution time being primary concerns."
                )

            return ' '.join(conclusions)

        def _generate_recommendations(self, results: dict[str, Any]) -> str:
            """Generate detailed recommendations"""
            recs = self._extract_recommendations(results)

            if not recs:
                return "No specific recommendations at this time."

            recommendation_text = "Based on the analysis, we recommend:\n\n"
            for i, rec in enumerate(recs, 1):
                recommendation_text += f"{i}. {rec}\n"

            return recommendation_text

        def _describe_methodology(self) -> str:
            """Describe analysis methodology"""
            return """
    The analysis was performed using the following methodology:

    1. Data Collection: VMP transformation data was loaded from JSONL format files containing 
       original assembly code and VMP-protected versions.

    2. Metrics Calculation: Multiple protection effectiveness metrics were calculated including:
       - Code expansion rate
       - Instruction diversity (Shannon entropy)
       - Control flow complexity (cyclomatic complexity)
       - Obfuscation strength
       - Anti-debugging feature count
       - VM handler complexity

    3. Performance Analysis: Estimated performance impact based on:
       - Instruction cycle counts
       - Memory overhead
       - Cache impact
       - Branch prediction effects

    4. Security Analysis: Detection of security mechanisms and vulnerability assessment.

    5. Statistical Analysis: Including correlation analysis, clustering, and anomaly detection
       using standard statistical methods and machine learning techniques.
    """

        # Table generation helper methods
        def _create_latex_table_performance(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for performance metrics"""
            if 'performance' not in results or not results['performance']:
                return "No performance data available."

            perf_data = results['performance'].get('aggregated_stats', {}).get('overall', {})

            if not perf_data:
                return "No aggregated performance statistics available."

            table = r"""
    \begin{table}[H]
    \centering
    \caption{Performance Impact Analysis}
    \begin{tabular}{lr}
    \toprule
    \textbf{Metric} & \textbf{Average Value} \\
    \midrule
    """

            metric_names = {
                'estimated_execution_overhead': 'Execution Overhead',
                'memory_overhead': 'Memory Overhead',
                'cache_impact_score': 'Cache Impact',
                'branch_prediction_impact': 'Branch Prediction Impact'
            }

            for metric_key, metric_name in metric_names.items():
                if metric_key in perf_data:
                    value = perf_data[metric_key].get('mean', 0)
                    table += f"{metric_name} & {value:.2f}x \\\\\n"

            table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
            return table

        def _create_latex_table_security(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for security mechanisms"""
            if 'security' not in results or not results['security']:
                return "No security analysis data available."

            patterns = results['security'].get('security_patterns', {})
            mechanisms = patterns.get('common_mechanisms', {})

            if not mechanisms:
                return "No security mechanisms detected."

            table = r"""
    \begin{longtable}{lr}
    \caption{Security Mechanisms Distribution} \\
    \toprule
    \textbf{Security Mechanism} & \textbf{Occurrences} \\
    \midrule
    \endfirsthead
    \multicolumn{2}{c}{\textit{Continued from previous page}} \\
    \toprule
    \textbf{Security Mechanism} & \textbf{Occurrences} \\
    \midrule
    \endhead
    \bottomrule
    \endfoot
    \bottomrule
    \endlastfoot
    """

            # Sort by frequency
            sorted_mechanisms = sorted(mechanisms.items(), key=lambda x: x[1], reverse=True)

            for mechanism, count in sorted_mechanisms[:20]:  # Top 20
                # Escape special LaTeX characters
                mechanism_escaped = mechanism.replace('_', r'\_').replace('&', r'\&')
                table += f"{mechanism_escaped} & {count} \\\\\n"

            table += r"\end{longtable}"

            return table

        def _create_latex_table_vulnerabilities(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for vulnerabilities"""
            if 'security' not in results or not results['security']:
                return "No vulnerability data available."

            patterns = results['security'].get('security_patterns', {})
            vulnerabilities = patterns.get('common_vulnerabilities', {})

            if not vulnerabilities:
                return "No vulnerabilities detected in the analyzed samples."

            table = r"""
    \begin{table}[H]
    \centering
    \caption{Detected Vulnerabilities}
    \begin{tabular}{lr}
    \toprule
    \textbf{Vulnerability Type} & \textbf{Occurrences} \\
    \midrule
    """

            for vuln, count in sorted(vulnerabilities.items(), key=lambda x: x[1], reverse=True):
                vuln_escaped = vuln.replace('_', r'\_')
                table += f"{vuln_escaped} & {count} \\\\\n"

            table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
            return table

        def _create_latex_table_categories(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for category analysis"""
            if 'statistics' not in results:
                return "No category analysis data available."

            comparisons = results['statistics'].get('category_comparison', {})

            if not comparisons:
                return "No category comparison data available."

            # Find a metric with category means
            sample_metric = None
            for metric, data in comparisons.items():
                if 'category_means' in data:
                    sample_metric = metric
                    break

            if not sample_metric:
                return "No category-specific statistics available."

            categories = comparisons[sample_metric]['category_means']

            table = r"""
    \begin{table}[H]
    \centering
    \caption{Protection Metrics by Function Category}
    \begin{tabular}{lrrrr}
    \toprule
    \textbf{Category} & \textbf{Samples} & \textbf{Code Expansion} & \textbf{Complexity} & \textbf{Obfuscation} \\
    \midrule
    """

            for category in sorted(categories.keys()):
                count = categories[category]['count']

                # Get metrics for this category
                expansion = 'N/A'
                complexity = 'N/A'
                obfuscation = 'N/A'

                if 'code_expansion_rate' in comparisons:
                    cat_data = comparisons['code_expansion_rate'].get('category_means', {}).get(category, {})
                    if 'mean' in cat_data:
                        expansion = f"{cat_data['mean']:.2f}"

                if 'control_flow_complexity' in comparisons:
                    cat_data = comparisons['control_flow_complexity'].get('category_means', {}).get(category, {})
                    if 'mean' in cat_data:
                        complexity = f"{cat_data['mean']:.2f}"

                if 'obfuscation_strength' in comparisons:
                    cat_data = comparisons['obfuscation_strength'].get('category_means', {}).get(category, {})
                    if 'mean' in cat_data:
                        obfuscation = f"{cat_data['mean']:.2f}"

                table += f"{category} & {count} & {expansion} & {complexity} & {obfuscation} \\\\\n"

            table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
            return table

        def _summarize_correlations(self, results: dict[str, Any]) -> str:
            """Summarize correlation findings"""
            if 'statistics' not in results or 'correlation_analysis' not in results['statistics']:
                return "No correlation analysis performed."

            corr_analysis = results['statistics']['correlation_analysis']
            strong_corrs = corr_analysis.get('strong_correlations', [])

            if not strong_corrs:
                return "No strong correlations found between metrics."

            summary = r"\begin{itemize}" + "\n"

            for corr in strong_corrs[:5]:  # Top 5
                metric1 = corr['metric1'].replace('_', ' ').title()
                metric2 = corr['metric2'].replace('_', ' ').title()
                value = corr['correlation']
                strength = corr['strength']

                summary += f"\\item {metric1} and {metric2}: r = {value:.3f} ({strength})\n"

            summary += r"\end{itemize}"

            return summary

        def _summarize_clustering(self, results: dict[str, Any]) -> str:
            """Summarize clustering results"""
            if 'statistics' not in results or 'clustering_results' not in results['statistics']:
                return "No clustering analysis performed."

            clustering = results['statistics']['clustering_results']

            if 'cluster_profiles' not in clustering:
                return "No cluster profiles available."

            profiles = clustering['cluster_profiles']
            n_clusters = clustering.get('n_clusters', len(profiles))

            summary = f"The analysis identified {n_clusters} distinct protection patterns:\n\n"
            summary += r"\begin{itemize}" + "\n"

            for profile in profiles:
                cluster_id = profile['cluster_id']
                size = profile['size']
                percentage = profile['percentage']
                dominant_cat = profile['dominant_category']

                summary += f"\\item Cluster {cluster_id}: {size} samples ({percentage:.1f}\\%), "
                summary += f"predominantly {dominant_cat} functions\n"

            summary += r"\end{itemize}"

            return summary

        def _summarize_outliers(self, results: dict[str, Any]) -> str:
            """Summarize outlier detection results"""
            if 'statistics' not in results or 'anomaly_detection' not in results['statistics']:
                return "No anomaly detection performed."

            anomalies = results['statistics']['anomaly_detection']
            outliers = anomalies.get('outliers', {})

            if not outliers:
                return "No significant outliers detected."

            summary = "Outliers were detected in the following metrics:\n\n"
            summary += r"\begin{itemize}" + "\n"

            for metric, info in outliers.items():
                count = info['count']
                metric_name = metric.replace('_', ' ').title()
                summary += f"\\item {metric_name}: {count} outliers detected\n"

            summary += r"\end{itemize}"

            return summary

        def _summarize_failed_protections(self, results: dict[str, Any]) -> str:
            """Summarize failed protection cases"""
            if 'statistics' not in results or 'anomaly_detection' not in results['statistics']:
                return "No failed protection analysis performed."

            anomalies = results['statistics']['anomaly_detection']
            failed = anomalies.get('failed_protections', [])

            if not failed:
                return "No failed protections detected."

            summary = f"{len(failed)} functions showed signs of failed protection:\n\n"
            summary += r"\begin{itemize}" + "\n"

            for failure in failed[:5]:  # Show top 5
                func = failure['function']
                category = failure['category']
                summary += f"\\item {func} ({category} category)\n"

            if len(failed) > 5:
                summary += f"\\item ... and {len(failed) - 5} more\n"

            summary += r"\end{itemize}"

            return summary

        # HTML generation helper methods
        def _generate_html_metrics_section(self, results: dict[str, Any]) -> str:
            """Generate HTML section for metrics analysis"""
            section = '<h2 id="metrics">Metrics Analysis</h2>\n'

            if 'statistics' in results and 'descriptive_stats' in results['statistics']:
                stats = results['statistics']['descriptive_stats']

                section += '<h3>Code Protection Metrics</h3>\n'
                section += '<table>\n<thead>\n<tr>\n'
                section += '<th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th>\n'
                section += '</tr>\n</thead>\n<tbody>\n'

                for metric, values in stats.items():
                    if isinstance(values, dict) and 'mean' in values:
                        metric_name = metric.replace('_', ' ').title()
                        section += f'<tr>\n'
                        section += f'<td>{metric_name}</td>\n'
                        section += f'<td>{values["mean"]:.2f}</td>\n'
                        section += f'<td>{values["median"]:.2f}</td>\n'
                        section += f'<td>{values["std"]:.2f}</td>\n'
                        section += f'<td>{values["min"]:.2f}</td>\n'
                        section += f'<td>{values["max"]:.2f}</td>\n'
                        section += '</tr>\n'

                section += '</tbody>\n</table>\n'

            return section

        def _generate_html_performance_section(self, results: dict[str, Any]) -> str:
            """Generate HTML section for performance analysis"""
            section = '<h2 id="performance">Performance Impact</h2>\n'

            if 'performance' not in results or not results['performance']:
                section += '<p>No performance analysis data available.</p>\n'
                return section

            perf_data = results['performance'].get('aggregated_stats', {}).get('overall', {})

            if perf_data:
                section += '<div class="metric-card">\n'
                section += '<h3>Average Performance Overhead</h3>\n'
                section += '<div class="summary-grid">\n'

                metrics = [
                    ('estimated_execution_overhead', 'Execution', 'x'),
                    ('memory_overhead', 'Memory', 'x'),
                    ('cache_impact_score', 'Cache Impact', ''),
                    ('branch_prediction_impact', 'Branch Pred.', '')
                ]

                for metric_key, label, suffix in metrics:
                    if metric_key in perf_data:
                        value = perf_data[metric_key].get('mean', 0)
                        section += f'<div class="summary-item">\n'
                        section += f'<h4>{label}</h4>\n'
                        section += f'<div class="metric-value">{value:.2f}{suffix}</div>\n'
                        section += '</div>\n'

                section += '</div>\n</div>\n'

            return section

        def _generate_html_security_section(self, results: dict[str, Any]) -> str:
            """Generate HTML section for security analysis"""
            section = '<h2 id="security">Security Analysis</h2>\n'

            if 'security' not in results or not results['security']:
                section += '<p>No security analysis data available.</p>\n'
                return section

            patterns = results['security'].get('security_patterns', {})

            # Common mechanisms
            if 'common_mechanisms' in patterns:
                mechanisms = patterns['common_mechanisms']
                section += '<h3>Top Security Mechanisms</h3>\n'
                section += '<table>\n<thead>\n<tr>\n'
                section += '<th>Mechanism</th><th>Occurrences</th>\n'
                section += '</tr>\n</thead>\n<tbody>\n'

                sorted_mechs = sorted(mechanisms.items(), key=lambda x: x[1], reverse=True)[:10]

                for mechanism, count in sorted_mechs:
                    section += f'<tr><td>{mechanism}</td><td>{count}</td></tr>\n'

                section += '</tbody>\n</table>\n'

            # Vulnerabilities
            if 'common_vulnerabilities' in patterns:
                vulns = patterns['common_vulnerabilities']
                if vulns:
                    section += '<div class="warning">\n'
                    section += '<h3>Detected Vulnerabilities</h3>\n'
                    section += '<ul>\n'

                    for vuln, count in sorted(vulns.items(), key=lambda x: x[1], reverse=True):
                        section += f'<li>{vuln}: {count} occurrences</li>\n'

                    section += '</ul>\n</div>\n'

            return section

        def _generate_html_statistics_section(self, results: dict[str, Any]) -> str:
            """Generate HTML section for statistical analysis"""
            section = '<h2 id="statistics">Statistical Analysis</h2>\n'

            if 'statistics' not in results:
                section += '<p>No statistical analysis data available.</p>\n'
                return section

            stats = results['statistics']

            # Clustering results
            if 'clustering_results' in stats:
                clustering = stats['clustering_results']
                section += '<h3>Protection Pattern Clustering</h3>\n'
                section += f'<p>Identified {clustering.get("n_clusters", 0)} distinct protection patterns '
                section += f'with {clustering.get("explained_variance", 0):.1%} variance explained.</p>\n'

            # Category comparisons
            if 'category_comparison' in stats:
                section += '<h3>Category Analysis</h3>\n'
                comparisons = stats['category_comparison']

                significant_diffs = [m for m, d in comparisons.items() if d.get('significant', False)]
                if significant_diffs:
                    section += '<p>Significant differences found in: '
                    section += ', '.join(m.replace('_', ' ').title() for m in significant_diffs)
                    section += '</p>\n'

            return section

        def _generate_pdf_report(self, results: dict[str, Any], timestamp: str) -> None:
            """Generate PDF report (requires LaTeX)"""
            # First generate LaTeX
            self._generate_latex_report(results, timestamp)

            # Note: PDF generation would require pdflatex to be installed
            logger.info("PDF generation requires LaTeX installation. "
                        "Please compile the generated .tex file manually.")
            """
    Report Generation Module for VMP Analysis
    """

    import logging
    import json
    from datetime import datetime
    from pathlib import Path
    from typing import Dict, Any, List
    import pandas as pd
    from jinja2 import Template

    logger = logging.getLogger(__name__)

    class ReportGenerator:
        """Generates comprehensive analysis reports"""

        def __init__(self, config: dict[str, Any]):
            self.config = config
            self.output_dir = Path('outputs/reports')
            self.output_dir.mkdir(parents=True, exist_ok=True)

        def generate_comprehensive_report(self, results: dict[str, Any]) -> None:
            """Generate comprehensive analysis report in multiple formats"""
            logger.info("Generating comprehensive report")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate different report formats
            if self.config.get('generate_latex', True):
                self._generate_latex_report(results, timestamp)

            if self.config.get('generate_html', True):
                self._generate_html_report(results, timestamp)

            if self.config.get('generate_pdf', False):
                self._generate_pdf_report(results, timestamp)

            # Always generate JSON summary
            self._generate_json_summary(results, timestamp)

        def _generate_latex_report(self, results: dict[str, Any], timestamp: str) -> None:
            """Generate LaTeX report"""
            latex_template = r"""
    \documentclass[11pt,a4paper]{article}
    \usepackage[utf8]{inputenc}
    \usepackage[T1]{fontenc}
    \usepackage{graphicx}
    \usepackage{booktabs}
    \usepackage{float}
    \usepackage{hyperref}
    \usepackage{geometry}
    \usepackage{longtable}
    \usepackage{array}
    \usepackage{xcolor}
    \usepackage{colortbl}

    \geometry{margin=1in}

    \title{VMP Transformation Analysis Report}
    \author{VMP Analysis Framework}
    \date{\today}

    \begin{document}

    \maketitle
    \tableofcontents
    \newpage

    \section{Executive Summary}

    This report presents a comprehensive analysis of Virtual Machine Protection (VMP) transformations 
    performed on {{ total_samples }} code samples across {{ num_categories }} function categories.

    \subsection{Key Findings}

    \begin{itemize}
        \item Average code expansion rate: {{ avg_expansion_rate }}x
        \item Average obfuscation strength: {{ avg_obfuscation_strength }}
        \item Total unique security mechanisms detected: {{ unique_security_mechanisms }}
        \item Functions with extreme protection (>95th percentile): {{ extreme_protection_count }}
    \end{itemize}

    \section{Metrics Analysis}

    \subsection{Code Expansion Analysis}

    {{ code_expansion_table }}

    \subsection{Protection Effectiveness Metrics}

    {{ protection_metrics_table }}

    \section{Performance Impact Analysis}

    \subsection{Estimated Performance Overhead}

    {{ performance_overhead_table }}

    \subsection{Memory Impact}

    The VMP transformations show significant memory overhead:
    \begin{itemize}
        \item Average memory overhead: {{ avg_memory_overhead }}x
        \item Maximum memory overhead observed: {{ max_memory_overhead }}x
    \end{itemize}

    \section{Security Analysis}

    \subsection{Security Mechanisms Distribution}

    {{ security_mechanisms_table }}

    \subsection{Vulnerability Assessment}

    {{ vulnerability_table }}

    \section{Statistical Analysis}

    \subsection{Correlation Analysis}

    Strong correlations (|r| > 0.7) were found between:
    {{ correlation_findings }}

    \subsection{Clustering Results}

    {{ clustering_summary }}

    \section{Category-Specific Analysis}

    {{ category_analysis_table }}

    \section{Anomaly Detection}

    \subsection{Outliers Detected}

    {{ outlier_summary }}

    \subsection{Failed Protections}

    {{ failed_protections_summary }}

    \section{Conclusions}

    {{ conclusions }}

    \section{Recommendations}

    {{ recommendations }}

    \appendix
    \section{Methodology}

    {{ methodology }}

    \end{document}
    """

            # Prepare template data
            template_data = self._prepare_latex_data(results)

            # Render template
            template = Template(latex_template)
            latex_content = template.render(**template_data)

            # Save LaTeX file
            output_path = self.output_dir / f"vmp_analysis_report_{timestamp}.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)

            logger.info(f"LaTeX report saved to {output_path}")

        def _prepare_latex_data(self, results: dict[str, Any]) -> dict[str, Any]:
            """Prepare data for LaTeX template"""
            data = {
                'total_samples': 0,
                'num_categories': 0,
                'avg_expansion_rate': 'N/A',
                'avg_obfuscation_strength': 'N/A',
                'unique_security_mechanisms': 0,
                'extreme_protection_count': 0,
                'avg_memory_overhead': 'N/A',
                'max_memory_overhead': 'N/A'
            }

            # Extract metrics results
            if 'metrics' in results and results['metrics']:
                metrics_df = pd.DataFrame([r['metrics'] for r in results['metrics'] if r])
                data['total_samples'] = len(metrics_df)

                if 'code_expansion_rate' in metrics_df.columns:
                    data['avg_expansion_rate'] = f"{metrics_df['code_expansion_rate'].mean():.2f}"

                if 'obfuscation_strength' in metrics_df.columns:
                    data['avg_obfuscation_strength'] = f"{metrics_df['obfuscation_strength'].mean():.2f}"

                # Count categories
                categories = set(r['function_category'] for r in results['metrics'] if r)
                data['num_categories'] = len(categories)

            # Extract security results
            if 'security' in results and results['security']:
                security_data = results['security']
                if 'individual_results' in security_data:
                    all_mechanisms = []
                    for r in security_data['individual_results']:
                        all_mechanisms.extend(r['security']['security_mechanisms'])
                    data['unique_security_mechanisms'] = len(set(all_mechanisms))

            # Create tables
            data['code_expansion_table'] = self._create_latex_table_expansion(results)
            data['protection_metrics_table'] = self._create_latex_table_protection(results)
            data['performance_overhead_table'] = self._create_latex_table_performance(results)
            data['security_mechanisms_table'] = self._create_latex_table_security(results)
            data['vulnerability_table'] = self._create_latex_table_vulnerabilities(results)
            data['category_analysis_table'] = self._create_latex_table_categories(results)

            # Analysis summaries
            data['correlation_findings'] = self._summarize_correlations(results)
            data['clustering_summary'] = self._summarize_clustering(results)
            data['outlier_summary'] = self._summarize_outliers(results)
            data['failed_protections_summary'] = self._summarize_failed_protections(results)

            # Conclusions and recommendations
            data['conclusions'] = self._generate_conclusions(results)
            data['recommendations'] = self._generate_recommendations(results)
            data['methodology'] = self._describe_methodology()

            return data

        def _create_latex_table_expansion(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for code expansion metrics"""
            if 'statistics' not in results or not results['statistics']:
                return "No code expansion data available."

            stats = results['statistics'].get('descriptive_stats', {})
            if 'code_expansion_rate' not in stats:
                return "No code expansion statistics available."

            expansion_stats = stats['code_expansion_rate']

            table = r"""
    \begin{table}[H]
    \centering
    \caption{Code Expansion Rate Statistics}
    \begin{tabular}{lr}
    \toprule
    \textbf{Statistic} & \textbf{Value} \\
    \midrule
    """

            table += f"Mean & {expansion_stats['mean']:.2f}x \\\\\n"
            table += f"Median & {expansion_stats['median']:.2f}x \\\\\n"
            table += f"Std. Deviation & {expansion_stats['std']:.2f} \\\\\n"
            table += f"Minimum & {expansion_stats['min']:.2f}x \\\\\n"
            table += f"Maximum & {expansion_stats['max']:.2f}x \\\\\n"
            table += f"25th Percentile & {expansion_stats['q1']:.2f}x \\\\\n"
            table += f"75th Percentile & {expansion_stats['q3']:.2f}x \\\\\n"

            table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
            return table

        def _create_latex_table_protection(self, results: dict[str, Any]) -> str:
            """Create LaTeX table for protection metrics"""
            if 'statistics' not in results or not results['statistics']:
                return "No protection metrics data available."

            stats = results['statistics'].get('descriptive_stats', {})

            metrics_to_show = [
                ('obfuscation_strength', 'Obfuscation Strength'),
                ('control_flow_complexity', 'Control Flow Complexity'),
                ('instruction_diversity', 'Instruction Diversity'),
                ('anti_debug_features', 'Anti-Debug Features'),
                ('vm_handler_count', 'VM Handler Count')
            ]

            table = r"""
    \begin{table}[H]
    \centering
    \caption{Protection Effectiveness Metrics}
    \begin{tabular}{lrrrrr}
    \toprule
    \textbf{Metric} & \textbf{Mean} & \textbf{Median} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
    \midrule
    """

            for metric_key, metric_name in metrics_to_show:
                if metric_key in stats:
                    s = stats[metric_key]
                    table += f"{metric_name} & {s['mean']:.2f} & {s['median']:.2f} & "
                    table += f"{s['std']:.2f} & {s['min']:.2f} & {s['max']:.2f} \\\\\n"

            table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
            return table

        def _generate_html_report(self, results: dict[str, Any], timestamp: str) -> None:
            """Generate HTML report"""
            html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VMP Analysis Report - {{ timestamp }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }

            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            h1, h2, h3 {
                color: #2c3e50;
            }

            h1 {
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }

            .metric-card {
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }

            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }

            th {
                background-color: #34495e;
                color: white;
                font-weight: bold;
            }

            tr:hover {
                background-color: #f5f5f5;
            }

            .warning {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }

            .success {
                background-color: #d4edda;
                border-left: 4px solid #28a745;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }

            .chart-container {
                margin: 20px 0;
                text-align: center;
            }

            .summary-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }

            .summary-item {
                text-align: center;
                padding: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
            }

            .toc {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }

            .toc ul {
                list-style-type: none;
                padding-left: 20px;
            }

            .toc a {
                color: #3498db;
                text-decoration: none;
            }

            .toc a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VMP Transformation Analysis Report</h1>
            <p><strong>Generated:</strong> {{ generation_date }}</p>

            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#summary">Executive Summary</a></li>
                    <li><a href="#metrics">Metrics Analysis</a></li>
                    <li><a href="#performance">Performance Impact</a></li>
                    <li><a href="#security">Security Analysis</a></li>
                    <li><a href="#statistics">Statistical Analysis</a></li>
                    <li><a href="#conclusions">Conclusions</a></li>
                </ul>
            </div>

            <h2 id="summary">Executive Summary</h2>

            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Samples</h3>
                    <div class="metric-value">{{ total_samples }}</div>
                </div>
                <div class="summary-item">
                    <h3>Categories</h3>
                    <div class="metric-value">{{ num_categories }}</div>
                </div>
                <div class="summary-item">
                    <h3>Avg Expansion</h3>
                    <div class="metric-value">{{ avg_expansion_rate }}x</div>
                </div>
                <div class="summary-item">
                    <h3>Security Mechanisms</h3>
                    <div class="metric-value">{{ unique_security_mechanisms }}</div>
                </div>
            </div>

            {{ content_sections }}

            <h2 id="conclusions">Conclusions</h2>
            <div class="metric-card">
                {{ conclusions }}
            </div>

            <h2>Recommendations</h2>
            <div class="success">
                {{ recommendations }}
            </div>

            <footer>
                <p style="text-align: center; color: #7f8c8d; margin-top: 50px;">
                    Generated by VMP Analysis Framework - {{ timestamp }}
                </p>
            </footer>
        </div>
    </body>
    </html>
    """

            # Prepare template data
            template_data = self._prepare_html_data(results)
            template_data['timestamp'] = timestamp
            template_data['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Render template
            template = Template(html_template)
            html_content = template.render(**template_data)

            # Save HTML file
            output_path = self.output_dir / f"vmp_analysis_report_{timestamp}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {output_path}")

        def _prepare_html_data(self, results: dict[str, Any]) -> dict[str, Any]:
            """Prepare data for HTML template"""
            data = self._prepare_latex_data(results)  # Reuse base data

            # Generate HTML-specific content sections
            content_sections = []

            # Metrics section
            if 'metrics' in results:
                content_sections.append(self._generate_html_metrics_section(results))

            # Performance section
            if 'performance' in results:
                content_sections.append(self._generate_html_performance_section(results))

            # Security section
            if 'security' in results:
                content_sections.append(self._generate_html_security_section(results))

            # Statistics section
            if 'statistics' in results:
                content_sections.append(self._generate_html_statistics_section(results))

            data['content_sections'] = '\n'.join(content_sections)

            return data

        def _generate_json_summary(self, results: dict[str, Any], timestamp: str) -> None:
            """Generate JSON summary of results"""
            summary = {
                'timestamp': timestamp,
                'generation_date': datetime.now().isoformat(),
                'summary_statistics': {},
                'key_findings': [],
                'recommendations': []
            }

            # Extract summary statistics
            if 'statistics' in results and 'descriptive_stats' in results['statistics']:
                stats = results['statistics']['descriptive_stats']

                for metric, values in stats.items():
                    if isinstance(values, dict) and 'mean' in values:
                        summary['summary_statistics'][metric] = {
                            'mean': values['mean'],
                            'median': values['median'],
                            'std': values['std'],
                            'min': values['min'],
                            'max': values['max']
                        }

            # Extract key findings
            summary['key_findings'] = self._extract_key_findings(results)

            # Generate recommendations
            summary['recommendations'] = self._extract_recommendations(results)

            # Add metadata
            summary['metadata'] = {
                'total_samples_analyzed': len(results.get('metrics', [])),
                'analysis_modules_run': list(results.keys()),
                'framework_version': '1.0.0'
            }

            # Save JSON file
            output_path = self.output_dir / f"vmp_analysis_summary_{timestamp}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"JSON summary saved to {output_path}")

        def _extract_key_findings(self, results: dict[str, Any]) -> List[str]:
            """Extract key findings from results"""
            findings = []

            # Analyze protection effectiveness
            if 'statistics' in results and 'protection_strength_analysis' in results['statistics']:
                protection_analysis = results['statistics']['protection_strength_analysis']
                if 'protection_distribution' in protection_analysis:
                    dist = protection_analysis['protection_distribution']
                    high_protection = dist.get('high', {}).get('percentage', 0)
                    findings.append(f"{high_protection:.1f}% of functions achieved high protection levels")

            # Analyze security mechanisms
            if 'security' in results and 'security_patterns' in results['security']:
                patterns = results['security']['security_patterns']
                if 'common_mechanisms' in patterns:
                    top_mechanisms = sorted(patterns['common_mechanisms'].items(),
                                            key=lambda x: x[1], reverse=True)[:3]
                    for mechanism, count in top_mechanisms:
                        findings.append(f"{mechanism} was detected in {count} functions")

            # Analyze anomalies
            if 'statistics' in results and 'anomaly_detection' in results['statistics']:
                anomalies = results['statistics']['anomaly_detection']
                if 'failed_protections' in anomalies:
                    failed_count = len(anomalies['failed_protections'])
                    if failed_count > 0:
                        findings.append(f"{failed_count} functions showed signs of failed protection")

            return findings

