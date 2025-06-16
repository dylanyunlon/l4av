"""
Visualization Module for VMP Analysis
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generates various plots for VMP analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path('outputs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def generate_all_plots(self, metrics_results: List[Dict[str, Any]],
                           stat_results: Dict[str, Any]) -> None:
        """Generate all visualization plots"""
        logger.info("Generating visualization plots")

        # Convert to DataFrame for easier plotting
        df = self._prepare_dataframe(metrics_results)

        if df.empty:
            logger.warning("No data available for plotting")
            return

        # Generate various plots
        self._plot_metric_distributions(df)
        self._plot_category_comparisons(df)
        self._plot_correlation_heatmap(stat_results)
        self._plot_protection_effectiveness(df)
        self._plot_anomaly_detection(df, stat_results)
        self._plot_clustering_results(stat_results)
        self._plot_performance_impact(df)

    def _prepare_dataframe(self, metrics_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert metrics results to DataFrame"""
        data = []

        for result in metrics_results:
            if result is None:
                continue

            row = {
                'function': result['function'],
                'category': result['function_category'],
                'bytecode_size': result['bytecode_size']
            }

            # Flatten metrics
            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    row[metric] = value

            data.append(row)

        return pd.DataFrame(data)

    def _plot_metric_distributions(self, df: pd.DataFrame) -> None:
        """Plot distributions of key metrics"""
        metrics = ['code_expansion_rate', 'instruction_diversity',
                   'control_flow_complexity', 'obfuscation_strength']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i]

                # Plot histogram with KDE
                df[metric].hist(bins=50, ax=ax, alpha=0.7, density=True)
                df[metric].plot.density(ax=ax, color='red', linewidth=2)

                ax.set_title(f'Distribution of {metric.replace("_", " ").title()}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel(metric.replace("_", " ").title())
                ax.set_ylabel('Density')

                # Add statistics
                mean = df[metric].mean()
                median = df[metric].median()
                ax.axvline(mean, color='green', linestyle='--',
                           label=f'Mean: {mean:.2f}')
                ax.axvline(median, color='orange', linestyle='--',
                           label=f'Median: {median:.2f}')
                ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_distributions.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_category_comparisons(self, df: pd.DataFrame) -> None:
        """Plot metric comparisons across categories"""
        metrics = ['code_expansion_rate', 'control_flow_complexity',
                   'obfuscation_strength', 'vm_handler_count']

        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]

            # Create box plot
            df.boxplot(column=metric, by='category', ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()} by Category')
            ax.set_xlabel('Category')
            ax.set_ylabel(metric.replace("_", " ").title())

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle('Metric Comparisons Across Function Categories')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_comparisons.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, stat_results: Dict[str, Any]) -> None:
        """Plot correlation heatmap of metrics"""
        if 'correlation_analysis' not in stat_results:
            return

        corr_data = stat_results['correlation_analysis'].get('correlation_matrix', {})
        if not corr_data:
            return

        # Convert to DataFrame
        corr_df = pd.DataFrame(corr_data)

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))

        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5)

        plt.title('Correlation Heatmap of VMP Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_protection_effectiveness(self, df: pd.DataFrame) -> None:
        """Plot protection effectiveness analysis"""
        # Create scatter plot matrix for key protection metrics
        protection_metrics = ['code_expansion_rate', 'obfuscation_strength',
                              'control_flow_complexity', 'anti_debug_features']

        available_metrics = [m for m in protection_metrics if m in df.columns]

        if len(available_metrics) >= 2:
            # Create interactive scatter matrix using plotly
            fig = px.scatter_matrix(df[available_metrics + ['category']],
                                    dimensions=available_metrics,
                                    color='category',
                                    title='Protection Metrics Scatter Matrix',
                                    height=1000)

            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.write_html(self.output_dir / 'protection_scatter_matrix.html')

        # Create radar chart for average protection by category
        if len(available_metrics) >= 3:
            categories = df['category'].unique()[:6]  # Limit to 6 categories

            fig = go.Figure()

            for category in categories:
                cat_data = df[df['category'] == category]
                if len(cat_data) > 0:
                    values = [cat_data[metric].mean() for metric in available_metrics]

                    # Normalize values to 0-1 range
                    max_vals = [df[metric].max() for metric in available_metrics]
                    normalized_values = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]

                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=available_metrics,
                        fill='toself',
                        name=category
                    ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Protection Profile by Category (Normalized)"
            )

            fig.write_html(self.output_dir / 'protection_radar_chart.html')

    def _plot_anomaly_detection(self, df: pd.DataFrame, stat_results: Dict[str, Any]) -> None:
        """Visualize anomaly detection results"""
        if 'anomaly_detection' not in stat_results:
            return

        anomalies = stat_results['anomaly_detection']

        # Plot outliers for key metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        metrics_to_plot = ['code_expansion_rate', 'control_flow_complexity',
                           'obfuscation_strength', 'bytecode_size']

        for i, metric in enumerate(metrics_to_plot):
            if metric in df.columns and metric in anomalies.get('outliers', {}):
                ax = axes[i]

                # Plot all data points
                ax.scatter(range(len(df)), df[metric], alpha=0.5, s=20)

                # Highlight outliers
                outlier_info = anomalies['outliers'][metric]
                outlier_indices = df[df['function'].isin(outlier_info['functions'])].index

                if len(outlier_indices) > 0:
                    ax.scatter(outlier_indices, df.loc[outlier_indices, metric],
                               color='red', s=50, marker='x', label='Outliers')

                ax.set_title(f'Outlier Detection: {metric.replace("_", " ").title()}')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_clustering_results(self, stat_results: Dict[str, Any]) -> None:
        """Plot clustering analysis results"""
        if 'clustering_results' not in stat_results:
            return

        clustering = stat_results['clustering_results']

        if 'pca_data' in clustering:
            # Create PCA scatter plot
            pca_data = clustering['pca_data']

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_data['x'], pca_data['y'],
                                  c=pca_data['clusters'], cmap='viridis',
                                  alpha=0.6, s=50)

            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('VMP Protection Patterns - Clustering Results\n' +
                      f'(Explained Variance: {clustering["explained_variance"]:.2%})')

            # Add cluster centers (approximate)
            for cluster_profile in clustering['cluster_profiles']:
                cluster_id = cluster_profile['cluster_id']
                cluster_points = [(x, y) for x, y, c in zip(pca_data['x'],
                                                            pca_data['y'],
                                                            pca_data['clusters'])
                                  if c == cluster_id]

                if cluster_points:
                    center_x = np.mean([p[0] for p in cluster_points])
                    center_y = np.mean([p[1] for p in cluster_points])

                    plt.plot(center_x, center_y, 'r*', markersize=15,
                             markeredgecolor='black', markeredgewidth=1)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'clustering_pca.png',
                        dpi=self.config['dpi'], bbox_inches='tight')
            plt.close()

            # Create cluster profile bar chart
            self._plot_cluster_profiles(clustering['cluster_profiles'])

    def _plot_cluster_profiles(self, cluster_profiles: List[Dict[str, Any]]) -> None:
        """Plot cluster profiles as grouped bar chart"""
        if not cluster_profiles:
            return

        # Extract metrics for each cluster
        clusters = []
        metrics = []

        for profile in cluster_profiles:
            clusters.append(f"Cluster {profile['cluster_id']}\n({profile['size']} samples)")
            if profile['avg_metrics']:
                if not metrics:
                    metrics = list(profile['avg_metrics'].keys())

        if not metrics:
            return

        # Create data for plotting
        n_clusters = len(clusters)
        n_metrics = len(metrics)
        x = np.arange(n_clusters)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(12, 8))

        for i, metric in enumerate(metrics):
            values = [profile['avg_metrics'].get(metric, 0) for profile in cluster_profiles]
            offset = (i - n_metrics / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Clusters')
        ax.set_ylabel('Average Metric Value')
        ax.set_title('Cluster Profiles - Average Metrics per Cluster')
        ax.set_xticks(x)
        ax.set_xticklabels(clusters)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cluster_profiles.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_performance_impact(self, df: pd.DataFrame) -> None:
        """Plot performance impact visualizations"""
        # Create subplot with performance-related metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Code Expansion vs Bytecode Size',
                            'Complexity vs Jump Density',
                            'VM Handlers Distribution',
                            'Protection Overhead by Category')
        )

        # 1. Code expansion vs bytecode size
        if 'code_expansion_rate' in df.columns and 'bytecode_size' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['bytecode_size'], y=df['code_expansion_rate'],
                           mode='markers', marker=dict(size=5, opacity=0.6),
                           name='Samples'),
                row=1, col=1
            )

        # 2. Complexity vs jump density
        if 'control_flow_complexity' in df.columns and 'jump_density' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['jump_density'], y=df['control_flow_complexity'],
                           mode='markers', marker=dict(size=5, opacity=0.6),
                           name='Samples'),
                row=1, col=2
            )

        # 3. VM handlers distribution
        if 'vm_handler_count' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['vm_handler_count'], nbinsx=20,
                             name='VM Handlers'),
                row=2, col=1
            )

        # 4. Protection overhead by category
        if 'code_expansion_rate' in df.columns:
            category_overhead = df.groupby('category')['code_expansion_rate'].mean().sort_values()

            fig.add_trace(
                go.Bar(x=category_overhead.index, y=category_overhead.values,
                       name='Avg Expansion'),
                row=2, col=2
            )

        fig.update_layout(height=1000, showlegend=False,
                          title_text="Performance Impact Analysis")
        fig.write_html(self.output_dir / 'performance_impact.html')


class HeatmapGenerator:
    """Generates heatmap visualizations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path('outputs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_protection_heatmap(self, metrics_results: List[Dict[str, Any]]) -> None:
        """Generate heatmap showing protection effectiveness"""
        logger.info("Generating protection heatmap")

        # Prepare data
        df = self._prepare_dataframe(metrics_results)

        if df.empty:
            logger.warning("No data available for heatmap")
            return

        # Create function-level protection heatmap
        self._create_function_heatmap(df)

        # Create category-level summary heatmap
        self._create_category_heatmap(df)

    def _prepare_dataframe(self, metrics_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert metrics results to DataFrame"""
        data = []

        for result in metrics_results:
            if result is None:
                continue

            row = {
                'function': result['function'],
                'category': result['function_category'],
            }

            # Extract key metrics
            metrics_to_include = [
                'code_expansion_rate', 'instruction_diversity',
                'control_flow_complexity', 'obfuscation_strength',
                'anti_debug_features', 'vm_handler_count'
            ]

            for metric in metrics_to_include:
                if metric in result['metrics']:
                    row[metric] = result['metrics'][metric]

            data.append(row)

        return pd.DataFrame(data)

    def _create_function_heatmap(self, df: pd.DataFrame) -> None:
        """Create heatmap for individual functions"""
        # Select top N functions by various criteria
        n_functions = min(50, len(df))

        # Get functions with highest protection scores
        metrics_cols = [col for col in df.columns if col not in ['function', 'category']]

        if not metrics_cols:
            return

        # Normalize metrics
        df_norm = df.copy()
        for col in metrics_cols:
            if df[col].std() > 0:
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
            else:
                df_norm[col] = 0

        # Calculate composite score and select top functions
        df_norm['composite_score'] = df_norm[metrics_cols].mean(axis=1)
        top_functions = df_norm.nlargest(n_functions, 'composite_score')

        # Create heatmap
        plt.figure(figsize=(10, 12))

        heatmap_data = top_functions[metrics_cols].values

        sns.heatmap(heatmap_data,
                    xticklabels=[col.replace('_', ' ').title() for col in metrics_cols],
                    yticklabels=top_functions['function'].values,
                    cmap='RdYlBu_r', center=0,
                    cbar_kws={'label': 'Normalized Score'})

        plt.title(f'Protection Metrics Heatmap - Top {n_functions} Functions',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Functions')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'function_protection_heatmap.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

    def _create_category_heatmap(self, df: pd.DataFrame) -> None:
        """Create category-level summary heatmap"""
        metrics_cols = [col for col in df.columns if col not in ['function', 'category']]

        if not metrics_cols:
            return

        # Calculate average metrics per category
        category_summary = df.groupby('category')[metrics_cols].mean()

        # Normalize for better visualization
        category_norm = category_summary.copy()
        for col in metrics_cols:
            if category_summary[col].std() > 0:
                category_norm[col] = (category_summary[col] - category_summary[col].mean()) / category_summary[
                    col].std()
            else:
                category_norm[col] = 0

        # Create heatmap
        plt.figure(figsize=(10, 8))

        sns.heatmap(category_norm,
                    xticklabels=[col.replace('_', ' ').title() for col in metrics_cols],
                    yticklabels=category_norm.index,
                    cmap='RdYlBu_r', center=0, annot=True, fmt='.2f',
                    cbar_kws={'label': 'Normalized Average Score'})

        plt.title('Protection Effectiveness by Function Category',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Protection Metrics')
        plt.ylabel('Function Category')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'category_protection_heatmap.png',
                    dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()