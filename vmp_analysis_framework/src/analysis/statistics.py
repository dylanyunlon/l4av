"""
Statistical Analysis Module for VMP Transformations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on VMP transformation metrics"""

    def __init__(self):
        self.significance_level = 0.05

    def analyze(self, metrics_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        logger.info("Starting statistical analysis")

        # Convert to DataFrame for easier analysis
        df = self._prepare_dataframe(metrics_results)

        if df.empty:
            logger.warning("No data available for statistical analysis")
            return {}

        results = {
            'descriptive_stats': self._calculate_descriptive_stats(df),
            'correlation_analysis': self._perform_correlation_analysis(df),
            'clustering_results': self._perform_clustering(df),
            'anomaly_detection': self._detect_anomalies(df),
            'category_comparison': self._compare_categories(df),
            'protection_strength_analysis': self._analyze_protection_strength(df),
            'trend_analysis': self._analyze_trends(df)
        }

        return results

    def _prepare_dataframe(self, metrics_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert metrics results to pandas DataFrame"""
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
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like register_usage_pattern)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            row[f"{metric}_{sub_key}"] = sub_value

            data.append(row)

        return pd.DataFrame(data)

    def _calculate_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate descriptive statistics for all metrics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats_dict = {}

        for col in numeric_cols:
            if col == 'bytecode_size':  # Skip non-metric columns
                continue

            stats_dict[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else np.inf
            }

        return stats_dict

    def _perform_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between metrics"""
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in ['bytecode_size']]

        if len(numeric_cols) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()

        # Find strong correlations
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': numeric_cols[i],
                        'metric2': numeric_cols[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })

        # Principal correlations with key metrics
        key_metrics = ['code_expansion_rate', 'control_flow_complexity', 'obfuscation_strength']
        principal_correlations = {}

        for key_metric in key_metrics:
            if key_metric in corr_matrix.columns:
                correlations = corr_matrix[key_metric].sort_values(ascending=False)
                principal_correlations[key_metric] = {
                    'top_positive': correlations[1:4].to_dict(),  # Skip self-correlation
                    'top_negative': correlations[-3:].to_dict()
                }

        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'principal_correlations': principal_correlations
        }

    def _perform_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis to identify protection patterns"""
        # Select features for clustering
        feature_cols = ['code_expansion_rate', 'instruction_diversity',
                        'control_flow_complexity', 'obfuscation_strength',
                        'anti_debug_features', 'jump_density']

        available_features = [col for col in feature_cols if col in df.columns]

        if len(available_features) < 2:
            return {}

        # Prepare data
        X = df[available_features].fillna(0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        n_clusters = min(5, len(df) // 10)  # Adaptive number of clusters
        if n_clusters < 2:
            return {}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        cluster_profiles = []
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_data = df[cluster_mask]

            profile = {
                'cluster_id': i,
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(df) * 100),
                'dominant_category': cluster_data['category'].mode()[0] if not cluster_data.empty else 'unknown',
                'avg_metrics': {}
            }

            # Calculate average metrics for cluster
            for col in available_features:
                profile['avg_metrics'][col] = float(cluster_data[col].mean())

            cluster_profiles.append(profile)

        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        return {
            'n_clusters': n_clusters,
            'cluster_profiles': cluster_profiles,
            'feature_importance': dict(zip(available_features, pca.components_[0])),
            'explained_variance': float(pca.explained_variance_ratio_.sum()),
            'pca_data': {
                'x': X_pca[:, 0].tolist(),
                'y': X_pca[:, 1].tolist(),
                'clusters': clusters.tolist()
            }
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous transformations"""
        anomalies = {
            'outliers': {},
            'failed_protections': [],
            'extreme_cases': []
        }

        # Statistical outlier detection for each metric
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in ['bytecode_size']]

        for col in numeric_cols:
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if not outliers.empty:
                anomalies['outliers'][col] = {
                    'count': len(outliers),
                    'functions': outliers['function'].tolist()[:10],  # Top 10
                    'values': outliers[col].tolist()[:10]
                }

        # Detect failed protections (very low metrics)
        protection_metrics = ['code_expansion_rate', 'obfuscation_strength', 'control_flow_complexity']
        available_metrics = [m for m in protection_metrics if m in df.columns]

        if available_metrics:
            # Functions with all protection metrics in bottom 10%
            failed_mask = True
            for metric in available_metrics:
                failed_mask &= df[metric] < df[metric].quantile(0.1)

            failed_protections = df[failed_mask]
            if not failed_protections.empty:
                anomalies['failed_protections'] = [
                    {
                        'function': row['function'],
                        'category': row['category'],
                        'metrics': {m: row[m] for m in available_metrics}
                    }
                    for _, row in failed_protections.head(20).iterrows()
                ]

        # Detect extreme cases (very high metrics)
        if 'code_expansion_rate' in df.columns:
            extreme_expansion = df[df['code_expansion_rate'] > df['code_expansion_rate'].quantile(0.99)]
            if not extreme_expansion.empty:
                anomalies['extreme_cases'] = [
                    {
                        'function': row['function'],
                        'expansion_rate': row['code_expansion_rate'],
                        'bytecode_size': row['bytecode_size']
                    }
                    for _, row in extreme_expansion.head(10).iterrows()
                ]

        return anomalies

    def _compare_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical comparison between function categories"""
        categories = df['category'].unique()

        if len(categories) < 2:
            return {}

        comparisons = {}

        # Key metrics to compare
        metrics_to_compare = ['code_expansion_rate', 'control_flow_complexity',
                              'obfuscation_strength', 'instruction_diversity']

        for metric in metrics_to_compare:
            if metric not in df.columns:
                continue

            # Perform ANOVA
            category_groups = [df[df['category'] == cat][metric].dropna()
                               for cat in categories]

            # Filter out empty groups
            category_groups = [g for g in category_groups if len(g) > 0]

            if len(category_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*category_groups)

                comparisons[metric] = {
                    'anova_f_statistic': float(f_stat),
                    'anova_p_value': float(p_value),
                    'significant': p_value < self.significance_level,
                    'category_means': {}
                }

                # Calculate means per category
                for cat in categories:
                    cat_data = df[df['category'] == cat][metric]
                    if not cat_data.empty:
                        comparisons[metric]['category_means'][cat] = {
                            'mean': float(cat_data.mean()),
                            'std': float(cat_data.std()),
                            'count': int(len(cat_data))
                        }

                # Post-hoc pairwise comparisons if significant
                if p_value < self.significance_level and len(categories) > 2:
                    pairwise = []
                    for i, cat1 in enumerate(categories):
                        for cat2 in categories[i + 1:]:
                            data1 = df[df['category'] == cat1][metric].dropna()
                            data2 = df[df['category'] == cat2][metric].dropna()

                            if len(data1) > 0 and len(data2) > 0:
                                t_stat, p_val = stats.ttest_ind(data1, data2)
                                pairwise.append({
                                    'category1': cat1,
                                    'category2': cat2,
                                    't_statistic': float(t_stat),
                                    'p_value': float(p_val),
                                    'significant': p_val < self.significance_level
                                })

                    comparisons[metric]['pairwise_comparisons'] = pairwise

        return comparisons

    def _analyze_protection_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall protection strength patterns"""
        # Create composite protection score
        protection_metrics = ['code_expansion_rate', 'instruction_diversity',
                              'control_flow_complexity', 'obfuscation_strength']

        available_metrics = [m for m in protection_metrics if m in df.columns]

        if not available_metrics:
            return {}

        # Normalize and combine metrics
        df_normalized = df.copy()
        for metric in available_metrics:
            # Min-max normalization
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df_normalized[metric + '_norm'] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_normalized[metric + '_norm'] = 0

        # Calculate composite score
        norm_cols = [m + '_norm' for m in available_metrics]
        df_normalized['protection_score'] = df_normalized[norm_cols].mean(axis=1)

        # Analyze protection levels
        protection_levels = {
            'low': df_normalized[df_normalized['protection_score'] < 0.33],
            'medium': df_normalized[(df_normalized['protection_score'] >= 0.33) &
                                    (df_normalized['protection_score'] < 0.67)],
            'high': df_normalized[df_normalized['protection_score'] >= 0.67]
        }

        analysis = {
            'protection_distribution': {
                level: {
                    'count': len(data),
                    'percentage': len(data) / len(df) * 100,
                    'avg_bytecode_size': data['bytecode_size'].mean() if not data.empty else 0,
                    'dominant_categories': data['category'].value_counts().head(3).to_dict() if not data.empty else {}
                }
                for level, data in protection_levels.items()
            },
            'protection_score_stats': {
                'mean': float(df_normalized['protection_score'].mean()),
                'std': float(df_normalized['protection_score'].std()),
                'median': float(df_normalized['protection_score'].median()),
                'distribution': np.histogram(df_normalized['protection_score'], bins=10)[0].tolist()
            }
        }

        return analysis

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in VMP transformations"""
        trends = {}

        # Size vs complexity relationship
        if 'bytecode_size' in df.columns and 'control_flow_complexity' in df.columns:
            correlation = df['bytecode_size'].corr(df['control_flow_complexity'])

            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            X = df['bytecode_size'].values.reshape(-1, 1)
            y = df['control_flow_complexity'].values

            mask = ~(np.isnan(X.flatten()) | np.isnan(y))
            if mask.sum() > 10:
                model = LinearRegression()
                model.fit(X[mask], y[mask])

                trends['size_complexity_relationship'] = {
                    'correlation': float(correlation),
                    'slope': float(model.coef_[0]),
                    'intercept': float(model.intercept_),
                    'r_squared': float(model.score(X[mask], y[mask]))
                }

        # Category-specific trends
        category_trends = {}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]

            if len(cat_data) > 5:
                category_trends[category] = {
                    'sample_count': len(cat_data),
                    'avg_code_expansion': float(cat_data['code_expansion_rate'].mean())
                    if 'code_expansion_rate' in cat_data.columns else None,
                    'protection_variance': float(cat_data['obfuscation_strength'].var())
                    if 'obfuscation_strength' in cat_data.columns else None
                }

        trends['category_trends'] = category_trends

        return trends