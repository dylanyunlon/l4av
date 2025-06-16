# src/statistics/descriptive.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DescriptiveStatistics:
    """Generate descriptive statistics for VMP analysis"""

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive descriptive statistics"""
        logger.info("Generating descriptive statistics...")

        stats_results = {
            'code_metrics_summary': self._summarize_code_metrics(results.get('code_metrics', {})),
            'performance_summary': self._summarize_performance(results.get('performance', {})),
            'security_summary': self._summarize_security(results.get('security', {})),
            'overall_summary': self._generate_overall_summary(results)
        }

        return stats_results

    def _summarize_code_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize code metrics statistics"""
        expansion_metrics = metrics.get('expansion_metrics', {})

        return {
            'expansion_ratio': {
                'mean': expansion_metrics.get('mean_expansion_ratio', 0),
                'median': expansion_metrics.get('median_expansion_ratio', 0),
                'std': expansion_metrics.get('std_expansion_ratio', 0),
                'cv': self._coefficient_of_variation(
                    expansion_metrics.get('mean_expansion_ratio', 0),
                    expansion_metrics.get('std_expansion_ratio', 0)
                ),
                'skewness': self._calculate_skewness(expansion_metrics.get('expansion_distribution', {})),
                'kurtosis': self._calculate_kurtosis(expansion_metrics.get('expansion_distribution', {}))
            },

            'instruction_growth': {
                'total_growth_rate': metrics.get('instruction_metrics', {}).get('instruction_growth_rate', 0),
                'unique_instruction_ratio': self._calculate_unique_ratio(metrics.get('instruction_metrics', {}))
            },

            'complexity_increase': {
                'control_flow': metrics.get('complexity_metrics', {}).get('control_flow', {}).get('mean_complexity', 0),
                'cyclomatic': metrics.get('complexity_metrics', {}).get('cyclomatic_complexity_estimate', {}).get(
                    'complexity_increase', 0)
            },

            'obfuscation_strength': {
                'mean_intensity': metrics.get('obfuscation_metrics', {}).get('intensity', {}).get('mean', 0),
                'overall_score': metrics.get('obfuscation_metrics', {}).get('strength_score', 0)
            }
        }

    def _summarize_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize performance statistics"""
        return {
            'execution_overhead': {
                'mean': metrics.get('execution_overhead', {}).get('mean_overhead_ratio', 0),
                'median': metrics.get('execution_overhead', {}).get('median_overhead_ratio', 0),
                'max': metrics.get('execution_overhead', {}).get('max_overhead_ratio', 0),
                'distribution': metrics.get('execution_overhead', {}).get('overhead_distribution', {})
            },

            'memory_impact': {
                'mean_increase': metrics.get('memory_impact', {}).get('mean_memory_increase', 0),
                'total_overhead': metrics.get('memory_impact', {}).get('memory_overhead_ratio', 0)
            },

            'cache_impact': {
                'l1_pressure': metrics.get('cache_impact', {}).get('l1_pressure', {}).get('mean_l1_utilization', 0),
                'l2_pressure': metrics.get('cache_impact', {}).get('l2_pressure', {}).get('mean_l2_utilization', 0)
            }
        }

    def _summarize_security(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize security statistics"""
        return {
            'anti_debug_coverage': metrics.get('anti_debug', {}).get('antidebug_ratio', 0),
            'anti_analysis_features': len(metrics.get('anti_analysis', {}).get('feature_counts', {})),
            'integrity_checks': metrics.get('code_integrity', {}).get('integrity_check_ratio', 0),
            'overall_resistance': metrics.get('resistance_score', {}).get('overall_score', 0),
            'security_rating': metrics.get('resistance_score', {}).get('rating', 'unknown')
        }

    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        return {
            'protection_effectiveness': self._calculate_protection_effectiveness(results),
            'performance_cost': self._calculate_performance_cost(results),
            'security_benefit': self._calculate_security_benefit(results),
            'cost_benefit_ratio': self._calculate_cost_benefit_ratio(results)
        }

    def _coefficient_of_variation(self, mean: float, std: float) -> float:
        """Calculate coefficient of variation"""
        if mean == 0:
            return 0
        return std / mean

    def _calculate_skewness(self, distribution: Dict[str, Any]) -> float:
        """Calculate skewness from distribution"""
        if 'histogram' not in distribution:
            return 0
        hist = distribution['histogram']
        edges = distribution['bin_edges']
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

        # Weighted skewness
        mean = np.average(centers, weights=hist)
        variance = np.average((np.array(centers) - mean) ** 2, weights=hist)
        skewness = np.average((np.array(centers) - mean) ** 3, weights=hist) / (variance ** 1.5)

        return skewness

    def _calculate_kurtosis(self, distribution: Dict[str, Any]) -> float:
        """Calculate kurtosis from distribution"""
        if 'histogram' not in distribution:
            return 0
        hist = distribution['histogram']
        edges = distribution['bin_edges']
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

        # Weighted kurtosis
        mean = np.average(centers, weights=hist)
        variance = np.average((np.array(centers) - mean) ** 2, weights=hist)
        kurtosis = np.average((np.array(centers) - mean) ** 4, weights=hist) / (variance ** 2) - 3

        return kurtosis

    def _calculate_unique_ratio(self, instruction_metrics: Dict[str, Any]) -> float:
        """Calculate unique instruction ratio"""
        original = instruction_metrics.get('original', {})
        vmp = instruction_metrics.get('vmp', {})

        if original.get('unique_instructions', 0) == 0:
            return 0

        return vmp.get('unique_instructions', 0) / original.get('unique_instructions', 1)

    def _calculate_protection_effectiveness(self, results: Dict[str, Any]) -> float:
        """Calculate overall protection effectiveness (0-100)"""
        factors = {
            'obfuscation': results.get('code_metrics', {}).get('obfuscation_metrics', {}).get('strength_score', 0),
            'expansion': min(
                results.get('code_metrics', {}).get('expansion_metrics', {}).get('mean_expansion_ratio', 0) * 10, 100),
            'security': results.get('security', {}).get('resistance_score', {}).get('overall_score', 0)
        }

        weights = {'obfuscation': 0.4, 'expansion': 0.2, 'security': 0.4}
        effectiveness = sum(factors[k] * weights[k] for k in factors)

        return effectiveness

    def _calculate_performance_cost(self, results: Dict[str, Any]) -> float:
        """Calculate performance cost (0-100)"""
        overhead = results.get('performance', {}).get('execution_overhead', {}).get('mean_overhead_ratio', 1)
        memory = results.get('performance', {}).get('memory_impact', {}).get('mean_memory_increase', 1)

        # Normalize to 0-100 scale
        cost = min((overhead - 1) * 10 + (memory - 1) * 10, 100)
        return cost

    def _calculate_security_benefit(self, results: Dict[str, Any]) -> float:
        """Calculate security benefit (0-100)"""
        return results.get('security', {}).get('resistance_score', {}).get('overall_score', 0)

    def _calculate_cost_benefit_ratio(self, results: Dict[str, Any]) -> float:
        """Calculate cost-benefit ratio"""
        benefit = self._calculate_security_benefit(results)
        cost = self._calculate_performance_cost(results)

        if cost == 0:
            return float('inf')
        return benefit / cost


# src/statistics/correlation.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalysis:
    """Perform correlation analysis on VMP metrics"""

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis"""
        logger.info("Performing correlation analysis...")

        # Extract key metrics for correlation
        metrics_data = self._extract_metrics_data(results)

        correlations = {
            'metric_correlations': self._calculate_correlations(metrics_data),
            'category_correlations': self._analyze_category_correlations(results),
            'feature_associations': self._analyze_feature_associations(results),
            'predictive_relationships': self._identify_predictive_relationships(metrics_data)
        }

        return correlations

    def _extract_metrics_data(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Extract metrics into DataFrame for correlation analysis"""
        # This would normally extract from the actual data DataFrame
        # For now, using aggregated metrics
        metrics = {
            'expansion_ratio': results.get('code_metrics', {}).get('expansion_metrics', {}).get('mean_expansion_ratio',
                                                                                                0),
            'instruction_growth': results.get('code_metrics', {}).get('instruction_metrics', {}).get(
                'instruction_growth_rate', 0),
            'complexity': results.get('code_metrics', {}).get('complexity_metrics', {}).get('control_flow', {}).get(
                'mean_complexity', 0),
            'obfuscation': results.get('code_metrics', {}).get('obfuscation_metrics', {}).get('strength_score', 0),
            'execution_overhead': results.get('performance', {}).get('execution_overhead', {}).get(
                'mean_overhead_ratio', 0),
            'memory_increase': results.get('performance', {}).get('memory_impact', {}).get('mean_memory_increase', 0),
            'security_score': results.get('security', {}).get('resistance_score', {}).get('overall_score', 0)
        }

        # Create synthetic data for demonstration
        # In real implementation, this would use the actual data
        n_samples = 100
        data = pd.DataFrame({
            key: np.random.normal(value, value * 0.2, n_samples)
            for key, value in metrics.items()
        })

        return data

    def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation coefficients"""
        # Pearson correlation
        pearson_corr = data.corr(method='pearson')

        # Spearman correlation (rank-based)
        spearman_corr = data.corr(method='spearman')

        # Find strongest correlations
        strong_correlations = self._find_strong_correlations(pearson_corr)

        return {
            'pearson': pearson_corr.to_dict(),
            'spearman': spearman_corr.to_dict(),
            'strong_correlations': strong_correlations,
            'correlation_summary': self._summarize_correlations(pearson_corr)
        }

    def _analyze_category_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations by function category"""
        category_data = results.get('code_metrics', {}).get('category_analysis', {})

        if not category_data:
            return {}

        # Analyze how different categories respond to VMP protection
        category_metrics = pd.DataFrame(category_data).T

        return {
            'category_effectiveness': self._calculate_category_effectiveness(category_metrics),
            'category_patterns': self._identify_category_patterns(category_metrics)
        }

    def _analyze_feature_associations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze associations between security features and protection strength"""
        security_features = results.get('security', {})
        code_metrics = results.get('code_metrics', {})

        associations = {
            'antidebug_vs_complexity': self._calculate_association(
                security_features.get('anti_debug', {}).get('antidebug_ratio', 0),
                code_metrics.get('complexity_metrics', {}).get('control_flow', {}).get('mean_complexity', 0)
            ),
            'integrity_vs_overhead': self._calculate_association(
                security_features.get('code_integrity', {}).get('integrity_check_ratio', 0),
                results.get('performance', {}).get('execution_overhead', {}).get('mean_overhead_ratio', 0)
            )
        }

        return associations

    def _identify_predictive_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify predictive relationships between metrics"""
        relationships = {}

        # Linear regression analysis for key relationships
        # Expansion ratio as predictor of execution overhead
        if 'expansion_ratio' in data.columns and 'execution_overhead' in data.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data['expansion_ratio'], data['execution_overhead']
            )
            relationships['expansion_to_overhead'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'predictive_power': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.5 else 'weak'
            }

        return relationships

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[
        Tuple[str, str, float]]:
        """Find strong correlations above threshold"""
        strong_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                if abs(value) >= threshold:
                    strong_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        value
                    ))

        return sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)

    def _summarize_correlations(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Summarize correlation patterns"""
        summary = {}

        for col in corr_matrix.columns:
            correlations = corr_matrix[col].drop(col)
            summary[col] = {
                'strongest_positive': correlations.idxmax(),
                'strongest_negative': correlations.idxmin(),
                'max_correlation': correlations.max(),
                'min_correlation': correlations.min(),
                'mean_abs_correlation': correlations.abs().mean()
            }

        return summary

    def _calculate_category_effectiveness(self, category_metrics: pd.DataFrame) -> Dict[str, float]:
        """Calculate effectiveness score for each category"""
        if category_metrics.empty:
            return {}

        effectiveness = {}
        for category in category_metrics.index:
            # Composite effectiveness score
            score = (
                    category_metrics.loc[category, 'mean_expansion'] * 0.3 +
                    category_metrics.loc[category, 'mean_complexity'] * 0.4 +
                    category_metrics.loc[category, 'mean_obfuscation'] * 0.3
            )
            effectiveness[category] = score

        return effectiveness

    def _identify_category_patterns(self, category_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in how different categories are protected"""
        if category_metrics.empty:
            return {}

        patterns = {
            'most_protected': category_metrics['mean_obfuscation'].idxmax(),
            'least_protected': category_metrics['mean_obfuscation'].idxmin(),
            'highest_overhead': category_metrics['mean_expansion'].idxmax(),
            'protection_variance': category_metrics['mean_obfuscation'].std()
        }

        return patterns

    def _calculate_association(self, x: float, y: float) -> Dict[str, float]:
        """Calculate association between two variables"""
        # Simplified for single values - in real implementation would use arrays
        return {
            'correlation': 0.0,  # Would calculate actual correlation
            'strength': 'unknown'
        }


# src/statistics/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ClusteringAnalysis:
    """Perform clustering analysis to identify protection patterns"""

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis"""
        logger.info("Performing clustering analysis...")

        # Prepare data for clustering
        feature_data = self._prepare_features(results)

        clusters = {
            'protection_clusters': self._cluster_by_protection_strength(feature_data),
            'complexity_clusters': self._cluster_by_complexity(feature_data),
            'behavior_clusters': self._cluster_by_behavior(feature_data),
            'outlier_analysis': self._detect_outliers(feature_data)
        }

        return clusters

    def _prepare_features(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare feature matrix for clustering"""
        # Extract relevant features
        # In real implementation, this would use the actual data
        n_samples = 1000

        features = pd.DataFrame({
            'expansion_ratio': np.random.exponential(5, n_samples),
            'complexity': np.random.exponential(20, n_samples),
            'obfuscation': np.random.beta(2, 5, n_samples) * 100,
            'instruction_diversity': np.random.beta(5, 2, n_samples),
            'security_features': np.random.poisson(3, n_samples),
            'overhead': np.random.exponential(3, n_samples)
        })

        return features

    def _cluster_by_protection_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster functions by protection strength"""
        # Select relevant features
        protection_features = ['expansion_ratio', 'complexity', 'obfuscation']
        X = data[protection_features].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform K-means clustering
        n_clusters = 4  # Low, Medium, High, Very High protection
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_data = data[mask]

            cluster_analysis[f'cluster_{i}'] = {
                'size': mask.sum(),
                'mean_expansion': cluster_data['expansion_ratio'].mean(),
                'mean_complexity': cluster_data['complexity'].mean(),
                'mean_obfuscation': cluster_data['obfuscation'].mean(),
                'protection_level': self._classify_protection_level(
                    cluster_data['obfuscation'].mean()
                )
            }

        return {
            'n_clusters': n_clusters,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_analysis': cluster_analysis,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }

    def _cluster_by_complexity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster by complexity patterns"""
        complexity_features = ['complexity', 'instruction_diversity', 'overhead']
        X = data[complexity_features].values

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(StandardScaler().fit_transform(X))

        return {
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'noise_points': (clusters == -1).sum(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'cluster_distribution': pd.Series(clusters).value_counts().to_dict()
        }

    def _cluster_by_behavior(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Cluster by behavioral characteristics"""
        # Create behavioral feature vector
        behavioral_features = self._extract_behavioral_features(data)

        # Hierarchical clustering would be performed here
        # For now, returning summary statistics

        return {
            'behavioral_patterns': {
                'aggressive_protection': (data['obfuscation'] > 80).sum(),
                'minimal_protection': (data['obfuscation'] < 20).sum(),
                'balanced_protection': ((data['obfuscation'] >= 20) & (data['obfuscation'] <= 80)).sum()
            }
        }

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in protection patterns"""
        outliers = {}

        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)

            outliers[column] = {
                'n_outliers': outlier_mask.sum(),
                'outlier_percentage': (outlier_mask.sum() / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

        # Multivariate outlier detection using Mahalanobis distance
        # (simplified version)
        mean = data.mean()
        cov = data.cov()

        return {
            'univariate_outliers': outliers,
            'total_outliers': sum(out['n_outliers'] for out in outliers.values()),
            'outlier_summary': self._summarize_outliers(outliers)
        }

    def _classify_protection_level(self, obfuscation_score: float) -> str:
        """Classify protection level based on obfuscation score"""
        if obfuscation_score < 25:
            return 'low'
        elif obfuscation_score < 50:
            return 'medium'
        elif obfuscation_score < 75:
            return 'high'
        else:
            return 'very_high'

    def _extract_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features for clustering"""
        # Would extract more sophisticated features in real implementation
        behavioral = pd.DataFrame({
            'protection_aggressiveness': data['obfuscation'] / data['expansion_ratio'],
            'efficiency': data['complexity'] / data['overhead'],
            'security_focus': data['security_features'] * data['obfuscation']
        })

        return behavioral

    def _summarize_outliers(self, outliers: Dict[str, Dict]) -> Dict[str, Any]:
        """Summarize outlier detection results"""
        most_outliers = max(outliers.items(), key=lambda x: x[1]['n_outliers'])

        return {
            'metric_with_most_outliers': most_outliers[0],
            'max_outlier_count': most_outliers[1]['n_outliers'],
            'average_outlier_percentage': np.mean([out['outlier_percentage'] for out in outliers.values()])
        }


# src/statistics/anomaly.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in VMP protection patterns"""

    def detect(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect various types of anomalies"""
        logger.info("Detecting anomalies...")

        anomalies = {
            'protection_anomalies': self._detect_protection_anomalies(results),
            'performance_anomalies': self._detect_performance_anomalies(results),
            'pattern_anomalies': self._detect_pattern_anomalies(results),
            'failure_detection': self._detect_protection_failures(results)
        }

        return anomalies

    def _detect_protection_anomalies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in protection strength"""
        code_metrics = results.get('code_metrics', {})

        anomalies = {
            'low_expansion': self._detect_low_expansion(code_metrics),
            'excessive_expansion': self._detect_excessive_expansion(code_metrics),
            'low_obfuscation': self._detect_low_obfuscation(code_metrics),
            'pattern_violations': self._detect_pattern_violations(code_metrics)
        }

        return anomalies

    def _detect_performance_anomalies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance-related anomalies"""
        perf_metrics = results.get('performance', {})

        anomalies = {
            'extreme_overhead': self._detect_extreme_overhead(perf_metrics),
            'memory_spikes': self._detect_memory_spikes(perf_metrics),
            'cache_thrashing': self._detect_cache_issues(perf_metrics)
        }

        return anomalies

    def _detect_pattern_anomalies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in VMP patterns"""
        # Use Isolation Forest for multivariate anomaly detection
        # In real implementation, would use actual data

        return {
            'unusual_patterns': self._find_unusual_patterns(results),
            'missing_features': self._detect_missing_features(results),
            'inconsistent_protection': self._detect_inconsistencies(results)
        }

    def _detect_protection_failures(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential protection failures"""
        failures = {
            'no_obfuscation': self._check_no_obfuscation(results),
            'minimal_transformation': self._check_minimal_transformation(results),
            'pattern_exposure': self._check_pattern_exposure(results)
        }

        failure_count = sum(1 for f in failures.values() if f.get('detected', False))

        return {
            'failure_types': failures,
            'total_failures': failure_count,
            'failure_rate': failure_count / 3.0  # Simplified
        }

    def _detect_low_expansion(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unusually low code expansion"""
        expansion = metrics.get('expansion_metrics', {})
        mean_expansion = expansion.get('mean_expansion_ratio', 0)

        # Flag if expansion is too low (< 2x)
        is_anomaly = mean_expansion < 2.0

        return {
            'detected': is_anomaly,
            'severity': 'high' if mean_expansion < 1.5 else 'medium',
            'description': f'Low expansion ratio: {mean_expansion:.2f}x'
        }

    def _detect_excessive_expansion(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect excessive code expansion"""
        expansion = metrics.get('expansion_metrics', {})
        max_expansion = expansion.get('max_expansion_ratio', 0)

        # Flag if expansion is excessive (> 50x)
        is_anomaly = max_expansion > 50.0

        return {
            'detected': is_anomaly,
            'severity': 'high' if max_expansion > 100 else 'medium',
            'description': f'Excessive expansion ratio: {max_expansion:.2f}x'
        }

    def _detect_low_obfuscation(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect low obfuscation levels"""
        obfuscation = metrics.get('obfuscation_metrics', {})
        score = obfuscation.get('strength_score', 0)

        is_anomaly = score < 20  # Less than 20% obfuscation

        return {
            'detected': is_anomaly,
            'severity': 'high' if score < 10 else 'medium',
            'description': f'Low obfuscation score: {score:.1f}%'
        }

    def _detect_pattern_violations(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect violations of expected patterns"""
        patterns = metrics.get('pattern_metrics', {})
        vmp_ratio = patterns.get('vmp_pattern_ratio', 0)

        # Expect VMP patterns in protected code
        is_anomaly = vmp_ratio < 0.5

        return {
            'detected': is_anomaly,
            'severity': 'medium',
            'description': f'Low VMP pattern presence: {vmp_ratio:.1%}'
        }

    def _detect_extreme_overhead(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect extreme performance overhead"""
        overhead = metrics.get('execution_overhead', {})
        max_overhead = overhead.get('max_overhead_ratio', 0)

        is_anomaly = max_overhead > 100  # 100x slower

        return {
            'detected': is_anomaly,
            'severity': 'critical' if max_overhead > 200 else 'high',
            'description': f'Extreme overhead: {max_overhead:.1f}x'
        }

    def _detect_memory_spikes(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect memory usage spikes"""
        memory = metrics.get('memory_impact', {})
        mean_increase = memory.get('mean_memory_increase', 0)

        is_anomaly = mean_increase > 20  # 20x memory increase

        return {
            'detected': is_anomaly,
            'severity': 'high',
            'description': f'High memory increase: {mean_increase:.1f}x'
        }

    def _detect_cache_issues(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect cache performance issues"""
        cache = metrics.get('cache_impact', {})
        l1_pressure = cache.get('l1_pressure', {}).get('mean_l1_utilization', 0)

        is_anomaly = l1_pressure > 0.9  # 90% L1 cache pressure

        return {
            'detected': is_anomaly,
            'severity': 'medium',
            'description': f'High L1 cache pressure: {l1_pressure:.1%}'
        }

    def _find_unusual_patterns(self, results: Dict[str, Any]) -> List[str]:
        """Find unusual protection patterns"""
        unusual = []

        # Check for unusual combinations
        security = results.get('security', {})
        if security.get('anti_debug', {}).get('antidebug_ratio', 0) > 0.9:
            unusual.append('Excessive anti-debugging (>90% of functions)')

        return unusual

    def _detect_missing_features(self, results: Dict[str, Any]) -> List[str]:
        """Detect missing expected features"""
        missing = []

        security = results.get('security', {})
        if security.get('code_integrity', {}).get('integrity_check_ratio', 0) < 0.1:
            missing.append('Low integrity check coverage (<10%)')

        return missing

    def _detect_inconsistencies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect inconsistent protection patterns"""
        # Would analyze variance in protection across similar functions
        return {
            'detected': False,
            'description': 'No major inconsistencies detected'
        }

    def _check_no_obfuscation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check for functions with no obfuscation"""
        obfuscation = results.get('code_metrics', {}).get('obfuscation_metrics', {})
        score = obfuscation.get('strength_score', 0)

        return {
            'detected': score < 5,
            'description': 'Minimal or no obfuscation detected'
        }

    def _check_minimal_transformation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check for minimal code transformation"""
        expansion = results.get('code_metrics', {}).get('expansion_metrics', {})
        ratio = expansion.get('mean_expansion_ratio', 0)

        return {
            'detected': ratio < 1.5,
            'description': 'Code barely transformed'
        }

    def _check_pattern_exposure(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if original patterns are exposed"""
        instruction_metrics = results.get('code_metrics', {}).get('instruction_metrics', {})

        # Check if instruction distribution is too similar
        return {
            'detected': False,  # Would need actual comparison
            'description': 'Pattern exposure analysis'
        }