class HeatmapGenerator:
    """Generates heatmap visualizations"""

    def __init__(self, config: dict[str, any]):
        self.config = config
        self.output_dir = Path('outputs/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_protection_heatmap(self, metrics_results: list[dict[str, any]]) -> None:
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

    def _prepare_dataframe(self, metrics_results: list[dict[str, any]]) -> pd.DataFrame:
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