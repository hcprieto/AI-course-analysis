"""
Fixed Enhanced Health and Well-being Analysis with Working Visualizations

This version fixes the HTML generation issues and provides comprehensive analysis
with properly embedded charts and detailed statistical reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class FixedHealthAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.output_dir = Path("../analysis-output-files")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_data(self):
        """Prepare and clean health-related data"""
        print("Preparing health data...")

        # Create a working copy
        self.health_df = self.df.copy()

        # Convert numeric columns
        numeric_cols = {
            'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction',
            'Life expectancy at birth (years)(2018)': 'life_expectancy',
            'Deaths Due to Self-Harm (2019)': 'self_harm_deaths',
            'Rate of Deaths Due to Self-Harm, per 100,000 people (2019)': 'self_harm_rate'
        }

        for orig_col, new_col in numeric_cols.items():
            if orig_col in self.health_df.columns:
                self.health_df[new_col] = pd.to_numeric(self.health_df[orig_col], errors='coerce')

        # Clean continent names
        self.health_df['continent'] = self.health_df['Continent']
        self.health_df['country'] = self.health_df['Country']

        print(f"Data prepared: {len(self.health_df)} countries")
        return self.health_df

    def create_distribution_plots(self):
        """Create distribution visualizations"""
        print("Creating distribution plots...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']

        # Create figure for distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Health and Well-being Indicators', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns:
                clean_data = self.health_df[indicator].dropna()

                if len(clean_data) > 0 and i < 3:
                    row, col = divmod(i, 2)
                    ax = axes[row, col]

                    # Histogram with better styling
                    ax.hist(clean_data, bins=20, alpha=0.7, density=True,
                           color=sns.color_palette()[i], edgecolor='black', linewidth=0.5)

                    # Add mean line
                    mean_val = clean_data.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_val:.2f}')

                    # Add median line
                    median_val = clean_data.median()
                    ax.axvline(median_val, color='orange', linestyle=':', linewidth=2,
                              label=f'Median: {median_val:.2f}')

                    # Formatting
                    title_map = {
                        'life_satisfaction': 'Life Satisfaction (Cantril Ladder)',
                        'life_expectancy': 'Life Expectancy (Years)',
                        'self_harm_rate': 'Self-harm Rate (per 100k)'
                    }
                    ax.set_title(title_map.get(indicator, indicator), fontweight='bold', fontsize=12)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

        # Remove empty subplot
        axes[1, 1].remove()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'health_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_regional_plots(self):
        """Create regional comparison visualizations"""
        print("Creating regional analysis plots...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']

        # Create regional comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Health Indicators by Continent', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns:
                ax = axes[i]
                plot_data = self.health_df[['continent', indicator]].dropna()

                if len(plot_data) > 0:
                    # Create boxplot
                    sns.boxplot(data=plot_data, x='continent', y=indicator, ax=ax,
                               palette='Set2', showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red"})

                    # Formatting
                    title_map = {
                        'life_satisfaction': 'Life Satisfaction',
                        'life_expectancy': 'Life Expectancy (Years)',
                        'self_harm_rate': 'Self-harm Rate (per 100k)'
                    }
                    ax.set_title(title_map.get(indicator, indicator), fontweight='bold')
                    ax.set_xlabel('Continent')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'regional_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_correlation_plots(self):
        """Create correlation visualizations"""
        print("Creating correlation plots...")

        # Select numeric indicators
        numeric_cols = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        available_cols = [col for col in numeric_cols if col in self.health_df.columns]

        if len(available_cols) >= 2:
            # Create correlation data
            corr_data = self.health_df[available_cols + ['continent', 'country']].dropna()

            # Calculate correlation matrix
            corr_matrix = corr_data[available_cols].corr()

            # Create correlation visualizations
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, ax=axes[0], fmt='.3f',
                       cbar_kws={'label': 'Correlation Coefficient'})
            axes[0].set_title('Correlation Matrix: Health Indicators', fontweight='bold')

            # Scatter plot for life satisfaction vs life expectancy
            if 'life_satisfaction' in available_cols and 'life_expectancy' in available_cols:
                # Create scatter plot by continent
                for continent in corr_data['continent'].unique():
                    if pd.notna(continent):
                        cont_data = corr_data[corr_data['continent'] == continent]
                        axes[1].scatter(cont_data['life_satisfaction'], cont_data['life_expectancy'],
                                       label=continent, alpha=0.7, s=60)

                # Add regression line
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    corr_data['life_satisfaction'], corr_data['life_expectancy'])
                x_range = np.linspace(corr_data['life_satisfaction'].min(),
                                     corr_data['life_satisfaction'].max(), 100)
                y_pred = slope * x_range + intercept
                axes[1].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                            label=f'R² = {r_value**2:.3f}')

                axes[1].set_xlabel('Life Satisfaction (Cantril Ladder)')
                axes[1].set_ylabel('Life Expectancy (Years)')
                axes[1].set_title('Life Satisfaction vs Life Expectancy by Continent', fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_outlier_plots(self):
        """Create outlier analysis visualizations"""
        print("Creating outlier analysis plots...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']

        # Create outlier visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns and i < 3:
                clean_data = self.health_df[['country', 'continent', indicator]].dropna()

                if len(clean_data) > 10:
                    row, col = divmod(i, 2)
                    ax = axes[row, col]

                    values = clean_data[indicator]

                    # Calculate outliers using Z-score method
                    z_scores = np.abs(stats.zscore(values))
                    outlier_mask = z_scores > 2.5

                    # Create box plot
                    bp = ax.boxplot([values], patch_artist=True, labels=[indicator])
                    bp['boxes'][0].set_facecolor(sns.color_palette()[i])

                    # Highlight outliers
                    outliers = clean_data[outlier_mask]
                    if len(outliers) > 0:
                        ax.scatter([1] * len(outliers), outliers[indicator],
                                 color='red', s=100, alpha=0.7, zorder=10,
                                 label=f'Outliers ({len(outliers)})')

                        # Annotate top outliers
                        sorted_outliers = outliers.sort_values(indicator, ascending=False)
                        for j, (idx, row) in enumerate(sorted_outliers.head(3).iterrows()):
                            ax.annotate(row['country'],
                                       (1, row[indicator]),
                                       xytext=(1.1, row[indicator]),
                                       fontsize=8, alpha=0.8)

                    title_map = {
                        'life_satisfaction': 'Life Satisfaction Outliers',
                        'life_expectancy': 'Life Expectancy Outliers',
                        'self_harm_rate': 'Self-harm Rate Outliers'
                    }
                    ax.set_title(title_map.get(indicator, indicator), fontweight='bold')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3, axis='y')

                    if len(outliers) > 0:
                        ax.legend()

        # Remove empty subplot
        axes[1, 1].remove()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_mortality_plots(self):
        """Create mortality analysis visualizations"""
        print("Creating mortality analysis plots...")

        # Collect all death causes
        death_causes = []
        cause_columns = ['Most Common Cause of Death (2018)',
                        'Second Most Common Cause of Death (2018)',
                        'Third Most Common Cause of Death (2018)']

        for col in cause_columns:
            if col in self.health_df.columns:
                causes = self.health_df[col].dropna().tolist()
                death_causes.extend(causes)

        if death_causes:
            # Overall analysis
            cause_counts = pd.Series(death_causes).value_counts()

            # Create mortality visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Mortality Patterns Analysis', fontsize=16, fontweight='bold')

            # Global top causes
            top_causes = cause_counts.head(8)
            axes[0, 0].barh(range(len(top_causes)), top_causes.values,
                           color=sns.color_palette("viridis", len(top_causes)))
            axes[0, 0].set_yticks(range(len(top_causes)))
            axes[0, 0].set_yticklabels(top_causes.index, fontsize=10)
            axes[0, 0].set_xlabel('Frequency')
            axes[0, 0].set_title('Top 8 Global Causes of Death', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, axis='x')

            # Cardiovascular disease prevalence by continent
            mortality_by_continent = {}
            cardio_by_continent = {}

            for continent in self.health_df['continent'].unique():
                if pd.notna(continent):
                    continent_df = self.health_df[self.health_df['continent'] == continent]
                    continent_causes = []

                    for col in cause_columns:
                        if col in continent_df.columns:
                            causes = continent_df[col].dropna().tolist()
                            continent_causes.extend(causes)

                    if continent_causes:
                        mortality_by_continent[continent] = pd.Series(continent_causes).value_counts()
                        cardio_count = pd.Series(continent_causes).value_counts().get('Cardiovascular Disease', 0)
                        total_count = len(continent_causes)
                        cardio_by_continent[continent] = (cardio_count / total_count) * 100 if total_count > 0 else 0

            if cardio_by_continent:
                continents = list(cardio_by_continent.keys())
                percentages = list(cardio_by_continent.values())

                axes[0, 1].bar(continents, percentages, color=sns.color_palette("Set2", len(continents)))
                axes[0, 1].set_ylabel('Percentage (%)')
                axes[0, 1].set_title('Cardiovascular Disease Prevalence by Continent', fontweight='bold')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3, axis='y')

            # Pie chart of top 5 causes
            top_5_causes = cause_counts.head(5)
            axes[1, 0].pie(top_5_causes.values, labels=top_5_causes.index, autopct='%1.1f%%',
                          colors=sns.color_palette("Set3", len(top_5_causes)))
            axes[1, 0].set_title('Distribution of Top 5 Causes of Death', fontweight='bold')

            # Remove empty subplot
            axes[1, 1].remove()

            plt.tight_layout()
            plt.savefig(self.output_dir / 'mortality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

    def perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        print("Performing statistical analysis...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        results = {}

        # Univariate analysis
        univariate_results = {}
        for indicator in indicators:
            if indicator in self.health_df.columns:
                clean_data = self.health_df[indicator].dropna()
                if len(clean_data) > 0:
                    stats_dict = {
                        'count': len(clean_data),
                        'mean': float(clean_data.mean()),
                        'median': float(clean_data.median()),
                        'std': float(clean_data.std()),
                        'min': float(clean_data.min()),
                        'max': float(clean_data.max()),
                        'q25': float(clean_data.quantile(0.25)),
                        'q75': float(clean_data.quantile(0.75)),
                        'skewness': float(stats.skew(clean_data)),
                        'kurtosis': float(stats.kurtosis(clean_data))
                    }

                    # Normality test
                    if len(clean_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(clean_data[:5000])
                        stats_dict['shapiro_stat'] = float(shapiro_stat)
                        stats_dict['shapiro_p'] = float(shapiro_p)
                        stats_dict['is_normal'] = shapiro_p > 0.05

                    univariate_results[indicator] = stats_dict

        results['univariate'] = univariate_results

        # Regional analysis (ANOVA)
        regional_results = {}
        for indicator in indicators:
            if indicator in self.health_df.columns:
                continent_groups = []
                continent_names = []

                for continent in self.health_df['continent'].unique():
                    if pd.notna(continent):
                        group_data = self.health_df[
                            (self.health_df['continent'] == continent) &
                            (self.health_df[indicator].notna())
                        ][indicator].values

                        if len(group_data) > 0:
                            continent_groups.append(group_data)
                            continent_names.append(continent)

                if len(continent_groups) > 1:
                    try:
                        anova_stat, anova_p = f_oneway(*continent_groups)

                        # Regional descriptive statistics
                        regional_stats = self.health_df.groupby('continent')[indicator].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(3)

                        regional_results[indicator] = {
                            'anova_stat': float(anova_stat),
                            'anova_p': float(anova_p),
                            'significant_difference': anova_p < 0.05,
                            'regional_stats': regional_stats.to_dict('index')
                        }

                    except Exception as e:
                        print(f"ANOVA failed for {indicator}: {e}")

        results['regional'] = regional_results

        # Correlation analysis
        numeric_cols = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        available_cols = [col for col in numeric_cols if col in self.health_df.columns]

        correlation_results = {}
        for i, var1 in enumerate(available_cols):
            for j, var2 in enumerate(available_cols):
                if i < j:
                    subset = self.health_df[[var1, var2]].dropna()
                    if len(subset) > 3:
                        pearson_r, pearson_p = stats.pearsonr(subset[var1], subset[var2])
                        spearman_r, spearman_p = stats.spearmanr(subset[var1], subset[var2])

                        correlation_results[f"{var1}_vs_{var2}"] = {
                            'pearson_r': float(pearson_r),
                            'pearson_p': float(pearson_p),
                            'spearman_r': float(spearman_r),
                            'spearman_p': float(spearman_p),
                            'r_squared': float(pearson_r**2),
                            'n_observations': len(subset),
                            'significant': pearson_p < 0.05
                        }

        results['correlations'] = correlation_results

        # Outlier analysis
        outlier_results = {}
        for indicator in indicators:
            if indicator in self.health_df.columns:
                clean_data = self.health_df[['country', 'continent', indicator]].dropna()

                if len(clean_data) > 10:
                    values = clean_data[indicator]
                    z_scores = np.abs(stats.zscore(values))
                    z_outliers = clean_data[z_scores > 2.5]

                    outlier_results[indicator] = {
                        'z_score_outliers': z_outliers[['country', indicator]].to_dict('records'),
                        'outlier_countries': z_outliers['country'].tolist()
                    }

        results['outliers'] = outlier_results

        self.results = results
        return results

    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        print("Generating HTML report...")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Health and Well-being Analysis - Global Thriving Study</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 40px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                .navigation {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .navigation a {{
                    display: inline-block;
                    margin: 5px 10px;
                    padding: 10px 20px;
                    background-color: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .navigation a:hover {{
                    background-color: #2980b9;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #3498db;
                }}
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                    display: block;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 30px 0;
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .chart-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 15px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 0.9em;
                }}
                th, td {{
                    border: 1px solid #bdc3c7;
                    padding: 10px;
                    text-align: left;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .highlight {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .success {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .alert {{
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .correlation-strong {{ background-color: #d4edda; }}
                .correlation-moderate {{ background-color: #fff3cd; }}
                .correlation-weak {{ background-color: #f8d7da; }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #bdc3c7;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="navigation">
                    <a href="index.html">← Back to Dashboard</a>
                    <a href="#executive-summary">Executive Summary</a>
                    <a href="#distributions">Distributions</a>
                    <a href="#regional">Regional Analysis</a>
                    <a href="#correlations">Correlations</a>
                    <a href="#outliers">Outliers</a>
                    <a href="#mortality">Mortality</a>
                </div>

                <h1>Enhanced Health and Well-being Analysis</h1>
                <p style="text-align: center;"><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

                <div class="highlight">
                    <h3>Key Research Questions</h3>
                    <ul>
                        <li>To what extent are global citizens thriving in terms of health and well-being?</li>
                        <li>How do health outcomes, life satisfaction, and mortality patterns compare across countries and regions?</li>
                        <li>What are the key relationships between different health indicators?</li>
                        <li>Which countries represent outliers in terms of health outcomes?</li>
                    </ul>
                </div>

                <section id="executive-summary">
                    <h2>Executive Summary</h2>
                    {self.generate_executive_summary()}
                </section>

                <section id="distributions">
                    <h2>Health Indicator Distributions</h2>
                    <p>The distribution analysis examines the shape, central tendency, and spread of each health indicator across all countries.</p>
                    <div class="chart-container">
                        <div class="chart-title">Health Indicator Distributions</div>
                        <img src="health_distributions.png" alt="Health Distributions">
                    </div>
                    {self.generate_univariate_table()}
                </section>

                <section id="regional">
                    <h2>Regional Analysis</h2>
                    <p>Regional analysis compares health indicators across continents using ANOVA tests to identify significant differences.</p>
                    <div class="chart-container">
                        <div class="chart-title">Health Indicators by Continent</div>
                        <img src="regional_analysis.png" alt="Regional Analysis">
                    </div>
                    {self.generate_anova_results()}
                </section>

                <section id="correlations">
                    <h2>Correlation Analysis</h2>
                    <p>Correlation analysis examines the relationships between different health indicators.</p>
                    <div class="chart-container">
                        <div class="chart-title">Correlation Analysis</div>
                        <img src="correlation_analysis.png" alt="Correlation Analysis">
                    </div>
                    {self.generate_correlation_table()}
                </section>

                <section id="outliers">
                    <h2>Outlier Analysis</h2>
                    <p>Outlier analysis identifies countries with unusual health indicator values using Z-score methodology.</p>
                    <div class="chart-container">
                        <div class="chart-title">Outlier Detection</div>
                        <img src="outlier_analysis.png" alt="Outlier Analysis">
                    </div>
                    {self.generate_outlier_section()}
                </section>

                <section id="mortality">
                    <h2>Mortality Patterns</h2>
                    <p>Mortality analysis examines the leading causes of death globally and by continent.</p>
                    <div class="chart-container">
                        <div class="chart-title">Mortality Patterns</div>
                        <img src="mortality_analysis.png" alt="Mortality Analysis">
                    </div>
                </section>

                <div class="footer">
                    <p>Generated by Enhanced Global Thriving Analysis System</p>
                    <p>Data Source: World Data 2.0-Data.csv</p>
                    <p>Analysis completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p>Statistical methods: ANOVA, correlation analysis, outlier detection, descriptive statistics</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Save the HTML file
        html_file = self.output_dir / "enhanced_health_analysis.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Enhanced HTML report saved to: {html_file}")
        return html_file

    def generate_executive_summary(self):
        """Generate executive summary section"""
        summary_html = """
        <div class="stats-grid">
        """

        if 'univariate' in self.results:
            for indicator, stats in self.results['univariate'].items():
                clean_name = indicator.replace('_', ' ').title()
                summary_html += f"""
                    <div class="stat-card">
                        <span class="stat-number">{stats['mean']:.2f}</span>
                        <div class="stat-label">{clean_name} (Mean)</div>
                    </div>
                """

        if 'correlations' in self.results:
            strong_corrs = sum(1 for corr in self.results['correlations'].values() if abs(corr.get('pearson_r', 0)) > 0.5)
            summary_html += f"""
                <div class="stat-card">
                    <span class="stat-number">{strong_corrs}</span>
                    <div class="stat-label">Strong Correlations</div>
                </div>
            """

        if 'regional' in self.results:
            significant_differences = sum(1 for analysis in self.results['regional'].values()
                                        if analysis.get('significant_difference', False))
            summary_html += f"""
                <div class="stat-card">
                    <span class="stat-number">{significant_differences}</span>
                    <div class="stat-label">Significant Regional Differences</div>
                </div>
            """

        summary_html += "</div>"
        return summary_html

    def generate_univariate_table(self):
        """Generate univariate statistics table"""
        if 'univariate' not in self.results:
            return ""

        table_html = """
        <h3>Statistical Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Indicator</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Skewness</th>
                    <th>Normal Distribution?</th>
                </tr>
            </thead>
            <tbody>
        """

        for indicator, stats in self.results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()
            is_normal = "Yes" if stats.get('is_normal', False) else "No"
            normal_class = "success" if stats.get('is_normal', False) else "alert"

            table_html += f"""
                <tr>
                    <td>{clean_name}</td>
                    <td>{stats['count']}</td>
                    <td>{stats['mean']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                    <td>{stats['skewness']:.3f}</td>
                    <td>{is_normal}</td>
                </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return table_html

    def generate_anova_results(self):
        """Generate ANOVA results section"""
        if 'regional' not in self.results:
            return ""

        section_html = "<h3>ANOVA Results</h3>"

        for indicator, analysis in self.results['regional'].items():
            clean_name = indicator.replace('_', ' ').title()
            anova_p = analysis.get('anova_p', 1)
            significant = analysis.get('significant_difference', False)

            status_class = "success" if significant else "alert"
            status_text = "Significant" if significant else "Not Significant"

            section_html += f"""
            <div class="{status_class}">
                <h4>{clean_name}</h4>
                <p><strong>ANOVA Result:</strong> {status_text} (p = {anova_p:.4f})</p>
                <p><strong>Interpretation:</strong> {'There are significant differences between continents.' if significant else 'No significant differences between continents.'}</p>
            </div>
            """

        return section_html

    def generate_correlation_table(self):
        """Generate correlation results table"""
        if 'correlations' not in self.results:
            return ""

        table_html = """
        <h3>Correlation Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Relationship</th>
                    <th>Correlation (r)</th>
                    <th>R²</th>
                    <th>P-value</th>
                    <th>Significance</th>
                    <th>Strength</th>
                </tr>
            </thead>
            <tbody>
        """

        for relationship, stats in self.results['correlations'].items():
            clean_rel = relationship.replace('_vs_', ' vs ').replace('_', ' ').title()
            r_value = stats['pearson_r']
            r_squared = stats['r_squared']
            p_value = stats['pearson_p']

            # Determine strength and class
            abs_r = abs(r_value)
            if abs_r > 0.7:
                strength = "Strong"
                corr_class = "correlation-strong"
            elif abs_r > 0.3:
                strength = "Moderate"
                corr_class = "correlation-moderate"
            else:
                strength = "Weak"
                corr_class = "correlation-weak"

            significance = "Significant" if p_value < 0.05 else "Not Significant"

            table_html += f"""
                <tr class="{corr_class}">
                    <td>{clean_rel}</td>
                    <td>{r_value:.3f}</td>
                    <td>{r_squared:.3f}</td>
                    <td>{p_value:.4f}</td>
                    <td>{significance}</td>
                    <td>{strength}</td>
                </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return table_html

    def generate_outlier_section(self):
        """Generate outlier section"""
        if 'outliers' not in self.results:
            return ""

        section_html = "<h3>Identified Outlier Countries</h3>"

        for indicator, outlier_data in self.results['outliers'].items():
            clean_name = indicator.replace('_', ' ').title()
            outlier_countries = outlier_data.get('outlier_countries', [])

            if outlier_countries:
                section_html += f"""
                <h4>{clean_name}</h4>
                <div class="alert">
                    <strong>Z-score Outliers (|z| > 2.5):</strong> {', '.join(outlier_countries)}
                </div>
                """
            else:
                section_html += f"""
                <h4>{clean_name}</h4>
                <div class="success">
                    <strong>No significant outliers detected.</strong>
                </div>
                """

        return section_html

    def run_complete_analysis(self):
        """Run all analysis components"""
        print("Starting enhanced health and well-being analysis...")

        self.prepare_data()
        self.create_distribution_plots()
        self.create_regional_plots()
        self.create_correlation_plots()
        self.create_outlier_plots()
        self.create_mortality_plots()
        self.perform_statistical_analysis()
        self.generate_html_report()

        print("Enhanced analysis complete!")
        return self.results

def load_data():
    """Load and clean the dataset"""
    try:
        source_path = "../source-files/World Data 2.0-Data.csv"
        print(f"Loading data from {source_path}...")

        df = pd.read_csv(source_path)
        print(f"Raw data shape: {df.shape}")

        # Filter to valid continents
        valid_continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        clean_df = df[df['Continent'].isin(valid_continents)].copy()
        clean_df = clean_df.drop_duplicates(subset=['Country'], keep='first')

        print(f"Clean data shape: {clean_df.shape}")
        return clean_df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("=" * 80)
    print("FIXED ENHANCED GLOBAL THRIVING ANALYSIS: HEALTH & WELL-BEING")
    print("=" * 80)

    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False

    # Run enhanced analysis
    analyzer = FixedHealthAnalyzer(df)
    results = analyzer.run_complete_analysis()

    print("\n" + "=" * 80)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("=" * 80)

    # Display key findings
    if 'univariate' in results:
        print("\n📊 HEALTH INDICATORS SUMMARY:")
        for indicator, stats in results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()
            print(f"   • {clean_name}: Mean = {stats['mean']:.2f} (±{stats['std']:.2f}), n = {stats['count']}")

    if 'correlations' in results:
        print("\n🔗 KEY CORRELATIONS:")
        for relationship, stats in results['correlations'].items():
            if stats['significant']:
                clean_rel = relationship.replace('_vs_', ' ↔ ').replace('_', ' ').title()
                print(f"   • {clean_rel}: r = {stats['pearson_r']:.3f} (p < 0.05)")

    if 'regional' in results:
        print("\n🌍 REGIONAL DIFFERENCES:")
        for indicator, analysis in results['regional'].items():
            clean_name = indicator.replace('_', ' ').title()
            if analysis.get('significant_difference', False):
                print(f"   • {clean_name}: Significant differences between continents (p = {analysis['anova_p']:.4f})")

    if 'outliers' in results:
        print("\n🎯 OUTLIER COUNTRIES:")
        for indicator, outlier_data in results['outliers'].items():
            outlier_countries = outlier_data.get('outlier_countries', [])
            if outlier_countries:
                clean_name = indicator.replace('_', ' ').title()
                print(f"   • {clean_name}: {', '.join(outlier_countries[:3])}{'...' if len(outlier_countries) > 3 else ''}")

    print(f"\n🌐 Enhanced report available at:")
    print(f"    {analyzer.output_dir / 'enhanced_health_analysis.html'}")

    print(f"\n📊 Generated visualizations:")
    visualizations = ['health_distributions.png', 'regional_analysis.png', 'correlation_analysis.png',
                     'outlier_analysis.png', 'mortality_analysis.png']
    for viz in visualizations:
        if (analyzer.output_dir / viz).exists():
            print(f"   ✅ {viz}")

    return True

if __name__ == "__main__":
    main()