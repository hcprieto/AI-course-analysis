"""
Health and Well-being Analysis Module

This module performs comprehensive analysis of health and well-being indicators
including life satisfaction, life expectancy, mortality patterns, and self-harm rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from data_loader import load_and_clean_data
from pathlib import Path

class HealthWellbeingAnalyzer:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.health_data = data_loader.get_health_indicators()
        self.results = {}

    def univariate_analysis(self):
        """Perform univariate analysis of health indicators"""
        print("Performing univariate analysis...")

        # Select numeric health indicators
        numeric_indicators = [
            'life_satisfaction_2018',
            'life_expectancy_2018',
            'self_harm_rate_2019'
        ]

        univariate_results = {}

        for indicator in numeric_indicators:
            if indicator in self.health_data.columns:
                # Remove missing values for analysis
                clean_data = self.health_data[indicator].dropna()

                if len(clean_data) > 0:
                    # Basic descriptive statistics
                    stats_dict = {
                        'count': len(clean_data),
                        'mean': clean_data.mean(),
                        'median': clean_data.median(),
                        'std': clean_data.std(),
                        'min': clean_data.min(),
                        'max': clean_data.max(),
                        'q25': clean_data.quantile(0.25),
                        'q75': clean_data.quantile(0.75),
                        'skewness': stats.skew(clean_data),
                        'kurtosis': stats.kurtosis(clean_data)
                    }

                    # Normality test
                    if len(clean_data) > 3:
                        shapiro_stat, shapiro_p = stats.shapiro(clean_data)
                        stats_dict['shapiro_stat'] = shapiro_stat
                        stats_dict['shapiro_p'] = shapiro_p
                        stats_dict['is_normal'] = shapiro_p > 0.05

                    univariate_results[indicator] = stats_dict

        self.results['univariate'] = univariate_results
        return univariate_results

    def bivariate_analysis(self):
        """Perform bivariate analysis between health indicators"""
        print("Performing bivariate analysis...")

        # Key relationships to examine
        relationships = [
            ('life_satisfaction_2018', 'life_expectancy_2018'),
            ('life_expectancy_2018', 'self_harm_rate_2019'),
            ('life_satisfaction_2018', 'self_harm_rate_2019')
        ]

        bivariate_results = {}

        for var1, var2 in relationships:
            if var1 in self.health_data.columns and var2 in self.health_data.columns:
                # Create subset with both variables present
                subset = self.health_data[[var1, var2, 'country', 'continent']].dropna()

                if len(subset) > 3:
                    # Correlation analysis
                    corr_pearson, p_pearson = stats.pearsonr(subset[var1], subset[var2])
                    corr_spearman, p_spearman = stats.spearmanr(subset[var1], subset[var2])

                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(subset[var1], subset[var2])

                    relationship_key = f"{var1}_vs_{var2}"
                    bivariate_results[relationship_key] = {
                        'n_observations': len(subset),
                        'pearson_r': corr_pearson,
                        'pearson_p': p_pearson,
                        'spearman_r': corr_spearman,
                        'spearman_p': p_spearman,
                        'regression_slope': slope,
                        'regression_intercept': intercept,
                        'r_squared': r_value**2,
                        'regression_p': p_value,
                        'data_subset': subset
                    }

        self.results['bivariate'] = bivariate_results
        return bivariate_results

    def regional_analysis(self):
        """Analyze health indicators by continent"""
        print("Performing regional analysis...")

        indicators = ['life_satisfaction_2018', 'life_expectancy_2018', 'self_harm_rate_2019']
        regional_results = {}

        for indicator in indicators:
            if indicator in self.health_data.columns:
                # Group by continent
                regional_stats = self.health_data.groupby('continent')[indicator].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(2)

                # ANOVA test
                continent_groups = []
                for continent in self.health_data['continent'].unique():
                    if pd.notna(continent):
                        group_data = self.health_data[
                            (self.health_data['continent'] == continent) &
                            (self.health_data[indicator].notna())
                        ][indicator]
                        if len(group_data) > 0:
                            continent_groups.append(group_data)

                if len(continent_groups) > 1:
                    anova_stat, anova_p = stats.f_oneway(*continent_groups)
                    regional_results[indicator] = {
                        'regional_stats': regional_stats,
                        'anova_stat': anova_stat,
                        'anova_p': anova_p,
                        'significant_difference': anova_p < 0.05
                    }

        self.results['regional'] = regional_results
        return regional_results

    def mortality_analysis(self):
        """Analyze mortality patterns and causes of death"""
        print("Performing mortality pattern analysis...")

        # Count most common causes of death
        death_causes = []
        for col in ['death_cause_1', 'death_cause_2', 'death_cause_3']:
            if col in self.health_data.columns:
                causes = self.health_data[col].dropna().tolist()
                death_causes.extend(causes)

        # Count frequency of each cause
        cause_counts = pd.Series(death_causes).value_counts()

        # Analyze top causes by continent
        mortality_by_continent = {}
        for continent in self.health_data['continent'].unique():
            if pd.notna(continent):
                continent_data = self.health_data[self.health_data['continent'] == continent]
                continent_causes = []
                for col in ['death_cause_1', 'death_cause_2', 'death_cause_3']:
                    if col in continent_data.columns:
                        causes = continent_data[col].dropna().tolist()
                        continent_causes.extend(causes)

                if continent_causes:
                    mortality_by_continent[continent] = pd.Series(continent_causes).value_counts().head(5)

        mortality_results = {
            'overall_top_causes': cause_counts.head(10),
            'by_continent': mortality_by_continent
        }

        self.results['mortality'] = mortality_results
        return mortality_results

    def identify_outliers(self):
        """Identify countries with unusual health patterns"""
        print("Identifying outliers...")

        outliers = {}
        indicators = ['life_satisfaction_2018', 'life_expectancy_2018', 'self_harm_rate_2019']

        for indicator in indicators:
            if indicator in self.health_data.columns:
                clean_data = self.health_data[['country', 'continent', indicator]].dropna()

                if len(clean_data) > 10:  # Need sufficient data for outlier detection
                    # Calculate IQR method outliers
                    Q1 = clean_data[indicator].quantile(0.25)
                    Q3 = clean_data[indicator].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_countries = clean_data[
                        (clean_data[indicator] < lower_bound) |
                        (clean_data[indicator] > upper_bound)
                    ]

                    # Z-score method outliers
                    z_scores = np.abs(stats.zscore(clean_data[indicator]))
                    z_outliers = clean_data[z_scores > 2.5]

                    outliers[indicator] = {
                        'iqr_outliers': outlier_countries,
                        'z_score_outliers': z_outliers,
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    }

        self.results['outliers'] = outliers
        return outliers

    def run_complete_analysis(self):
        """Run all analysis components"""
        print("Starting complete health and well-being analysis...")

        self.univariate_analysis()
        self.bivariate_analysis()
        self.regional_analysis()
        self.mortality_analysis()
        self.identify_outliers()

        print("Analysis complete!")
        return self.results

def create_health_visualizations(analyzer, output_dir):
    """Create visualizations for health analysis"""
    print("Creating health visualizations...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    health_data = analyzer.health_data
    results = analyzer.results

    # Set style
    plt.style.use('seaborn-v0_8')

    # 1. Distribution plots for key indicators
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Health and Well-being Indicators', fontsize=16, fontweight='bold')

    indicators = [
        ('life_satisfaction_2018', 'Life Satisfaction (Cantril Ladder)', axes[0,0]),
        ('life_expectancy_2018', 'Life Expectancy (Years)', axes[0,1]),
        ('self_harm_rate_2019', 'Self-harm Rate (per 100k)', axes[1,0])
    ]

    for indicator, title, ax in indicators:
        if indicator in health_data.columns:
            clean_data = health_data[indicator].dropna()
            if len(clean_data) > 0:
                ax.hist(clean_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

    # Remove empty subplot
    axes[1,1].remove()

    plt.tight_layout()
    plt.savefig(output_path / 'health_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Regional comparison boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Health Indicators by Continent', fontsize=16, fontweight='bold')

    indicators = [
        ('life_satisfaction_2018', 'Life Satisfaction', axes[0]),
        ('life_expectancy_2018', 'Life Expectancy', axes[1]),
        ('self_harm_rate_2019', 'Self-harm Rate', axes[2])
    ]

    for indicator, title, ax in indicators:
        if indicator in health_data.columns:
            clean_data = health_data[['continent', indicator]].dropna()
            if len(clean_data) > 0:
                sns.boxplot(data=clean_data, x='continent', y=indicator, ax=ax)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Continent')
                ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / 'health_by_continent.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Correlation scatter plots
    if 'bivariate' in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Health Indicator Relationships', fontsize=16, fontweight='bold')

        plot_configs = [
            ('life_satisfaction_2018_vs_life_expectancy_2018', 'Life Satisfaction vs Life Expectancy', axes[0]),
            ('life_expectancy_2018_vs_self_harm_rate_2019', 'Life Expectancy vs Self-harm Rate', axes[1]),
            ('life_satisfaction_2018_vs_self_harm_rate_2019', 'Life Satisfaction vs Self-harm Rate', axes[2])
        ]

        for key, title, ax in plot_configs:
            if key in results['bivariate']:
                data_subset = results['bivariate'][key]['data_subset']
                var1, var2 = key.split('_vs_')

                # Color by continent
                continents = data_subset['continent'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(continents)))

                for i, continent in enumerate(continents):
                    cont_data = data_subset[data_subset['continent'] == continent]
                    ax.scatter(cont_data[var1], cont_data[var2],
                             label=continent, alpha=0.7, color=colors[i])

                # Add regression line
                r_squared = results['bivariate'][key]['r_squared']
                slope = results['bivariate'][key]['regression_slope']
                intercept = results['bivariate'][key]['regression_intercept']

                x_range = np.linspace(data_subset[var1].min(), data_subset[var1].max(), 100)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, 'r--', alpha=0.8,
                       label=f'R² = {r_squared:.3f}')

                ax.set_title(title, fontweight='bold')
                ax.set_xlabel(var1.replace('_', ' ').title())
                ax.set_ylabel(var2.replace('_', ' ').title())
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'health_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_path}")

if __name__ == "__main__":
    # Load data and run analysis
    source_path = "../source-files/World Data 2.0-Data.csv"
    loader, clean_data = load_and_clean_data(source_path)

    if loader:
        # Create analyzer and run analysis
        analyzer = HealthWellbeingAnalyzer(loader)
        results = analyzer.run_complete_analysis()

        # Create visualizations
        create_health_visualizations(analyzer, "../analysis-output-files/")

        print("\nHealth and Well-being Analysis Summary:")
        print("=" * 50)

        if 'univariate' in results:
            print("\nUnivariate Analysis Results:")
            for indicator, stats in results['univariate'].items():
                print(f"\n{indicator.replace('_', ' ').title()}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")

        if 'bivariate' in results:
            print("\nKey Correlations:")
            for relationship, stats in results['bivariate'].items():
                print(f"\n{relationship.replace('_', ' ').title()}:")
                print(f"  Correlation (r): {stats['pearson_r']:.3f}")
                print(f"  R-squared: {stats['r_squared']:.3f}")
                print(f"  P-value: {stats['pearson_p']:.3f}")
    else:
        print("Failed to load data for analysis")