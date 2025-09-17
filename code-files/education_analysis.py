"""
Education Analysis Module for Global Thriving Study

This module performs comprehensive analysis of educational attainment indicators,
examining learning-adjusted years of schooling and its relationships with other
development indicators across countries and continents.

Research Questions:
- What is the average number of learning-adjusted school years across countries?
- How does educational attainment correlate with GDP per capita and life expectancy?
- Do countries with higher education levels also report higher life satisfaction?
- Are there significant regional differences in educational outcomes?
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

class EducationAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.output_dir = Path("../analysis-output-files")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_education_data(self):
        """Prepare and clean education-related data"""
        print("Preparing education data...")

        # Create a working copy
        self.education_df = self.df.copy()

        # Convert numeric columns with proper handling
        numeric_cols = {
            'Learning-Adjusted Years of School (2020)': 'education_years',
            'GDP per Capita (2018)': 'gdp_per_capita',
            'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction',
            'Life expectancy at birth (years)(2018)': 'life_expectancy',
            'Unemployment Rate (2018)': 'unemployment_rate',
            'Total population (2018)': 'population'
        }

        for orig_col, new_col in numeric_cols.items():
            if orig_col in self.education_df.columns:
                # Clean and convert to numeric
                if orig_col in ['Total population (2018)', 'GDP per Capita (2018)']:
                    # Handle columns with potential formatting issues
                    self.education_df[new_col] = (
                        self.education_df[orig_col]
                        .astype(str)
                        .str.replace('"', '')
                        .str.replace(',', '')
                        .str.strip()
                    )
                else:
                    self.education_df[new_col] = self.education_df[orig_col]

                # Convert to numeric
                self.education_df[new_col] = pd.to_numeric(self.education_df[new_col], errors='coerce')

        # Clean continent and country names
        self.education_df['continent'] = self.education_df['Continent']
        self.education_df['country'] = self.education_df['Country']

        # Remove rows with missing education data
        self.education_df = self.education_df.dropna(subset=['education_years'])

        print(f"Education data prepared: {len(self.education_df)} countries with education data")
        return self.education_df

    def univariate_education_analysis(self):
        """Perform univariate analysis of education indicators"""
        print("Performing univariate education analysis...")

        clean_data = self.education_df['education_years'].dropna()

        if len(clean_data) > 0:
            # Calculate comprehensive statistics
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

            # Education level categorization
            low_education = clean_data[clean_data <= clean_data.quantile(0.33)]
            medium_education = clean_data[(clean_data > clean_data.quantile(0.33)) &
                                        (clean_data <= clean_data.quantile(0.67))]
            high_education = clean_data[clean_data > clean_data.quantile(0.67)]

            stats_dict['education_categories'] = {
                'low_education_threshold': float(clean_data.quantile(0.33)),
                'high_education_threshold': float(clean_data.quantile(0.67)),
                'low_education_count': len(low_education),
                'medium_education_count': len(medium_education),
                'high_education_count': len(high_education)
            }

            self.results['univariate'] = stats_dict
            print(f"Education years: Mean = {stats_dict['mean']:.2f}, Std = {stats_dict['std']:.2f}")

        return stats_dict

    def regional_education_analysis(self):
        """Analyze education by continent with ANOVA"""
        print("Performing regional education analysis...")

        # Prepare data for ANOVA
        continent_groups = []
        continent_names = []

        for continent in self.education_df['continent'].unique():
            if pd.notna(continent):
                group_data = self.education_df[
                    (self.education_df['continent'] == continent) &
                    (self.education_df['education_years'].notna())
                ]['education_years'].values

                if len(group_data) > 0:
                    continent_groups.append(group_data)
                    continent_names.append(continent)

        regional_results = {}

        # Perform ANOVA
        if len(continent_groups) > 1:
            try:
                anova_stat, anova_p = f_oneway(*continent_groups)

                # Calculate effect size (eta-squared)
                all_values = np.concatenate(continent_groups)
                grand_mean = np.mean(all_values)
                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in continent_groups)
                ss_total = sum((x - grand_mean)**2 for x in all_values)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                # Regional descriptive statistics
                regional_stats = self.education_df.groupby('continent')['education_years'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(3)

                # Post-hoc pairwise comparisons
                pairwise_results = {}
                for i in range(len(continent_names)):
                    for j in range(i+1, len(continent_names)):
                        t_stat, t_p = stats.ttest_ind(continent_groups[i], continent_groups[j])
                        pairwise_results[f"{continent_names[i]}_vs_{continent_names[j]}"] = {
                            't_stat': float(t_stat),
                            'p_value': float(t_p),
                            'significant': t_p < 0.05,
                            'effect_size': abs(float(t_stat)) / np.sqrt(len(continent_groups[i]) + len(continent_groups[j]))
                        }

                regional_results = {
                    'anova_stat': float(anova_stat),
                    'anova_p': float(anova_p),
                    'significant_difference': anova_p < 0.05,
                    'eta_squared': float(eta_squared),
                    'regional_stats': regional_stats.to_dict('index'),
                    'pairwise_comparisons': pairwise_results,
                    'continent_names': continent_names
                }

                print(f"Regional ANOVA: F = {anova_stat:.3f}, p = {anova_p:.4f}")

            except Exception as e:
                print(f"ANOVA failed: {e}")

        self.results['regional'] = regional_results
        return regional_results

    def education_correlations(self):
        """Analyze correlations between education and other indicators"""
        print("Performing education correlation analysis...")

        # Select indicators for correlation analysis
        correlation_vars = ['education_years', 'gdp_per_capita', 'life_satisfaction',
                           'life_expectancy', 'unemployment_rate']

        available_vars = [var for var in correlation_vars if var in self.education_df.columns]

        if len(available_vars) < 2:
            return {}

        # Create correlation subset
        corr_data = self.education_df[available_vars + ['continent', 'country']].dropna()

        if len(corr_data) < 10:
            return {}

        # Calculate correlation matrix
        corr_matrix = corr_data[available_vars].corr()

        # Detailed pairwise correlations
        correlation_results = {}
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i < j and var1 == 'education_years':  # Focus on education correlations
                    subset = corr_data[[var1, var2]].dropna()
                    if len(subset) > 3:
                        pearson_r, pearson_p = stats.pearsonr(subset[var1], subset[var2])
                        spearman_r, spearman_p = stats.spearmanr(subset[var1], subset[var2])

                        # Linear regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(subset[var1], subset[var2])

                        correlation_results[f"education_vs_{var2}"] = {
                            'pearson_r': float(pearson_r),
                            'pearson_p': float(pearson_p),
                            'spearman_r': float(spearman_r),
                            'spearman_p': float(spearman_p),
                            'r_squared': float(r_value**2),
                            'regression_slope': float(slope),
                            'regression_intercept': float(intercept),
                            'n_observations': len(subset),
                            'significant': pearson_p < 0.05,
                            'correlation_strength': self.get_correlation_strength(abs(pearson_r))
                        }

        # Store correlation matrix
        correlation_results['correlation_matrix'] = corr_matrix.to_dict()
        correlation_results['available_variables'] = available_vars

        self.results['correlations'] = correlation_results
        return correlation_results

    def education_outliers(self):
        """Identify countries with unusual education patterns"""
        print("Identifying education outliers...")

        clean_data = self.education_df[['country', 'continent', 'education_years']].dropna()

        if len(clean_data) < 10:
            return {}

        values = clean_data['education_years']

        # Z-score method
        z_scores = np.abs(stats.zscore(values))
        z_outliers = clean_data[z_scores > 2.5].copy()
        z_outliers['z_score'] = z_scores[z_scores > 2.5]

        # IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_outliers = clean_data[
            (values < lower_bound) | (values > upper_bound)
        ].copy()

        # Categorize outliers
        high_performers = z_outliers[z_outliers['education_years'] > values.mean()]
        low_performers = z_outliers[z_outliers['education_years'] < values.mean()]

        outlier_results = {
            'z_score_outliers': z_outliers.to_dict('records'),
            'iqr_outliers': iqr_outliers.to_dict('records'),
            'high_performers': high_performers[['country', 'education_years']].to_dict('records'),
            'low_performers': low_performers[['country', 'education_years']].to_dict('records'),
            'outlier_countries': {
                'z_score': z_outliers['country'].tolist(),
                'iqr': iqr_outliers['country'].tolist(),
                'high_education': high_performers['country'].tolist(),
                'low_education': low_performers['country'].tolist()
            },
            'bounds': {
                'iqr_lower': float(lower_bound),
                'iqr_upper': float(upper_bound),
                'mean': float(values.mean()),
                'std': float(values.std())
            }
        }

        self.results['outliers'] = outlier_results
        return outlier_results

    def get_correlation_strength(self, r_value):
        """Categorize correlation strength"""
        if r_value >= 0.7:
            return "Strong"
        elif r_value >= 0.3:
            return "Moderate"
        else:
            return "Weak"

    def create_education_visualizations(self):
        """Create comprehensive education visualizations"""
        print("Creating education visualizations...")

        # 1. Distribution and Summary Statistics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Education Analysis: Learning-Adjusted Years of School', fontsize=16, fontweight='bold')

        # Distribution plot
        clean_data = self.education_df['education_years'].dropna()
        axes[0, 0].hist(clean_data, bins=20, alpha=0.7, density=True,
                       color='skyblue', edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(clean_data.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {clean_data.mean():.2f}')
        axes[0, 0].axvline(clean_data.median(), color='orange', linestyle=':', linewidth=2,
                          label=f'Median: {clean_data.median():.2f}')
        axes[0, 0].set_title('Distribution of Learning-Adjusted Years of School', fontweight='bold')
        axes[0, 0].set_xlabel('Years of Education')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Regional comparison
        plot_data = self.education_df[['continent', 'education_years']].dropna()
        sns.boxplot(data=plot_data, x='continent', y='education_years', ax=axes[0, 1],
                   palette='Set2', showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red"})
        axes[0, 1].set_title('Education by Continent', fontweight='bold')
        axes[0, 1].set_xlabel('Continent')
        axes[0, 1].set_ylabel('Learning-Adjusted Years')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Top and bottom countries
        top_countries = self.education_df.nlargest(10, 'education_years')
        bottom_countries = self.education_df.nsmallest(10, 'education_years')

        y_pos_top = np.arange(len(top_countries))
        axes[1, 0].barh(y_pos_top, top_countries['education_years'],
                       color='green', alpha=0.7)
        axes[1, 0].set_yticks(y_pos_top)
        axes[1, 0].set_yticklabels(top_countries['country'], fontsize=9)
        axes[1, 0].set_xlabel('Years of Education')
        axes[1, 0].set_title('Top 10 Countries by Education', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        y_pos_bottom = np.arange(len(bottom_countries))
        axes[1, 1].barh(y_pos_bottom, bottom_countries['education_years'],
                       color='red', alpha=0.7)
        axes[1, 1].set_yticks(y_pos_bottom)
        axes[1, 1].set_yticklabels(bottom_countries['country'], fontsize=9)
        axes[1, 1].set_xlabel('Years of Education')
        axes[1, 1].set_title('Bottom 10 Countries by Education', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'education_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Correlation Analysis Visualization
        if 'correlations' in self.results and 'available_variables' in self.results['correlations']:
            available_vars = self.results['correlations']['available_variables']
            if len(available_vars) >= 3:

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle('Education Correlations with Development Indicators', fontsize=16, fontweight='bold')

                # Correlation heatmap
                corr_matrix = pd.DataFrame(self.results['correlations']['correlation_matrix'])
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                           square=True, ax=axes[0], fmt='.3f')
                axes[0].set_title('Correlation Matrix', fontweight='bold')

                # Education vs GDP scatter plot
                if 'gdp_per_capita' in available_vars:
                    corr_data = self.education_df[['education_years', 'gdp_per_capita', 'continent']].dropna()

                    for continent in corr_data['continent'].unique():
                        if pd.notna(continent):
                            cont_data = corr_data[corr_data['continent'] == continent]
                            axes[1].scatter(cont_data['education_years'], cont_data['gdp_per_capita'],
                                           label=continent, alpha=0.7, s=60)

                    # Add regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        corr_data['education_years'], corr_data['gdp_per_capita'])
                    x_range = np.linspace(corr_data['education_years'].min(),
                                         corr_data['education_years'].max(), 100)
                    y_pred = slope * x_range + intercept
                    axes[1].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                                label=f'R² = {r_value**2:.3f}')

                    axes[1].set_xlabel('Learning-Adjusted Years of School')
                    axes[1].set_ylabel('GDP per Capita')
                    axes[1].set_title('Education vs Economic Development', fontweight='bold')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                # Education vs Life Satisfaction scatter plot
                if 'life_satisfaction' in available_vars:
                    corr_data = self.education_df[['education_years', 'life_satisfaction', 'continent']].dropna()

                    for continent in corr_data['continent'].unique():
                        if pd.notna(continent):
                            cont_data = corr_data[corr_data['continent'] == continent]
                            axes[2].scatter(cont_data['education_years'], cont_data['life_satisfaction'],
                                           label=continent, alpha=0.7, s=60)

                    # Add regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        corr_data['education_years'], corr_data['life_satisfaction'])
                    x_range = np.linspace(corr_data['education_years'].min(),
                                         corr_data['education_years'].max(), 100)
                    y_pred = slope * x_range + intercept
                    axes[2].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                                label=f'R² = {r_value**2:.3f}')

                    axes[2].set_xlabel('Learning-Adjusted Years of School')
                    axes[2].set_ylabel('Life Satisfaction (Cantril Ladder)')
                    axes[2].set_title('Education vs Life Satisfaction', fontweight='bold')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(self.output_dir / 'education_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()

        # 3. Outlier Analysis
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Education Outlier Analysis', fontsize=16, fontweight='bold')

        # Box plot with outliers
        clean_data = self.education_df[['country', 'education_years']].dropna()
        bp = axes[0].boxplot([clean_data['education_years']], patch_artist=True,
                            labels=['Learning-Adjusted Years'])
        bp['boxes'][0].set_facecolor('lightblue')

        # Highlight outliers
        if 'outliers' in self.results:
            z_outliers = pd.DataFrame(self.results['outliers']['z_score_outliers'])
            if len(z_outliers) > 0:
                axes[0].scatter([1] * len(z_outliers), z_outliers['education_years'],
                               color='red', s=100, alpha=0.7, zorder=10,
                               label=f'Outliers ({len(z_outliers)})')

                # Annotate top outliers
                sorted_outliers = z_outliers.sort_values('education_years', ascending=False)
                for i, (idx, row) in enumerate(sorted_outliers.head(5).iterrows()):
                    axes[0].annotate(row['country'],
                                   (1, row['education_years']),
                                   xytext=(1.1, row['education_years']),
                                   fontsize=8, alpha=0.8)

        axes[0].set_title('Education Distribution with Outliers', fontweight='bold')
        axes[0].set_ylabel('Learning-Adjusted Years of School')
        axes[0].grid(True, alpha=0.3, axis='y')
        if 'outliers' in self.results and len(z_outliers) > 0:
            axes[0].legend()

        # Regional performance ranking
        if 'regional' in self.results and 'regional_stats' in self.results['regional']:
            regional_stats = pd.DataFrame(self.results['regional']['regional_stats']).T
            regional_means = regional_stats['mean'].sort_values(ascending=True)

            y_pos = np.arange(len(regional_means))
            bars = axes[1].barh(y_pos, regional_means.values, color='orange', alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(regional_means.index)
            axes[1].set_xlabel('Mean Learning-Adjusted Years')
            axes[1].set_title('Average Education by Continent', fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1].text(width + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{width:.1f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'education_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Education visualizations saved successfully!")

    def run_complete_analysis(self):
        """Run all education analysis components"""
        print("Starting comprehensive education analysis...")

        self.prepare_education_data()
        self.univariate_education_analysis()
        self.regional_education_analysis()
        self.education_correlations()
        self.education_outliers()
        self.create_education_visualizations()

        print("Education analysis complete!")
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
    print("GLOBAL THRIVING ANALYSIS: EDUCATION")
    print("=" * 80)

    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False

    # Run education analysis
    analyzer = EducationAnalyzer(df)
    results = analyzer.run_complete_analysis()

    print("\n" + "=" * 80)
    print("EDUCATION ANALYSIS COMPLETE!")
    print("=" * 80)

    # Display key findings
    if 'univariate' in results:
        print("\n📚 EDUCATION STATISTICS:")
        stats = results['univariate']
        print(f"   • Countries with education data: {stats['count']}")
        print(f"   • Global average: {stats['mean']:.2f} years (±{stats['std']:.2f})")
        print(f"   • Range: {stats['min']:.2f} - {stats['max']:.2f} years")
        print(f"   • Education categories:")
        print(f"     - Low education (≤{stats['education_categories']['low_education_threshold']:.1f} years): {stats['education_categories']['low_education_count']} countries")
        print(f"     - Medium education: {stats['education_categories']['medium_education_count']} countries")
        print(f"     - High education (>{stats['education_categories']['high_education_threshold']:.1f} years): {stats['education_categories']['high_education_count']} countries")

    if 'correlations' in results:
        print("\n🔗 KEY EDUCATION CORRELATIONS:")
        for relationship, stats in results['correlations'].items():
            if relationship.startswith('education_vs_') and isinstance(stats, dict) and stats.get('significant', False):
                var_name = relationship.replace('education_vs_', '').replace('_', ' ').title()
                print(f"   • Education ↔ {var_name}: r = {stats['pearson_r']:.3f} ({stats['correlation_strength']}, p < 0.05)")

    if 'regional' in results and results['regional'].get('significant_difference', False):
        print("\n🌍 REGIONAL DIFFERENCES:")
        regional_stats = results['regional']['regional_stats']
        sorted_regions = sorted(regional_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        print(f"   • Significant differences between continents (p = {results['regional']['anova_p']:.4f})")
        print("   • Regional ranking (highest to lowest):")
        for i, (continent, stats) in enumerate(sorted_regions, 1):
            print(f"     {i}. {continent}: {stats['mean']:.2f} years")

    if 'outliers' in results:
        print("\n🎯 EDUCATION OUTLIERS:")
        outliers = results['outliers']
        if outliers['outlier_countries']['high_education']:
            print(f"   • High performers: {', '.join(outliers['outlier_countries']['high_education'][:5])}")
        if outliers['outlier_countries']['low_education']:
            print(f"   • Low performers: {', '.join(outliers['outlier_countries']['low_education'][:5])}")

    print(f"\n📊 Generated visualizations:")
    visualizations = ['education_overview.png', 'education_correlations.png', 'education_outliers.png']
    for viz in visualizations:
        if (analyzer.output_dir / viz).exists():
            print(f"   ✅ {viz}")

    return True

if __name__ == "__main__":
    main()