"""
Enhanced Health and Well-being Analysis with Visualizations

This module performs comprehensive health analysis with:
- Statistical visualizations (distributions, correlations, regional comparisons)
- Regional analysis with ANOVA tests
- Outlier detection and identification
- Interactive charts and comprehensive reporting
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
import base64
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class EnhancedHealthAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.charts = {}

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

    def univariate_analysis(self):
        """Comprehensive univariate analysis with visualizations"""
        print("Performing univariate analysis...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        univariate_results = {}

        # Create figure for distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Health and Well-being Indicators', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns:
                clean_data = self.health_df[indicator].dropna()

                if len(clean_data) > 0:
                    # Calculate statistics
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

                    # Normality tests
                    if len(clean_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(clean_data[:5000])  # Limit for shapiro
                        stats_dict['shapiro_stat'] = float(shapiro_stat)
                        stats_dict['shapiro_p'] = float(shapiro_p)
                        stats_dict['is_normal'] = shapiro_p > 0.05

                    univariate_results[indicator] = stats_dict

                    # Create visualization
                    if i < 3:  # We have 3 subplots for 3 indicators
                        row, col = divmod(i, 2)
                        ax = axes[row, col]

                        # Histogram with KDE
                        ax.hist(clean_data, bins=20, alpha=0.7, density=True,
                               color=sns.color_palette()[i], edgecolor='black')

                        # Add KDE line
                        try:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(clean_data)
                            x_range = np.linspace(clean_data.min(), clean_data.max(), 100)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                        except:
                            pass

                        # Add mean line
                        ax.axvline(stats_dict['mean'], color='red', linestyle='--',
                                  label=f'Mean: {stats_dict["mean"]:.2f}')

                        # Formatting
                        title_map = {
                            'life_satisfaction': 'Life Satisfaction (Cantril Ladder)',
                            'life_expectancy': 'Life Expectancy (Years)',
                            'self_harm_rate': 'Self-harm Rate (per 100k)'
                        }
                        ax.set_title(title_map.get(indicator, indicator), fontweight='bold')
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Density')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

        # Remove empty subplot
        axes[1, 1].remove()

        plt.tight_layout()
        self.save_chart(plt, 'health_distributions')

        self.results['univariate'] = univariate_results
        return univariate_results

    def regional_analysis(self):
        """Comprehensive regional analysis with ANOVA"""
        print("Performing regional analysis...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        regional_results = {}

        # Create regional comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Health Indicators by Continent', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns:
                # Prepare data for ANOVA
                continent_data = []
                continent_names = []

                for continent in self.health_df['continent'].unique():
                    if pd.notna(continent):
                        group_data = self.health_df[
                            (self.health_df['continent'] == continent) &
                            (self.health_df[indicator].notna())
                        ][indicator].values

                        if len(group_data) > 0:
                            continent_data.append(group_data)
                            continent_names.append(continent)

                # Perform ANOVA
                if len(continent_data) > 1:
                    try:
                        anova_stat, anova_p = f_oneway(*continent_data)

                        # Post-hoc analysis (pairwise comparisons)
                        pairwise_results = {}
                        for j in range(len(continent_names)):
                            for k in range(j+1, len(continent_names)):
                                t_stat, t_p = stats.ttest_ind(continent_data[j], continent_data[k])
                                pairwise_results[f"{continent_names[j]}_vs_{continent_names[k]}"] = {
                                    't_stat': float(t_stat),
                                    'p_value': float(t_p),
                                    'significant': t_p < 0.05
                                }

                        # Regional descriptive statistics
                        regional_stats = self.health_df.groupby('continent')[indicator].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(3)

                        regional_results[indicator] = {
                            'anova_stat': float(anova_stat),
                            'anova_p': float(anova_p),
                            'significant_difference': anova_p < 0.05,
                            'regional_stats': regional_stats.to_dict('index'),
                            'pairwise_comparisons': pairwise_results,
                            'effect_size': self.calculate_eta_squared(continent_data, anova_stat)
                        }

                    except Exception as e:
                        print(f"ANOVA failed for {indicator}: {e}")
                        continue

                # Create boxplot
                ax = axes[i]
                plot_data = self.health_df[['continent', indicator]].dropna()

                if len(plot_data) > 0:
                    sns.boxplot(data=plot_data, x='continent', y=indicator, ax=ax)

                    # Add mean points
                    means = plot_data.groupby('continent')[indicator].mean()
                    for j, (continent, mean_val) in enumerate(means.items()):
                        ax.plot(j, mean_val, 'ro', markersize=8, label='Mean' if j == 0 else "")

                    # Formatting
                    title_map = {
                        'life_satisfaction': 'Life Satisfaction',
                        'life_expectancy': 'Life Expectancy (Years)',
                        'self_harm_rate': 'Self-harm Rate (per 100k)'
                    }
                    ax.set_title(title_map.get(indicator, indicator), fontweight='bold')
                    ax.set_xlabel('Continent')
                    ax.tick_params(axis='x', rotation=45)

                    # Add ANOVA result to plot
                    if indicator in regional_results:
                        anova_text = f"ANOVA p = {regional_results[indicator]['anova_p']:.4f}"
                        sig_text = " (Significant)" if regional_results[indicator]['significant_difference'] else " (Not Significant)"
                        ax.text(0.02, 0.98, anova_text + sig_text, transform=ax.transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        self.save_chart(plt, 'regional_analysis')

        self.results['regional'] = regional_results
        return regional_results

    def outlier_analysis(self):
        """Comprehensive outlier detection and analysis"""
        print("Performing outlier analysis...")

        indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        outlier_results = {}

        # Create outlier visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Outlier Analysis: Z-scores and Box Plots', fontsize=16, fontweight='bold')

        for i, indicator in enumerate(indicators):
            if indicator in self.health_df.columns:
                clean_data = self.health_df[['country', 'continent', indicator]].dropna()

                if len(clean_data) > 10:
                    values = clean_data[indicator]

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

                    # Modified Z-score (using median)
                    median = values.median()
                    mad = np.median(np.abs(values - median))
                    modified_z = 0.6745 * (values - median) / mad
                    modified_outliers = clean_data[np.abs(modified_z) > 3.5].copy()

                    outlier_results[indicator] = {
                        'z_score_outliers': z_outliers.to_dict('records'),
                        'iqr_outliers': iqr_outliers.to_dict('records'),
                        'modified_z_outliers': modified_outliers.to_dict('records'),
                        'bounds': {
                            'iqr_lower': float(lower_bound),
                            'iqr_upper': float(upper_bound),
                            'z_threshold': 2.5,
                            'modified_z_threshold': 3.5
                        },
                        'outlier_countries': {
                            'z_score': z_outliers['country'].tolist(),
                            'iqr': iqr_outliers['country'].tolist(),
                            'modified_z': modified_outliers['country'].tolist()
                        }
                    }

                    # Create visualization
                    if i < 3:
                        row, col = divmod(i, 2)
                        ax = axes[row, col]

                        # Box plot with outliers highlighted
                        box_data = [values]
                        bp = ax.boxplot(box_data, patch_artist=True, labels=[indicator])
                        bp['boxes'][0].set_facecolor(sns.color_palette()[i])

                        # Highlight extreme outliers
                        if len(z_outliers) > 0:
                            ax.scatter([1] * len(z_outliers), z_outliers[indicator],
                                     color='red', s=100, alpha=0.7, label=f'Z-score outliers ({len(z_outliers)})')

                        title_map = {
                            'life_satisfaction': 'Life Satisfaction Outliers',
                            'life_expectancy': 'Life Expectancy Outliers',
                            'self_harm_rate': 'Self-harm Rate Outliers'
                        }
                        ax.set_title(title_map.get(indicator, indicator), fontweight='bold')
                        ax.set_ylabel('Value')

                        if len(z_outliers) > 0:
                            ax.legend()

        # Remove empty subplot
        axes[1, 1].remove()

        plt.tight_layout()
        self.save_chart(plt, 'outlier_analysis')

        self.results['outliers'] = outlier_results
        return outlier_results

    def correlation_analysis(self):
        """Enhanced correlation analysis with visualizations"""
        print("Performing correlation analysis...")

        # Select numeric indicators
        numeric_cols = ['life_satisfaction', 'life_expectancy', 'self_harm_rate']
        available_cols = [col for col in numeric_cols if col in self.health_df.columns]

        if len(available_cols) < 2:
            return {}

        # Create correlation data
        corr_data = self.health_df[available_cols + ['continent', 'country']].dropna()

        # Calculate correlation matrix
        corr_matrix = corr_data[available_cols].corr()

        # Create correlation visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=axes[0], fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
        axes[0].set_title('Correlation Matrix: Health Indicators', fontweight='bold')

        # Scatter plot for strongest correlation
        if len(available_cols) >= 2:
            # Find strongest correlation
            corr_vals = corr_matrix.values
            np.fill_diagonal(corr_vals, 0)  # Remove diagonal
            max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_vals)), corr_vals.shape)

            var1 = available_cols[max_corr_idx[0]]
            var2 = available_cols[max_corr_idx[1]]
            correlation = corr_matrix.loc[var1, var2]

            # Create scatter plot
            for continent in corr_data['continent'].unique():
                if pd.notna(continent):
                    cont_data = corr_data[corr_data['continent'] == continent]
                    axes[1].scatter(cont_data[var1], cont_data[var2],
                                   label=continent, alpha=0.7, s=60)

            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(corr_data[var1], corr_data[var2])
            x_range = np.linspace(corr_data[var1].min(), corr_data[var1].max(), 100)
            y_pred = slope * x_range + intercept
            axes[1].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                        label=f'R² = {r_value**2:.3f}')

            axes[1].set_xlabel(var1.replace('_', ' ').title())
            axes[1].set_ylabel(var2.replace('_', ' ').title())
            axes[1].set_title(f'Strongest Correlation: {var1.replace("_", " ").title()} vs {var2.replace("_", " ").title()}',
                             fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_chart(plt, 'correlation_analysis')

        # Store detailed correlation results
        correlation_results = {}
        for i, var1 in enumerate(available_cols):
            for j, var2 in enumerate(available_cols):
                if i < j:  # Avoid duplicates
                    subset = corr_data[[var1, var2]].dropna()
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

        self.results['correlations'] = correlation_results
        return correlation_results

    def mortality_analysis(self):
        """Enhanced mortality pattern analysis"""
        print("Performing mortality analysis...")

        # Collect all death causes
        death_causes = []
        cause_columns = ['Most Common Cause of Death (2018)',
                        'Second Most Common Cause of Death (2018)',
                        'Third Most Common Cause of Death (2018)']

        for col in cause_columns:
            if col in self.health_df.columns:
                causes = self.health_df[col].dropna().tolist()
                death_causes.extend(causes)

        if not death_causes:
            return {}

        # Overall analysis
        cause_counts = pd.Series(death_causes).value_counts()

        # By continent analysis
        mortality_by_continent = {}
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

        # Create mortality visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Mortality Patterns Analysis', fontsize=16, fontweight='bold')

        # Global top causes
        top_causes = cause_counts.head(10)
        axes[0, 0].barh(range(len(top_causes)), top_causes.values,
                       color=sns.color_palette("viridis", len(top_causes)))
        axes[0, 0].set_yticks(range(len(top_causes)))
        axes[0, 0].set_yticklabels(top_causes.index, fontsize=10)
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_title('Top 10 Global Causes of Death', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # Cardiovascular disease prevalence by continent
        cardio_by_continent = {}
        for continent, causes in mortality_by_continent.items():
            cardio_count = causes.get('Cardiovascular Disease', 0)
            total_count = causes.sum()
            cardio_by_continent[continent] = (cardio_count / total_count) * 100 if total_count > 0 else 0

        if cardio_by_continent:
            continents = list(cardio_by_continent.keys())
            percentages = list(cardio_by_continent.values())

            axes[0, 1].bar(continents, percentages, color=sns.color_palette("Set2", len(continents)))
            axes[0, 1].set_ylabel('Percentage (%)')
            axes[0, 1].set_title('Cardiovascular Disease Prevalence by Continent', fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Top causes by continent (stacked bar)
        continent_cause_matrix = pd.DataFrame()
        top_global_causes = cause_counts.head(5).index

        for continent, causes in mortality_by_continent.items():
            continent_percentages = []
            total_causes = causes.sum()
            for cause in top_global_causes:
                percentage = (causes.get(cause, 0) / total_causes) * 100 if total_causes > 0 else 0
                continent_percentages.append(percentage)
            continent_cause_matrix[continent] = continent_percentages

        if not continent_cause_matrix.empty:
            continent_cause_matrix.index = top_global_causes
            continent_cause_matrix.plot(kind='bar', stacked=True, ax=axes[1, 0],
                                       colormap='Set3', legend=True)
            axes[1, 0].set_ylabel('Percentage (%)')
            axes[1, 0].set_title('Top 5 Causes Distribution by Continent', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Remove empty subplot and adjust
        axes[1, 1].remove()
        plt.tight_layout()
        self.save_chart(plt, 'mortality_analysis')

        mortality_results = {
            'global_top_causes': cause_counts.head(10).to_dict(),
            'by_continent': {k: v.head(5).to_dict() for k, v in mortality_by_continent.items()},
            'cardiovascular_prevalence': cardio_by_continent,
            'total_records': len(death_causes)
        }

        self.results['mortality'] = mortality_results
        return mortality_results

    def calculate_eta_squared(self, groups, f_stat):
        """Calculate eta-squared (effect size) for ANOVA"""
        try:
            # Calculate total sum of squares and within-group sum of squares
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)

            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)

            # Total sum of squares
            ss_total = sum((x - grand_mean)**2 for x in all_values)

            # Eta-squared
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            return float(eta_squared)
        except:
            return 0.0

    def save_chart(self, plt_obj, filename):
        """Save chart as base64 string for HTML embedding"""
        buffer = BytesIO()
        plt_obj.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt_obj.close()

        self.charts[filename] = chart_base64

        # Also save as file
        output_dir = Path("../analysis-output-files")
        output_dir.mkdir(exist_ok=True)

        # Recreate and save the plot
        # Note: This is a simplified approach - in practice you'd want to save before converting to base64

    def run_complete_analysis(self):
        """Run all analysis components"""
        print("Starting enhanced health and well-being analysis...")

        self.prepare_data()
        self.univariate_analysis()
        self.regional_analysis()
        self.outlier_analysis()
        self.correlation_analysis()
        self.mortality_analysis()

        print("Enhanced analysis complete!")
        return self.results, self.charts

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

def generate_enhanced_html_report(results, charts, output_dir):
    """Generate comprehensive HTML report with embedded visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

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
            .outlier-card {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }}
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
                {generate_executive_summary_section(results)}
            </section>

            <section id="distributions">
                <h2>Health Indicator Distributions</h2>
                {generate_distributions_section(results, charts)}
            </section>

            <section id="regional">
                <h2>Regional Analysis</h2>
                {generate_regional_section(results, charts)}
            </section>

            <section id="correlations">
                <h2>Correlation Analysis</h2>
                {generate_correlations_section(results, charts)}
            </section>

            <section id="outliers">
                <h2>Outlier Analysis</h2>
                {generate_outliers_section(results, charts)}
            </section>

            <section id="mortality">
                <h2>Mortality Patterns</h2>
                {generate_mortality_section(results, charts)}
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
    html_file = output_path / "enhanced_health_analysis.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Enhanced HTML report saved to: {html_file}")
    return html_file

def generate_executive_summary_section(results):
    """Generate executive summary section"""
    summary_html = """
    <div class="stats-grid">
    """

    if 'univariate' in results:
        for indicator, stats in results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()
            summary_html += f"""
                <div class="stat-card">
                    <span class="stat-number">{stats['mean']:.2f}</span>
                    <div class="stat-label">{clean_name} (Mean)</div>
                </div>
            """

    if 'correlations' in results:
        strong_corrs = sum(1 for corr in results['correlations'].values() if abs(corr.get('pearson_r', 0)) > 0.5)
        summary_html += f"""
            <div class="stat-card">
                <span class="stat-number">{strong_corrs}</span>
                <div class="stat-label">Strong Correlations</div>
            </div>
        """

    if 'regional' in results:
        significant_differences = sum(1 for analysis in results['regional'].values()
                                    if analysis.get('significant_difference', False))
        summary_html += f"""
            <div class="stat-card">
                <span class="stat-number">{significant_differences}</span>
                <div class="stat-label">Significant Regional Differences</div>
            </div>
        """

    summary_html += "</div>"
    return summary_html

def generate_distributions_section(results, charts):
    """Generate distributions section"""
    section_html = """
    <p>The distribution analysis examines the shape, central tendency, and spread of each health indicator across all countries.</p>
    """

    if 'health_distributions' in charts:
        section_html += f"""
        <div class="chart-container">
            <div class="chart-title">Health Indicator Distributions</div>
            <img src="data:image/png;base64,{charts['health_distributions']}" alt="Health Distributions">
        </div>
        """

    if 'univariate' in results:
        section_html += """
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

        for indicator, stats in results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()
            is_normal = "Yes" if stats.get('is_normal', False) else "No"
            normal_class = "success" if stats.get('is_normal', False) else "alert"

            section_html += f"""
                <tr class="{normal_class}">
                    <td>{clean_name}</td>
                    <td>{stats['count']}</td>
                    <td>{stats['mean']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                    <td>{stats['skewness']:.3f}</td>
                    <td>{is_normal}</td>
                </tr>
            """

        section_html += """
            </tbody>
        </table>
        """

    return section_html

def generate_regional_section(results, charts):
    """Generate regional analysis section"""
    section_html = """
    <p>Regional analysis compares health indicators across continents using ANOVA tests to identify significant differences.</p>
    """

    if 'regional_analysis' in charts:
        section_html += f"""
        <div class="chart-container">
            <div class="chart-title">Health Indicators by Continent</div>
            <img src="data:image/png;base64,{charts['regional_analysis']}" alt="Regional Analysis">
        </div>
        """

    if 'regional' in results:
        section_html += "<h3>ANOVA Results</h3>"

        for indicator, analysis in results['regional'].items():
            clean_name = indicator.replace('_', ' ').title()
            anova_p = analysis.get('anova_p', 1)
            significant = analysis.get('significant_difference', False)
            effect_size = analysis.get('effect_size', 0)

            status_class = "success" if significant else "alert"
            status_text = "Significant" if significant else "Not Significant"

            section_html += f"""
            <div class="{status_class}">
                <h4>{clean_name}</h4>
                <p><strong>ANOVA Result:</strong> {status_text} (p = {anova_p:.4f})</p>
                <p><strong>Effect Size (η²):</strong> {effect_size:.3f}</p>
            </div>
            """

    return section_html

def generate_correlations_section(results, charts):
    """Generate correlations section"""
    section_html = """
    <p>Correlation analysis examines the relationships between different health indicators.</p>
    """

    if 'correlation_analysis' in charts:
        section_html += f"""
        <div class="chart-container">
            <div class="chart-title">Correlation Analysis</div>
            <img src="data:image/png;base64,{charts['correlation_analysis']}" alt="Correlation Analysis">
        </div>
        """

    if 'correlations' in results:
        section_html += """
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

        for relationship, stats in results['correlations'].items():
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

            section_html += f"""
                <tr class="{corr_class}">
                    <td>{clean_rel}</td>
                    <td>{r_value:.3f}</td>
                    <td>{r_squared:.3f}</td>
                    <td>{p_value:.4f}</td>
                    <td>{significance}</td>
                    <td>{strength}</td>
                </tr>
            """

        section_html += """
            </tbody>
        </table>
        """

    return section_html

def generate_outliers_section(results, charts):
    """Generate outliers section"""
    section_html = """
    <p>Outlier analysis identifies countries with unusual health indicator values using multiple statistical methods.</p>
    """

    if 'outlier_analysis' in charts:
        section_html += f"""
        <div class="chart-container">
            <div class="chart-title">Outlier Detection</div>
            <img src="data:image/png;base64,{charts['outlier_analysis']}" alt="Outlier Analysis">
        </div>
        """

    if 'outliers' in results:
        section_html += "<h3>Identified Outlier Countries</h3>"

        for indicator, outlier_data in results['outliers'].items():
            clean_name = indicator.replace('_', ' ').title()

            section_html += f"<h4>{clean_name}</h4>"

            # Z-score outliers
            z_countries = outlier_data.get('outlier_countries', {}).get('z_score', [])
            if z_countries:
                section_html += f"""
                <div class="outlier-card">
                    <strong>Z-score Outliers (|z| > 2.5):</strong> {', '.join(z_countries)}
                </div>
                """

            # IQR outliers
            iqr_countries = outlier_data.get('outlier_countries', {}).get('iqr', [])
            if iqr_countries:
                section_html += f"""
                <div class="outlier-card">
                    <strong>IQR Outliers:</strong> {', '.join(iqr_countries)}
                </div>
                """

    return section_html

def generate_mortality_section(results, charts):
    """Generate mortality section"""
    section_html = """
    <p>Mortality analysis examines the leading causes of death globally and by continent.</p>
    """

    if 'mortality_analysis' in charts:
        section_html += f"""
        <div class="chart-container">
            <div class="chart-title">Mortality Patterns</div>
            <img src="data:image/png;base64,{charts['mortality_analysis']}" alt="Mortality Analysis">
        </div>
        """

    if 'mortality' in results:
        global_causes = results['mortality'].get('global_top_causes', {})
        if global_causes:
            section_html += """
            <h3>Global Leading Causes of Death</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Cause of Death</th>
                        <th>Frequency</th>
                    </tr>
                </thead>
                <tbody>
            """

            for rank, (cause, frequency) in enumerate(global_causes.items(), 1):
                section_html += f"""
                    <tr>
                        <td>{rank}</td>
                        <td>{cause}</td>
                        <td>{frequency}</td>
                    </tr>
                """

            section_html += """
                </tbody>
            </table>
            """

    return section_html

def main():
    print("=" * 80)
    print("ENHANCED GLOBAL THRIVING ANALYSIS: HEALTH & WELL-BEING")
    print("=" * 80)

    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False

    # Run enhanced analysis
    analyzer = EnhancedHealthAnalyzer(df)
    results, charts = analyzer.run_complete_analysis()

    # Generate enhanced HTML report
    output_dir = "../analysis-output-files"
    html_file = generate_enhanced_html_report(results, charts, output_dir)

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
            z_outliers = outlier_data.get('outlier_countries', {}).get('z_score', [])
            if z_outliers:
                clean_name = indicator.replace('_', ' ').title()
                print(f"   • {clean_name}: {', '.join(z_outliers[:3])}{'...' if len(z_outliers) > 3 else ''}")

    print(f"\n🌐 Enhanced report available at:")
    print(f"    {html_file.absolute()}")

    return True

if __name__ == "__main__":
    main()