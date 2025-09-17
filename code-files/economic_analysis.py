"""
Economic Analysis Module for Global Thriving Study

This module performs comprehensive analysis of economic indicators including
GDP per capita, unemployment rates, and their relationships with well-being
and development indicators across countries and regions.

Research Questions:
- How do GDP per capita levels differ across countries and regions?
- What is the relationship between unemployment and life satisfaction?
- Do wealthier countries show higher thriving on other dimensions (education, health)?
- Can we identify distinct economic clusters of countries?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class EconomicAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.output_dir = Path("../analysis-output-files")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_economic_data(self):
        """Prepare and clean economic data"""
        print("Preparing economic data...")

        # Create working copy
        self.economic_df = self.df.copy()

        # Define economic and related indicators
        numeric_cols = {
            'GDP per Capita (2018)': 'gdp_per_capita',
            'Unemployment Rate (2018)': 'unemployment_rate',
            'Total population (2018)': 'population',
            'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction',
            'Life expectancy at birth (years)(2018)': 'life_expectancy',
            'Learning-Adjusted Years of School (2020)': 'education_years'
        }

        for orig_col, new_col in numeric_cols.items():
            if orig_col in self.economic_df.columns:
                # Handle formatting issues for GDP and population
                if orig_col in ['GDP per Capita (2018)', 'Total population (2018)']:
                    self.economic_df[new_col] = (
                        self.economic_df[orig_col]
                        .astype(str)
                        .str.replace('"', '')
                        .str.replace(',', '')
                        .str.strip()
                    )
                else:
                    self.economic_df[new_col] = self.economic_df[orig_col]

                # Convert to numeric
                self.economic_df[new_col] = pd.to_numeric(self.economic_df[new_col], errors='coerce')

        # Clean location data
        self.economic_df['continent'] = self.economic_df['Continent']
        self.economic_df['country'] = self.economic_df['Country']

        # Create economic development categories
        self.categorize_economic_development()

        # Log transform GDP for better visualization
        self.economic_df['log_gdp'] = np.log10(self.economic_df['gdp_per_capita'].replace(0, np.nan))

        print(f"Economic data prepared: {len(self.economic_df)} countries")
        return self.economic_df

    def categorize_economic_development(self):
        """Categorize countries by economic development level"""
        gdp_data = self.economic_df['gdp_per_capita'].dropna()

        if len(gdp_data) > 0:
            # World Bank income classifications (approximate)
            low_income_threshold = 1045
            lower_middle_threshold = 4095
            upper_middle_threshold = 12695

            def categorize_income(gdp):
                if pd.isna(gdp):
                    return 'Unknown'
                elif gdp <= low_income_threshold:
                    return 'Low Income'
                elif gdp <= lower_middle_threshold:
                    return 'Lower Middle Income'
                elif gdp <= upper_middle_threshold:
                    return 'Upper Middle Income'
                else:
                    return 'High Income'

            self.economic_df['income_category'] = self.economic_df['gdp_per_capita'].apply(categorize_income)

            # Alternative quintile-based categorization
            quintiles = gdp_data.quantile([0.2, 0.4, 0.6, 0.8])

            def categorize_quintile(gdp):
                if pd.isna(gdp):
                    return 'Unknown'
                elif gdp <= quintiles[0.2]:
                    return 'Bottom 20%'
                elif gdp <= quintiles[0.4]:
                    return 'Second 20%'
                elif gdp <= quintiles[0.6]:
                    return 'Middle 20%'
                elif gdp <= quintiles[0.8]:
                    return 'Fourth 20%'
                else:
                    return 'Top 20%'

            self.economic_df['wealth_quintile'] = self.economic_df['gdp_per_capita'].apply(categorize_quintile)

    def univariate_economic_analysis(self):
        """Perform univariate analysis of economic indicators"""
        print("Performing univariate economic analysis...")

        indicators = ['gdp_per_capita', 'unemployment_rate', 'population']
        univariate_results = {}

        for indicator in indicators:
            if indicator in self.economic_df.columns:
                clean_data = self.economic_df[indicator].dropna()

                if len(clean_data) > 0:
                    # Basic statistics
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

                    # Economic-specific metrics
                    if indicator == 'gdp_per_capita':
                        # Gini-like inequality measure
                        sorted_gdp = np.sort(clean_data)
                        n = len(sorted_gdp)
                        cumsum = np.cumsum(sorted_gdp)
                        gini_approx = (2 * np.sum((np.arange(1, n+1) * sorted_gdp))) / (n * cumsum[-1]) - (n + 1) / n
                        stats_dict['gini_approximation'] = float(gini_approx)

                        # Income category distribution
                        category_counts = self.economic_df['income_category'].value_counts()
                        stats_dict['income_distribution'] = category_counts.to_dict()

                    univariate_results[indicator] = stats_dict
                    print(f"{indicator}: Mean = {stats_dict['mean']:.2f}, Count = {stats_dict['count']}")

        self.results['univariate'] = univariate_results
        return univariate_results

    def regional_economic_analysis(self):
        """Analyze economic indicators by continent"""
        print("Performing regional economic analysis...")

        indicators = ['gdp_per_capita', 'unemployment_rate']
        regional_results = {}

        for indicator in indicators:
            if indicator in self.economic_df.columns:
                # Prepare continent groups
                continent_groups = []
                continent_names = []

                for continent in self.economic_df['continent'].unique():
                    if pd.notna(continent):
                        group_data = self.economic_df[
                            (self.economic_df['continent'] == continent) &
                            (self.economic_df[indicator].notna())
                        ][indicator].values

                        if len(group_data) > 0:
                            continent_groups.append(group_data)
                            continent_names.append(continent)

                # ANOVA analysis
                if len(continent_groups) > 1:
                    try:
                        anova_stat, anova_p = f_oneway(*continent_groups)

                        # Effect size (eta-squared)
                        all_values = np.concatenate(continent_groups)
                        grand_mean = np.mean(all_values)
                        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in continent_groups)
                        ss_total = sum((x - grand_mean)**2 for x in all_values)
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0

                        # Regional statistics
                        regional_stats = self.economic_df.groupby('continent')[indicator].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(3)

                        # Pairwise comparisons
                        pairwise_results = {}
                        for i in range(len(continent_names)):
                            for j in range(i+1, len(continent_names)):
                                t_stat, t_p = stats.ttest_ind(continent_groups[i], continent_groups[j])
                                pairwise_results[f"{continent_names[i]}_vs_{continent_names[j]}"] = {
                                    't_stat': float(t_stat),
                                    'p_value': float(t_p),
                                    'significant': t_p < 0.05,
                                    'mean_diff': float(np.mean(continent_groups[i]) - np.mean(continent_groups[j]))
                                }

                        regional_results[indicator] = {
                            'anova_stat': float(anova_stat),
                            'anova_p': float(anova_p),
                            'significant_difference': anova_p < 0.05,
                            'eta_squared': float(eta_squared),
                            'regional_stats': regional_stats.to_dict('index'),
                            'pairwise_comparisons': pairwise_results
                        }

                    except Exception as e:
                        print(f"ANOVA failed for {indicator}: {e}")

        self.results['regional'] = regional_results
        return regional_results

    def economic_correlations(self):
        """Analyze correlations between economic and well-being indicators"""
        print("Performing economic correlation analysis...")

        # Core variables for correlation analysis
        correlation_vars = ['gdp_per_capita', 'unemployment_rate', 'population',
                          'life_satisfaction', 'life_expectancy', 'education_years']

        available_vars = [var for var in correlation_vars if var in self.economic_df.columns]

        if len(available_vars) < 3:
            return {}

        # Create correlation subset
        corr_data = self.economic_df[available_vars + ['continent', 'country']].dropna()

        if len(corr_data) < 10:
            return {}

        # Correlation matrix
        corr_matrix = corr_data[available_vars].corr()

        # Detailed economic correlations
        correlation_results = {}
        economic_vars = ['gdp_per_capita', 'unemployment_rate']

        for econ_var in economic_vars:
            if econ_var in available_vars:
                for other_var in available_vars:
                    if other_var != econ_var and other_var not in ['population']:  # Skip population for clarity
                        subset = corr_data[[econ_var, other_var]].dropna()
                        if len(subset) > 3:
                            pearson_r, pearson_p = stats.pearsonr(subset[econ_var], subset[other_var])
                            spearman_r, spearman_p = stats.spearmanr(subset[econ_var], subset[other_var])

                            # Linear regression
                            slope, intercept, r_value, p_value, std_err = stats.linregress(subset[econ_var], subset[other_var])

                            correlation_results[f"{econ_var}_vs_{other_var}"] = {
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

        # Store results
        correlation_results['correlation_matrix'] = corr_matrix.to_dict()
        correlation_results['available_variables'] = available_vars

        self.results['correlations'] = correlation_results
        return correlation_results

    def economic_clustering(self):
        """Perform cluster analysis to identify economic country groups"""
        print("Performing economic clustering analysis...")

        # Select variables for clustering
        cluster_vars = ['gdp_per_capita', 'unemployment_rate']

        # Include well-being indicators if available
        if 'life_satisfaction' in self.economic_df.columns:
            cluster_vars.append('life_satisfaction')
        if 'education_years' in self.economic_df.columns:
            cluster_vars.append('education_years')

        # Create clustering dataset
        cluster_data = self.economic_df[cluster_vars + ['country', 'continent']].dropna()

        if len(cluster_data) < 10:
            return {}

        # Prepare data for clustering
        X = cluster_data[cluster_vars].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, min(8, len(cluster_data)//3))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Choose optimal k (simple elbow detection)
        if len(inertias) >= 3:
            # Find elbow point
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            optimal_k = K_range[np.argmax(second_diffs) + 1] if len(second_diffs) > 0 else 3
        else:
            optimal_k = 3

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to data
        cluster_data['cluster'] = cluster_labels

        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_countries = cluster_data[cluster_data['cluster'] == cluster_id]

            cluster_stats = {}
            for var in cluster_vars:
                cluster_stats[var] = {
                    'mean': float(cluster_countries[var].mean()),
                    'std': float(cluster_countries[var].std()),
                    'count': len(cluster_countries)
                }

            cluster_analysis[f'cluster_{cluster_id}'] = {
                'countries': cluster_countries['country'].tolist(),
                'statistics': cluster_stats,
                'continent_distribution': cluster_countries['continent'].value_counts().to_dict()
            }

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        clustering_results = {
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_analysis': cluster_analysis,
            'cluster_data': cluster_data,
            'pca_components': X_pca.tolist(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'cluster_variables': cluster_vars,
            'inertias': inertias,
            'k_range': list(K_range)
        }

        self.results['clustering'] = clustering_results
        return clustering_results

    def economic_outliers(self):
        """Identify countries with unusual economic patterns"""
        print("Identifying economic outliers...")

        indicators = ['gdp_per_capita', 'unemployment_rate']
        outlier_results = {}

        for indicator in indicators:
            if indicator in self.economic_df.columns:
                clean_data = self.economic_df[['country', 'continent', indicator]].dropna()

                if len(clean_data) < 10:
                    continue

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

                # Categorize outliers by type
                if indicator == 'gdp_per_capita':
                    high_performers = z_outliers[z_outliers[indicator] > values.mean()]
                    low_performers = z_outliers[z_outliers[indicator] < values.mean()]
                elif indicator == 'unemployment_rate':
                    # For unemployment, low is good, high is concerning
                    low_unemployment = z_outliers[z_outliers[indicator] < values.mean()]
                    high_unemployment = z_outliers[z_outliers[indicator] > values.mean()]
                    high_performers = low_unemployment
                    low_performers = high_unemployment

                outlier_results[indicator] = {
                    'z_score_outliers': z_outliers.to_dict('records'),
                    'iqr_outliers': iqr_outliers.to_dict('records'),
                    'high_performers': high_performers[['country', indicator]].to_dict('records') if len(high_performers) > 0 else [],
                    'low_performers': low_performers[['country', indicator]].to_dict('records') if len(low_performers) > 0 else [],
                    'outlier_countries': {
                        'z_score': z_outliers['country'].tolist(),
                        'iqr': iqr_outliers['country'].tolist(),
                        'high_performance': high_performers['country'].tolist() if len(high_performers) > 0 else [],
                        'low_performance': low_performers['country'].tolist() if len(low_performers) > 0 else []
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

    def create_economic_visualizations(self):
        """Create comprehensive economic visualizations"""
        print("Creating economic visualizations...")

        # 1. Economic Overview: Distributions and Comparisons
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Economic Analysis: GDP per Capita and Unemployment', fontsize=16, fontweight='bold')

        # GDP distribution
        gdp_data = self.economic_df['gdp_per_capita'].dropna()
        if len(gdp_data) > 0:
            axes[0, 0].hist(gdp_data, bins=25, alpha=0.7, color='green', edgecolor='black')
            axes[0, 0].axvline(gdp_data.mean(), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: ${gdp_data.mean():.0f}')
            axes[0, 0].axvline(gdp_data.median(), color='orange', linestyle=':', linewidth=2,
                              label=f'Median: ${gdp_data.median():.0f}')
            axes[0, 0].set_title('GDP per Capita Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('GDP per Capita (USD)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Log GDP distribution (for better visualization)
        log_gdp_data = self.economic_df['log_gdp'].dropna()
        if len(log_gdp_data) > 0:
            axes[0, 1].hist(log_gdp_data, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
            axes[0, 1].axvline(log_gdp_data.mean(), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {log_gdp_data.mean():.2f}')
            axes[0, 1].set_title('Log GDP per Capita Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('Log10(GDP per Capita)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Unemployment distribution
        unemp_data = self.economic_df['unemployment_rate'].dropna()
        if len(unemp_data) > 0:
            axes[0, 2].hist(unemp_data, bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[0, 2].axvline(unemp_data.mean(), color='darkred', linestyle='--', linewidth=2,
                              label=f'Mean: {unemp_data.mean():.1f}%')
            axes[0, 2].axvline(unemp_data.median(), color='orange', linestyle=':', linewidth=2,
                              label=f'Median: {unemp_data.median():.1f}%')
            axes[0, 2].set_title('Unemployment Rate Distribution', fontweight='bold')
            axes[0, 2].set_xlabel('Unemployment Rate (%)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # GDP by continent
        gdp_continent_data = self.economic_df[['continent', 'gdp_per_capita']].dropna()
        if len(gdp_continent_data) > 0:
            sns.boxplot(data=gdp_continent_data, x='continent', y='gdp_per_capita', ax=axes[1, 0],
                       palette='Set2', showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red"})
            axes[1, 0].set_title('GDP per Capita by Continent', fontweight='bold')
            axes[1, 0].set_xlabel('Continent')
            axes[1, 0].set_ylabel('GDP per Capita (USD)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Unemployment by continent
        unemp_continent_data = self.economic_df[['continent', 'unemployment_rate']].dropna()
        if len(unemp_continent_data) > 0:
            sns.boxplot(data=unemp_continent_data, x='continent', y='unemployment_rate', ax=axes[1, 1],
                       palette='Set3', showmeans=True, meanprops={"marker": "o", "markerfacecolor": "red"})
            axes[1, 1].set_title('Unemployment Rate by Continent', fontweight='bold')
            axes[1, 1].set_xlabel('Continent')
            axes[1, 1].set_ylabel('Unemployment Rate (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Income category distribution
        if 'income_category' in self.economic_df.columns:
            income_counts = self.economic_df['income_category'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            wedges, texts, autotexts = axes[1, 2].pie(income_counts.values, labels=income_counts.index,
                                                     autopct='%1.1f%%', colors=colors[:len(income_counts)])
            axes[1, 2].set_title('Income Category Distribution', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'economic_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Economic Correlations and Relationships
        if 'correlations' in self.results and 'available_variables' in self.results['correlations']:
            available_vars = self.results['correlations']['available_variables']

            if len(available_vars) >= 4:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Economic Correlations with Well-being Indicators', fontsize=16, fontweight='bold')

                # Correlation heatmap
                corr_matrix = pd.DataFrame(self.results['correlations']['correlation_matrix'])
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                           square=True, ax=axes[0, 0], fmt='.3f')
                axes[0, 0].set_title('Economic Correlation Matrix', fontweight='bold')

                # GDP vs Life Satisfaction
                if 'life_satisfaction' in available_vars:
                    scatter_data = self.economic_df[['gdp_per_capita', 'life_satisfaction', 'continent']].dropna()

                    for continent in scatter_data['continent'].unique():
                        if pd.notna(continent):
                            cont_data = scatter_data[scatter_data['continent'] == continent]
                            axes[0, 1].scatter(cont_data['gdp_per_capita'], cont_data['life_satisfaction'],
                                             label=continent, alpha=0.7, s=60)

                    # Add regression line
                    if len(scatter_data) > 3:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            scatter_data['gdp_per_capita'], scatter_data['life_satisfaction'])
                        x_range = np.linspace(scatter_data['gdp_per_capita'].min(),
                                             scatter_data['gdp_per_capita'].max(), 100)
                        y_pred = slope * x_range + intercept
                        axes[0, 1].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                                       label=f'R² = {r_value**2:.3f}')

                    axes[0, 1].set_xlabel('GDP per Capita (USD)')
                    axes[0, 1].set_ylabel('Life Satisfaction')
                    axes[0, 1].set_title('Economic Development vs Life Satisfaction', fontweight='bold')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)

                # Unemployment vs Life Satisfaction
                if 'life_satisfaction' in available_vars:
                    scatter_data = self.economic_df[['unemployment_rate', 'life_satisfaction', 'continent']].dropna()

                    for continent in scatter_data['continent'].unique():
                        if pd.notna(continent):
                            cont_data = scatter_data[scatter_data['continent'] == continent]
                            axes[1, 0].scatter(cont_data['unemployment_rate'], cont_data['life_satisfaction'],
                                             label=continent, alpha=0.7, s=60)

                    # Add regression line
                    if len(scatter_data) > 3:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            scatter_data['unemployment_rate'], scatter_data['life_satisfaction'])
                        x_range = np.linspace(scatter_data['unemployment_rate'].min(),
                                             scatter_data['unemployment_rate'].max(), 100)
                        y_pred = slope * x_range + intercept
                        axes[1, 0].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                                       label=f'R² = {r_value**2:.3f}')

                    axes[1, 0].set_xlabel('Unemployment Rate (%)')
                    axes[1, 0].set_ylabel('Life Satisfaction')
                    axes[1, 0].set_title('Unemployment vs Life Satisfaction', fontweight='bold')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)

                # GDP vs Education
                if 'education_years' in available_vars:
                    scatter_data = self.economic_df[['gdp_per_capita', 'education_years', 'continent']].dropna()

                    for continent in scatter_data['continent'].unique():
                        if pd.notna(continent):
                            cont_data = scatter_data[scatter_data['continent'] == continent]
                            axes[1, 1].scatter(cont_data['gdp_per_capita'], cont_data['education_years'],
                                             label=continent, alpha=0.7, s=60)

                    # Add regression line
                    if len(scatter_data) > 3:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            scatter_data['gdp_per_capita'], scatter_data['education_years'])
                        x_range = np.linspace(scatter_data['gdp_per_capita'].min(),
                                             scatter_data['gdp_per_capita'].max(), 100)
                        y_pred = slope * x_range + intercept
                        axes[1, 1].plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                                       label=f'R² = {r_value**2:.3f}')

                    axes[1, 1].set_xlabel('GDP per Capita (USD)')
                    axes[1, 1].set_ylabel('Learning-Adjusted Years of School')
                    axes[1, 1].set_title('Economic Development vs Education', fontweight='bold')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(self.output_dir / 'economic_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()

        # 3. Economic Clustering and Outliers
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Economic Clustering and Outlier Analysis', fontsize=16, fontweight='bold')

        # Clustering visualization
        if 'clustering' in self.results:
            cluster_data = self.results['clustering']['cluster_data']
            pca_components = np.array(self.results['clustering']['pca_components'])
            cluster_labels = self.results['clustering']['cluster_labels']

            scatter = axes[0, 0].scatter(pca_components[:, 0], pca_components[:, 1],
                                        c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
            axes[0, 0].set_xlabel(f'PC1 ({self.results["clustering"]["pca_explained_variance"][0]*100:.1f}% variance)')
            axes[0, 0].set_ylabel(f'PC2 ({self.results["clustering"]["pca_explained_variance"][1]*100:.1f}% variance)')
            axes[0, 0].set_title('Economic Country Clusters (PCA)', fontweight='bold')
            plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
            axes[0, 0].grid(True, alpha=0.3)

            # Elbow curve
            k_range = self.results['clustering']['k_range']
            inertias = self.results['clustering']['inertias']
            axes[0, 1].plot(k_range, inertias, 'bo-')
            axes[0, 1].axvline(self.results['clustering']['optimal_k'], color='red', linestyle='--',
                              label=f'Optimal k = {self.results["clustering"]["optimal_k"]}')
            axes[0, 1].set_xlabel('Number of Clusters (k)')
            axes[0, 1].set_ylabel('Inertia')
            axes[0, 1].set_title('Elbow Method for Optimal k', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # GDP outliers
        if 'outliers' in self.results and 'gdp_per_capita' in self.results['outliers']:
            gdp_data = self.economic_df[['country', 'gdp_per_capita']].dropna()
            gdp_outliers = pd.DataFrame(self.results['outliers']['gdp_per_capita']['z_score_outliers'])

            bp = axes[1, 0].boxplot([gdp_data['gdp_per_capita']], patch_artist=True,
                                   labels=['GDP per Capita'])
            bp['boxes'][0].set_facecolor('lightgreen')

            if len(gdp_outliers) > 0:
                axes[1, 0].scatter([1] * len(gdp_outliers), gdp_outliers['gdp_per_capita'],
                                  color='red', s=100, alpha=0.7, zorder=10,
                                  label=f'Outliers ({len(gdp_outliers)})')

                # Annotate top outliers
                sorted_outliers = gdp_outliers.sort_values('gdp_per_capita', ascending=False)
                for i, (idx, row) in enumerate(sorted_outliers.head(3).iterrows()):
                    axes[1, 0].annotate(row['country'],
                                       (1, row['gdp_per_capita']),
                                       xytext=(1.1, row['gdp_per_capita']),
                                       fontsize=8, alpha=0.8)

            axes[1, 0].set_title('GDP per Capita Outliers', fontweight='bold')
            axes[1, 0].set_ylabel('GDP per Capita (USD)')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            if len(gdp_outliers) > 0:
                axes[1, 0].legend()

        # Unemployment outliers
        if 'outliers' in self.results and 'unemployment_rate' in self.results['outliers']:
            unemp_data = self.economic_df[['country', 'unemployment_rate']].dropna()
            unemp_outliers = pd.DataFrame(self.results['outliers']['unemployment_rate']['z_score_outliers'])

            bp = axes[1, 1].boxplot([unemp_data['unemployment_rate']], patch_artist=True,
                                   labels=['Unemployment Rate'])
            bp['boxes'][0].set_facecolor('lightcoral')

            if len(unemp_outliers) > 0:
                axes[1, 1].scatter([1] * len(unemp_outliers), unemp_outliers['unemployment_rate'],
                                  color='darkred', s=100, alpha=0.7, zorder=10,
                                  label=f'Outliers ({len(unemp_outliers)})')

                # Annotate outliers
                for i, (idx, row) in enumerate(unemp_outliers.iterrows()):
                    if i < 5:  # Limit annotations
                        axes[1, 1].annotate(row['country'],
                                           (1, row['unemployment_rate']),
                                           xytext=(1.1, row['unemployment_rate']),
                                           fontsize=8, alpha=0.8)

            axes[1, 1].set_title('Unemployment Rate Outliers', fontweight='bold')
            axes[1, 1].set_ylabel('Unemployment Rate (%)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            if len(unemp_outliers) > 0:
                axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'economic_clustering_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Economic visualizations saved successfully!")

    def run_complete_analysis(self):
        """Run all economic analysis components"""
        print("Starting comprehensive economic analysis...")

        self.prepare_economic_data()
        self.univariate_economic_analysis()
        self.regional_economic_analysis()
        self.economic_correlations()
        self.economic_clustering()
        self.economic_outliers()
        self.create_economic_visualizations()

        print("Economic analysis complete!")
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
    print("GLOBAL THRIVING ANALYSIS: ECONOMIC INDICATORS")
    print("=" * 80)

    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False

    # Run economic analysis
    analyzer = EconomicAnalyzer(df)
    results = analyzer.run_complete_analysis()

    print("\n" + "=" * 80)
    print("ECONOMIC ANALYSIS COMPLETE!")
    print("=" * 80)

    # Display key findings
    if 'univariate' in results:
        print("\n💰 ECONOMIC STATISTICS:")
        if 'gdp_per_capita' in results['univariate']:
            gdp_stats = results['univariate']['gdp_per_capita']
            print(f"   • GDP per Capita:")
            print(f"     - Countries with data: {gdp_stats['count']}")
            print(f"     - Global average: ${gdp_stats['mean']:,.0f} (±${gdp_stats['std']:,.0f})")
            print(f"     - Range: ${gdp_stats['min']:,.0f} - ${gdp_stats['max']:,.0f}")

            if 'income_distribution' in gdp_stats:
                print(f"     - Income categories:")
                for category, count in gdp_stats['income_distribution'].items():
                    print(f"       * {category}: {count} countries")

        if 'unemployment_rate' in results['univariate']:
            unemp_stats = results['univariate']['unemployment_rate']
            print(f"   • Unemployment Rate:")
            print(f"     - Countries with data: {unemp_stats['count']}")
            print(f"     - Global average: {unemp_stats['mean']:.1f}% (±{unemp_stats['std']:.1f}%)")
            print(f"     - Range: {unemp_stats['min']:.1f}% - {unemp_stats['max']:.1f}%")

    if 'correlations' in results:
        print("\n🔗 KEY ECONOMIC CORRELATIONS:")
        for relationship, stats in results['correlations'].items():
            if relationship.startswith(('gdp_per_capita_vs_', 'unemployment_rate_vs_')) and isinstance(stats, dict) and stats.get('significant', False):
                econ_var = relationship.split('_vs_')[0].replace('_', ' ').title()
                other_var = relationship.split('_vs_')[1].replace('_', ' ').title()
                print(f"   • {econ_var} ↔ {other_var}: r = {stats['pearson_r']:.3f} ({stats['correlation_strength']}, p < 0.05)")

    if 'regional' in results:
        print("\n🌍 REGIONAL ECONOMIC DIFFERENCES:")
        for indicator, analysis in results['regional'].items():
            if analysis.get('significant_difference', False):
                clean_name = indicator.replace('_', ' ').title()
                print(f"   • {clean_name}: Significant differences between continents (p = {analysis['anova_p']:.4f})")

    if 'clustering' in results:
        optimal_k = results['clustering']['optimal_k']
        print(f"\n🎯 ECONOMIC CLUSTERING:")
        print(f"   • Optimal number of clusters: {optimal_k}")
        print(f"   • Countries grouped by economic and well-being patterns")

    if 'outliers' in results:
        print("\n📈 ECONOMIC OUTLIERS:")
        for indicator, outlier_data in results['outliers'].items():
            clean_name = indicator.replace('_', ' ').title()
            high_performers = outlier_data.get('outlier_countries', {}).get('high_performance', [])
            low_performers = outlier_data.get('outlier_countries', {}).get('low_performance', [])

            if high_performers:
                print(f"   • {clean_name} - High performers: {', '.join(high_performers[:3])}{'...' if len(high_performers) > 3 else ''}")
            if low_performers:
                print(f"   • {clean_name} - Needs attention: {', '.join(low_performers[:3])}{'...' if len(low_performers) > 3 else ''}")

    print(f"\n📊 Generated visualizations:")
    visualizations = ['economic_overview.png', 'economic_correlations.png', 'economic_clustering_outliers.png']
    for viz in visualizations:
        if (analyzer.output_dir / viz).exists():
            print(f"   ✅ {viz}")

    return True

if __name__ == "__main__":
    main()