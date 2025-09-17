"""
Infrastructure Analysis Module for Global Thriving Study

This module performs comprehensive analysis of infrastructure indicators including
electricity access, telecommunications, transportation, and their relationships 
with development and well-being indicators across countries and regions.

Research Questions:
- How does electricity access vary across countries and regions?
- What is the relationship between infrastructure development and economic prosperity?
- Do countries with better infrastructure show higher thriving on other dimensions?
- Can we identify distinct infrastructure development clusters of countries?
- How does infrastructure access relate to life satisfaction and development outcomes?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class InfrastructureAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.output_dir = Path("analysis-output-files")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_infrastructure_data(self):
        """Prepare and clean infrastructure data"""
        print("Preparing infrastructure data...")

        # Create working copy
        self.infrastructure_df = self.df.copy()

        # Define infrastructure and related indicators
        numeric_cols = {
            ' Number of people with access to electricity (2020) ': 'electricity_access_absolute',
            '% of population with access to electricty (2020)': 'electricity_access_percent',
            ' Total population (2018) ': 'population',
            'GDP per Capita (2018)': 'gdp_per_capita',
            'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction',
            'Life expectancy at birth (years)(2018)': 'life_expectancy',
            'Learning-Adjusted Years of School (2020)': 'education_years',
            'Unemployment Rate (2018)': 'unemployment_rate'
        }

        for orig_col, new_col in numeric_cols.items():
            if orig_col in self.infrastructure_df.columns:
                # Handle formatting issues for specific columns
                if orig_col in [' Number of people with access to electricity (2020) ', ' Total population (2018) ', 'GDP per Capita (2018)']:
                    self.infrastructure_df[new_col] = (
                        self.infrastructure_df[orig_col]
                        .astype(str)
                        .str.replace('"', '')
                        .str.replace(',', '')
                        .str.strip()
                    )
                else:
                    self.infrastructure_df[new_col] = self.infrastructure_df[orig_col]

                # Convert to numeric
                self.infrastructure_df[new_col] = pd.to_numeric(self.infrastructure_df[new_col], errors='coerce')

        # Clean location data
        self.infrastructure_df['continent'] = self.infrastructure_df['Continent']
        self.infrastructure_df['country'] = self.infrastructure_df['Country']

        # Calculate infrastructure development metrics
        self.calculate_infrastructure_metrics()

        # Categorize infrastructure development
        self.categorize_infrastructure_development()

        print(f"Infrastructure data prepared: {len(self.infrastructure_df)} countries")
        return self.infrastructure_df

    def calculate_infrastructure_metrics(self):
        """Calculate additional infrastructure metrics"""
        # Infrastructure gap (people without electricity access)
        self.infrastructure_df['electricity_gap_absolute'] = (
            self.infrastructure_df['population'] - self.infrastructure_df['electricity_access_absolute']
        )
        
        # Infrastructure gap percentage
        self.infrastructure_df['electricity_gap_percent'] = (
            100 - self.infrastructure_df['electricity_access_percent']
        )
        
        # Infrastructure efficiency (access per capita relative to GDP)
        self.infrastructure_df['infrastructure_efficiency'] = (
            self.infrastructure_df['electricity_access_percent'] / 
            (self.infrastructure_df['gdp_per_capita'] / 1000)
        ).replace([np.inf, -np.inf], np.nan)

    def categorize_infrastructure_development(self):
        """Categorize countries by infrastructure development level"""
        electricity_data = self.infrastructure_df['electricity_access_percent'].dropna()

        if len(electricity_data) > 0:
            # Infrastructure development categories based on electricity access
            def categorize_infrastructure(access_rate):
                if pd.isna(access_rate):
                    return 'Unknown'
                elif access_rate >= 99:
                    return 'Universal Access'
                elif access_rate >= 90:
                    return 'High Access'
                elif access_rate >= 70:
                    return 'Medium Access'
                elif access_rate >= 50:
                    return 'Low Access'
                else:
                    return 'Very Low Access'

            self.infrastructure_df['infrastructure_category'] = (
                self.infrastructure_df['electricity_access_percent'].apply(categorize_infrastructure)
            )

    def analyze_regional_infrastructure(self):
        """Analyze infrastructure patterns by region"""
        print("Analyzing regional infrastructure patterns...")
        
        regional_stats = []
        
        for continent in self.infrastructure_df['continent'].unique():
            if pd.isna(continent):
                continue
                
            continent_data = self.infrastructure_df[
                self.infrastructure_df['continent'] == continent
            ]
            
            stats_dict = {
                'continent': continent,
                'countries': len(continent_data),
                'avg_electricity_access': continent_data['electricity_access_percent'].mean(),
                'median_electricity_access': continent_data['electricity_access_percent'].median(),
                'std_electricity_access': continent_data['electricity_access_percent'].std(),
                'min_electricity_access': continent_data['electricity_access_percent'].min(),
                'max_electricity_access': continent_data['electricity_access_percent'].max(),
                'avg_gdp': continent_data['gdp_per_capita'].mean(),
                'avg_life_satisfaction': continent_data['life_satisfaction'].mean(),
                'countries_universal_access': len(continent_data[continent_data['electricity_access_percent'] >= 99])
            }
            
            regional_stats.append(stats_dict)
        
        regional_df = pd.DataFrame(regional_stats)
        
        # Perform ANOVA test
        continent_groups = [
            group['electricity_access_percent'].dropna() 
            for name, group in self.infrastructure_df.groupby('continent')
            if len(group['electricity_access_percent'].dropna()) > 0
        ]
        
        if len(continent_groups) > 1:
            f_stat, p_value = f_oneway(*continent_groups)
            
            print(f"\n🌍 REGIONAL INFRASTRUCTURE ANALYSIS:")
            print(f"   ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("   ✅ Significant regional differences in electricity access")
            else:
                print("   ❌ No significant regional differences in electricity access")
        
        self.results['regional_analysis'] = {
            'statistics': regional_df,
            'anova_f_stat': f_stat if 'f_stat' in locals() else None,
            'anova_p_value': p_value if 'p_value' in locals() else None
        }
        
        return regional_df

    def analyze_infrastructure_correlations(self):
        """Analyze correlations between infrastructure and other indicators"""
        print("Analyzing infrastructure correlations...")
        
        # Define correlation pairs
        correlation_vars = [
            'electricity_access_percent',
            'gdp_per_capita',
            'life_satisfaction',
            'life_expectancy',
            'education_years',
            'unemployment_rate'
        ]
        
        # Calculate correlation matrix
        correlation_data = self.infrastructure_df[correlation_vars].dropna()
        correlation_matrix = correlation_data.corr()
        
        # Find strongest correlations with infrastructure
        infrastructure_correlations = []
        
        for var in correlation_vars:
            if var != 'electricity_access_percent':
                corr_coef = correlation_matrix.loc['electricity_access_percent', var]
                if not pd.isna(corr_coef):
                    # Calculate p-value
                    valid_data = self.infrastructure_df[['electricity_access_percent', var]].dropna()
                    if len(valid_data) > 2:
                        _, p_val = pearsonr(valid_data['electricity_access_percent'], valid_data[var])
                        infrastructure_correlations.append({
                            'variable': var,
                            'correlation': corr_coef,
                            'p_value': p_val,
                            'strength': 'Strong' if abs(corr_coef) > 0.7 else 'Moderate' if abs(corr_coef) > 0.5 else 'Weak'
                        })
        
        # Sort by absolute correlation strength
        infrastructure_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print(f"\n🔗 INFRASTRUCTURE CORRELATIONS:")
        for corr in infrastructure_correlations[:5]:
            direction = "↗️" if corr['correlation'] > 0 else "↘️"
            print(f"   {direction} {corr['variable']}: r = {corr['correlation']:.3f} ({corr['strength']}, p = {corr['p_value']:.3f})")
        
        self.results['correlations'] = {
            'correlation_matrix': correlation_matrix,
            'infrastructure_correlations': infrastructure_correlations
        }
        
        return correlation_matrix, infrastructure_correlations

    def analyze_infrastructure_clusters(self):
        """Perform clustering analysis on infrastructure and development indicators"""
        print("Performing infrastructure clustering analysis...")
        
        # Prepare clustering data
        cluster_vars = [
            'electricity_access_percent',
            'gdp_per_capita',
            'life_satisfaction',
            'life_expectancy',
            'education_years'
        ]
        
        cluster_data = self.infrastructure_df[cluster_vars].dropna()
        
        if len(cluster_data) < 10:
            print("❌ Insufficient data for clustering analysis")
            return None
        
        # Standardize the data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(8, len(cluster_data) // 3))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_data_scaled)
            score = silhouette_score(cluster_data_scaled, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to dataframe
        cluster_results = cluster_data.copy()
        cluster_results['cluster'] = cluster_labels
        cluster_results['country'] = self.infrastructure_df.loc[cluster_data.index, 'country']
        cluster_results['continent'] = self.infrastructure_df.loc[cluster_data.index, 'continent']
        
        # Analyze cluster characteristics
        cluster_summary = []
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_countries = cluster_results[cluster_results['cluster'] == cluster_id]
            
            summary = {
                'cluster_id': cluster_id,
                'size': np.sum(cluster_mask),
                'countries': cluster_countries['country'].tolist(),
                'avg_electricity_access': cluster_countries['electricity_access_percent'].mean(),
                'avg_gdp': cluster_countries['gdp_per_capita'].mean(),
                'avg_life_satisfaction': cluster_countries['life_satisfaction'].mean(),
                'avg_life_expectancy': cluster_countries['life_expectancy'].mean(),
                'avg_education': cluster_countries['education_years'].mean()
            }
            
            cluster_summary.append(summary)
        
        print(f"\n🎯 INFRASTRUCTURE CLUSTERING:")
        print(f"   Optimal clusters: {optimal_k}")
        print(f"   Silhouette score: {max(silhouette_scores):.3f}")
        
        for summary in cluster_summary:
            print(f"   Cluster {summary['cluster_id']}: {summary['size']} countries")
            print(f"      Avg electricity access: {summary['avg_electricity_access']:.1f}%")
            print(f"      Avg GDP per capita: ${summary['avg_gdp']:,.0f}")
        
        self.results['clustering'] = {
            'optimal_k': optimal_k,
            'silhouette_scores': silhouette_scores,
            'cluster_results': cluster_results,
            'cluster_summary': cluster_summary
        }
        
        return cluster_results, cluster_summary

    def analyze_infrastructure_outliers(self):
        """Identify countries with exceptional infrastructure patterns"""
        print("Analyzing infrastructure outliers...")
        
        outliers = []
        
        # High electricity access despite low GDP
        high_access_low_gdp = self.infrastructure_df[
            (self.infrastructure_df['electricity_access_percent'] >= 95) &
            (self.infrastructure_df['gdp_per_capita'] < 5000)
        ]
        
        for _, country in high_access_low_gdp.iterrows():
            outliers.append({
                'country': country['country'],
                'type': 'High Access, Low GDP',
                'electricity_access': country['electricity_access_percent'],
                'gdp_per_capita': country['gdp_per_capita'],
                'description': f"Achieves {country['electricity_access_percent']:.1f}% electricity access with GDP of ${country['gdp_per_capita']:,.0f}"
            })
        
        # Low electricity access despite high GDP
        low_access_high_gdp = self.infrastructure_df[
            (self.infrastructure_df['electricity_access_percent'] < 80) &
            (self.infrastructure_df['gdp_per_capita'] > 10000)
        ]
        
        for _, country in low_access_high_gdp.iterrows():
            outliers.append({
                'country': country['country'],
                'type': 'Low Access, High GDP',
                'electricity_access': country['electricity_access_percent'],
                'gdp_per_capita': country['gdp_per_capita'],
                'description': f"Only {country['electricity_access_percent']:.1f}% electricity access despite GDP of ${country['gdp_per_capita']:,.0f}"
            })
        
        # Infrastructure efficiency outliers
        efficiency_data = self.infrastructure_df['infrastructure_efficiency'].dropna()
        if len(efficiency_data) > 0:
            efficiency_threshold = efficiency_data.quantile(0.95)
            high_efficiency = self.infrastructure_df[
                self.infrastructure_df['infrastructure_efficiency'] >= efficiency_threshold
            ]
            
            for _, country in high_efficiency.iterrows():
                outliers.append({
                    'country': country['country'],
                    'type': 'High Infrastructure Efficiency',
                    'electricity_access': country['electricity_access_percent'],
                    'gdp_per_capita': country['gdp_per_capita'],
                    'efficiency': country['infrastructure_efficiency'],
                    'description': f"High infrastructure efficiency: {country['infrastructure_efficiency']:.1f}"
                })
        
        print(f"\n🎯 INFRASTRUCTURE OUTLIERS:")
        for outlier in outliers[:10]:  # Show top 10
            print(f"   • {outlier['country']} ({outlier['type']}): {outlier['description']}")
        
        self.results['outliers'] = outliers
        return outliers

    def create_infrastructure_visualizations(self):
        """Create comprehensive infrastructure visualizations"""
        print("Creating infrastructure visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Regional Infrastructure Overview
        plt.subplot(4, 3, 1)
        regional_data = self.results['regional_analysis']['statistics']
        bars = plt.bar(regional_data['continent'], regional_data['avg_electricity_access'], 
                      color=sns.color_palette("husl", len(regional_data)))
        plt.title('Average Electricity Access by Region', fontsize=14, fontweight='bold')
        plt.xlabel('Continent')
        plt.ylabel('Electricity Access (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Infrastructure vs GDP Scatter
        plt.subplot(4, 3, 2)
        continents = self.infrastructure_df['continent'].unique()
        colors = sns.color_palette("husl", len(continents))
        
        for i, continent in enumerate(continents):
            if pd.notna(continent):
                continent_data = self.infrastructure_df[self.infrastructure_df['continent'] == continent]
                plt.scatter(continent_data['gdp_per_capita'], continent_data['electricity_access_percent'],
                           alpha=0.7, label=continent, color=colors[i], s=60)
        
        plt.xlabel('GDP per Capita (USD)')
        plt.ylabel('Electricity Access (%)')
        plt.title('Infrastructure vs Economic Development', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. Infrastructure Distribution
        plt.subplot(4, 3, 3)
        electricity_data = self.infrastructure_df['electricity_access_percent'].dropna()
        plt.hist(electricity_data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axvline(electricity_data.mean(), color='red', linestyle='--', 
                   label=f'Mean: {electricity_data.mean():.1f}%')
        plt.axvline(electricity_data.median(), color='orange', linestyle='--', 
                   label=f'Median: {electricity_data.median():.1f}%')
        plt.xlabel('Electricity Access (%)')
        plt.ylabel('Number of Countries')
        plt.title('Distribution of Electricity Access', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Infrastructure Categories
        plt.subplot(4, 3, 4)
        category_counts = self.infrastructure_df['infrastructure_category'].value_counts()
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#DC143C']
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Countries by Infrastructure Development Level', fontsize=14, fontweight='bold')
        
        # 5. Correlation Heatmap
        plt.subplot(4, 3, 5)
        correlation_matrix = self.results['correlations']['correlation_matrix']
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Infrastructure & Development Correlations', fontsize=14, fontweight='bold')
        
        # 6. Top/Bottom Performers
        plt.subplot(4, 3, 6)
        sorted_countries = self.infrastructure_df.sort_values('electricity_access_percent', ascending=False)
        top_10 = sorted_countries.head(10)
        bottom_10 = sorted_countries.tail(10)
        
        # Plot top 10
        y_pos = np.arange(len(top_10))
        plt.barh(y_pos, top_10['electricity_access_percent'], color='green', alpha=0.7)
        plt.yticks(y_pos, top_10['country'])
        plt.xlabel('Electricity Access (%)')
        plt.title('Top 10 Countries - Electricity Access', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 7. Infrastructure Gap Analysis
        plt.subplot(4, 3, 7)
        gap_data = self.infrastructure_df['electricity_gap_percent'].dropna()
        continent_gaps = self.infrastructure_df.groupby('continent')['electricity_gap_percent'].mean().sort_values(ascending=False)
        
        bars = plt.bar(range(len(continent_gaps)), continent_gaps.values, 
                      color=sns.color_palette("Reds_r", len(continent_gaps)))
        plt.xticks(range(len(continent_gaps)), continent_gaps.index, rotation=45)
        plt.ylabel('Average Electricity Gap (%)')
        plt.title('Infrastructure Gap by Region', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 8. Clustering Results
        plt.subplot(4, 3, 8)
        if 'clustering' in self.results:
            cluster_data = self.results['clustering']['cluster_results']
            scatter = plt.scatter(cluster_data['gdp_per_capita'], cluster_data['electricity_access_percent'],
                                c=cluster_data['cluster'], cmap='viridis', alpha=0.7, s=60)
            plt.xlabel('GDP per Capita (USD)')
            plt.ylabel('Electricity Access (%)')
            plt.title('Infrastructure Development Clusters', fontsize=14, fontweight='bold')
            plt.xscale('log')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
        
        # 9. Infrastructure vs Life Satisfaction
        plt.subplot(4, 3, 9)
        valid_data = self.infrastructure_df[['electricity_access_percent', 'life_satisfaction']].dropna()
        plt.scatter(valid_data['electricity_access_percent'], valid_data['life_satisfaction'],
                   alpha=0.6, color='purple', s=60)
        
        # Add trend line
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['electricity_access_percent'], valid_data['life_satisfaction'], 1)
            p = np.poly1d(z)
            plt.plot(valid_data['electricity_access_percent'], 
                    p(valid_data['electricity_access_percent']), "r--", alpha=0.8)
        
        plt.xlabel('Electricity Access (%)')
        plt.ylabel('Life Satisfaction (Cantril Ladder)')
        plt.title('Infrastructure vs Life Satisfaction', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 10. Bottom Performers Detail
        plt.subplot(4, 3, 10)
        y_pos = np.arange(len(bottom_10))
        bars = plt.barh(y_pos, bottom_10['electricity_access_percent'], color='red', alpha=0.7)
        plt.yticks(y_pos, bottom_10['country'])
        plt.xlabel('Electricity Access (%)')
        plt.title('Bottom 10 Countries - Electricity Access', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 11. Infrastructure Efficiency
        plt.subplot(4, 3, 11)
        efficiency_data = self.infrastructure_df['infrastructure_efficiency'].dropna()
        top_efficient = self.infrastructure_df.nlargest(15, 'infrastructure_efficiency')
        
        y_pos = np.arange(len(top_efficient))
        bars = plt.barh(y_pos, top_efficient['infrastructure_efficiency'], color='orange', alpha=0.7)
        plt.yticks(y_pos, top_efficient['country'])
        plt.xlabel('Infrastructure Efficiency Score')
        plt.title('Most Infrastructure Efficient Countries', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 12. Regional Comparison Box Plot
        plt.subplot(4, 3, 12)
        continent_data = [
            group['electricity_access_percent'].dropna().values
            for name, group in self.infrastructure_df.groupby('continent')
            if len(group['electricity_access_percent'].dropna()) > 0
        ]
        continent_names = [
            name for name, group in self.infrastructure_df.groupby('continent')
            if len(group['electricity_access_percent'].dropna()) > 0
        ]
        
        plt.boxplot(continent_data, labels=continent_names)
        plt.xticks(rotation=45)
        plt.ylabel('Electricity Access (%)')
        plt.title('Regional Infrastructure Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / "infrastructure_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Infrastructure overview saved to: {output_path}")
        
        plt.show()
        
        # Create additional specialized plots
        self.create_outlier_analysis_plot()
        
        self.results['visualizations'] = {
            'overview_plot': str(output_path),
            'outlier_plot': str(self.output_dir / "infrastructure_outliers.png")
        }

    def create_outlier_analysis_plot(self):
        """Create detailed outlier analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. High access, low GDP countries
        high_access_low_gdp = self.infrastructure_df[
            (self.infrastructure_df['electricity_access_percent'] >= 95) &
            (self.infrastructure_df['gdp_per_capita'] < 5000)
        ]
        
        ax1.scatter(self.infrastructure_df['gdp_per_capita'], 
                   self.infrastructure_df['electricity_access_percent'],
                   alpha=0.6, color='lightblue', s=50, label='All countries')
        ax1.scatter(high_access_low_gdp['gdp_per_capita'], 
                   high_access_low_gdp['electricity_access_percent'],
                   color='red', s=100, label='High access, low GDP', zorder=5)
        
        for _, country in high_access_low_gdp.iterrows():
            ax1.annotate(country['country'], 
                        (country['gdp_per_capita'], country['electricity_access_percent']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('GDP per Capita (USD)')
        ax1.set_ylabel('Electricity Access (%)')
        ax1.set_title('Countries with High Access Despite Low GDP', fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Low access, high GDP countries
        low_access_high_gdp = self.infrastructure_df[
            (self.infrastructure_df['electricity_access_percent'] < 80) &
            (self.infrastructure_df['gdp_per_capita'] > 10000)
        ]
        
        ax2.scatter(self.infrastructure_df['gdp_per_capita'], 
                   self.infrastructure_df['electricity_access_percent'],
                   alpha=0.6, color='lightblue', s=50, label='All countries')
        ax2.scatter(low_access_high_gdp['gdp_per_capita'], 
                   low_access_high_gdp['electricity_access_percent'],
                   color='orange', s=100, label='Low access, high GDP', zorder=5)
        
        for _, country in low_access_high_gdp.iterrows():
            ax2.annotate(country['country'], 
                        (country['gdp_per_capita'], country['electricity_access_percent']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('GDP per Capita (USD)')
        ax2.set_ylabel('Electricity Access (%)')
        ax2.set_title('Countries with Low Access Despite High GDP', fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Infrastructure efficiency leaders
        efficiency_data = self.infrastructure_df['infrastructure_efficiency'].dropna()
        if len(efficiency_data) > 0:
            top_efficient = self.infrastructure_df.nlargest(10, 'infrastructure_efficiency')
            
            bars = ax3.bar(range(len(top_efficient)), top_efficient['infrastructure_efficiency'], 
                          color='green', alpha=0.7)
            ax3.set_xticks(range(len(top_efficient)))
            ax3.set_xticklabels(top_efficient['country'], rotation=45, ha='right')
            ax3.set_ylabel('Infrastructure Efficiency Score')
            ax3.set_title('Most Infrastructure Efficient Countries', fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Regional efficiency comparison
        regional_efficiency = self.infrastructure_df.groupby('continent')['infrastructure_efficiency'].mean().sort_values(ascending=False)
        regional_efficiency = regional_efficiency.dropna()
        
        bars = ax4.bar(range(len(regional_efficiency)), regional_efficiency.values, 
                      color=sns.color_palette("viridis", len(regional_efficiency)))
        ax4.set_xticks(range(len(regional_efficiency)))
        ax4.set_xticklabels(regional_efficiency.index, rotation=45, ha='right')
        ax4.set_ylabel('Average Infrastructure Efficiency')
        ax4.set_title('Regional Infrastructure Efficiency', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / "infrastructure_outliers.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Infrastructure outliers analysis saved to: {output_path}")
        
        plt.show()

    def generate_summary_report(self):
        """Generate a comprehensive summary of all analyses"""
        print("\n" + "="*80)
        print("INFRASTRUCTURE ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_countries = len(self.infrastructure_df)
        avg_access = self.infrastructure_df['electricity_access_percent'].mean()
        median_access = self.infrastructure_df['electricity_access_percent'].median()
        universal_access = len(self.infrastructure_df[self.infrastructure_df['electricity_access_percent'] >= 99])
        
        print(f"\n📊 GLOBAL INFRASTRUCTURE OVERVIEW:")
        print(f"   • Total countries analyzed: {total_countries}")
        print(f"   • Average electricity access: {avg_access:.1f}%")
        print(f"   • Median electricity access: {median_access:.1f}%")
        print(f"   • Countries with universal access (≥99%): {universal_access}")
        
        # Regional insights
        if 'regional_analysis' in self.results:
            regional_stats = self.results['regional_analysis']['statistics']
            best_region = regional_stats.loc[regional_stats['avg_electricity_access'].idxmax()]
            worst_region = regional_stats.loc[regional_stats['avg_electricity_access'].idxmin()]
            
            print(f"\n🌍 REGIONAL INSIGHTS:")
            print(f"   • Best performing region: {best_region['continent']} ({best_region['avg_electricity_access']:.1f}% avg access)")
            print(f"   • Lowest performing region: {worst_region['continent']} ({worst_region['avg_electricity_access']:.1f}% avg access)")
            print(f"   • Largest regional gap: {best_region['avg_electricity_access'] - worst_region['avg_electricity_access']:.1f} percentage points")
        
        # Correlation insights
        if 'correlations' in self.results:
            correlations = self.results['correlations']['infrastructure_correlations']
            strongest_corr = max(correlations, key=lambda x: abs(x['correlation']))
            
            print(f"\n🔗 KEY RELATIONSHIPS:")
            print(f"   • Strongest correlation: {strongest_corr['variable']} (r = {strongest_corr['correlation']:.3f})")
            
            strong_correlations = [c for c in correlations if abs(c['correlation']) > 0.6]
            print(f"   • Strong correlations found: {len(strong_correlations)}")
            
            for corr in strong_correlations[:3]:
                direction = "positively" if corr['correlation'] > 0 else "negatively"
                print(f"     - {corr['variable']}: {direction} correlated (r = {corr['correlation']:.3f})")
        
        # Clustering insights
        if 'clustering' in self.results:
            clustering_info = self.results['clustering']
            print(f"\n🎯 DEVELOPMENT PATTERNS:")
            print(f"   • Identified {clustering_info['optimal_k']} distinct infrastructure clusters")
            print(f"   • Clustering quality score: {max(clustering_info['silhouette_scores']):.3f}")
            
            # Show cluster characteristics
            for summary in clustering_info['cluster_summary']:
                print(f"   • Cluster {summary['cluster_id']}: {summary['size']} countries")
                print(f"     - Avg electricity access: {summary['avg_electricity_access']:.1f}%")
                print(f"     - Avg GDP: ${summary['avg_gdp']:,.0f}")
        
        # Outlier insights
        if 'outliers' in self.results:
            outliers = self.results['outliers']
            high_access_low_gdp = [o for o in outliers if o['type'] == 'High Access, Low GDP']
            low_access_high_gdp = [o for o in outliers if o['type'] == 'Low Access, High GDP']
            
            print(f"\n🎯 NOTABLE PATTERNS:")
            print(f"   • Countries with high access despite low GDP: {len(high_access_low_gdp)}")
            if high_access_low_gdp:
                examples = high_access_low_gdp[:3]
                for example in examples:
                    print(f"     - {example['country']}: {example['electricity_access']:.1f}% access, ${example['gdp_per_capita']:,.0f} GDP")
            
            print(f"   • Countries with low access despite high GDP: {len(low_access_high_gdp)}")
            if low_access_high_gdp:
                examples = low_access_high_gdp[:3]
                for example in examples:
                    print(f"     - {example['country']}: {example['electricity_access']:.1f}% access, ${example['gdp_per_capita']:,.0f} GDP")
        
        # Performance insights
        top_performers = self.infrastructure_df.nlargest(5, 'electricity_access_percent')
        bottom_performers = self.infrastructure_df.nsmallest(5, 'electricity_access_percent')
        
        print(f"\n🏆 TOP PERFORMERS:")
        for _, country in top_performers.iterrows():
            print(f"   • {country['country']}: {country['electricity_access_percent']:.1f}% access")
        
        print(f"\n⚠️  IMPROVEMENT OPPORTUNITIES:")
        for _, country in bottom_performers.iterrows():
            print(f"   • {country['country']}: {country['electricity_access_percent']:.1f}% access")
        
        print("\n" + "="*80)

    def run_complete_analysis(self):
        """Run the complete infrastructure analysis pipeline"""
        print("Starting infrastructure analysis...")
        
        # Prepare data
        self.prepare_infrastructure_data()
        
        # Run all analyses
        self.analyze_regional_infrastructure()
        self.analyze_infrastructure_correlations()
        self.analyze_infrastructure_clusters()
        self.analyze_infrastructure_outliers()
        
        # Create visualizations
        self.create_infrastructure_visualizations()
        
        # Generate summary
        self.generate_summary_report()
        
        print(f"\n✅ Infrastructure analysis complete!")
        print(f"📊 Results saved to: {self.output_dir}")
        
        return self.results

def load_data():
    """Load and clean the dataset"""
    try:
        source_path = "source-files/World Data 2.0-Data.csv"
        print(f"Loading data from {source_path}...")
        
        df = pd.read_csv(source_path)
        print(f"Raw data shape: {df.shape}")
        
        # Filter to valid continents
        valid_continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']
        df_clean = df[df['Continent'].isin(valid_continents)].copy()
        
        print(f"Clean data shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    """Main execution function"""
    print("="*80)
    print("GLOBAL THRIVING ANALYSIS: INFRASTRUCTURE DIMENSION")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return
    
    # Run analysis
    analyzer = InfrastructureAnalyzer(df)
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎉 Analysis complete! Check the '{analyzer.output_dir}' directory for outputs.")

if __name__ == "__main__":
    main()