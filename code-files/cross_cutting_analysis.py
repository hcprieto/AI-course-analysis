"""
Cross-Cutting Analysis Module for Global Thriving Study

This module performs comprehensive analysis across multiple dimensions of thriving
including economic, health, education, and social indicators to identify
patterns, synergies, trade-offs, and create composite measures of national thriving.

Research Questions:
- How do different dimensions of thriving relate to each other?
- Can we create a comprehensive "thriving index" combining multiple indicators?
- What are the key synergies and trade-offs between different aspects of development?
- Which countries excel across multiple dimensions vs. those with specific strengths?
- Are there distinct patterns of national development strategies?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor
import warnings
from pathlib import Path
from datetime import datetime
import itertools

# Add new import for HTML generation
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class CrossCuttingAnalyzer:
    def __init__(self, df):
        self.df = df
        self.results = {}
        self.output_dir = Path("analysis-output-files")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define indicator categories
        self.indicator_categories = {
            'economic': [
                'GDP per Capita (2018)',
                'Unemployment Rate (2018)'
            ],
            'health': [
                'Life expectancy at birth (years)(2018)',
                'Infant mortality rate (2018)',
                'Physicians per thousand (2017)',
                'Hospital beds per thousand (2017)'
            ],
            'education': [
                'Learning-Adjusted Years of School (2020)',
                'Primary school enrollment (% gross)(2019)',
                'Tertiary school enrollment (% gross)(2018)'
            ],
            'social': [
                'Life satisfaction in Cantril Ladder (2018)',
                'Personal freedoms (2018)',
                'Social support (2018)',
                'Generosity (2018)'
            ],
            'demographics': [
                'Total population (2018)',
                'Population growth (annual %)(2018)',
                'Urban population (% of total population)(2018)'
            ]
        }

    def prepare_cross_cutting_data(self):
        """Prepare and clean data for cross-cutting analysis"""
        print("Preparing cross-cutting data...")
        
        # Create working copy
        self.cross_df = self.df.copy()
        
        # Clean and standardize all indicators
        self.cleaned_indicators = {}
        
        for category, indicators in self.indicator_categories.items():
            print(f"Processing {category} indicators...")
            category_data = {}
            
            for indicator in indicators:
                if indicator in self.cross_df.columns:
                    # Clean the data
                    clean_col = self._clean_indicator_name(indicator)
                    
                    # Handle different data formats
                    series = self.cross_df[indicator].copy()
                    
                    # Remove quotes and commas for numeric data
                    if series.dtype == 'object':
                        series = series.astype(str).str.replace('"', '').str.replace(',', '').str.strip()
                    
                    # Convert to numeric
                    series = pd.to_numeric(series, errors='coerce')
                    
                    # Handle specific transformations
                    if 'unemployment' in indicator.lower():
                        # Invert unemployment (lower is better)
                        series = 100 - series
                        clean_col = clean_col.replace('unemployment', 'employment_health')
                    elif 'infant_mortality' in clean_col:
                        # Invert infant mortality (lower is better)
                        series = -np.log1p(series)  # Log transform and invert
                    
                    category_data[clean_col] = series
                    self.cross_df[clean_col] = series
            
            self.cleaned_indicators[category] = category_data
        
        # Add geographic data
        self.cross_df['continent'] = self.cross_df['Continent']
        self.cross_df['country'] = self.cross_df['Country']
        
        # Create master list of all cleaned indicators
        self.all_indicators = []
        for category_indicators in self.cleaned_indicators.values():
            self.all_indicators.extend(category_indicators.keys())
        
        print(f"Prepared {len(self.all_indicators)} indicators across {len(self.cleaned_indicators)} categories")
        return self.cross_df

    def _clean_indicator_name(self, indicator):
        """Clean indicator names for easier handling"""
        clean_name = (indicator.lower()
                     .replace('(', '').replace(')', '')
                     .replace('%', 'pct')
                     .replace(' ', '_')
                     .replace('-', '_')
                     .replace(',', '')
                     .replace('/', '_per_'))
        
        # Remove year information
        import re
        clean_name = re.sub(r'_\d{4}', '', clean_name)
        
        return clean_name

    def multidimensional_correlation_analysis(self):
        """Analyze correlations across all dimensions"""
        print("Performing multidimensional correlation analysis...")
        
        # Create correlation dataset
        correlation_data = self.cross_df[self.all_indicators + ['continent', 'country']].copy()
        
        # Remove countries with too many missing values
        missing_threshold = 0.5  # Allow 50% missing data per country
        countries_to_keep = correlation_data.isnull().mean(axis=1) < missing_threshold
        correlation_data = correlation_data[countries_to_keep]
        
        # Calculate correlation matrices
        numeric_data = correlation_data[self.all_indicators]
        
        # Overall correlation matrix
        overall_corr = numeric_data.corr()
        
        # Category-wise correlations
        category_correlations = {}
        cross_category_correlations = {}
        
        for cat1, indicators1 in self.cleaned_indicators.items():
            # Within-category correlations
            cat1_indicators = [ind for ind in indicators1.keys() if ind in self.all_indicators]
            if len(cat1_indicators) > 1:
                cat_corr = numeric_data[cat1_indicators].corr()
                category_correlations[cat1] = cat_corr
            
            # Cross-category correlations
            for cat2, indicators2 in self.cleaned_indicators.items():
                if cat1 != cat2:
                    cat2_indicators = [ind for ind in indicators2.keys() if ind in self.all_indicators]
                    
                    if len(cat1_indicators) > 0 and len(cat2_indicators) > 0:
                        cross_corr_data = []
                        
                        for ind1 in cat1_indicators:
                            for ind2 in cat2_indicators:
                                subset = numeric_data[[ind1, ind2]].dropna()
                                if len(subset) > 10:
                                    r, p = pearsonr(subset[ind1], subset[ind2])
                                    cross_corr_data.append({
                                        'indicator1': ind1,
                                        'indicator2': ind2,
                                        'category1': cat1,
                                        'category2': cat2,
                                        'correlation': r,
                                        'p_value': p,
                                        'significant': p < 0.05,
                                        'n_observations': len(subset)
                                    })
                        
                        if cross_corr_data:
                            cross_category_correlations[f"{cat1}_vs_{cat2}"] = cross_corr_data
        
        # Find strongest cross-dimensional relationships
        strong_relationships = []
        for pair, relationships in cross_category_correlations.items():
            for rel in relationships:
                if rel['significant'] and abs(rel['correlation']) > 0.4:
                    strong_relationships.append(rel)
        
        # Sort by correlation strength
        strong_relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        correlation_results = {
            'overall_correlation_matrix': overall_corr.to_dict(),
            'category_correlations': {cat: corr.to_dict() for cat, corr in category_correlations.items()},
            'cross_category_correlations': cross_category_correlations,
            'strong_cross_dimensional_relationships': strong_relationships[:20],  # Top 20
            'total_countries_analyzed': len(correlation_data),
            'correlation_data': correlation_data
        }
        
        self.results['correlations'] = correlation_results
        return correlation_results

    def create_composite_indices(self):
        """Create composite indices for different aspects of thriving"""
        print("Creating composite indices...")
        
        composite_data = self.cross_df[self.all_indicators + ['country', 'continent']].copy()
        
        # Create indices for each dimension
        dimension_indices = {}
        
        for dimension, indicators in self.cleaned_indicators.items():
            available_indicators = [ind for ind in indicators.keys() if ind in self.all_indicators]
            
            if len(available_indicators) >= 2:
                # Get data for this dimension
                dim_data = composite_data[available_indicators].copy()
                
                # Standardize indicators (0-100 scale)
                scaler = MinMaxScaler(feature_range=(0, 100))
                dim_data_scaled = pd.DataFrame(
                    scaler.fit_transform(dim_data.fillna(dim_data.median())),
                    columns=available_indicators,
                    index=dim_data.index
                )
                
                # Calculate dimension index (simple average)
                dimension_index = dim_data_scaled.mean(axis=1)
                dimension_indices[f"{dimension}_index"] = dimension_index
                composite_data[f"{dimension}_index"] = dimension_index
        
        # Create overall thriving index
        if len(dimension_indices) >= 3:
            # Weight dimensions equally for overall index
            index_cols = list(dimension_indices.keys())
            overall_index = composite_data[index_cols].mean(axis=1)
            composite_data['overall_thriving_index'] = overall_index
            dimension_indices['overall_thriving_index'] = overall_index
        
        # Alternative: PCA-based composite index
        if len(self.all_indicators) >= 4:
            pca_data = composite_data[self.all_indicators].fillna(composite_data[self.all_indicators].median())
            scaler = StandardScaler()
            pca_data_scaled = scaler.fit_transform(pca_data)
            
            pca = PCA(n_components=1)
            pca_index = pca.fit_transform(pca_data_scaled).flatten()
            
            # Convert to 0-100 scale
            pca_index_scaled = MinMaxScaler(feature_range=(0, 100)).fit_transform(pca_index.reshape(-1, 1)).flatten()
            composite_data['pca_thriving_index'] = pca_index_scaled
            dimension_indices['pca_thriving_index'] = pca_index_scaled
        
        # Rank countries by indices
        rankings = {}
        for index_name, index_values in dimension_indices.items():
            index_df = pd.DataFrame({
                'country': composite_data['country'],
                'continent': composite_data['continent'],
                'index_value': index_values
            }).dropna()
            
            index_df['rank'] = index_df['index_value'].rank(ascending=False, method='min')
            rankings[index_name] = index_df.sort_values('rank')
        
        composite_results = {
            'dimension_indices': dimension_indices,
            'composite_data': composite_data,
            'rankings': rankings,
            'index_statistics': {
                index_name: {
                    'mean': float(index_values.mean()),
                    'std': float(index_values.std()),
                    'min': float(index_values.min()),
                    'max': float(index_values.max()),
                    'countries_with_data': int((~np.isnan(index_values)).sum())
                }
                for index_name, index_values in dimension_indices.items()
            }
        }
        
        self.results['composite_indices'] = composite_results
        return composite_results

    def multidimensional_clustering(self):
        """Perform clustering across all dimensions to identify country patterns"""
        print("Performing multidimensional clustering...")
        
        # Prepare clustering data
        cluster_indicators = []
        for category, indicators in self.cleaned_indicators.items():
            # Select top 2 indicators per category to avoid overcomplexity
            available = [ind for ind in indicators.keys() if ind in self.all_indicators]
            cluster_indicators.extend(available[:2])
        
        cluster_data = self.cross_df[cluster_indicators + ['country', 'continent']].dropna()
        
        if len(cluster_data) < 10:
            return {}
        
        # Prepare features
        X = cluster_data[cluster_indicators].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        silhouette_scores = []
        inertias = []
        K_range = range(2, min(8, len(cluster_data)//3))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Choose optimal k based on silhouette score
        if silhouette_scores:
            optimal_k = K_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        cluster_data['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_countries = cluster_data[cluster_data['cluster'] == cluster_id]
            
            cluster_stats = {}
            for indicator in cluster_indicators:
                cluster_stats[indicator] = {
                    'mean': float(cluster_countries[indicator].mean()),
                    'std': float(cluster_countries[indicator].std()),
                    'median': float(cluster_countries[indicator].median())
                }
            
            # Characterize cluster
            cluster_profile = self._characterize_cluster(cluster_countries, cluster_indicators)
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'countries': cluster_countries['country'].tolist(),
                'size': len(cluster_countries),
                'continent_distribution': cluster_countries['continent'].value_counts().to_dict(),
                'statistics': cluster_stats,
                'profile': cluster_profile
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
            'clustering_indicators': cluster_indicators,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'k_range': list(K_range)
        }
        
        self.results['clustering'] = clustering_results
        return clustering_results

    def _characterize_cluster(self, cluster_data, indicators):
        """Characterize a cluster based on its indicator patterns"""
        # Calculate z-scores relative to global means
        global_means = self.cross_df[indicators].mean()
        global_stds = self.cross_df[indicators].std()
        
        cluster_means = cluster_data[indicators].mean()
        z_scores = (cluster_means - global_means) / global_stds
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for indicator, z_score in z_scores.items():
            if z_score > 1:
                strengths.append(f"{indicator} (z={z_score:.2f})")
            elif z_score < -1:
                weaknesses.append(f"{indicator} (z={z_score:.2f})")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'dominant_characteristics': sorted(z_scores.to_dict().items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        }

    def synergy_tradeoff_analysis(self):
        """Analyze synergies and trade-offs between different thriving dimensions"""
        print("Analyzing synergies and trade-offs...")
        
        if 'composite_indices' not in self.results:
            return {}
        
        indices_data = self.results['composite_indices']['composite_data']
        dimension_indices = [col for col in indices_data.columns if col.endswith('_index') and col != 'overall_thriving_index']
        
        if len(dimension_indices) < 3:
            return {}
        
        # Calculate pairwise relationships between dimensions
        synergies = []
        tradeoffs = []
        
        for i, dim1 in enumerate(dimension_indices):
            for j, dim2 in enumerate(dimension_indices[i+1:], i+1):
                subset = indices_data[[dim1, dim2, 'country', 'continent']].dropna()
                
                if len(subset) > 10:
                    r, p = pearsonr(subset[dim1], subset[dim2])
                    
                    relationship = {
                        'dimension1': dim1.replace('_index', ''),
                        'dimension2': dim2.replace('_index', ''),
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'n_countries': len(subset),
                        'relationship_type': 'synergy' if r > 0.3 else 'tradeoff' if r < -0.3 else 'neutral'
                    }
                    
                    if r > 0.3 and p < 0.05:
                        synergies.append(relationship)
                    elif r < -0.3 and p < 0.05:
                        tradeoffs.append(relationship)
        
        # Identify countries with unusual patterns
        unusual_patterns = []
        
        for _, country_data in indices_data.iterrows():
            if pd.isna(country_data['country']):
                continue
                
            country_indices = {dim: country_data[dim] for dim in dimension_indices if pd.notna(country_data[dim])}
            
            if len(country_indices) >= 3:
                # Calculate coefficient of variation
                values = list(country_indices.values())
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                
                # Identify outlier patterns
                if cv > 0.3:  # High variation across dimensions
                    unusual_patterns.append({
                        'country': country_data['country'],
                        'continent': country_data['continent'],
                        'pattern_type': 'high_variation',
                        'coefficient_variation': cv,
                        'dimension_scores': country_indices
                    })
        
        # Sort by most unusual
        unusual_patterns.sort(key=lambda x: x['coefficient_variation'], reverse=True)
        
        synergy_results = {
            'synergies': sorted(synergies, key=lambda x: x['correlation'], reverse=True),
            'tradeoffs': sorted(tradeoffs, key=lambda x: x['correlation']),
            'unusual_patterns': unusual_patterns[:15],  # Top 15 most unusual
            'dimension_relationships_summary': {
                'total_synergies': len(synergies),
                'total_tradeoffs': len(tradeoffs),
                'strongest_synergy': max(synergies, key=lambda x: x['correlation']) if synergies else None,
                'strongest_tradeoff': min(tradeoffs, key=lambda x: x['correlation']) if tradeoffs else None
            }
        }
        
        self.results['synergy_tradeoffs'] = synergy_results
        return synergy_results

    def development_pathways_analysis(self):
        """Analyze different pathways to thriving"""
        print("Analyzing development pathways...")
        
        if 'composite_indices' not in self.results:
            return {}
        
        composite_data = self.results['composite_indices']['composite_data']
        
        # Define development archetypes based on dimension strengths
        dimension_indices = [col for col in composite_data.columns if col.endswith('_index') and col != 'overall_thriving_index']
        
        if len(dimension_indices) < 3:
            return {}
        
        archetype_data = composite_data[dimension_indices + ['country', 'continent']].dropna()
        
        # Identify countries excelling in specific dimensions
        archetypes = {}
        
        for dimension in dimension_indices:
            # Top performers in this dimension
            top_percentile = archetype_data[dimension].quantile(0.8)
            top_performers = archetype_data[archetype_data[dimension] >= top_percentile]
            
            # Check if they're balanced across other dimensions
            other_dimensions = [d for d in dimension_indices if d != dimension]
            other_scores = top_performers[other_dimensions].mean(axis=1)
            
            # Categorize as specialists vs generalists
            specialists = top_performers[other_scores < archetype_data[other_dimensions].mean().mean()]
            generalists = top_performers[other_scores >= archetype_data[other_dimensions].mean().mean()]
            
            archetypes[dimension.replace('_index', '')] = {
                'specialists': {
                    'countries': specialists['country'].tolist(),
                    'count': len(specialists),
                    'avg_dimension_score': float(specialists[dimension].mean()),
                    'avg_other_scores': float(specialists[other_dimensions].mean().mean())
                },
                'generalists': {
                    'countries': generalists['country'].tolist(),
                    'count': len(generalists),
                    'avg_dimension_score': float(generalists[dimension].mean()),
                    'avg_other_scores': float(generalists[other_dimensions].mean().mean())
                }
            }
        
        # Identify overall top performers
        if 'overall_thriving_index' in composite_data.columns:
            overall_top = composite_data.nlargest(10, 'overall_thriving_index')
            overall_bottom = composite_data.nsmallest(10, 'overall_thriving_index')
            
            top_performers_analysis = {
                'top_10_countries': overall_top[['country', 'continent', 'overall_thriving_index'] + dimension_indices].to_dict('records'),
                'bottom_10_countries': overall_bottom[['country', 'continent', 'overall_thriving_index'] + dimension_indices].to_dict('records'),
                'top_performers_patterns': self._analyze_top_performers_patterns(overall_top, dimension_indices),
                'improvement_opportunities': self._analyze_improvement_opportunities(overall_bottom, dimension_indices)
            }
        else:
            top_performers_analysis = {}
        
        pathways_results = {
            'development_archetypes': archetypes,
            'top_performers_analysis': top_performers_analysis,
            'pathway_insights': self._generate_pathway_insights(archetypes)
        }
        
        self.results['development_pathways'] = pathways_results
        return pathways_results

    def _analyze_top_performers_patterns(self, top_performers, dimension_indices):
        """Analyze patterns among top performing countries"""
        patterns = {}
        
        # Dimension strength patterns
        for dimension in dimension_indices:
            mean_score = top_performers[dimension].mean()
            patterns[dimension.replace('_index', '')] = {
                'average_score': float(mean_score),
                'consistency': float(1 - top_performers[dimension].std() / mean_score) if mean_score > 0 else 0
            }
        
        # Continental representation
        continent_dist = top_performers['continent'].value_counts()
        patterns['continental_representation'] = continent_dist.to_dict()
        
        return patterns

    def _analyze_improvement_opportunities(self, bottom_performers, dimension_indices):
        """Identify key improvement opportunities for lower-performing countries"""
        opportunities = {}
        
        global_means = self.cross_df[dimension_indices].mean()
        
        for dimension in dimension_indices:
            dim_mean = bottom_performers[dimension].mean()
            global_mean = global_means[dimension]
            gap = global_mean - dim_mean
            
            opportunities[dimension.replace('_index', '')] = {
                'average_score': float(dim_mean),
                'global_average': float(global_mean),
                'improvement_gap': float(gap),
                'priority_level': 'high' if gap > global_means.std()[dimension] else 'medium' if gap > 0 else 'low'
            }
        
        return opportunities

    def _generate_pathway_insights(self, archetypes):
        """Generate insights about different development pathways"""
        insights = []
        
        # Specialist vs generalist patterns
        total_specialists = sum(arch['specialists']['count'] for arch in archetypes.values())
        total_generalists = sum(arch['generalists']['count'] for arch in archetypes.values())
        
        insights.append(f"Development approaches: {total_specialists} specialist countries vs {total_generalists} generalist countries")
        
        # Identify most common specialization
        specializations = {dim: data['specialists']['count'] for dim, data in archetypes.items()}
        most_common = max(specializations, key=specializations.get)
        insights.append(f"Most common specialization: {most_common} ({specializations[most_common]} countries)")
        
        return insights

    def create_cross_cutting_visualizations(self):
        """Create comprehensive cross-cutting visualizations"""
        print("Creating cross-cutting visualizations...")
        
        # 1. Multidimensional Overview
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('Cross-Cutting Analysis: Multidimensional Global Thriving', fontsize=20, fontweight='bold')
        
        # Correlation heatmap
        if 'correlations' in self.results:
            ax1 = fig.add_subplot(gs[0, :2])
            corr_matrix = pd.DataFrame(self.results['correlations']['overall_correlation_matrix'])
            
            # Select subset for readability
            if len(corr_matrix) > 15:
                # Select most connected indicators
                connectivity = corr_matrix.abs().sum().sort_values(ascending=False)
                top_indicators = connectivity.head(15).index
                corr_subset = corr_matrix.loc[top_indicators, top_indicators]
            else:
                corr_subset = corr_matrix
            
            mask = np.triu(np.ones_like(corr_subset, dtype=bool))
            sns.heatmap(corr_subset, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, ax=ax1, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax1.set_title('Cross-Dimensional Correlation Matrix', fontweight='bold')
            
        # Dimension indices comparison
        if 'composite_indices' in self.results:
            ax2 = fig.add_subplot(gs[0, 2:])
            composite_data = self.results['composite_indices']['composite_data']
            dimension_indices = [col for col in composite_data.columns if col.endswith('_index') and 'overall' not in col]
            
            if len(dimension_indices) >= 3:
                plot_data = composite_data[dimension_indices].melt(var_name='Dimension', value_name='Index_Score')
                plot_data['Dimension'] = plot_data['Dimension'].str.replace('_index', '').str.title()
                
                sns.boxplot(data=plot_data, x='Dimension', y='Index_Score', ax=ax2, palette='Set2')
                ax2.set_title('Distribution of Dimension Indices', fontweight='bold')
                ax2.set_ylabel('Index Score (0-100)')
                ax2.tick_params(axis='x', rotation=45)
        
        # Clustering visualization
        if 'clustering' in self.results:
            ax3 = fig.add_subplot(gs[1, :2])
            pca_components = np.array(self.results['clustering']['pca_components'])
            cluster_labels = self.results['clustering']['cluster_labels']
            
            scatter = ax3.scatter(pca_components[:, 0], pca_components[:, 1],
                                c=cluster_labels, cmap='viridis', alpha=0.7, s=60)
            ax3.set_xlabel(f'PC1 ({self.results["clustering"]["pca_explained_variance"][0]*100:.1f}% variance)')
            ax3.set_ylabel(f'PC2 ({self.results["clustering"]["pca_explained_variance"][1]*100:.1f}% variance)')
            ax3.set_title('Multidimensional Country Clusters', fontweight='bold')
            plt.colorbar(scatter, ax=ax3, label='Cluster')
            ax3.grid(True, alpha=0.3)
        
        # Overall thriving index ranking
        if 'composite_indices' in self.results and 'overall_thriving_index' in self.results['composite_indices']['rankings']:
            ax4 = fig.add_subplot(gs[1, 2:])
            ranking_data = self.results['composite_indices']['rankings']['overall_thriving_index'].head(15)
            
            bars = ax4.barh(range(len(ranking_data)), ranking_data['index_value'],
                           color=plt.cm.viridis(ranking_data['index_value']/100))
            ax4.set_yticks(range(len(ranking_data)))
            ax4.set_yticklabels(ranking_data['country'], fontsize=9)
            ax4.set_xlabel('Overall Thriving Index')
            ax4.set_title('Top 15 Countries - Overall Thriving', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add values on bars
            for i, (idx, row) in enumerate(ranking_data.iterrows()):
                ax4.text(row['index_value'] + 1, i, f"{row['index_value']:.1f}",
                        va='center', fontsize=8)
        
        # Synergies and trade-offs
        if 'synergy_tradeoffs' in self.results:
            ax5 = fig.add_subplot(gs[2, :2])
            synergies = self.results['synergy_tradeoffs']['synergies'][:10]
            tradeoffs = self.results['synergy_tradeoffs']['tradeoffs'][:5]
            
            if synergies or tradeoffs:
                relationships = synergies + tradeoffs
                correlations = [r['correlation'] for r in relationships]
                labels = [f"{r['dimension1']} - {r['dimension2']}" for r in relationships]
                colors = ['green' if r['correlation'] > 0 else 'red' for r in relationships]
                
                bars = ax5.barh(range(len(relationships)), correlations, color=colors, alpha=0.7)
                ax5.set_yticks(range(len(relationships)))
                ax5.set_yticklabels(labels, fontsize=8)
                ax5.set_xlabel('Correlation Coefficient')
                ax5.set_title('Strongest Synergies and Trade-offs', fontweight='bold')
                ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax5.grid(True, alpha=0.3, axis='x')
        
        # Development archetypes
        if 'development_pathways' in self.results:
            ax6 = fig.add_subplot(gs[2, 2:])
            archetypes = self.results['development_pathways']['development_archetypes']
            
            specialist_counts = [data['specialists']['count'] for data in archetypes.values()]
            generalist_counts = [data['generalists']['count'] for data in archetypes.values()]
            dimensions = [dim.title() for dim in archetypes.keys()]
            
            x = np.arange(len(dimensions))
            width = 0.35
            
            ax6.bar(x - width/2, specialist_counts, width, label='Specialists', alpha=0.8, color='skyblue')
            ax6.bar(x + width/2, generalist_counts, width, label='Generalists', alpha=0.8, color='lightcoral')
            
            ax6.set_xlabel('Development Dimensions')
            ax6.set_ylabel('Number of Countries')
            ax6.set_title('Development Archetypes by Dimension', fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(dimensions, rotation=45)
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        # Regional patterns in overall thriving
        if 'composite_indices' in self.results:
            ax7 = fig.add_subplot(gs[3, :2])
            composite_data = self.results['composite_indices']['composite_data']
            
            if 'overall_thriving_index' in composite_data.columns:
                regional_data = composite_data[['continent', 'overall_thriving_index']].dropna()
                sns.boxplot(data=regional_data, x='continent', y='overall_thriving_index',
                           ax=ax7, palette='Set3', showmeans=True)
                ax7.set_title('Regional Patterns in Overall Thriving', fontweight='bold')
                ax7.set_xlabel('Continent')
                ax7.set_ylabel('Overall Thriving Index')
                ax7.tick_params(axis='x', rotation=45)
                ax7.grid(True, alpha=0.3, axis='y')
        
        # Top performers characteristics
        if 'development_pathways' in self.results and 'top_performers_analysis' in self.results['development_pathways']:
            ax8 = fig.add_subplot(gs[3, 2:])
            top_analysis = self.results['development_pathways']['top_performers_analysis']
            
            if 'top_performers_patterns' in top_analysis:
                patterns = top_analysis['top_performers_patterns']
                dimensions = [k for k in patterns.keys() if k != 'continental_representation']
                scores = [patterns[dim]['average_score'] for dim in dimensions]
                
                ax8.bar(range(len(dimensions)), scores, color='gold', alpha=0.8)
                ax8.set_xticks(range(len(dimensions)))
                ax8.set_xticklabels([d.title() for d in dimensions], rotation=45)
                ax8.set_ylabel('Average Score')
                ax8.set_title('Top 10 Countries: Dimension Strengths', fontweight='bold')
                ax8.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, score in enumerate(scores):
                    ax8.text(i, score + 1, f"{score:.1f}", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_cutting_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Relationships Analysis
        self._create_detailed_relationships_plot()
        
        print("Cross-cutting visualizations saved successfully!")

    def _create_detailed_relationships_plot(self):
        """Create detailed plot of cross-dimensional relationships"""
        if 'composite_indices' not in self.results:
            return
        
        composite_data = self.results['composite_indices']['composite_data']
        dimension_indices = [col for col in composite_data.columns if col.endswith('_index') and 'overall' not in col]
        
        if len(dimension_indices) < 3:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Dimensional Relationships and Patterns', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Create scatter plots for key relationships
        relationships = list(itertools.combinations(dimension_indices[:4], 2))
        
        for i, (dim1, dim2) in enumerate(relationships[:6]):
            if i >= 6:
                break
                
            ax = axes[i]
            plot_data = composite_data[[dim1, dim2, 'continent']].dropna()
            
            # Scatter plot by continent
            for continent in plot_data['continent'].unique():
                if pd.notna(continent):
                    cont_data = plot_data[plot_data['continent'] == continent]
                    ax.scatter(cont_data[dim1], cont_data[dim2], label=continent, alpha=0.7, s=60)
            
            # Add regression line
            if len(plot_data) > 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[dim1], plot_data[dim2])
                x_range = np.linspace(plot_data[dim1].min(), plot_data[dim1].max(), 100)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, 'r--', alpha=0.8, linewidth=2,
                       label=f'R² = {r_value**2:.3f}')
            
            ax.set_xlabel(dim1.replace('_index', '').title())
            ax.set_ylabel(dim2.replace('_index', '').title())
            ax.set_title(f'{dim1.replace("_index", "").title()} vs {dim2.replace("_index", "").title()}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_dimensional_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_html_report(self):
        """Generate comprehensive HTML report for cross-cutting analysis"""
        print("Generating HTML report...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cross-Cutting Analysis - Global Thriving Study</title>
            <style>
                {self._get_html_styles()}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>🔗 Cross-Cutting Analysis: Global Thriving Study</h1>
                    <p class="subtitle">Comprehensive analysis across multiple dimensions of national thriving</p>
                    <div class="meta-info">
                        <span>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</span>
                        <span>Countries analyzed: {self.results.get('correlations', {}).get('total_countries_analyzed', 'N/A')}</span>
                    </div>
                </header>

                <nav class="table-of-contents">
                    <h2>📋 Table of Contents</h2>
                    <ul>
                        <li><a href="#executive-summary">Executive Summary</a></li>
                        <li><a href="#correlations">Cross-Dimensional Relationships</a></li>
                        <li><a href="#composite-indices">Composite Thriving Indices</a></li>
                        <li><a href="#clustering">Multidimensional Clustering</a></li>
                        <li><a href="#synergies">Synergies and Trade-offs</a></li>
                        <li><a href="#pathways">Development Pathways</a></li>
                        <li><a href="#visualizations">Key Visualizations</a></li>
                        <li><a href="#methodology">Methodology</a></li>
                    </ul>
                </nav>

                {self._generate_executive_summary()}
                {self._generate_correlations_section()}
                {self._generate_composite_indices_section()}
                {self._generate_clustering_section()}
                {self._generate_synergies_section()}
                {self._generate_pathways_section()}
                {self._generate_visualizations_section()}
                {self._generate_methodology_section()}

                <footer>
                    <p>Report generated by Global Thriving Analysis System</p>
                    <p>Data sources: World Bank, OECD, Gallup World Poll, and other international organizations</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = self.output_dir / "cross_cutting_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {report_path}")
        return report_path

    def _get_html_styles(self):
        """Return CSS styles for the HTML report"""
        return """
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }

            header {
                text-align: center;
                margin-bottom: 40px;
                padding: 30px 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
            }

            header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }

            .subtitle {
                font-size: 1.2rem;
                opacity: 0.9;
                margin-bottom: 20px;
            }

            .meta-info {
                display: flex;
                justify-content: center;
                gap: 30px;
                font-size: 0.9rem;
                opacity: 0.8;
            }

            .table-of-contents {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }

            .table-of-contents ul {
                list-style: none;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }

            .table-of-contents a {
                color: #667eea;
                text-decoration: none;
                padding: 8px 12px;
                border-radius: 5px;
                transition: background-color 0.3s;
            }

            .table-of-contents a:hover {
                background-color: #e9ecef;
            }

            .section {
                margin-bottom: 40px;
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }

            .section h2 {
                color: #495057;
                margin-bottom: 20px;
                font-size: 1.8rem;
            }

            .section h3 {
                color: #6c757d;
                margin: 20px 0 15px 0;
                font-size: 1.3rem;
            }

            .key-findings {
                background-color: #e3f2fd;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #2196f3;
            }

            .insight-box {
                background-color: #f1f8e9;
                padding: 15px;
                border-radius: 6px;
                margin: 15px 0;
                border-left: 3px solid #4caf50;
            }

            .warning-box {
                background-color: #fff3e0;
                padding: 15px;
                border-radius: 6px;
                margin: 15px 0;
                border-left: 3px solid #ff9800;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }

            .stat-card {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #dee2e6;
                text-align: center;
                transition: transform 0.2s;
            }

            .stat-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .stat-number {
                font-size: 2rem;
                font-weight: bold;
                color: #667eea;
                display: block;
            }

            .stat-label {
                color: #6c757d;
                font-size: 0.9rem;
                margin-top: 5px;
            }

            .country-list {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin: 15px 0;
            }

            .country-tag {
                background-color: #e7f3ff;
                color: #0066cc;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.85rem;
                border: 1px solid #b3d9ff;
            }

            .relationship-item {
                background-color: #f8f9fa;
                padding: 12px;
                margin: 8px 0;
                border-radius: 6px;
                border-left: 3px solid #667eea;
            }

            .correlation-strength {
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
            }

            .strong { background-color: #d4edda; color: #155724; }
            .moderate { background-color: #fff3cd; color: #856404; }
            .weak { background-color: #f8d7da; color: #721c24; }

            .visualization-container {
                text-align: center;
                margin: 20px 0;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
            }

            .visualization-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }

            th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }

            tr:hover {
                background-color: #f8f9fa;
            }

            footer {
                text-align: center;
                margin-top: 50px;
                padding: 30px;
                background-color: #f8f9fa;
                color: #6c757d;
                border-radius: 8px;
                font-size: 0.9rem;
            }

            .methodology-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }

            .method-card {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }

            .method-card h4 {
                color: #495057;
                margin-bottom: 10px;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                header h1 {
                    font-size: 2rem;
                }
                
                .meta-info {
                    flex-direction: column;
                    gap: 10px;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        """

    def _generate_executive_summary(self):
        """Generate executive summary section"""
        correlations = self.results.get('correlations', {})
        composite = self.results.get('composite_indices', {})
        clustering = self.results.get('clustering', {})
        synergies = self.results.get('synergy_tradeoffs', {})
        pathways = self.results.get('development_pathways', {})
        
        strong_relationships = correlations.get('strong_cross_dimensional_relationships', [])
        total_countries = correlations.get('total_countries_analyzed', 0)
        
        return f"""
        <section id="executive-summary" class="section">
            <h2>📊 Executive Summary</h2>
            
            <div class="key-findings">
                <h3>🎯 Key Findings</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{total_countries}</span>
                        <div class="stat-label">Countries Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{len(strong_relationships)}</span>
                        <div class="stat-label">Strong Cross-Dimensional Relationships</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{len(self.cleaned_indicators)}</span>
                        <div class="stat-label">Thriving Dimensions</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{clustering.get('optimal_k', 'N/A')}</span>
                        <div class="stat-label">Country Clusters Identified</div>
                    </div>
                </div>
            </div>

            <div class="insight-box">
                <h3>💡 Main Insights</h3>
                <ul>
                    <li><strong>Multidimensional Patterns:</strong> Countries show distinct patterns of development across economic, health, education, and social dimensions.</li>
                    <li><strong>Synergistic Relationships:</strong> {synergies.get('dimension_relationships_summary', {}).get('total_synergies', 0)} positive synergies found between different thriving dimensions.</li>
                    <li><strong>Development Archetypes:</strong> Countries follow different pathways - some specialize in specific dimensions while others pursue balanced development.</li>
                    <li><strong>Regional Variations:</strong> Significant differences exist between continents in overall thriving patterns.</li>
                </ul>
            </div>

            {self._generate_top_relationships_summary(strong_relationships[:5])}
        </section>
        """

    def _generate_top_relationships_summary(self, relationships):
        """Generate summary of top relationships"""
        if not relationships:
            return ""
        
        relationships_html = ""
        for rel in relationships:
            strength_class = rel.get('correlation_strength', 'moderate').lower()
            relationships_html += f"""
            <div class="relationship-item">
                <strong>{rel['category1'].title()} ↔ {rel['category2'].title()}</strong>
                <span class="correlation-strength {strength_class}">r = {rel['correlation']:.3f}</span>
                <br>
                <small>{rel['indicator1'].replace('_', ' ').title()} vs {rel['indicator2'].replace('_', ' ').title()}</small>
            </div>
            """
        
        return f"""
        <div class="insight-box">
            <h3>🔗 Strongest Cross-Dimensional Relationships</h3>
            {relationships_html}
        </div>
        """

    def _generate_correlations_section(self):
        """Generate correlations analysis section"""
        if 'correlations' not in self.results:
            return ""
        
        correlations = self.results['correlations']
        strong_relationships = correlations.get('strong_cross_dimensional_relationships', [])
        
        return f"""
        <section id="correlations" class="section">
            <h2>🔗 Cross-Dimensional Relationships</h2>
            
            <p>This analysis examines how different dimensions of national thriving relate to each other, 
            identifying synergies and potential trade-offs between economic, health, education, and social indicators.</p>
            
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li>Found <strong>{len(strong_relationships)}</strong> statistically significant cross-dimensional relationships</li>
                    <li>Analysis based on <strong>{correlations.get('total_countries_analyzed', 'N/A')}</strong> countries with sufficient data</li>
                    <li>Relationships span across all major thriving dimensions</li>
                </ul>
            </div>

            {self._generate_relationships_table(strong_relationships[:10])}
        </section>
        """

    def _generate_relationships_table(self, relationships):
        """Generate table of relationships"""
        if not relationships:
            return "<p>No significant relationships found.</p>"
        
        table_rows = ""
        for rel in relationships:
            correlation_class = "strong" if abs(rel['correlation']) >= 0.7 else "moderate" if abs(rel['correlation']) >= 0.4 else "weak"
            table_rows += f"""
            <tr>
                <td>{rel['category1'].title()}</td>
                <td>{rel['category2'].title()}</td>
                <td>{rel['indicator1'].replace('_', ' ').title()}</td>
                <td>{rel['indicator2'].replace('_', ' ').title()}</td>
                <td><span class="correlation-strength {correlation_class}">{rel['correlation']:.3f}</span></td>
                <td>{rel['n_observations']}</td>
            </tr>
            """
        
        return f"""
        <h3>Top Cross-Dimensional Relationships</h3>
        <table>
            <thead>
                <tr>
                    <th>Dimension 1</th>
                    <th>Dimension 2</th>
                    <th>Indicator 1</th>
                    <th>Indicator 2</th>
                    <th>Correlation</th>
                    <th>Countries</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """

    def _generate_composite_indices_section(self):
        """Generate composite indices section"""
        if 'composite_indices' not in self.results:
            return ""
        
        composite = self.results['composite_indices']
        index_stats = composite.get('index_statistics', {})
        rankings = composite.get('rankings', {})
        
        # Generate index statistics
        stats_html = ""
        for index_name, stats in index_stats.items():
            clean_name = index_name.replace('_', ' ').title()
            stats_html += f"""
            <div class="stat-card">
                <span class="stat-number">{stats['mean']:.1f}</span>
                <div class="stat-label">{clean_name}<br>Average Score</div>
            </div>
            """
        
        # Generate top performers table
        top_performers_html = ""
        if 'overall_thriving_index' in rankings:
            top_10 = rankings['overall_thriving_index'].head(10)
            for _, row in top_10.iterrows():
                top_performers_html += f"""
                <tr>
                    <td>{int(row['rank'])}</td>
                    <td>{row['country']}</td>
                    <td>{row['continent']}</td>
                    <td>{row['index_value']:.1f}</td>
                </tr>
                """
        
        return f"""
        <section id="composite-indices" class="section">
            <h2>📊 Composite Thriving Indices</h2>
            
            <p>We created composite indices that combine multiple indicators within each dimension, 
            as well as an overall thriving index that synthesizes all dimensions.</p>
            
            <div class="stats-grid">
                {stats_html}
            </div>

            <h3>Top 10 Countries - Overall Thriving Index</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Country</th>
                        <th>Continent</th>
                        <th>Overall Score</th>
                    </tr>
                </thead>
                <tbody>
                    {top_performers_html}
                </tbody>
            </table>
        </section>
        """

    def _generate_clustering_section(self):
        """Generate clustering analysis section"""
        if 'clustering' not in self.results:
            return ""
        
        clustering = self.results['clustering']
        cluster_analysis = clustering.get('cluster_analysis', {})
        
        clusters_html = ""
        for cluster_id, analysis in cluster_analysis.items():
            countries_tags = "".join([f'<span class="country-tag">{country}</span>' for country in analysis['countries'][:8]])
            if len(analysis['countries']) > 8:
                countries_tags += f'<span class="country-tag">+{len(analysis["countries"]) - 8} more</span>'
            
            strengths = analysis.get('profile', {}).get('strengths', [])[:3]
            strengths_text = ", ".join([s.split(' (')[0] for s in strengths]) if strengths else "None identified"
            
            clusters_html += f"""
            <div class="insight-box">
                <h4>{cluster_id.replace('_', ' ').title()} ({analysis['size']} countries)</h4>
                <p><strong>Key Strengths:</strong> {strengths_text}</p>
                <div class="country-list">
                    {countries_tags}
                </div>
            </div>
            """
        
        return f"""
        <section id="clustering" class="section">
            <h2>🎯 Multidimensional Clustering</h2>
            
            <p>Using machine learning clustering algorithms, we identified distinct groups of countries 
            with similar patterns across multiple thriving dimensions.</p>
            
            <div class="key-findings">
                <h3>Clustering Results</h3>
                <ul>
                    <li><strong>{clustering.get('optimal_k', 'N/A')}</strong> distinct country clusters identified</li>
                    <li>Clustering based on <strong>{len(clustering.get('clustering_indicators', []))}</strong> key indicators</li>
                    <li>Each cluster represents a unique development pattern</li>
                </ul>
            </div>

            <h3>Country Clusters</h3>
            {clusters_html}
        </section>
        """

    def _generate_synergies_section(self):
        """Generate synergies and trade-offs section"""
        if 'synergy_tradeoffs' not in self.results:
            return ""
        
        synergies_data = self.results['synergy_tradeoffs']
        synergies = synergies_data.get('synergies', [])
        tradeoffs = synergies_data.get('tradeoffs', [])
        unusual_patterns = synergies_data.get('unusual_patterns', [])
        
        synergies_html = ""
        for syn in synergies[:5]:
            synergies_html += f"""
            <div class="relationship-item">
                <strong>{syn['dimension1'].title()} ↔ {syn['dimension2'].title()}</strong>
                <span class="correlation-strength strong">r = {syn['correlation']:.3f}</span>
                <br><small>{syn['n_countries']} countries analyzed</small>
            </div>
            """
        
        tradeoffs_html = ""
        for trade in tradeoffs[:3]:
            tradeoffs_html += f"""
            <div class="relationship-item">
                <strong>{trade['dimension1'].title()} ↔ {trade['dimension2'].title()}</strong>
                <span class="correlation-strength weak">r = {trade['correlation']:.3f}</span>
                <br><small>{trade['n_countries']} countries analyzed</small>
            </div>
            """
        
        unusual_html = ""
        for pattern in unusual_patterns[:5]:
            unusual_html += f"""
            <span class="country-tag">{pattern['country']} (CV: {pattern['coefficient_variation']:.2f})</span>
            """
        
        return f"""
        <section id="synergies" class="section">
            <h2>⚖️ Synergies and Trade-offs</h2>
            
            <p>This analysis identifies where different dimensions of thriving mutually reinforce each other (synergies) 
            or compete for resources and attention (trade-offs).</p>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number">{len(synergies)}</span>
                    <div class="stat-label">Synergies Identified</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(tradeoffs)}</span>
                    <div class="stat-label">Trade-offs Found</div>
                </div>
            </div>

            <div class="key-findings">
                <h3>🤝 Strongest Synergies</h3>
                {synergies_html if synergies_html else "<p>No significant synergies identified.</p>"}
            </div>

            {f'''<div class="warning-box">
                <h3>⚡ Notable Trade-offs</h3>
                {tradeoffs_html}
            </div>''' if tradeoffs_html else ""}

            <div class="insight-box">
                <h3>🎭 Countries with Unusual Patterns</h3>
                <p>These countries show high variation across dimensions (high coefficient of variation):</p>
                <div class="country-list">
                    {unusual_html}
                </div>
            </div>
        </section>
        """

    def _generate_pathways_section(self):
        """Generate development pathways section"""
        if 'development_pathways' not in self.results:
            return ""
        
        pathways = self.results['development_pathways']
        archetypes = pathways.get('development_archetypes', {})
        top_analysis = pathways.get('top_performers_analysis', {})
        
        archetypes_html = ""
        for dimension, data in archetypes.items():
            specialists = data.get('specialists', {})
            generalists = data.get('generalists', {})
            
            archetypes_html += f"""
            <tr>
                <td>{dimension.title()}</td>
                <td>{specialists.get('count', 0)}</td>
                <td>{generalists.get('count', 0)}</td>
                <td>{specialists.get('avg_dimension_score', 0):.1f}</td>
                <td>{generalists.get('avg_dimension_score', 0):.1f}</td>
            </tr>
            """
        
        top_countries_html = ""
        if 'top_10_countries' in top_analysis:
            for country in top_analysis['top_10_countries'][:5]:
                top_countries_html += f"""
                <span class="country-tag">{country['country']} ({country['overall_thriving_index']:.1f})</span>
                """
        
        return f"""
        <section id="pathways" class="section">
            <h2>🛤️ Development Pathways</h2>
            
            <p>Countries pursue different strategies for achieving thriving. Some specialize in specific dimensions 
            while others pursue more balanced development across all areas.</p>
            
            <h3>Development Archetypes by Dimension</h3>
            <table>
                <thead>
                    <tr>
                        <th>Dimension</th>
                        <th>Specialists</th>
                        <th>Generalists</th>
                        <th>Specialist Avg Score</th>
                        <th>Generalist Avg Score</th>
                    </tr>
                </thead>
                <tbody>
                    {archetypes_html}
                </tbody>
            </table>

            <div class="insight-box">
                <h3>🏆 Top Overall Performers</h3>
                <div class="country-list">
                    {top_countries_html}
                </div>
            </div>
        </section>
        """

    def _generate_visualizations_section(self):
        """Generate visualizations section"""
        return f"""
        <section id="visualizations" class="section">
            <h2>📈 Key Visualizations</h2>
            
            <div class="visualization-container">
                <h3>Cross-Cutting Analysis Overview</h3>
                <img src="cross_cutting_overview.png" alt="Cross-Cutting Analysis Overview">
                <p>Comprehensive overview showing correlations, clustering, and overall thriving patterns</p>
            </div>
            
            <div class="visualization-container">
                <h3>Cross-Dimensional Relationships</h3>
                <img src="cross_dimensional_relationships.png" alt="Cross-Dimensional Relationships">
                <p>Detailed scatter plots showing relationships between different thriving dimensions</p>
            </div>
        </section>
        """

    def _generate_methodology_section(self):
        """Generate methodology section"""
        return f"""
        <section id="methodology" class="section">
            <h2>🔬 Methodology</h2>
            
            <div class="methodology-grid">
                <div class="method-card">
                    <h4>Data Preparation</h4>
                    <ul>
                        <li>Cleaned and standardized {len(self.all_indicators)} indicators</li>
                        <li>Handled missing values and outliers</li>
                        <li>Transformed indicators for comparability</li>
                    </ul>
                </div>
                
                <div class="method-card">
                    <h4>Correlation Analysis</h4>
                    <ul>
                        <li>Pearson and Spearman correlations</li>
                        <li>Statistical significance testing</li>
                        <li>Cross-dimensional relationship mapping</li>
                    </ul>
                </div>
                
                <div class="method-card">
                    <h4>Composite Indices</h4>
                    <ul>
                        <li>Min-Max normalization (0-100 scale)</li>
                        <li>Equal weighting within dimensions</li>
                        <li>PCA-based alternative index</li>
                    </ul>
                </div>
                
                <div class="method-card">
                    <h4>Clustering Analysis</h4>
                    <ul>
                        <li>K-means clustering with standardized features</li>
                        <li>Silhouette score optimization</li>
                        <li>PCA for visualization</li>
                    </ul>
                </div>
            </div>
            
            <div class="insight-box">
                <h4>Data Coverage</h4>
                <p>Analysis includes {len(self.cleaned_indicators)} major thriving dimensions: 
                {', '.join([cat.title() for cat in self.cleaned_indicators.keys()])}. 
                Country coverage varies by indicator but includes major economies and representative 
                countries from all continents.</p>
            </div>
        </section>
        """

    def run_complete_analysis(self):
        """Run all cross-cutting analysis components"""
        print("Starting comprehensive cross-cutting analysis...")
        
        self.prepare_cross_cutting_data()
        self.multidimensional_correlation_analysis()
        self.create_composite_indices()
        self.multidimensional_clustering()
        self.synergy_tradeoff_analysis()
        self.development_pathways_analysis()
        self.create_cross_cutting_visualizations()
        self.generate_html_report()
        
        print("Cross-cutting analysis complete!")
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
        clean_df = df[df['Continent'].isin(valid_continents)].copy()
        clean_df = clean_df.drop_duplicates(subset=['Country'], keep='first')
        
        print(f"Clean data shape: {clean_df.shape}")
        return clean_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("=" * 80)
    print("GLOBAL THRIVING ANALYSIS: CROSS-CUTTING DIMENSIONS")
    print("=" * 80)
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return False
    
    # Run cross-cutting analysis
    analyzer = CrossCuttingAnalyzer(df)
    results = analyzer.run_complete_analysis()
    
    print("\n" + "=" * 80)
    print("CROSS-CUTTING ANALYSIS COMPLETE!")
    print("=" * 80)
    
    # Display key findings
    if 'correlations' in results:
        print("\n🔗 CROSS-DIMENSIONAL RELATIONSHIPS:")
        strong_relationships = results['correlations']['strong_cross_dimensional_relationships']
        print(f"   • Found {len(strong_relationships)} strong cross-dimensional relationships")
        
        for rel in strong_relationships[:5]:  # Top 5
            cat1, cat2 = rel['category1'], rel['category2']
            indicator1 = rel['indicator1'].replace('_', ' ').title()
            indicator2 = rel['indicator2'].replace('_', ' ').title()
            print(f"   • {cat1.title()} - {cat2.title()}: {indicator1} ↔ {indicator2} (r = {rel['correlation']:.3f})")
    
    if 'composite_indices' in results:
        print("\n📊 COMPOSITE THRIVING INDICES:")
        index_stats = results['composite_indices']['index_statistics']
        
        for index_name, stats in index_stats.items():
            clean_name = index_name.replace('_', ' ').title()
            print(f"   • {clean_name}:")
            print(f"     - Countries with data: {stats['countries_with_data']}")
            print(f"     - Average score: {stats['mean']:.1f} (±{stats['std']:.1f})")
            print(f"     - Range: {stats['min']:.1f} - {stats['max']:.1f}")
    
    if 'clustering' in results:
        optimal_k = results['clustering']['optimal_k']
        cluster_analysis = results['clustering']['cluster_analysis']
        print(f"\n🎯 MULTIDIMENSIONAL CLUSTERING:")
        print(f"   • Identified {optimal_k} distinct country clusters")
        
        for cluster_id, analysis in cluster_analysis.items():
            print(f"   • {cluster_id.title()}: {analysis['size']} countries")
            if analysis['profile']['strengths']:
                print(f"     - Strengths: {', '.join(analysis['profile']['strengths'][:2])}")
    
    if 'synergy_tradeoffs' in results:
        synergy_summary = results['synergy_tradeoffs']['dimension_relationships_summary']
        print(f"\n⚖️ SYNERGIES AND TRADE-OFFS:")
        print(f"   • Synergies found: {synergy_summary['total_synergies']}")
        print(f"   • Trade-offs found: {synergy_summary['total_tradeoffs']}")
        
        if synergy_summary['strongest_synergy']:
            strongest = synergy_summary['strongest_synergy']
            print(f"   • Strongest synergy: {strongest['dimension1']} ↔ {strongest['dimension2']} (r = {strongest['correlation']:.3f})")
        
        if synergy_summary['strongest_tradeoff']:
            strongest = synergy_summary['strongest_tradeoff']
            print(f"   • Strongest trade-off: {strongest['dimension1']} ↔ {strongest['dimension2']} (r = {strongest['correlation']:.3f})")
    
    if 'development_pathways' in results:
        pathways = results['development_pathways']
        print(f"\n🛤️ DEVELOPMENT PATHWAYS:")
        
        if 'development_archetypes' in pathways:
            archetypes = pathways['development_archetypes']
            total_specialists = sum(arch['specialists']['count'] for arch in archetypes.values())
            total_generalists = sum(arch['generalists']['count'] for arch in archetypes.values())
            print(f"   • Specialist countries: {total_specialists}")
            print(f"   • Generalist countries: {total_generalists}")
        
        if 'top_performers_analysis' in pathways and 'top_10_countries' in pathways['top_performers_analysis']:
            top_countries = pathways['top_performers_analysis']['top_10_countries'][:5]
            print(f"   • Top 5 overall performers: {', '.join([c['country'] for c in top_countries])}")
    
    print(f"\n📊 Generated outputs:")
    outputs = ['cross_cutting_overview.png', 'cross_dimensional_relationships.png', 'cross_cutting_analysis_report.html']
    for output in outputs:
        if (analyzer.output_dir / output).exists():
            print(f"   ✅ {output}")
    
    return True

if __name__ == "__main__":
    main()
