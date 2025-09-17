"""
Generate Infrastructure Analysis HTML Report

This script creates a comprehensive HTML report for the infrastructure analysis,
integrating the statistical results with professional visualizations.
"""

import json
from pathlib import Path
from datetime import datetime
from infrastructure_analysis import InfrastructureAnalyzer, load_data

def generate_infrastructure_html_report(analyzer):
    """Generate comprehensive HTML report for infrastructure analysis"""

    results = analyzer.results
    output_dir = analyzer.output_dir

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Infrastructure Analysis - Global Thriving Study</title>
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
                border-bottom: 3px solid #8e44ad;
                padding-bottom: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #8e44ad;
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
                background-color: #8e44ad;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            .navigation a:hover {{
                background-color: #7d3c98;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: #f4f0f8;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #8e44ad;
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
                margin: 30px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .insights {{
                background-color: #e8f6f3;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #16a085;
                margin: 20px 0;
            }}
            .insights h3 {{
                color: #16a085;
                margin-top: 0;
            }}
            .correlation-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }}
            .correlation-table th, .correlation-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .correlation-table th {{
                background-color: #8e44ad;
                color: white;
            }}
            .correlation-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .cluster-info {{
                background-color: #fff3cd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }}
            .outlier-info {{
                background-color: #f8d7da;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #dc3545;
                margin: 15px 0;
            }}
            .methodology {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 30px 0;
            }}
            .country-list {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 10px;
                margin: 15px 0;
            }}
            .country-item {{
                background-color: #e9ecef;
                padding: 8px;
                border-radius: 4px;
                text-align: center;
                font-size: 0.9em;
            }}
            ul.key-findings {{
                background-color: #d1ecf1;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #17a2b8;
            }}
            ul.key-findings li {{
                margin: 10px 0;
            }}
            .metric-comparison {{
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .metric-comparison .metric-name {{
                font-weight: bold;
            }}
            .metric-comparison .metric-value {{
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏗️ Infrastructure Analysis - Global Thriving Study</h1>
            
            <div class="navigation">
                <a href="#overview">Overview</a>
                <a href="#regional">Regional Analysis</a>
                <a href="#correlations">Correlations</a>
                <a href="#clusters">Development Clusters</a>
                <a href="#outliers">Notable Patterns</a>
                <a href="#insights">Key Insights</a>
                <a href="#methodology">Methodology</a>
            </div>

            <div id="overview">
                <h2>📊 Infrastructure Overview</h2>
                <div class="stats-grid">
    """

    # Add overview statistics
    total_countries = len(analyzer.infrastructure_df)
    avg_access = analyzer.infrastructure_df['electricity_access_percent'].mean()
    median_access = analyzer.infrastructure_df['electricity_access_percent'].median()
    universal_access = len(analyzer.infrastructure_df[analyzer.infrastructure_df['electricity_access_percent'] >= 99])
    
    html_content += f"""
                    <div class="stat-card">
                        <span class="stat-number">{total_countries}</span>
                        <div class="stat-label">Countries Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{avg_access:.1f}%</span>
                        <div class="stat-label">Average Electricity Access</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{median_access:.1f}%</span>
                        <div class="stat-label">Median Electricity Access</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{universal_access}</span>
                        <div class="stat-label">Universal Access Countries</div>
                    </div>
    """

    html_content += """
                </div>
                
                <div class="chart-container">
                    <img src="infrastructure_overview.png" alt="Infrastructure Overview Visualization">
                </div>
                
                <div class="insights">
                    <h3>🔍 Overview Insights</h3>
                    <ul>
    """

    # Add overview insights
    if 'regional_analysis' in results:
        regional_stats = results['regional_analysis']['statistics']
        best_region = regional_stats.loc[regional_stats['avg_electricity_access'].idxmax()]
        worst_region = regional_stats.loc[regional_stats['avg_electricity_access'].idxmin()]
        
        html_content += f"""
                        <li><strong>Global Status:</strong> {avg_access:.1f}% average electricity access across {total_countries} countries</li>
                        <li><strong>Universal Access:</strong> {universal_access} countries achieve near-universal electricity access (≥99%)</li>
                        <li><strong>Regional Leaders:</strong> {best_region['continent']} leads with {best_region['avg_electricity_access']:.1f}% average access</li>
                        <li><strong>Development Gap:</strong> {best_region['avg_electricity_access'] - worst_region['avg_electricity_access']:.1f} percentage point gap between best and worst regions</li>
        """

    html_content += """
                    </ul>
                </div>
            </div>

            <div id="regional">
                <h2>🌍 Regional Analysis</h2>
    """

    # Add regional analysis
    if 'regional_analysis' in results:
        regional_stats = results['regional_analysis']['statistics']
        html_content += """
                <table class="correlation-table">
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Countries</th>
                            <th>Avg Access (%)</th>
                            <th>Median Access (%)</th>
                            <th>Universal Access</th>
                            <th>Avg GDP per Capita</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for _, row in regional_stats.iterrows():
            html_content += f"""
                        <tr>
                            <td>{row['continent']}</td>
                            <td>{row['countries']}</td>
                            <td>{row['avg_electricity_access']:.1f}%</td>
                            <td>{row['median_electricity_access']:.1f}%</td>
                            <td>{row['countries_universal_access']}</td>
                            <td>${row['avg_gdp']:,.0f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
        """

        # Add ANOVA results if available
        if 'anova_p_value' in results['regional_analysis'] and results['regional_analysis']['anova_p_value'] is not None:
            p_value = results['regional_analysis']['anova_p_value']
            f_stat = results['regional_analysis']['anova_f_stat']
            
            html_content += f"""
                <div class="insights">
                    <h3>📈 Statistical Analysis</h3>
                    <p><strong>ANOVA Test Results:</strong></p>
                    <ul>
                        <li>F-statistic: {f_stat:.3f}</li>
                        <li>P-value: {p_value:.6f}</li>
                        <li>Result: {'Significant regional differences' if p_value < 0.05 else 'No significant regional differences'} in electricity access (α = 0.05)</li>
                    </ul>
                </div>
            """

    html_content += """
            </div>

            <div id="correlations">
                <h2>🔗 Infrastructure Correlations</h2>
    """

    # Add correlation analysis
    if 'correlations' in results:
        correlations = results['correlations']['infrastructure_correlations']
        
        html_content += """
                <table class="correlation-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Correlation with Electricity Access</th>
                            <th>Strength</th>
                            <th>P-value</th>
                            <th>Significance</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for corr in correlations:
            significance = "✅ Significant" if corr['p_value'] < 0.05 else "❌ Not Significant"
            direction = "↗️" if corr['correlation'] > 0 else "↘️"
            
            html_content += f"""
                        <tr>
                            <td>{direction} {corr['variable'].replace('_', ' ').title()}</td>
                            <td>{corr['correlation']:.3f}</td>
                            <td>{corr['strength']}</td>
                            <td>{corr['p_value']:.3f}</td>
                            <td>{significance}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
                
                <div class="insights">
                    <h3>🔍 Correlation Insights</h3>
                    <ul>
        """
        
        strong_correlations = [c for c in correlations if abs(c['correlation']) > 0.6]
        html_content += f"<li><strong>Strong Relationships:</strong> Found {len(strong_correlations)} strong correlations with electricity access</li>"
        
        if correlations:
            strongest = max(correlations, key=lambda x: abs(x['correlation']))
            direction = "positively" if strongest['correlation'] > 0 else "negatively"
            html_content += f"<li><strong>Strongest Correlation:</strong> {strongest['variable'].replace('_', ' ').title()} is {direction} correlated (r = {strongest['correlation']:.3f})</li>"
        
        for corr in strong_correlations[:3]:
            direction = "increases" if corr['correlation'] > 0 else "decreases"
            html_content += f"<li><strong>{corr['variable'].replace('_', ' ').title()}:</strong> As electricity access improves, {corr['variable'].replace('_', ' ')} typically {direction}</li>"
        
        html_content += """
                    </ul>
                </div>
    """

    html_content += """
            </div>

            <div id="clusters">
                <h2>🎯 Infrastructure Development Clusters</h2>
    """

    # Add clustering analysis
    if 'clustering' in results:
        clustering_info = results['clustering']
        
        html_content += f"""
                <div class="cluster-info">
                    <h3>Clustering Overview</h3>
                    <p><strong>Number of Clusters:</strong> {clustering_info['optimal_k']}</p>
                    <p><strong>Clustering Quality (Silhouette Score):</strong> {max(clustering_info['silhouette_scores']):.3f}</p>
                </div>
        """
        
        for summary in clustering_info['cluster_summary']:
            html_content += f"""
                <div class="cluster-info">
                    <h3>Cluster {summary['cluster_id']} - {summary['size']} Countries</h3>
                    <div class="metric-comparison">
                        <span class="metric-name">Average Electricity Access:</span>
                        <span class="metric-value">{summary['avg_electricity_access']:.1f}%</span>
                    </div>
                    <div class="metric-comparison">
                        <span class="metric-name">Average GDP per Capita:</span>
                        <span class="metric-value">${summary['avg_gdp']:,.0f}</span>
                    </div>
                    <div class="metric-comparison">
                        <span class="metric-name">Average Life Satisfaction:</span>
                        <span class="metric-value">{summary['avg_life_satisfaction']:.2f}</span>
                    </div>
                    <div class="metric-comparison">
                        <span class="metric-name">Average Life Expectancy:</span>
                        <span class="metric-value">{summary['avg_life_expectancy']:.1f} years</span>
                    </div>
                    <div class="metric-comparison">
                        <span class="metric-name">Average Education Years:</span>
                        <span class="metric-value">{summary['avg_education']:.1f} years</span>
                    </div>
                    
                    <h4>Countries in this cluster:</h4>
                    <div class="country-list">
            """
            
            for country in summary['countries']:
                html_content += f'<div class="country-item">{country}</div>'
            
            html_content += """
                    </div>
                </div>
            """

    html_content += """
            </div>

            <div id="outliers">
                <h2>🎯 Notable Infrastructure Patterns</h2>
    """

    # Add outlier analysis
    if 'outliers' in results:
        outliers = results['outliers']
        
        # Group outliers by type
        outlier_types = {}
        for outlier in outliers:
            if outlier['type'] not in outlier_types:
                outlier_types[outlier['type']] = []
            outlier_types[outlier['type']].append(outlier)
        
        for outlier_type, type_outliers in outlier_types.items():
            html_content += f"""
                <div class="outlier-info">
                    <h3>{outlier_type}</h3>
                    <ul>
            """
            
            for outlier in type_outliers[:5]:  # Show top 5 per type
                html_content += f"<li><strong>{outlier['country']}:</strong> {outlier['description']}</li>"
            
            html_content += """
                    </ul>
                </div>
            """

        html_content += """
                <div class="chart-container">
                    <img src="infrastructure_outliers.png" alt="Infrastructure Outliers Analysis">
                </div>
        """

    html_content += """
            </div>

            <div id="insights">
                <h2>💡 Key Insights</h2>
                <ul class="key-findings">
    """

    # Generate key insights
    top_countries = analyzer.infrastructure_df.nlargest(3, 'electricity_access_percent')
    bottom_countries = analyzer.infrastructure_df.nsmallest(3, 'electricity_access_percent')
    
    html_content += f"""
                    <li><strong>Global Infrastructure Status:</strong> Average electricity access stands at {avg_access:.1f}%, with {universal_access} countries achieving near-universal access</li>
                    <li><strong>Top Performers:</strong> {', '.join(top_countries['country'].tolist())} lead in electricity access</li>
                    <li><strong>Improvement Opportunities:</strong> {', '.join(bottom_countries['country'].tolist())} have the lowest electricity access rates</li>
    """

    if 'correlations' in results and results['correlations']['infrastructure_correlations']:
        strongest_corr = max(results['correlations']['infrastructure_correlations'], key=lambda x: abs(x['correlation']))
        html_content += f"""
                    <li><strong>Key Relationship:</strong> Electricity access shows strongest correlation with {strongest_corr['variable'].replace('_', ' ')} (r = {strongest_corr['correlation']:.3f})</li>
        """

    if 'clustering' in results:
        html_content += f"""
                    <li><strong>Development Patterns:</strong> Countries group into {results['clustering']['optimal_k']} distinct infrastructure development clusters</li>
        """

    if 'outliers' in results:
        high_access_low_gdp = [o for o in results['outliers'] if o['type'] == 'High Access, Low GDP']
        if high_access_low_gdp:
            html_content += f"""
                    <li><strong>Efficiency Examples:</strong> {len(high_access_low_gdp)} countries achieve high electricity access despite lower GDP levels</li>
            """

    html_content += """
                </ul>
            </div>

            <div id="methodology">
                <h2>🔬 Methodology</h2>
                <div class="methodology">
                    <h3>Data Sources & Indicators</h3>
                    <ul>
                        <li><strong>Primary Infrastructure Indicator:</strong> Percentage of population with access to electricity (2020)</li>
                        <li><strong>Supporting Indicators:</strong> Number of people with electricity access, GDP per capita, life satisfaction, life expectancy, education years</li>
                        <li><strong>Geographic Coverage:</strong> Global analysis across major continents</li>
                    </ul>
                    
                    <h3>Analytical Methods</h3>
                    <ul>
                        <li><strong>Regional Analysis:</strong> ANOVA testing for significant regional differences</li>
                        <li><strong>Correlation Analysis:</strong> Pearson correlation coefficients with significance testing</li>
                        <li><strong>Clustering Analysis:</strong> K-means clustering with silhouette score optimization</li>
                        <li><strong>Outlier Detection:</strong> Statistical and logical outlier identification</li>
                        <li><strong>Infrastructure Metrics:</strong> Custom efficiency and gap calculations</li>
                    </ul>
                    
                    <h3>Infrastructure Categories</h3>
                    <ul>
                        <li><strong>Universal Access:</strong> ≥99% electricity access</li>
                        <li><strong>High Access:</strong> 90-98% electricity access</li>
                        <li><strong>Medium Access:</strong> 70-89% electricity access</li>
                        <li><strong>Low Access:</strong> 50-69% electricity access</li>
                        <li><strong>Very Low Access:</strong> <50% electricity access</li>
                    </ul>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                <p><strong>Infrastructure Analysis Report</strong></p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Global Thriving Study - Infrastructure Dimension</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save the HTML report
    output_path = output_dir / "infrastructure_analysis.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Infrastructure HTML report saved to: {output_path}")
    return output_path

def main():
    """Generate the infrastructure analysis HTML report"""
    print("="*80)
    print("GENERATING INFRASTRUCTURE ANALYSIS HTML REPORT")
    print("="*80)
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ Failed to load data")
        return
    
    # Create analyzer and run analysis
    analyzer = InfrastructureAnalyzer(df)
    
    # Check if analysis results exist, if not run the analysis
    if not analyzer.results:
        print("Running infrastructure analysis first...")
        analyzer.run_complete_analysis()
    
    # Generate HTML report
    html_path = generate_infrastructure_html_report(analyzer)
    
    print(f"\n✅ Infrastructure HTML report generated successfully!")
    print(f"📄 Report saved to: {html_path}")
    print(f"🌐 Open {html_path} in your browser to view the report")

if __name__ == "__main__":
    main()