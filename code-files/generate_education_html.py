"""
Generate Education Analysis HTML Report

This script creates a comprehensive HTML report for the education analysis,
integrating the statistical results with professional visualizations.
"""

import json
from pathlib import Path
from datetime import datetime
from education_analysis import EducationAnalyzer, load_data

def generate_education_html_report(analyzer):
    """Generate comprehensive HTML report for education analysis"""

    results = analyzer.results
    output_dir = analyzer.output_dir

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Education Analysis - Global Thriving Study</title>
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
                border-bottom: 3px solid #e67e22;
                padding-bottom: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #e67e22;
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
                background-color: #e67e22;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }}
            .navigation a:hover {{
                background-color: #d35400;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: #fdf2e9;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #e67e22;
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
                background-color: #e67e22;
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
            .ranking-table {{
                background-color: #fdf2e9;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
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
                <a href="enhanced_health_analysis.html">🏥 Health Analysis</a>
                <a href="#executive-summary">Executive Summary</a>
                <a href="#overview">Education Overview</a>
                <a href="#regional">Regional Analysis</a>
                <a href="#correlations">Correlations</a>
                <a href="#outliers">Outliers</a>
            </div>

            <h1>📚 Education Analysis: Learning-Adjusted Years of School</h1>
            <p style="text-align: center;"><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

            <div class="highlight">
                <h3>Key Research Questions</h3>
                <ul>
                    <li>What is the average number of learning-adjusted school years across countries?</li>
                    <li>How does educational attainment correlate with GDP per capita and life expectancy?</li>
                    <li>Do countries with higher education levels also report higher life satisfaction?</li>
                    <li>Are there significant regional differences in educational outcomes?</li>
                </ul>
            </div>

            <section id="executive-summary">
                <h2>Executive Summary</h2>
                {generate_education_summary(results)}
            </section>

            <section id="overview">
                <h2>Education Overview</h2>
                <p>This analysis examines learning-adjusted years of schooling across {results.get('univariate', {}).get('count', 0)} countries,
                providing insights into global educational attainment patterns and their relationships with development indicators.</p>

                <div class="chart-container">
                    <div class="chart-title">Education Overview: Distributions and Rankings</div>
                    <img src="education_overview.png" alt="Education Overview">
                </div>

                {generate_univariate_education_section(results)}
            </section>

            <section id="regional">
                <h2>Regional Analysis</h2>
                <p>Regional analysis reveals significant differences in educational attainment across continents,
                with ANOVA testing showing statistical significance in these variations.</p>

                {generate_regional_education_section(results)}
            </section>

            <section id="correlations">
                <h2>Education Correlations with Development Indicators</h2>
                <p>Education shows strong relationships with multiple development indicators,
                suggesting its central role in human thriving and national development.</p>

                <div class="chart-container">
                    <div class="chart-title">Education Correlations Analysis</div>
                    <img src="education_correlations.png" alt="Education Correlations">
                </div>

                {generate_correlation_education_section(results)}
            </section>

            <section id="outliers">
                <h2>Education Outliers and Performance Analysis</h2>
                <p>Outlier analysis identifies countries with exceptional educational performance,
                both high and low, providing insights into educational success factors and challenges.</p>

                <div class="chart-container">
                    <div class="chart-title">Education Outlier Analysis</div>
                    <img src="education_outliers.png" alt="Education Outliers">
                </div>

                {generate_outlier_education_section(results)}
            </section>

            <div class="footer">
                <p>Generated by Global Thriving Analysis System - Education Module</p>
                <p>Data Source: World Data 2.0-Data.csv | Learning-Adjusted Years of School (2020)</p>
                <p>Analysis completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                <p>Statistical methods: ANOVA, correlation analysis, outlier detection, regression analysis</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save the HTML file
    html_file = output_dir / "education_analysis.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Education HTML report saved to: {html_file}")
    return html_file

def generate_education_summary(results):
    """Generate executive summary section"""
    if 'univariate' not in results:
        return "<p>Analysis results not available.</p>"

    stats = results['univariate']

    summary_html = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-number">{stats['count']}</span>
            <div class="stat-label">Countries Analyzed</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">{stats['mean']:.2f}</span>
            <div class="stat-label">Global Average (Years)</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">{stats['max']:.1f}</span>
            <div class="stat-label">Highest Achievement</div>
        </div>
        <div class="stat-card">
            <span class="stat-number">{stats['min']:.1f}</span>
            <div class="stat-label">Lowest Achievement</div>
        </div>
    """

    if 'correlations' in results:
        strong_corrs = sum(1 for rel, data in results['correlations'].items()
                          if isinstance(data, dict) and data.get('correlation_strength') == 'Strong')
        summary_html += f"""
        <div class="stat-card">
            <span class="stat-number">{strong_corrs}</span>
            <div class="stat-label">Strong Correlations Found</div>
        </div>
        """

    if 'regional' in results and results['regional'].get('significant_difference'):
        summary_html += f"""
        <div class="stat-card">
            <span class="stat-number">Significant</span>
            <div class="stat-label">Regional Differences</div>
        </div>
        """

    summary_html += "</div>"

    # Key insights
    summary_html += f"""
    <div class="success">
        <h3>Key Findings</h3>
        <ul>
            <li><strong>Global Education:</strong> Average of {stats['mean']:.2f} learning-adjusted years across {stats['count']} countries</li>
            <li><strong>Range:</strong> Substantial variation from {stats['min']:.1f} to {stats['max']:.1f} years</li>
            <li><strong>Distribution:</strong> Education is {'normally' if stats.get('is_normal', False) else 'not normally'} distributed across countries</li>
    """

    if 'correlations' in results:
        for rel, data in results['correlations'].items():
            if rel.startswith('education_vs_') and isinstance(data, dict) and data.get('significant'):
                var_name = rel.replace('education_vs_', '').replace('_', ' ').title()
                summary_html += f"<li><strong>Strong Correlation:</strong> Education and {var_name} (r = {data['pearson_r']:.3f})</li>"

    summary_html += """
        </ul>
    </div>
    """

    return summary_html

def generate_univariate_education_section(results):
    """Generate univariate statistics section"""
    if 'univariate' not in results:
        return ""

    stats = results['univariate']
    categories = stats.get('education_categories', {})

    section_html = f"""
    <h3>Descriptive Statistics</h3>
    <table>
        <thead>
            <tr>
                <th>Statistic</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Mean</td>
                <td>{stats['mean']:.2f} years</td>
                <td>Global average educational attainment</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{stats['median']:.2f} years</td>
                <td>Middle value when countries are ranked</td>
            </tr>
            <tr>
                <td>Standard Deviation</td>
                <td>{stats['std']:.2f} years</td>
                <td>Measure of variability across countries</td>
            </tr>
            <tr>
                <td>Range</td>
                <td>{stats['min']:.1f} - {stats['max']:.1f} years</td>
                <td>Span from lowest to highest achieving country</td>
            </tr>
            <tr>
                <td>Skewness</td>
                <td>{stats['skewness']:.3f}</td>
                <td>{'Right-skewed' if stats['skewness'] > 0.5 else 'Left-skewed' if stats['skewness'] < -0.5 else 'Approximately symmetric'} distribution</td>
            </tr>
        </tbody>
    </table>

    <h3>Education Level Categories</h3>
    <div class="ranking-table">
        <p>Countries are categorized into three education levels based on learning-adjusted years:</p>
        <ul>
            <li><strong>Low Education</strong> (≤{categories.get('low_education_threshold', 0):.1f} years): {categories.get('low_education_count', 0)} countries</li>
            <li><strong>Medium Education</strong> ({categories.get('low_education_threshold', 0):.1f} - {categories.get('high_education_threshold', 0):.1f} years): {categories.get('medium_education_count', 0)} countries</li>
            <li><strong>High Education</strong> (>{categories.get('high_education_threshold', 0):.1f} years): {categories.get('high_education_count', 0)} countries</li>
        </ul>
    </div>
    """

    return section_html

def generate_regional_education_section(results):
    """Generate regional analysis section"""
    if 'regional' not in results:
        return ""

    regional = results['regional']

    section_html = f"""
    <div class="{'success' if regional.get('significant_difference') else 'alert'}">
        <h3>ANOVA Results</h3>
        <p><strong>Statistical Test:</strong> {'Significant' if regional.get('significant_difference') else 'Not significant'}
        differences between continents (F = {regional.get('anova_stat', 0):.3f}, p = {regional.get('anova_p', 1):.4f})</p>
        <p><strong>Effect Size:</strong> η² = {regional.get('eta_squared', 0):.3f}
        ({'Large' if regional.get('eta_squared', 0) > 0.14 else 'Medium' if regional.get('eta_squared', 0) > 0.06 else 'Small'} effect)</p>
    </div>
    """

    if 'regional_stats' in regional:
        stats_df = regional['regional_stats']

        # Sort by mean education years
        sorted_regions = sorted(stats_df.items(), key=lambda x: x[1]['mean'], reverse=True)

        section_html += f"""
        <h3>Regional Educational Attainment Ranking</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Continent</th>
                    <th>Countries</th>
                    <th>Mean Years</th>
                    <th>Std Dev</th>
                    <th>Range</th>
                </tr>
            </thead>
            <tbody>
        """

        for rank, (continent, stats) in enumerate(sorted_regions, 1):
            section_html += f"""
                <tr>
                    <td>{rank}</td>
                    <td><strong>{continent}</strong></td>
                    <td>{stats['count']}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['min']:.1f} - {stats['max']:.1f}</td>
                </tr>
            """

        section_html += """
            </tbody>
        </table>
        """

        # Regional insights
        section_html += f"""
        <div class="highlight">
            <h4>Regional Insights</h4>
            <ul>
                <li><strong>Highest:</strong> {sorted_regions[0][0]} leads with {sorted_regions[0][1]['mean']:.2f} years average</li>
                <li><strong>Lowest:</strong> {sorted_regions[-1][0]} has {sorted_regions[-1][1]['mean']:.2f} years average</li>
                <li><strong>Gap:</strong> {sorted_regions[0][1]['mean'] - sorted_regions[-1][1]['mean']:.1f} years difference between highest and lowest regions</li>
            </ul>
        </div>
        """

    return section_html

def generate_correlation_education_section(results):
    """Generate correlation analysis section"""
    if 'correlations' not in results:
        return ""

    section_html = """
    <h3>Correlation Results</h3>
    <table>
        <thead>
            <tr>
                <th>Relationship</th>
                <th>Correlation (r)</th>
                <th>R²</th>
                <th>Strength</th>
                <th>Significance</th>
                <th>Interpretation</th>
            </tr>
        </thead>
        <tbody>
    """

    for relationship, stats in results['correlations'].items():
        if relationship.startswith('education_vs_') and isinstance(stats, dict):
            var_name = relationship.replace('education_vs_', '').replace('_', ' ').title()
            r_value = stats.get('pearson_r', 0)
            r_squared = stats.get('r_squared', 0)
            strength = stats.get('correlation_strength', 'Unknown')
            significant = stats.get('significant', False)

            # Determine row class based on strength
            if strength == 'Strong':
                row_class = 'correlation-strong'
            elif strength == 'Moderate':
                row_class = 'correlation-moderate'
            else:
                row_class = 'correlation-weak'

            # Generate interpretation
            direction = "positive" if r_value > 0 else "negative"
            interpretation = f"{strength} {direction} relationship"

            section_html += f"""
                <tr class="{row_class}">
                    <td>Education ↔ {var_name}</td>
                    <td>{r_value:.3f}</td>
                    <td>{r_squared:.3f}</td>
                    <td>{strength}</td>
                    <td>{'Significant' if significant else 'Not Significant'}</td>
                    <td>{interpretation}</td>
                </tr>
            """

    section_html += """
        </tbody>
    </table>
    """

    # Key correlations insights
    section_html += """
    <div class="success">
        <h4>Key Correlation Insights</h4>
        <ul>
    """

    for relationship, stats in results['correlations'].items():
        if relationship.startswith('education_vs_') and isinstance(stats, dict) and stats.get('significant'):
            var_name = relationship.replace('education_vs_', '').replace('_', ' ').title()
            r_value = stats.get('pearson_r', 0)
            r_squared = stats.get('r_squared', 0)
            strength = stats.get('correlation_strength', 'Unknown')

            section_html += f"""
            <li><strong>{var_name}:</strong> {strength} correlation (r = {r_value:.3f}) -
            Education explains {r_squared*100:.1f}% of the variance in {var_name.lower()}</li>
            """

    section_html += """
        </ul>
    </div>
    """

    return section_html

def generate_outlier_education_section(results):
    """Generate outlier analysis section"""
    if 'outliers' not in results:
        return ""

    outliers = results['outliers']

    section_html = "<h3>Education Performance Outliers</h3>"

    # High performers
    high_performers = outliers.get('outlier_countries', {}).get('high_education', [])
    if high_performers:
        section_html += f"""
        <div class="success">
            <h4>🌟 High-Performing Countries</h4>
            <p><strong>Exceptional Educational Achievement:</strong> {', '.join(high_performers)}</p>
            <p>These countries demonstrate learning-adjusted years of schooling significantly above the global average.</p>
        </div>
        """

    # Low performers
    low_performers = outliers.get('outlier_countries', {}).get('low_education', [])
    if low_performers:
        section_html += f"""
        <div class="alert">
            <h4>⚠️ Countries Needing Educational Support</h4>
            <p><strong>Below-Average Educational Attainment:</strong> {', '.join(low_performers)}</p>
            <p>These countries show learning-adjusted years of schooling significantly below the global average,
            indicating potential areas for educational development assistance.</p>
        </div>
        """

    # Statistical bounds
    bounds = outliers.get('bounds', {})
    section_html += f"""
    <h4>Statistical Thresholds</h4>
    <div class="ranking-table">
        <ul>
            <li><strong>Global Mean:</strong> {bounds.get('mean', 0):.2f} years</li>
            <li><strong>Standard Deviation:</strong> {bounds.get('std', 0):.2f} years</li>
            <li><strong>IQR Bounds:</strong> {bounds.get('iqr_lower', 0):.2f} - {bounds.get('iqr_upper', 0):.2f} years</li>
        </ul>
        <p><em>Outliers are identified using Z-score methodology (|z| > 2.5) and IQR analysis.</em></p>
    </div>
    """

    return section_html

def main():
    """Generate education HTML report"""
    print("Generating Education Analysis HTML Report...")

    # Load data and run analysis
    df = load_data()
    if df is None:
        print("❌ Failed to load data for HTML generation")
        return False

    # Create analyzer and get results
    analyzer = EducationAnalyzer(df)
    results = analyzer.run_complete_analysis()

    # Generate HTML report
    html_file = generate_education_html_report(analyzer)

    print(f"🌐 Education analysis report available at:")
    print(f"    {html_file.absolute()}")

    return True

if __name__ == "__main__":
    main()