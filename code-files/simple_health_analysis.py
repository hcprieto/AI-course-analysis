"""
Simplified Health and Well-being Analysis

This version performs the core statistical analysis without heavy visualization dependencies,
generating HTML reports with summary statistics and findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def load_and_clean_data_simple(source_path):
    """Simplified data loading function"""
    try:
        # Load the CSV data
        print(f"Loading data from {source_path}...")
        df = pd.read_csv(source_path)
        print(f"Raw data shape: {df.shape}")

        # Get the first few rows to understand structure
        print("Sample data:")
        print(df.head(2))

        # Basic cleaning - get countries with valid continent data
        valid_continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']

        # Filter to rows with valid continents
        clean_df = df[df['Continent'].isin(valid_continents)].copy()

        # Remove duplicates by country
        clean_df = clean_df.drop_duplicates(subset=['Country'], keep='first')

        print(f"Clean data shape: {clean_df.shape}")
        print(f"Continents found: {clean_df['Continent'].value_counts().to_dict()}")

        return clean_df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_health_indicators(df):
    """Perform health indicator analysis"""
    print("\nAnalyzing health indicators...")

    # Define the health-related columns (using original column names)
    health_columns = {
        'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction',
        'Life expectancy at birth (years)(2018)': 'life_expectancy',
        'Deaths Due to Self-Harm (2019)': 'self_harm_deaths',
        'Most Common Cause of Death (2018)': 'death_cause_1',
        'Second Most Common Cause of Death (2018)': 'death_cause_2',
        'Third Most Common Cause of Death (2018)': 'death_cause_3'
    }

    results = {
        'total_countries': len(df),
        'continents': df['Continent'].value_counts().to_dict(),
        'indicators': {}
    }

    # Analyze numeric indicators
    numeric_indicators = ['life_satisfaction', 'life_expectancy', 'self_harm_deaths']

    for orig_col, clean_name in health_columns.items():
        if orig_col in df.columns and clean_name in numeric_indicators:
            # Convert to numeric, handling errors
            numeric_data = pd.to_numeric(df[orig_col], errors='coerce')
            valid_data = numeric_data.dropna()

            if len(valid_data) > 0:
                stats = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'median': float(valid_data.median()),
                    'std': float(valid_data.std()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'q25': float(valid_data.quantile(0.25)),
                    'q75': float(valid_data.quantile(0.75))
                }
                results['indicators'][clean_name] = stats

                print(f"{clean_name}: {stats['count']} countries, mean = {stats['mean']:.2f}")

    # Analyze correlations if we have life satisfaction and life expectancy
    if 'Life satisfaction in Cantril Ladder (2018)' in df.columns and 'Life expectancy at birth (years)(2018)' in df.columns:
        # Create subset with both variables
        subset = df[['Country', 'Continent',
                    'Life satisfaction in Cantril Ladder (2018)',
                    'Life expectancy at birth (years)(2018)']].copy()

        # Convert to numeric
        subset['life_sat'] = pd.to_numeric(subset['Life satisfaction in Cantril Ladder (2018)'], errors='coerce')
        subset['life_exp'] = pd.to_numeric(subset['Life expectancy at birth (years)(2018)'], errors='coerce')

        # Remove rows with missing data
        subset = subset.dropna(subset=['life_sat', 'life_exp'])

        if len(subset) > 3:
            correlation = subset['life_sat'].corr(subset['life_exp'])
            results['correlation_life_sat_life_exp'] = {
                'correlation': float(correlation),
                'n_countries': len(subset),
                'r_squared': float(correlation**2)
            }
            print(f"Life satisfaction vs Life expectancy: r = {correlation:.3f} (n = {len(subset)})")

    # Analyze regional differences
    regional_analysis = {}
    for orig_col, clean_name in health_columns.items():
        if orig_col in df.columns and clean_name in numeric_indicators:
            # Group by continent
            continent_stats = df.groupby('Continent')[orig_col].agg(['count', 'mean', 'std']).round(2)
            regional_analysis[clean_name] = continent_stats.to_dict('index')

    results['regional_analysis'] = regional_analysis

    # Analyze mortality patterns
    mortality_analysis = {}
    death_causes = []

    for col in ['Most Common Cause of Death (2018)', 'Second Most Common Cause of Death (2018)', 'Third Most Common Cause of Death (2018)']:
        if col in df.columns:
            causes = df[col].dropna().tolist()
            death_causes.extend(causes)

    if death_causes:
        cause_counts = pd.Series(death_causes).value_counts()
        mortality_analysis['top_causes'] = cause_counts.head(10).to_dict()

        # By continent
        mortality_by_continent = {}
        for continent in df['Continent'].unique():
            if pd.notna(continent):
                continent_df = df[df['Continent'] == continent]
                continent_causes = []
                for col in ['Most Common Cause of Death (2018)', 'Second Most Common Cause of Death (2018)', 'Third Most Common Cause of Death (2018)']:
                    if col in continent_df.columns:
                        causes = continent_df[col].dropna().tolist()
                        continent_causes.extend(causes)

                if continent_causes:
                    mortality_by_continent[continent] = pd.Series(continent_causes).value_counts().head(5).to_dict()

        mortality_analysis['by_continent'] = mortality_by_continent

    results['mortality_analysis'] = mortality_analysis

    return results

def generate_simple_html_report(results, output_dir):
    """Generate a simple HTML report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Health and Well-being Analysis - Global Thriving Study</title>
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
                max-width: 1200px;
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
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 25px;
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
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #bdc3c7;
                padding: 12px;
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
            <h1>Health and Well-being Analysis</h1>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

            <div class="highlight">
                <h3>Key Research Question</h3>
                <p>To what extent are global citizens thriving in terms of health and well-being?
                How do health outcomes, life satisfaction, and mortality patterns compare across countries?</p>
            </div>

            <h2>Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number">{results['total_countries']}</span>
                    <div class="stat-label">Countries Analyzed</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(results['continents'])}</span>
                    <div class="stat-label">Continents</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(results['indicators'])}</span>
                    <div class="stat-label">Health Indicators</div>
                </div>
    """

    # Add correlation if available
    if 'correlation_life_sat_life_exp' in results:
        corr_data = results['correlation_life_sat_life_exp']
        html_content += f"""
                <div class="stat-card">
                    <span class="stat-number">{corr_data['correlation']:.3f}</span>
                    <div class="stat-label">Life Satisfaction ↔ Life Expectancy</div>
                </div>
        """

    html_content += """
            </div>

            <h2>Health Indicator Statistics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th>Countries</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add indicator statistics
    for indicator, stats in results['indicators'].items():
        clean_name = indicator.replace('_', ' ').title()
        html_content += f"""
                    <tr>
                        <td>{clean_name}</td>
                        <td>{stats['count']}</td>
                        <td>{stats['mean']:.2f}</td>
                        <td>{stats['median']:.2f}</td>
                        <td>{stats['std']:.2f}</td>
                        <td>{stats['min']:.2f}</td>
                        <td>{stats['max']:.2f}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
    """

    # Add correlation analysis if available
    if 'correlation_life_sat_life_exp' in results:
        corr_data = results['correlation_life_sat_life_exp']
        strength = "Strong" if abs(corr_data['correlation']) > 0.7 else "Moderate" if abs(corr_data['correlation']) > 0.3 else "Weak"

        html_content += f"""
            <h2>Key Relationship: Life Satisfaction vs Life Expectancy</h2>
            <div class="success">
                <h3>Correlation Analysis</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{corr_data['correlation']:.3f}</span>
                        <div class="stat-label">Pearson Correlation</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{corr_data['r_squared']:.3f}</span>
                        <div class="stat-label">R-squared</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{strength}</span>
                        <div class="stat-label">Correlation Strength</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{corr_data['n_countries']}</span>
                        <div class="stat-label">Countries with Data</div>
                    </div>
                </div>
                <p><strong>Interpretation:</strong> There is a {strength.lower()} correlation between life satisfaction and life expectancy,
                explaining {corr_data['r_squared']*100:.1f}% of the variance across {corr_data['n_countries']} countries.</p>
            </div>
        """

    # Add continental breakdown
    html_content += """
            <h2>Countries by Continent</h2>
            <table>
                <thead>
                    <tr>
                        <th>Continent</th>
                        <th>Number of Countries</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
    """

    total_countries = results['total_countries']
    for continent, count in sorted(results['continents'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_countries) * 100
        html_content += f"""
                    <tr>
                        <td>{continent}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
    """

    # Add mortality analysis if available
    if 'mortality_analysis' in results and 'top_causes' in results['mortality_analysis']:
        html_content += """
            <h2>Leading Causes of Death Globally</h2>
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

        for rank, (cause, count) in enumerate(results['mortality_analysis']['top_causes'].items(), 1):
            html_content += f"""
                        <tr>
                            <td>{rank}</td>
                            <td>{cause}</td>
                            <td>{count}</td>
                        </tr>
            """

        html_content += """
                </tbody>
            </table>
        """

    # Close HTML
    html_content += f"""
            <div class="footer">
                <p>Generated by Global Thriving Analysis System</p>
                <p>Data Source: World Data 2.0-Data.csv</p>
                <p>Analysis completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save the HTML file
    html_file = output_path / "health_wellbeing.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report saved to: {html_file}")
    return html_file

def main():
    print("=" * 60)
    print("GLOBAL THRIVING ANALYSIS: HEALTH & WELL-BEING")
    print("=" * 60)

    # Define paths
    source_path = "../source-files/World Data 2.0-Data.csv"
    output_dir = "../analysis-output-files"

    # Load data
    df = load_and_clean_data_simple(source_path)
    if df is None:
        print("❌ Failed to load data")
        return False

    # Analyze health indicators
    results = analyze_health_indicators(df)

    # Generate HTML report
    html_file = generate_simple_html_report(results, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📊 Countries analyzed: {results['total_countries']}")
    print(f"🌍 Continents: {', '.join(results['continents'].keys())}")
    print(f"📈 Health indicators: {len(results['indicators'])}")

    if 'correlation_life_sat_life_exp' in results:
        corr = results['correlation_life_sat_life_exp']
        print(f"🔗 Life Satisfaction ↔ Life Expectancy: r = {corr['correlation']:.3f}")

    print(f"\n🌐 Open this file in your browser:")
    print(f"    {html_file.absolute()}")

    return True

if __name__ == "__main__":
    main()