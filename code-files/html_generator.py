"""
HTML Report Generator for Global Thriving Analysis

This module generates comprehensive HTML reports for the analysis results,
creating linked pages with visualizations and interactive content.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class HTMLReportGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_css_style(self):
        """Generate CSS styling for the reports"""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }
            h3 {
                color: #2c3e50;
                margin-top: 25px;
            }
            .navigation {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }
            .navigation a {
                display: inline-block;
                margin: 5px 10px 5px 0;
                padding: 8px 15px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .navigation a:hover {
                background-color: #2980b9;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .stat-card {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #3498db;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
                display: block;
            }
            .stat-label {
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 5px;
            }
            .table-container {
                overflow-x: auto;
                margin: 20px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #bdc3c7;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #34495e;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .image-container {
                text-align: center;
                margin: 30px 0;
            }
            .image-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .highlight {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }
            .alert {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }
            .success {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
            }
            .correlation-strong { background-color: #d4edda; }
            .correlation-moderate { background-color: #fff3cd; }
            .correlation-weak { background-color: #f8d7da; }
            .footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
            }
        </style>
        """

    def generate_navigation(self, current_page=""):
        """Generate navigation menu"""
        pages = [
            ("index.html", "Home"),
            ("health_wellbeing.html", "Health & Well-being"),
            ("data_overview.html", "Data Overview"),
            ("methodology.html", "Methodology")
        ]

        nav_html = '<div class="navigation">'
        for url, title in pages:
            active_class = ' style="background-color: #2c3e50;"' if current_page == url else ''
            nav_html += f'<a href="{url}"{active_class}>{title}</a>'
        nav_html += '</div>'

        return nav_html

    def generate_health_wellbeing_report(self, analyzer):
        """Generate the Health and Well-being analysis report"""
        results = analyzer.results
        health_data = analyzer.health_data

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Health and Well-being Analysis - Global Thriving Study</title>
            {self.generate_css_style()}
        </head>
        <body>
            <div class="container">
                {self.generate_navigation("health_wellbeing.html")}

                <h1>Health and Well-being Analysis</h1>
                <p><strong>Analysis Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

                <div class="highlight">
                    <h3>Key Research Question</h3>
                    <p>To what extent are global citizens thriving in terms of health and well-being?
                    How do health outcomes, life satisfaction, and mortality patterns compare across countries?</p>
                </div>

                <h2>Executive Summary</h2>
                {self._generate_health_summary(results, health_data)}

                <h2>Univariate Analysis</h2>
                {self._generate_univariate_section(results)}

                <h2>Bivariate Analysis</h2>
                {self._generate_bivariate_section(results)}

                <h2>Regional Analysis</h2>
                {self._generate_regional_section(results)}

                <h2>Mortality Patterns</h2>
                {self._generate_mortality_section(results)}

                <h2>Outlier Analysis</h2>
                {self._generate_outlier_section(results)}

                <h2>Visualizations</h2>
                <div class="image-container">
                    <h3>Distribution of Health Indicators</h3>
                    <img src="health_distributions.png" alt="Health Indicator Distributions">
                </div>

                <div class="image-container">
                    <h3>Health Indicators by Continent</h3>
                    <img src="health_by_continent.png" alt="Health by Continent">
                </div>

                <div class="image-container">
                    <h3>Health Indicator Relationships</h3>
                    <img src="health_correlations.png" alt="Health Correlations">
                </div>

                <div class="footer">
                    <p>Generated by Global Thriving Analysis System</p>
                    <p>Data Source: World Data 2.0-Data.csv</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Save the HTML file
        with open(self.output_dir / "health_wellbeing.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Health and Well-being report saved to {self.output_dir / 'health_wellbeing.html'}")

    def _generate_health_summary(self, results, health_data):
        """Generate executive summary for health analysis"""
        total_countries = len(health_data)
        continents = health_data['continent'].nunique()

        summary_stats = []
        if 'univariate' in results:
            for indicator, stats in results['univariate'].items():
                summary_stats.append((
                    indicator.replace('_', ' ').title(),
                    stats.get('count', 0),
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('std', 0):.2f}"
                ))

        summary_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{total_countries}</span>
                <div class="stat-label">Countries Analyzed</div>
            </div>
            <div class="stat-card">
                <span class="stat-number">{continents}</span>
                <div class="stat-label">Continents</div>
            </div>
        """

        if 'bivariate' in results:
            strong_correlations = sum(1 for r in results['bivariate'].values()
                                    if abs(r.get('pearson_r', 0)) > 0.5)
            summary_html += f"""
            <div class="stat-card">
                <span class="stat-number">{strong_correlations}</span>
                <div class="stat-label">Strong Correlations Found</div>
            </div>
            """

        summary_html += "</div>"

        if summary_stats:
            summary_html += """
            <div class="table-container">
                <h3>Indicator Summary Statistics</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Indicator</th>
                            <th>Countries with Data</th>
                            <th>Mean</th>
                            <th>Standard Deviation</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for name, count, mean, std in summary_stats:
                summary_html += f"""
                        <tr>
                            <td>{name}</td>
                            <td>{count}</td>
                            <td>{mean}</td>
                            <td>{std}</td>
                        </tr>
                """
            summary_html += """
                    </tbody>
                </table>
            </div>
            """

        return summary_html

    def _generate_univariate_section(self, results):
        """Generate univariate analysis section"""
        if 'univariate' not in results:
            return "<p>No univariate analysis results available.</p>"

        html = "<h3>Descriptive Statistics</h3>"

        for indicator, stats in results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()

            # Determine normality status
            normality_class = "success" if stats.get('is_normal', False) else "alert"
            normality_text = "normally distributed" if stats.get('is_normal', False) else "not normally distributed"

            html += f"""
            <h4>{clean_name}</h4>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number">{stats.get('count', 0)}</span>
                    <div class="stat-label">Valid Observations</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{stats.get('mean', 0):.2f}</span>
                    <div class="stat-label">Mean</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{stats.get('median', 0):.2f}</span>
                    <div class="stat-label">Median</div>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{stats.get('std', 0):.2f}</span>
                    <div class="stat-label">Standard Deviation</div>
                </div>
            </div>
            <div class="{normality_class}">
                <strong>Distribution:</strong> Data is {normality_text}
                (Shapiro-Wilk p-value: {stats.get('shapiro_p', 0):.4f})
            </div>
            """

        return html

    def _generate_bivariate_section(self, results):
        """Generate bivariate analysis section"""
        if 'bivariate' not in results:
            return "<p>No bivariate analysis results available.</p>"

        html = "<h3>Correlation Analysis</h3>"

        for relationship, stats in results['bivariate'].items():
            var1, var2 = relationship.split('_vs_')
            clean_relationship = f"{var1.replace('_', ' ').title()} vs {var2.replace('_', ' ').title()}"

            # Determine correlation strength
            r_value = abs(stats.get('pearson_r', 0))
            if r_value > 0.7:
                strength = "Strong"
                corr_class = "correlation-strong"
            elif r_value > 0.3:
                strength = "Moderate"
                corr_class = "correlation-moderate"
            else:
                strength = "Weak"
                corr_class = "correlation-weak"

            # Significance
            p_value = stats.get('pearson_p', 1)
            significance = "Significant" if p_value < 0.05 else "Not Significant"

            html += f"""
            <h4>{clean_relationship}</h4>
            <div class="{corr_class}">
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{stats.get('pearson_r', 0):.3f}</span>
                        <div class="stat-label">Pearson Correlation</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{stats.get('r_squared', 0):.3f}</span>
                        <div class="stat-label">R-squared</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{strength}</span>
                        <div class="stat-label">Correlation Strength</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{significance}</span>
                        <div class="stat-label">Statistical Significance</div>
                    </div>
                </div>
                <p><strong>Interpretation:</strong> {strength} correlation (r = {stats.get('pearson_r', 0):.3f}),
                explaining {stats.get('r_squared', 0)*100:.1f}% of the variance. {significance} at α = 0.05 level.</p>
            </div>
            """

        return html

    def _generate_regional_section(self, results):
        """Generate regional analysis section"""
        if 'regional' not in results:
            return "<p>No regional analysis results available.</p>"

        html = "<h3>Continental Comparisons</h3>"

        for indicator, analysis in results['regional'].items():
            clean_name = indicator.replace('_', ' ').title()
            anova_p = analysis.get('anova_p', 1)
            significant = analysis.get('significant_difference', False)

            status_class = "success" if significant else "alert"
            status_text = "Significant differences found" if significant else "No significant differences"

            html += f"""
            <h4>{clean_name} by Continent</h4>
            <div class="{status_class}">
                <strong>ANOVA Result:</strong> {status_text} between continents (p = {anova_p:.4f})
            </div>
            """

            if 'regional_stats' in analysis:
                stats_df = analysis['regional_stats']
                html += """
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Continent</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Median</th>
                                <th>Std Dev</th>
                                <th>Min</th>
                                <th>Max</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for continent, row in stats_df.iterrows():
                    html += f"""
                            <tr>
                                <td>{continent}</td>
                                <td>{row['count']}</td>
                                <td>{row['mean']}</td>
                                <td>{row['median']}</td>
                                <td>{row['std']}</td>
                                <td>{row['min']}</td>
                                <td>{row['max']}</td>
                            </tr>
                    """
                html += """
                        </tbody>
                    </table>
                </div>
                """

        return html

    def _generate_mortality_section(self, results):
        """Generate mortality analysis section"""
        if 'mortality' not in results:
            return "<p>No mortality analysis results available.</p>"

        html = "<h3>Leading Causes of Death</h3>"

        mortality = results['mortality']

        if 'overall_top_causes' in mortality:
            html += """
            <h4>Global Top 10 Causes of Death</h4>
            <div class="table-container">
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
            for rank, (cause, count) in enumerate(mortality['overall_top_causes'].items(), 1):
                html += f"""
                        <tr>
                            <td>{rank}</td>
                            <td>{cause}</td>
                            <td>{count}</td>
                        </tr>
                """
            html += """
                    </tbody>
                </table>
            </div>
            """

        if 'by_continent' in mortality:
            html += "<h4>Top Causes by Continent</h4>"
            for continent, causes in mortality['by_continent'].items():
                html += f"""
                <h5>{continent}</h5>
                <div class="table-container">
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
                for rank, (cause, count) in enumerate(causes.items(), 1):
                    html += f"""
                            <tr>
                                <td>{rank}</td>
                                <td>{cause}</td>
                                <td>{count}</td>
                            </tr>
                    """
                html += """
                        </tbody>
                    </table>
                </div>
                """

        return html

    def _generate_outlier_section(self, results):
        """Generate outlier analysis section"""
        if 'outliers' not in results:
            return "<p>No outlier analysis results available.</p>"

        html = "<h3>Outlier Countries</h3>"

        for indicator, outlier_data in results['outliers'].items():
            clean_name = indicator.replace('_', ' ').title()

            html += f"<h4>{clean_name} Outliers</h4>"

            if 'iqr_outliers' in outlier_data and len(outlier_data['iqr_outliers']) > 0:
                outliers_df = outlier_data['iqr_outliers']
                html += """
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Country</th>
                                <th>Continent</th>
                                <th>Value</th>
                                <th>Type</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                bounds = outlier_data.get('bounds', {})
                lower_bound = bounds.get('lower', 0)
                upper_bound = bounds.get('upper', 0)

                for _, row in outliers_df.iterrows():
                    outlier_type = "High" if row[indicator] > upper_bound else "Low"
                    html += f"""
                            <tr>
                                <td>{row['country']}</td>
                                <td>{row['continent']}</td>
                                <td>{row[indicator]:.2f}</td>
                                <td>{outlier_type}</td>
                            </tr>
                    """
                html += """
                        </tbody>
                    </table>
                </div>
                """
            else:
                html += "<p>No significant outliers detected using IQR method.</p>"

        return html

if __name__ == "__main__":
    # This would be called from the main analysis script
    pass