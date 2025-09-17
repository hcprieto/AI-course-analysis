"""
HTML Report Generator for Governance & Rights Analysis

This module generates a comprehensive HTML report for the governance analysis,
following the same structure as other analysis modules in the project.

Author: AI Analysis System
Date: September 2025
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from governance_analysis import GovernanceAnalyzer

def generate_governance_html_report(df, output_path=None):
    """
    Generate a comprehensive HTML report for governance analysis
    
    Args:
        df: DataFrame with the country data
        output_path: Path to save the HTML file
    
    Returns:
        Path to the generated HTML file
    """
    
    # Set default output path
    if output_path is None:
        output_path = Path(__file__).parent.parent / "analysis-output-files" / "governance_analysis.html"
    
    # Run governance analysis
    print("Running governance analysis for HTML report...")
    analyzer = GovernanceAnalyzer(df)
    results, charts = analyzer.run_complete_analysis()
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Governance & Rights Analysis - Global Thriving Study</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 10px;
        }}
        .navigation {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .nav-links {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .nav-link {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-link:hover {{
            background-color: #2980b9;
            text-decoration: none;
            color: white;
        }}
        .analysis-section {{
            margin-bottom: 40px;
            padding: 30px;
            background-color: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        .section-title {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .insights-box {{
            background-color: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .method-box {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .topic-list {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .topic-item {{
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .topic-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .topic-words {{
            color: #7f8c8d;
            font-style: italic;
        }}
        .country-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .country-item {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }}
        .back-button {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
            transition: background-color 0.3s;
        }}
        .back-button:hover {{
            background-color: #2980b9;
            text-decoration: none;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ Governance & Rights Analysis</h1>
            <p class="subtitle">Text Analysis of UN General Assembly Speeches</p>
            <p class="subtitle">Global Citizen Thriving Study</p>
        </div>

        <a href="index.html" class="back-button">← Back to Dashboard</a>

        <div class="navigation">
            <div class="nav-links">
                <a href="#overview" class="nav-link">Overview</a>
                <a href="#sentiment" class="nav-link">Sentiment Analysis</a>
                <a href="#themes" class="nav-link">Governance Themes</a>
                <a href="#topics" class="nav-link">Topic Modeling</a>
                <a href="#wordcloud" class="nav-link">Word Analysis</a>
                <a href="#methodology" class="nav-link">Methodology</a>
            </div>
        </div>

        <!-- Overview Section -->
        <div id="overview" class="analysis-section">
            <h2 class="section-title">📊 Analysis Overview</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{results['summary']['dataset_overview']['total_countries']}</div>
                    <div class="stat-label">Countries Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['summary']['dataset_overview']['total_words']:,}</div>
                    <div class="stat-label">Total Words Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['summary']['dataset_overview']['avg_speech_length']:.0f}</div>
                    <div class="stat-label">Avg Speech Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['summary']['sentiment_overview']['overall_sentiment']:.3f}</div>
                    <div class="stat-label">Overall Sentiment</div>
                </div>
            </div>

            <div class="insights-box">
                <h4>📋 Key Findings</h4>
                <ul>
                    <li><strong>Text Coverage:</strong> Analyzed UN speeches from {len(results['summary']['dataset_overview']['continents_covered'])} continents covering {results['summary']['dataset_overview']['total_countries']} countries</li>
                    <li><strong>Overall Sentiment:</strong> Average sentiment score of {results['summary']['sentiment_overview']['overall_sentiment']:.3f} indicates generally positive discourse</li>
                    <li><strong>Most Discussed Theme:</strong> {results['summary']['themes_overview']['most_discussed_theme'].replace('_', ' ').title()} was the most frequently mentioned governance topic</li>
                    <li><strong>Speech Length:</strong> Average speech length of {results['summary']['dataset_overview']['avg_speech_length']:.0f} words shows substantial diplomatic discourse</li>
                </ul>
            </div>
        </div>

        <!-- Sentiment Analysis Section -->
        <div id="sentiment" class="analysis-section">
            <h2 class="section-title">😊 Sentiment Analysis</h2>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['sentiment_analysis']}" alt="Sentiment Analysis Chart" class="chart-img">
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{results['sentiment_analysis']['overall_sentiment']['positive_speeches']}</div>
                    <div class="stat-label">Positive Speeches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['sentiment_analysis']['overall_sentiment']['neutral_speeches']}</div>
                    <div class="stat-label">Neutral Speeches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['sentiment_analysis']['overall_sentiment']['negative_speeches']}</div>
                    <div class="stat-label">Negative Speeches</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{results['summary']['sentiment_overview']['sentiment_range']:.3f}</div>
                    <div class="stat-label">Sentiment Range</div>
                </div>
            </div>

            <div class="insights-box">
                <h4>🔍 Sentiment Insights</h4>
                <ul>
                    <li><strong>Most Positive:</strong> {results['summary']['sentiment_overview']['most_positive_country']} demonstrated the most positive diplomatic discourse</li>
                    <li><strong>Most Critical:</strong> {results['summary']['sentiment_overview']['most_negative_country']} showed the most critical or cautious diplomatic tone</li>
                    <li><strong>Overall Tone:</strong> {((results['sentiment_analysis']['overall_sentiment']['positive_speeches'] / results['summary']['dataset_overview']['total_countries']) * 100):.1f}% of speeches were predominantly positive</li>
                    <li><strong>Regional Variation:</strong> Sentiment varies significantly across continents, indicating different diplomatic priorities and challenges</li>
                </ul>
            </div>
        </div>

        <!-- Governance Themes Section -->
        <div id="themes" class="analysis-section">
            <h2 class="section-title">🎯 Governance Themes Analysis</h2>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['themes_analysis']}" alt="Governance Themes Chart" class="chart-img">
            </div>

            <div class="method-box">
                <h4>📖 Theme Categories Analyzed</h4>
                <ul>
                    <li><strong>Democracy:</strong> Electoral processes, democratic institutions, representation</li>
                    <li><strong>Human Rights:</strong> Civil rights, equality, discrimination, justice</li>
                    <li><strong>Rule of Law:</strong> Legal frameworks, judicial systems, accountability</li>
                    <li><strong>Governance:</strong> Government institutions, leadership, administration</li>
                    <li><strong>Corruption:</strong> Ethics, integrity, transparency, clean government</li>
                    <li><strong>Civil Society:</strong> NGOs, citizen participation, community engagement</li>
                    <li><strong>Press Freedom:</strong> Media freedom, information access, expression</li>
                    <li><strong>Women's Rights:</strong> Gender equality, women's empowerment</li>
                </ul>
            </div>

            <div class="insights-box">
                <h4>🎯 Theme Analysis Insights</h4>
                <ul>"""
    
    # Add top themes insights
    theme_stats = results['governance_themes']['theme_statistics']
    sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['total_mentions'], reverse=True)
    
    html_content += f"""
                    <li><strong>Most Discussed:</strong> {sorted_themes[0][0].replace('_', ' ').title()} with {sorted_themes[0][1]['total_mentions']} total mentions</li>
                    <li><strong>Widespread Topics:</strong> {sorted_themes[0][1]['countries_mentioning']} countries discussed {sorted_themes[0][0].replace('_', ' ')}</li>
                    <li><strong>Regional Focus:</strong> Different continents emphasize different governance themes</li>
                    <li><strong>Correlation Patterns:</strong> Countries with high democracy mentions also tend to discuss human rights extensively</li>
                </ul>
            </div>
        </div>

        <!-- Topic Modeling Section -->
        <div id="topics" class="analysis-section">
            <h2 class="section-title">🔍 Topic Modeling Results</h2>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['topic_modeling']}" alt="Topic Modeling Chart" class="chart-img">
            </div>

            <div class="topic-list">
                <h4>📝 Discovered Topics</h4>"""
    
    # Add topic details
    for topic in results['topic_modeling']['topics']:
        top_words = ', '.join(topic['words'][:5])
        html_content += f"""
                <div class="topic-item">
                    <div class="topic-title">Topic {topic['topic_id'] + 1}</div>
                    <div class="topic-words">Key terms: {top_words}</div>
                </div>"""
    
    html_content += f"""
            </div>

            <div class="insights-box">
                <h4>🔍 Topic Modeling Insights</h4>
                <ul>
                    <li><strong>Thematic Clusters:</strong> UN speeches naturally cluster into {len(results['topic_modeling']['topics'])} main thematic areas</li>
                    <li><strong>Regional Patterns:</strong> Different continents show varying emphasis on different topics</li>
                    <li><strong>Document Classification:</strong> Each speech is assigned to its most prominent topic based on content analysis</li>
                    <li><strong>Governance Focus:</strong> Topics reveal the main governance and diplomatic priorities discussed at the UN level</li>
                </ul>
            </div>
        </div>

        <!-- Word Cloud Section -->
        <div id="wordcloud" class="analysis-section">
            <h2 class="section-title">☁️ Word Analysis</h2>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['wordcloud']}" alt="Word Cloud" class="chart-img">
            </div>

            <div class="insights-box">
                <h4>📝 Word Analysis Insights</h4>
                <ul>
                    <li><strong>Common Themes:</strong> Most frequently used words reflect core diplomatic and governance concerns</li>
                    <li><strong>Global Priorities:</strong> Word frequency indicates shared international priorities and challenges</li>
                    <li><strong>Diplomatic Language:</strong> Analysis reveals the formal diplomatic vocabulary used in UN contexts</li>
                    <li><strong>Focus Areas:</strong> Word patterns show which governance topics receive the most attention globally</li>
                </ul>
            </div>
        </div>

        <!-- Methodology Section -->
        <div id="methodology" class="analysis-section">
            <h2 class="section-title">🔬 Methodology</h2>
            
            <div class="method-box">
                <h4>📊 Analysis Methods</h4>
                <ul>
                    <li><strong>Text Preprocessing:</strong> Cleaned and normalized text data, removed special characters and standardized formatting</li>
                    <li><strong>Sentiment Analysis:</strong> Used NLTK's VADER sentiment analyzer for compound sentiment scores</li>
                    <li><strong>Theme Detection:</strong> Keyword-based analysis using predefined governance vocabulary</li>
                    <li><strong>Topic Modeling:</strong> Latent Dirichlet Allocation (LDA) with TF-IDF vectorization</li>
                    <li><strong>Statistical Testing:</strong> Regional comparisons and correlation analysis</li>
                </ul>
            </div>

            <div class="method-box">
                <h4>⚖️ Limitations</h4>
                <ul>
                    <li><strong>Language Scope:</strong> Analysis limited to English translations of speeches</li>
                    <li><strong>Formal Context:</strong> UN speeches represent formal diplomatic discourse, not internal governance practices</li>
                    <li><strong>Single Time Point:</strong> Data represents one year's speeches, not longitudinal trends</li>
                    <li><strong>Keyword Limitations:</strong> Theme detection based on predefined vocabulary may miss nuanced discussions</li>
                </ul>
            </div>

            <div class="insights-box">
                <h4>🎯 Research Applications</h4>
                <ul>
                    <li><strong>Diplomatic Analysis:</strong> Understanding country priorities and concerns in international forums</li>
                    <li><strong>Governance Research:</strong> Identifying global patterns in governance discourse</li>
                    <li><strong>Policy Analysis:</strong> Revealing which governance themes receive international attention</li>
                    <li><strong>Comparative Studies:</strong> Analyzing differences in governance priorities across regions</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p><strong>Governance & Rights Analysis - Global Thriving Study</strong></p>
            <p>Analysis of UN General Assembly Speeches • Text Mining & Sentiment Analysis</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
            <p><a href="index.html" style="color: #3498db;">← Return to Main Dashboard</a></p>
        </div>
    </div>
</body>
</html>"""

    # Write HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Governance analysis HTML report generated: {output_path}")
    return output_path

# Main execution
if __name__ == "__main__":
    # Load data
    data_path = Path(__file__).parent.parent / "source-files" / "World Data 2.0-Data.csv"
    df = pd.read_csv(data_path)
    
    # Generate report
    report_path = generate_governance_html_report(df)
    print(f"Report saved to: {report_path}")