"""
Main script to run Health and Well-being Analysis

This script orchestrates the complete health and well-being analysis pipeline:
1. Load and clean data
2. Perform statistical analysis
3. Generate visualizations
4. Create HTML reports
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import load_and_clean_data
from health_wellbeing_analysis import HealthWellbeingAnalyzer, create_health_visualizations
from html_generator import HTMLReportGenerator

def main():
    print("=" * 60)
    print("GLOBAL THRIVING ANALYSIS: HEALTH & WELL-BEING")
    print("=" * 60)

    # Define paths
    source_path = Path("../source-files/World Data 2.0-Data.csv")
    output_dir = Path("../analysis-output-files")

    print(f"\n1. Loading data from: {source_path}")
    print("-" * 40)

    # Load and clean data
    loader, clean_data = load_and_clean_data(source_path)

    if not loader:
        print("❌ Failed to load data. Exiting.")
        return False

    print("✅ Data loaded successfully!")
    summary = loader.get_data_summary()
    print(f"   • Total countries: {summary['total_countries']}")
    print(f"   • Continents: {list(summary['continents'].keys())}")

    print(f"\n2. Running Health & Well-being Analysis")
    print("-" * 40)

    # Create analyzer and run analysis
    analyzer = HealthWellbeingAnalyzer(loader)
    results = analyzer.run_complete_analysis()

    print("✅ Analysis completed!")

    print(f"\n3. Generating Visualizations")
    print("-" * 40)

    # Create visualizations
    create_health_visualizations(analyzer, output_dir)
    print("✅ Visualizations created!")

    print(f"\n4. Generating HTML Report")
    print("-" * 40)

    # Generate HTML report
    html_generator = HTMLReportGenerator(output_dir)
    html_generator.generate_health_wellbeing_report(analyzer)
    print("✅ HTML report generated!")

    print(f"\n5. Analysis Results Summary")
    print("=" * 40)

    # Display key findings
    if 'univariate' in results:
        print("\n📊 KEY HEALTH INDICATORS:")
        for indicator, stats in results['univariate'].items():
            clean_name = indicator.replace('_', ' ').title()
            print(f"   • {clean_name}:")
            print(f"     - Countries with data: {stats['count']}")
            print(f"     - Mean: {stats['mean']:.2f} (±{stats['std']:.2f})")
            print(f"     - Range: {stats['min']:.2f} - {stats['max']:.2f}")

    if 'bivariate' in results:
        print("\n🔗 KEY RELATIONSHIPS:")
        for relationship, stats in results['bivariate'].items():
            clean_rel = relationship.replace('_vs_', ' vs ').replace('_', ' ').title()
            correlation = stats['pearson_r']
            r_squared = stats['r_squared']
            significance = "significant" if stats['pearson_p'] < 0.05 else "not significant"

            print(f"   • {clean_rel}:")
            print(f"     - Correlation: r = {correlation:.3f} ({significance})")
            print(f"     - Variance explained: {r_squared*100:.1f}%")

    if 'regional' in results:
        print("\n🌍 REGIONAL DIFFERENCES:")
        for indicator, analysis in results['regional'].items():
            clean_name = indicator.replace('_', ' ').title()
            significant = analysis.get('significant_difference', False)
            status = "Significant differences" if significant else "No significant differences"
            print(f"   • {clean_name}: {status} between continents")

    if 'mortality' in results and 'overall_top_causes' in results['mortality']:
        print("\n💀 TOP CAUSES OF DEATH (Global):")
        top_causes = results['mortality']['overall_top_causes'].head(5)
        for rank, (cause, count) in enumerate(top_causes.items(), 1):
            print(f"   {rank}. {cause} ({count} countries)")

    print(f"\n6. Output Files Created")
    print("=" * 40)
    output_files = [
        "health_wellbeing.html",
        "health_distributions.png",
        "health_by_continent.png",
        "health_correlations.png"
    ]

    for filename in output_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (missing)")

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print(f"🌐 Open health_wellbeing.html in your browser to view the full report")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)