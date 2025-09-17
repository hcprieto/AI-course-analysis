# Global Thriving Analysis: Health & Well-being Study

A comprehensive statistical analysis of global health and well-being indicators across 132 countries, examining the extent to which global citizens are thriving.

## 🌐 Live Analysis Dashboard

**View the interactive analysis:** [https://[your-username].github.io/AI-course-analysis/analysis-output-files/](https://[your-username].github.io/AI-course-analysis/analysis-output-files/)

*Replace `[your-username]` with your actual GitHub username after publishing*

## 📊 Analysis Overview

This study examines multiple dimensions of global citizen thriving through statistical analysis of health and well-being indicators:

### Key Research Question
**"To what extent are global citizens thriving? How do these data compare by country?"**

### Dataset
- **132 countries** across **6 continents**
- **Multiple health indicators** from 2018-2020
- Data source: World Data 2.0-Data.csv

## 🔍 Analysis Features

### Health & Well-being Analysis
- **Life Satisfaction** (Cantril Ladder scale)
- **Life Expectancy** at birth
- **Self-harm Rates** per 100,000 people
- **Mortality Patterns** and leading causes of death

### Statistical Methods
- **Univariate Analysis**: Descriptive statistics, normality testing
- **Bivariate Analysis**: Correlation analysis, regression
- **Regional Analysis**: ANOVA tests comparing continents
- **Outlier Detection**: Z-score methodology
- **Mortality Analysis**: Global and regional patterns

### Key Findings
- **Strong correlation** (r = 0.72) between life satisfaction and life expectancy
- **Significant regional differences** across all health indicators
- **Outlier countries** identified for each indicator
- **Statistical significance** testing for all relationships

## 📈 Visualizations

The analysis includes comprehensive visualizations:
- Distribution plots with normality testing
- Regional comparison box plots
- Correlation matrices and scatter plots
- Outlier detection charts
- Mortality pattern analysis

## 🚀 Two Analysis Levels

### Enhanced Analysis (Recommended)
- Interactive charts and visualizations
- Comprehensive statistical testing
- Outlier identification with country annotations
- Regional ANOVA analysis
- Professional presentation

### Basic Analysis
- Summary statistics
- Key correlations
- Simplified presentation
- Quick overview format

## 📁 Repository Structure

```
AI-course-analysis/
├── analysis-output-files/          # Web-ready analysis reports
│   ├── index.html                  # Main dashboard
│   ├── enhanced_health_analysis.html  # Full statistical analysis
│   ├── health_wellbeing.html       # Basic analysis
│   ├── *.png                       # Generated visualizations
├── code-files/                     # Analysis scripts
│   ├── enhanced_health_analysis.py # Main analysis script
│   ├── simple_health_analysis.py   # Basic analysis
│   ├── data_loader.py              # Data processing
│   └── html_generator.py           # Report generation
├── source-files/                   # Raw data
│   ├── World Data 2.0-Data.csv     # Main dataset
│   └── Analysis plan.csv           # Research framework
├── CLAUDE.md                       # Project documentation
└── README.md                       # This file
```

## 🛠 Technical Implementation

### Technologies Used
- **Python**: Data analysis and statistical computing
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical testing (ANOVA, correlations)
- **Matplotlib/Seaborn**: Data visualization
- **HTML/CSS**: Interactive reporting
- **GitHub Pages**: Web hosting

### Analysis Pipeline
1. **Data Loading & Cleaning**: Handle CSV parsing issues, standardize formats
2. **Statistical Analysis**: Univariate, bivariate, and multivariate analysis
3. **Visualization Generation**: Create publication-quality charts
4. **Report Generation**: Compile results into interactive HTML
5. **Web Deployment**: Serve via GitHub Pages

## 📖 How to Use

### View Online (Recommended)
Simply visit the GitHub Pages link above to explore the interactive analysis.

### Run Locally
1. Clone this repository
2. Install required packages: `pip install pandas matplotlib seaborn scipy`
3. Run the analysis: `python code-files/enhanced_health_analysis.py`
4. Open `analysis-output-files/index.html` in your browser

### Reproduce Analysis
All analysis scripts are included and documented. The main analysis can be reproduced by running:
```bash
cd code-files
python enhanced_health_analysis.py
```

## 🎯 Key Insights

### Global Health Patterns
- **Life satisfaction** varies significantly by continent (p < 0.001)
- **Life expectancy** shows strong regional clustering
- **Self-harm rates** identify concerning outlier countries

### Statistical Relationships
- **Life satisfaction ↔ Life expectancy**: Strong positive correlation (r = 0.72)
- **Regional differences**: All indicators show significant continental variation
- **Outlier identification**: Afghanistan (life satisfaction), multiple countries (self-harm rates)

### Mortality Analysis
- **Cardiovascular disease** dominates globally
- **Regional patterns** in leading causes of death
- **Distribution analysis** across continents

## 📚 Methodology

The analysis follows a systematic approach:
1. **Exploratory Data Analysis**: Understand distributions and patterns
2. **Statistical Testing**: Apply appropriate tests for data types
3. **Effect Size Calculation**: Measure practical significance
4. **Outlier Detection**: Multiple methods for robust identification
5. **Visualization**: Clear, publication-ready charts
6. **Interpretation**: Context-aware analysis of results

## 🤝 Contributing

This is an educational project demonstrating statistical analysis techniques. Feel free to:
- Explore the analysis methods
- Suggest improvements to visualizations
- Propose additional research questions
- Report any issues with the code

## 📄 License

This project is for educational purposes. Data source attributions are maintained in the analysis reports.

## 🙏 Acknowledgments

- Data analysis techniques adapted from best practices in statistical computing
- Visualization design inspired by modern data science standards
- HTML templates designed for accessibility and user experience

---

**View the live analysis:** [GitHub Pages Link](https://[your-username].github.io/AI-course-analysis/analysis-output-files/)

*Generated using Claude Code for comprehensive statistical analysis*