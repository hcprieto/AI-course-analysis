# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains, in the /source-files folder, a csv files with information about different indicators from countries.
Claude Code is intended to help with the analysis of these data to support an evaluation, includding Univariate analysis, Bivariate analysis and Multivariate (multiple regression, ANOVA/ANCOVA, etc).

- You will help me identify the best analysis based on my goal/question, which is to respond to the question: **To what extent are global citizens thriving? How do these data compare by country?**
- Check that an analysis is in alignment with the data types I'll be using (if not, what the alternatives are).
- Ensuring the data viz approach I'm planning is the best way to clearly and simply visualize my findings.


## Process and file naming conventions

You will be requested to perform analysis of this data. 
To do so, you could create scripts (eg: python) in the /code-files directory, as required, or use and/or modify the scripts contained there. 

## Main Source files

Main data files included in the source-files directory are:

### 1. Analysis plan.csv
This file contains a structured research framework that maps out analytical approaches for examining global citizen thriving across different constructs. It includes:

**Research Question:** To what extent are global citizens thriving? How do these data compare by country?

**Structure:**
- **Construct:** Different dimensions of thriving (Health/Access to basic needs, Economic, Education, Environment, Equity, Freedom/Rights/Governance)
- **Measure(s):** Specific indicators available for analysis
- **Question:** Focused research questions for each construct
- **Analysis type:** Statistical methods to be applied (correlation, regression, clustering, etc.)
- **Visualization:** Recommended chart types and visual approaches
- **Results:** Space for findings and statistical outcomes
- **Insights:** Reflections and interpretations of results

### 2. World Data 2.0-Data.csv
This is the main dataset containing country-level indicators across multiple dimensions of global thriving. It includes data for countries worldwide with their corresponding values across various well-being and development metrics.

**KEY INFORMATION COLUMNS**

**Identification:**
- Country: Country name
- ISO3: Three-letter country code
- Continent: Geographic region

**Demographics & Economics:**
- Total population (2018)
- GDP per Capita (2018)
- Unemployment Rate (2018)

**Health & Well-being:**
- Life satisfaction in Cantril Ladder (2018): Subjective well-being measure
- Life expectancy at birth (years) (2018)
- Most/Second/Third Common Cause of Death (2018)
- Deaths Due to Self-Harm (2019) and rate per 100,000 people

**Infrastructure & Education:**
- Number of people with access to electricity (2020)
- % of population with access to electricity (2020)
- Learning-Adjusted Years of School (2020)

**Additional Data:**
- Text: Contains UN General Assembly speeches for qualitative analysis
- Correlation coefficient (0.54): Pre-calculated relationship between specific variables 


## Analysis Plan

### Data Overview
- **Dataset Size**: ~3,570 observations (includes duplicates with UN speech data)
- **Geographic Coverage**: Global with continental groupings (Africa: 39, Europe: 36, Asia: 35, Americas: 20, Oceania: 2)
- **Data Quality**: Requires cleaning due to CSV parsing issues and embedded text data

### Analysis Framework

Based on the Analysis plan.csv, the study will examine 6 key constructs of global thriving:

#### 1. Health & Well-being Analysis
**Available Indicators:**
- Life satisfaction (Cantril Ladder 2018)
- Life expectancy at birth (2018)
- Mortality patterns (top 3 causes of death)
- Self-harm rates (2019)

**Analysis Approach:**
- Descriptive statistics and distribution analysis
- Correlation between subjective (life satisfaction) and objective (life expectancy) measures
- Regional comparisons of health outcomes
- Identification of health outliers

#### 2. Economic Indicators Analysis
**Available Indicators:**
- GDP per capita (2018)
- Unemployment rate (2018)
- Population size (2018)

**Analysis Approach:**
- Economic development clustering
- Relationship between economic indicators and life satisfaction
- Unemployment-wellbeing correlations
- Economic outlier identification

#### 3. Infrastructure & Basic Needs
**Available Indicators:**
- Access to electricity (% population and absolute numbers, 2020)

**Analysis Approach:**
- Infrastructure access patterns by region
- Correlation with life satisfaction (target R² = 0.54 as noted in plan)
- Infrastructure gaps identification

#### 4. Education Analysis
**Available Indicators:**
- Learning-Adjusted Years of School (2020)

**Analysis Approach:**
- Educational attainment by region
- Education-development relationships
- Education-wellbeing correlations

#### 5. Governance & Rights (Qualitative)
**Available Data:**
- UN General Assembly speeches (text analysis)

**Analysis Approach:**
- Content analysis of speeches for governance themes
- Sentiment analysis
- Topic modeling for rights/democracy themes

#### 6. Environmental & Equity Indicators
**Status:** Limited data available (noted as gaps in analysis plan)

### Statistical Methods

#### Univariate Analysis
- Descriptive statistics (mean, median, standard deviation)
- Distribution analysis and normality testing
- Box plots for outlier identification

#### Bivariate Analysis
- Correlation analysis between key indicators
- Scatterplot analysis with trend lines
- Regional comparisons using ANOVA

#### Multivariate Analysis
- Multiple regression models predicting life satisfaction
- Principal component analysis for dimensionality reduction
- Cluster analysis to identify country typologies

### Implementation Approach

**Technology Stack:**
- **Python scripts** for data processing and analysis (pandas, scipy, matplotlib, seaborn)
- **HTML output** with interactive visualizations for presentation
- **Jupyter notebooks** for exploratory analysis documentation

**Key Processing Steps:**
1. Data cleaning and standardization (address CSV parsing issues)
2. Missing data analysis and treatment
3. Statistical analysis execution
4. Visualization generation
5. HTML report compilation

### Outputs
Generate comprehensive analysis as linked HTML pages within /analysis-output-files directory:

1. **Executive Summary** (index.html)
2. **Data Overview & Quality Assessment**
3. **Individual Construct Analysis** (6 separate pages)
4. **Cross-cutting Analysis** (correlations, clusters)
5. **Country Profiles** (outliers and notable cases)
6. **Methodology & Technical Appendix**


