#!/usr/bin/env python3
"""
Comprehensive Digital Grants Outcomes Analysis
Created: 2025.07.23.12.27

This script performs a comprehensive analysis of Digital theme grant outcomes at three levels:
1. Individual Grant Analysis
2. Sub-Portfolio Analysis  
3. Theme-Wide Analysis

The analysis focuses particularly on detailed Outcome Basket (OB) usage patterns,
classifying them as default-for-assigned-domain, default-for-other-domain, or custom.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class DigitalGrantsOutcomesAnalyzer:
    """
    Comprehensive analyzer for Digital theme grant outcomes with enhanced OB analysis
    """
    
    def __init__(self, source_dir="source-files", output_dir="analysis-output-files"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.timestamp = "2025.07.23.12.27"
        
        # Data containers
        self.outcomes_df = None
        self.grants_df = None
        self.key_concepts_df = None
        self.portfolio_descriptions_df = None
        
        # OB classification data
        self.default_obs_by_domain = {}
        self.all_default_obs = set()
        
        # Analysis results
        self.individual_analyses = {}
        self.subportfolio_analyses = {}
        self.theme_analysis = {}
        
        # Domains of Change mapping
        self.domains = {
            'D1': 'D1 - Stronger agency of young people of all backgrounds to shape their digital futures',
            'D2': 'D2 - Expanded equitable access to services for young people', 
            'D3': 'D3 - Collective efforts by relevant stakeholders for joint learning, knowledge exchange and strategic alignment',
            'D4': 'D4 - Increased collaborative efforts towards strengthened human rights-based digital governance frameworks'
        }
        
        # Sub-portfolios
        self.subportfolios = [
            'Digital Health sub-portfolio',
            'Digital Rights & Governance sub-portfolio', 
            'Digital Literacy sub-portfolio',
            'Digital Innovation sub-portfolio'
        ]
    
    def load_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load outcomes data
        outcomes_path = self.source_dir / "FB-Digital-Grants-Outcomes-Data.v2.csv"
        self.outcomes_df = pd.read_csv(outcomes_path, encoding='utf-8-sig')
        print(f"Loaded {len(self.outcomes_df)} outcome records")
        
        # Load grants data 
        grants_path = self.source_dir / "FBs-Digital-Grants-Data.v5.2.csv"
        self.grants_df = pd.read_csv(grants_path, encoding='utf-8-sig', header=0)
        print(f"Loaded {len(self.grants_df)} grant records")
        
        # Load key concepts for OB definitions
        concepts_path = self.source_dir / "Key-Concepts-Descriptors.v5.csv"
        self.key_concepts_df = pd.read_csv(concepts_path, encoding='utf-8-sig')
        
        # Load portfolio descriptions
        portfolio_path = self.source_dir / "Portfolio-descriptions.v5.csv"
        self.portfolio_descriptions_df = pd.read_csv(portfolio_path, encoding='utf-8-sig')
        
        print("Data loading completed successfully")
    
    def extract_default_outcome_baskets(self):
        """Extract default outcome baskets for each Domain of Change from Key-Concepts file"""
        print("Extracting default outcome baskets...")
        
        # Find the section with DEFAULT OUTCOME BASKETS
        obs_section_start = None
        for i, row in self.key_concepts_df.iterrows():
            if pd.notna(row.iloc[0]) and 'DEFAULT OUTCOME BASKETS' in str(row.iloc[0]):
                obs_section_start = i
                break
        
        if obs_section_start is None:
            raise ValueError("Could not find DEFAULT OUTCOME BASKETS section")
        
        # Find the header row with domain names
        domain_header_row = obs_section_start + 2  # Skip empty row
        domain_headers = self.key_concepts_df.iloc[domain_header_row]
        
        # Map columns to domains
        domain_columns = {}
        for col_idx, header in enumerate(domain_headers):
            if pd.notna(header):
                header_str = str(header).strip()
                if 'DoC 1' in header_str:
                    domain_columns['D1'] = col_idx
                elif 'DoC 2' in header_str:
                    domain_columns['D2'] = col_idx
                elif 'DoC 3' in header_str:
                    domain_columns['D3'] = col_idx
                elif 'DoC 4' in header_str:
                    domain_columns['D4'] = col_idx
        
        # Extract OBs for each domain
        for domain, col_idx in domain_columns.items():
            obs_list = []
            # Read OBs from subsequent rows
            for i in range(domain_header_row + 1, len(self.key_concepts_df)):
                cell_value = self.key_concepts_df.iloc[i, col_idx]
                if pd.notna(cell_value) and str(cell_value).strip():
                    ob = str(cell_value).strip()
                    if ob and ob not in ['', 'nan']:
                        obs_list.append(ob)
                
                # Stop if we hit another section or empty rows
                if i > domain_header_row + 15:  # Reasonable limit
                    break
            
            self.default_obs_by_domain[domain] = obs_list
            self.all_default_obs.update(obs_list)
        
        print(f"Extracted default OBs for {len(self.default_obs_by_domain)} domains")
        for domain, obs in self.default_obs_by_domain.items():
            print(f"  {domain}: {len(obs)} default OBs")
    
    def parse_outcome_baskets(self, ob_string):
        """Parse semi-colon separated outcome baskets"""
        if pd.isna(ob_string) or not str(ob_string).strip():
            return []
        
        # Split by semicolon and clean
        obs = [ob.strip() for ob in str(ob_string).split(';') if ob.strip()]
        return obs
    
    def classify_outcome_basket(self, ob, assigned_domain):
        """
        Classify an outcome basket as:
        - 'default_assigned': Default OB for the assigned domain
        - 'default_other': Default OB for a different domain  
        - 'custom': Custom OB not in any default list
        """
        # Extract domain key from full domain string
        domain_key = assigned_domain.split(' - ')[0] if ' - ' in assigned_domain else assigned_domain
        
        if ob in self.default_obs_by_domain.get(domain_key, []):
            return 'default_assigned'
        elif ob in self.all_default_obs:
            # Find which domain this OB belongs to
            for domain, obs_list in self.default_obs_by_domain.items():
                if ob in obs_list:
                    return f'default_other_{domain}'
            return 'default_other'
        else:
            return 'custom'
    
    def analyze_outcome_baskets_for_outcomes(self, outcomes_subset):
        """Analyze outcome baskets for a subset of outcomes"""
        ob_analysis = {
            'default_assigned': Counter(),
            'default_other': Counter(),
            'custom': Counter(),
            'unused_defaults': {},
            'total_outcomes': len(outcomes_subset)
        }
        
        # Track which domains are represented
        domains_in_subset = set(outcomes_subset['Domain of Change'].dropna())
        
        # Initialize unused defaults tracking for represented domains
        for domain_full in domains_in_subset:
            domain_key = domain_full.split(' - ')[0] if ' - ' in domain_full else domain_full
            if domain_key in self.default_obs_by_domain:
                ob_analysis['unused_defaults'][domain_key] = set(self.default_obs_by_domain[domain_key])
        
        # Process each outcome
        for _, outcome in outcomes_subset.iterrows():
            assigned_domain = outcome['Domain of Change']
            ob_string = outcome['Outcome Basket names']
            
            obs = self.parse_outcome_baskets(ob_string)
            
            for ob in obs:
                classification = self.classify_outcome_basket(ob, assigned_domain)
                
                if classification == 'default_assigned':
                    ob_analysis['default_assigned'][ob] += 1
                    # Remove from unused defaults
                    domain_key = assigned_domain.split(' - ')[0] if ' - ' in assigned_domain else assigned_domain
                    if domain_key in ob_analysis['unused_defaults']:
                        ob_analysis['unused_defaults'][domain_key].discard(ob)
                        
                elif classification.startswith('default_other'):
                    ob_analysis['default_other'][ob] += 1
                    
                elif classification == 'custom':
                    ob_analysis['custom'][ob] += 1
        
        return ob_analysis
    
    def create_individual_grant_analysis(self, gams_code):
        """Create detailed analysis for an individual grant"""
        # Get grant info
        grant_info = self.grants_df[self.grants_df['GAMS Code'] == gams_code].iloc[0]
        
        # Get outcomes for this grant
        grant_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'] == gams_code]
        
        if len(grant_outcomes) == 0:
            return None  # No outcomes for this grant
        
        # Basic grant metadata
        analysis = {
            'gams_code': gams_code,
            'grant_name': grant_info.get('Grant\'s Name', 'N/A'),
            'short_name': grant_info.get('Grant\'s short name', 'N/A'),
            'budget': grant_info.get('Budget (CHF)', 'N/A'),
            'duration': grant_info.get('Duration (months)', 'N/A'),
            'countries': grant_info.get('Countries', 'N/A'),
            'outcomes_count': len(grant_outcomes),
            'outcomes_data': grant_outcomes
        }
        
        # Sub-portfolio assignments
        analysis['subportfolios'] = []
        for sp in self.subportfolios:
            if sp in grant_info and pd.notna(grant_info[sp]) and str(grant_info[sp]).strip().upper() in ['PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)']:
                analysis['subportfolios'].append(sp)
        
        # Performance ratings
        performance_cols = ['Implementation', 'Contribution to strategic intent', 'Learning', 
                          'Sustainability of changes', 'Stakeholder engagement and collaboration', 
                          'Meaningful youth participation']
        analysis['performance'] = {}
        for col in performance_cols:
            if col in grant_info:
                analysis['performance'][col] = grant_info.get(col, 'N/A')
        
        # Domain distribution
        domain_counts = grant_outcomes['Domain of Change'].value_counts()
        analysis['domain_distribution'] = domain_counts.to_dict()
        
        # Significance distribution
        significance_counts = grant_outcomes['Significance Level'].value_counts()
        analysis['significance_distribution'] = significance_counts.to_dict()
        
        # Countries analysis
        countries_list = []
        for country_str in grant_outcomes['Country(ies) / Global'].dropna():
            countries_list.extend([c.strip() for c in str(country_str).split(',') if c.strip()])
        analysis['outcome_countries'] = Counter(countries_list)
        
        # Special cases analysis
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        analysis['special_cases'] = {}
        for col in special_cases_cols:
            if col in grant_outcomes.columns:
                count = (grant_outcomes[col] == 1).sum()
                if count > 0:
                    analysis['special_cases'][col] = count
        
        # Enhanced OB analysis
        analysis['ob_analysis'] = self.analyze_outcome_baskets_for_outcomes(grant_outcomes)
        
        return analysis
    
    def create_subportfolio_analysis(self, subportfolio_name):
        """Create detailed analysis for a sub-portfolio"""
        # Clean subportfolio name for comparison
        sp_clean = subportfolio_name.replace(' sub-portfolio', '').replace('&', 'and')
        
        # Find grants in this sub-portfolio
        subportfolio_grants = []
        subportfolio_gams_codes = []
        
        for _, grant in self.grants_df.iterrows():
            if pd.notna(grant.get(subportfolio_name)) and str(grant[subportfolio_name]).strip().upper() in ['PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)']:
                gams_code = grant['GAMS Code']
                if gams_code in self.individual_analyses:
                    subportfolio_grants.append(grant)
                    subportfolio_gams_codes.append(gams_code)
        
        if not subportfolio_grants:
            return None
        
        # Get all outcomes for grants in this sub-portfolio
        subportfolio_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].isin(subportfolio_gams_codes)]
        
        # Basic sub-portfolio info
        analysis = {
            'name': subportfolio_name,
            'grants_count': len(subportfolio_grants),
            'grants_list': [(g['GAMS Code'], g.get('Grant\'s short name', 'N/A')) for g in subportfolio_grants],
            'total_outcomes': len(subportfolio_outcomes),
            'outcomes_data': subportfolio_outcomes
        }
        
        # Get sub-portfolio description from portfolio descriptions file
        sp_desc_row = None
        for _, row in self.portfolio_descriptions_df.iterrows():
            if pd.notna(row.get('NAME')) and sp_clean.lower() in str(row['NAME']).lower():
                sp_desc_row = row
                break
        
        if sp_desc_row is not None:
            analysis['description'] = sp_desc_row.get('CORE FOCUS', 'N/A')
            analysis['focus_details'] = sp_desc_row.get('Focus of grants', 'N/A')
        else:
            analysis['description'] = 'Description not found'
            analysis['focus_details'] = 'Details not available'
        
        # Domain distribution across sub-portfolio
        domain_counts = subportfolio_outcomes['Domain of Change'].value_counts()
        analysis['domain_distribution'] = domain_counts.to_dict()
        
        # Budget and duration aggregation
        total_budget = 0
        valid_budgets = 0
        total_duration = 0
        valid_durations = 0
        
        for grant in subportfolio_grants:
            budget = grant.get('Budget (CHF)')
            if pd.notna(budget) and str(budget).replace(',', '').replace(' ', '').isdigit():
                total_budget += int(str(budget).replace(',', '').replace(' ', ''))
                valid_budgets += 1
            
            duration = grant.get('Duration (months)')
            if pd.notna(duration):
                try:
                    total_duration += float(duration)
                    valid_durations += 1
                except:
                    pass
        
        analysis['total_budget'] = total_budget if valid_budgets > 0 else 'N/A'
        analysis['avg_budget'] = total_budget / valid_budgets if valid_budgets > 0 else 'N/A'
        analysis['avg_duration'] = total_duration / valid_durations if valid_durations > 0 else 'N/A'
        
        # Analyze grant overlap patterns (single vs multi-portfolio)
        single_portfolio_grants = []
        multi_portfolio_grants = []
        
        for grant in subportfolio_grants:
            portfolio_count = 0
            grant_portfolios = []
            for sp in self.subportfolios:
                if pd.notna(grant.get(sp)) and str(grant[sp]).strip().upper() in ['PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)']:
                    portfolio_count += 1
                    grant_portfolios.append(sp.replace(' sub-portfolio', ''))
            
            grant_info = {
                'gams_code': grant['GAMS Code'], 
                'short_name': grant.get('Grant\'s short name', 'N/A'),
                'portfolios': grant_portfolios
            }
            
            if portfolio_count == 1:
                single_portfolio_grants.append(grant_info)
            else:
                multi_portfolio_grants.append(grant_info)
        
        analysis['single_portfolio_grants'] = single_portfolio_grants
        analysis['multi_portfolio_grants'] = multi_portfolio_grants
        
        # Geographic distribution
        countries_list = []
        for country_str in subportfolio_outcomes['Country(ies) / Global'].dropna():
            countries_list.extend([c.strip() for c in str(country_str).split(',') if c.strip()])
        analysis['geographic_distribution'] = Counter(countries_list)
        
        # Significance analysis
        significance_counts = subportfolio_outcomes['Significance Level'].value_counts()
        analysis['significance_distribution'] = significance_counts.to_dict()
        
        # Special cases aggregated
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        analysis['special_cases'] = {}
        for col in special_cases_cols:
            if col in subportfolio_outcomes.columns:
                count = (subportfolio_outcomes[col] == 1).sum()
                if count > 0:
                    analysis['special_cases'][col] = count
        
        # Enhanced OB analysis for sub-portfolio
        analysis['ob_analysis'] = self.analyze_outcome_baskets_for_outcomes(subportfolio_outcomes)
        
        # Compare single vs multi-portfolio grants OB usage
        if single_portfolio_grants and multi_portfolio_grants:
            single_gams = [g['gams_code'] for g in single_portfolio_grants]
            multi_gams = [g['gams_code'] for g in multi_portfolio_grants]
            
            single_outcomes = subportfolio_outcomes[subportfolio_outcomes['GAMS code'].isin(single_gams)]
            multi_outcomes = subportfolio_outcomes[subportfolio_outcomes['GAMS code'].isin(multi_gams)]
            
            analysis['single_portfolio_ob_analysis'] = self.analyze_outcome_baskets_for_outcomes(single_outcomes)
            analysis['multi_portfolio_ob_analysis'] = self.analyze_outcome_baskets_for_outcomes(multi_outcomes)
        
        return analysis
    
    def create_theme_analysis(self):
        """Create comprehensive theme-wide analysis"""
        analysis = {
            'total_grants': len(self.individual_analyses),
            'total_outcomes': len(self.outcomes_df),
            'outcomes_data': self.outcomes_df
        }
        
        # Overall domain distribution
        domain_counts = self.outcomes_df['Domain of Change'].value_counts()
        analysis['domain_distribution'] = domain_counts.to_dict()
        
        # Sub-portfolio distribution analysis
        subportfolio_grant_counts = {}
        for sp in self.subportfolios:
            count = 0
            for _, grant in self.grants_df.iterrows():
                if pd.notna(grant.get(sp)) and str(grant[sp]).strip().upper() in ['PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)']:
                    count += 1
            subportfolio_grant_counts[sp] = count
        analysis['subportfolio_distribution'] = subportfolio_grant_counts
        
        # Budget analysis across theme
        total_budget = 0
        valid_budgets = 0
        budget_by_portfolio = {sp: 0 for sp in self.subportfolios}
        
        for _, grant in self.grants_df.iterrows():
            budget = grant.get('Budget (CHF)')
            if pd.notna(budget) and str(budget).replace(',', '').replace(' ', '').isdigit():
                budget_val = int(str(budget).replace(',', '').replace(' ', ''))
                total_budget += budget_val
                valid_budgets += 1
                
                # Add to relevant sub-portfolios
                for sp in self.subportfolios:
                    if pd.notna(grant.get(sp)) and str(grant[sp]).strip().upper() in ['PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)']:
                        budget_by_portfolio[sp] += budget_val
        
        analysis['total_theme_budget'] = total_budget
        analysis['avg_grant_budget'] = total_budget / valid_budgets if valid_budgets > 0 else 0
        analysis['budget_by_subportfolio'] = budget_by_portfolio
        
        # Geographic spread across theme
        countries_list = []
        for country_str in self.outcomes_df['Country(ies) / Global'].dropna():
            countries_list.extend([c.strip() for c in str(country_str).split(',') if c.strip()])
        analysis['geographic_distribution'] = Counter(countries_list)
        
        # Theme-wide significance patterns
        significance_counts = self.outcomes_df['Significance Level'].value_counts()
        analysis['significance_distribution'] = significance_counts.to_dict()
        
        # Special cases theme-wide
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        analysis['special_cases'] = {}
        for col in special_cases_cols:
            if col in self.outcomes_df.columns:
                count = (self.outcomes_df[col] == 1).sum()
                if count > 0:
                    analysis['special_cases'][col] = count
        
        # Comprehensive theme-wide OB analysis
        analysis['ob_analysis'] = self.analyze_outcome_baskets_for_outcomes(self.outcomes_df)
        
        # Cross-portfolio OB usage patterns
        analysis['ob_patterns_by_subportfolio'] = {}
        for sp_name in self.subportfolios:
            sp_analysis = self.subportfolio_analyses.get(sp_name)
            if sp_analysis:
                analysis['ob_patterns_by_subportfolio'][sp_name] = sp_analysis['ob_analysis']
        
        # Identify most common custom OBs theme-wide (candidates for standardization)
        all_custom_obs = analysis['ob_analysis']['custom']
        analysis['top_custom_obs'] = dict(all_custom_obs.most_common(10))
        
        return analysis

    def run_analysis(self):
        """Run the complete analysis workflow"""
        print("Starting comprehensive outcomes analysis...")
        
        # Load data
        self.load_data()
        
        # Extract default outcome baskets
        self.extract_default_outcome_baskets()
        
        # Individual grant analyses
        print("Analyzing individual grants...")
        unique_gams_codes = self.outcomes_df['GAMS code'].unique()
        
        for gams_code in unique_gams_codes:
            if pd.notna(gams_code):
                analysis = self.create_individual_grant_analysis(gams_code)
                if analysis:
                    self.individual_analyses[gams_code] = analysis
        
        print(f"Completed analysis for {len(self.individual_analyses)} grants")
        
        # Sub-portfolio analyses
        print("Analyzing sub-portfolios...")
        for subportfolio in self.subportfolios:
            analysis = self.create_subportfolio_analysis(subportfolio)
            if analysis:
                self.subportfolio_analyses[subportfolio] = analysis
        
        print(f"Completed analysis for {len(self.subportfolio_analyses)} sub-portfolios")
        
        # Theme-wide analysis
        print("Analyzing theme-wide patterns...")
        self.theme_analysis = self.create_theme_analysis()
        print("Theme-wide analysis completed")
    
    def generate_individual_grant_report(self, gams_code, analysis):
        """Generate markdown report for individual grant"""
        report = []
        
        # Header
        report.append(f"# Grant Analysis: {analysis['short_name']}")
        report.append(f"**GAMS Code:** {gams_code}")
        report.append(f"**Full Grant Name:** {analysis['grant_name']}")
        report.append("")
        
        # Basic metadata
        report.append("## Grant Metadata")
        report.append(f"- **Budget (CHF):** {analysis['budget']}")
        report.append(f"- **Duration (months):** {analysis['duration']}")
        report.append(f"- **Countries:** {analysis['countries']}")
        report.append(f"- **Total Outcomes:** {analysis['outcomes_count']}")
        report.append("")
        
        # Sub-portfolios
        if analysis['subportfolios']:
            report.append("## Sub-Portfolio Assignments")
            for sp in analysis['subportfolios']:
                report.append(f"- {sp}")
            report.append("")
        
        # Performance ratings
        if analysis['performance']:
            report.append("## Performance Ratings")
            for metric, rating in analysis['performance'].items():
                report.append(f"- **{metric}:** {rating}")
            report.append("")
        
        # Domain distribution
        report.append("## Outcomes by Domain of Change")
        total_outcomes = sum(analysis['domain_distribution'].values())
        for domain, count in analysis['domain_distribution'].items():
            percentage = (count / total_outcomes) * 100
            report.append(f"- **{domain}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Enhanced OB Analysis
        ob_analysis = analysis['ob_analysis']
        report.append("## Outcome Basket Usage Analysis")
        report.append("")
        
        # Default OBs for assigned domain
        if ob_analysis['default_assigned']:
            total_default_assigned = sum(ob_analysis['default_assigned'].values())
            percentage_default_assigned = (total_default_assigned / analysis['outcomes_count']) * 100
            
            report.append(f"### Default OBs for Assigned Domain: {total_default_assigned} occurrences ({percentage_default_assigned:.1f}%)")
            report.append("**Used Default OBs:**")
            for ob, count in ob_analysis['default_assigned'].most_common():
                ob_percentage = (count / analysis['outcomes_count']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Unused defaults
        if ob_analysis['unused_defaults']:
            report.append("**Unused Default OBs:**")
            for domain, unused_obs in ob_analysis['unused_defaults'].items():
                if unused_obs:
                    report.append(f"- **{domain}:**")
                    for ob in sorted(unused_obs):
                        report.append(f"  - \"{ob}\"")
            report.append("")
        
        # Default OBs from other domains
        if ob_analysis['default_other']:
            total_default_other = sum(ob_analysis['default_other'].values())
            percentage_default_other = (total_default_other / analysis['outcomes_count']) * 100
            
            report.append(f"### Default OBs from Different Domains: {total_default_other} occurrences ({percentage_default_other:.1f}%)")
            for ob, count in ob_analysis['default_other'].most_common():
                ob_percentage = (count / analysis['outcomes_count']) * 100
                # Find which domain this OB belongs to
                source_domain = "Unknown"
                for domain, obs_list in self.default_obs_by_domain.items():
                    if ob in obs_list:
                        source_domain = domain
                        break
                report.append(f"- \"{ob}\" (default for {source_domain}): {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Custom OBs
        if ob_analysis['custom']:
            total_custom = sum(ob_analysis['custom'].values())
            percentage_custom = (total_custom / analysis['outcomes_count']) * 100
            
            report.append(f"### Custom OBs: {total_custom} occurrences ({percentage_custom:.1f}%)")
            for ob, count in ob_analysis['custom'].most_common():
                ob_percentage = (count / analysis['outcomes_count']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Significance distribution
        if analysis['significance_distribution']:
            report.append("## Significance Level Distribution")
            for level, count in analysis['significance_distribution'].items():
                percentage = (count / analysis['outcomes_count']) * 100
                report.append(f"- **{level}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Countries involved in outcomes
        if analysis['outcome_countries']:
            report.append("## Countries/Regions in Outcomes")
            for country, count in analysis['outcome_countries'].most_common():
                percentage = (count / analysis['outcomes_count']) * 100
                report.append(f"- **{country}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Special cases
        if analysis['special_cases']:
            report.append("## Special Case Outcomes")
            for case_type, count in analysis['special_cases'].items():
                percentage = (count / analysis['outcomes_count']) * 100
                clean_type = case_type.replace('Special case - ', '')
                report.append(f"- **{clean_type}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Raw outcomes data
        report.append("## All Outcome Records")
        report.append("")
        
        # Create a simplified table of outcomes
        outcomes_data = analysis['outcomes_data']
        key_columns = ['Description of the Reported Outcome', 'Domain of Change', 
                      'Outcome Basket names', 'Significance Level', 'Country(ies) / Global']
        
        for i, (_, outcome) in enumerate(outcomes_data.iterrows(), 1):
            report.append(f"### Outcome {i}")
            for col in key_columns:
                if col in outcome and pd.notna(outcome[col]):
                    report.append(f"**{col}:** {outcome[col]}")
            report.append("")
        
        return "\n".join(report)
    
    def generate_subportfolio_report(self, subportfolio_name, analysis):
        """Generate markdown report for sub-portfolio"""
        report = []
        
        # Header
        clean_name = subportfolio_name.replace(' sub-portfolio', '')
        report.append(f"# Sub-Portfolio Analysis: {clean_name}")
        report.append("")
        
        # Description
        report.append("## Sub-Portfolio Description")
        report.append(f"**Core Focus:** {analysis['description']}")
        report.append("")
        report.append(f"**Focus Details:** {analysis['focus_details']}")
        report.append("")
        
        # Basic metrics
        report.append("## Overview Metrics")
        report.append(f"- **Total Grants:** {analysis['grants_count']}")
        report.append(f"- **Total Outcomes:** {analysis['total_outcomes']}")
        if analysis['total_budget'] != 'N/A':
            report.append(f"- **Total Budget (CHF):** {analysis['total_budget']:,}")
            report.append(f"- **Average Grant Budget (CHF):** {analysis['avg_budget']:,.0f}")
        if analysis['avg_duration'] != 'N/A':
            report.append(f"- **Average Grant Duration (months):** {analysis['avg_duration']:.1f}")
        report.append("")
        
        # Grants list
        report.append("## Grants in Sub-Portfolio")
        
        # Single portfolio grants
        if analysis['single_portfolio_grants']:
            report.append(f"### Single Portfolio Grants ({len(analysis['single_portfolio_grants'])})")
            for grant in analysis['single_portfolio_grants']:
                report.append(f"- **{grant['gams_code']}:** {grant['short_name']}")
            report.append("")
        
        # Multi-portfolio grants
        if analysis['multi_portfolio_grants']:
            report.append(f"### Multi-Portfolio Grants ({len(analysis['multi_portfolio_grants'])})")
            for grant in analysis['multi_portfolio_grants']:
                portfolios_str = ", ".join(grant['portfolios'])
                report.append(f"- **{grant['gams_code']}:** {grant['short_name']} (also in: {portfolios_str})")
            report.append("")
        
        # Domain distribution
        report.append("## Outcomes by Domain of Change")
        total_outcomes = sum(analysis['domain_distribution'].values())
        for domain, count in analysis['domain_distribution'].items():
            percentage = (count / total_outcomes) * 100
            report.append(f"- **{domain}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Enhanced OB Analysis for sub-portfolio
        ob_analysis = analysis['ob_analysis']
        report.append("## Sub-Portfolio Outcome Basket Usage Analysis")
        report.append("")
        
        # Default OBs for assigned domain
        if ob_analysis['default_assigned']:
            total_default_assigned = sum(ob_analysis['default_assigned'].values())
            percentage_default_assigned = (total_default_assigned / analysis['total_outcomes']) * 100
            
            report.append(f"### Default OBs for Assigned Domains: {total_default_assigned} occurrences ({percentage_default_assigned:.1f}%)")
            report.append("**Most Used Default OBs:**")
            for ob, count in ob_analysis['default_assigned'].most_common(10):
                ob_percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Unused defaults across sub-portfolio
        if ob_analysis['unused_defaults']:
            report.append("**Unused Default OBs in Sub-Portfolio:**")
            for domain, unused_obs in ob_analysis['unused_defaults'].items():
                if unused_obs:
                    report.append(f"- **{domain} (unused: {len(unused_obs)}):**")
                    for ob in sorted(list(unused_obs)[:5]):  # Show first 5
                        report.append(f"  - \"{ob}\"")
                    if len(unused_obs) > 5:
                        report.append(f"  - ... and {len(unused_obs) - 5} more")
            report.append("")
        
        # Default OBs from other domains
        if ob_analysis['default_other']:
            total_default_other = sum(ob_analysis['default_other'].values())
            percentage_default_other = (total_default_other / analysis['total_outcomes']) * 100
            
            report.append(f"### Cross-Domain Default OBs: {total_default_other} occurrences ({percentage_default_other:.1f}%)")
            for ob, count in ob_analysis['default_other'].most_common(10):
                ob_percentage = (count / analysis['total_outcomes']) * 100
                # Find source domain
                source_domain = "Unknown"
                for domain, obs_list in self.default_obs_by_domain.items():
                    if ob in obs_list:
                        source_domain = domain
                        break
                report.append(f"- \"{ob}\" (default for {source_domain}): {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Custom OBs
        if ob_analysis['custom']:
            total_custom = sum(ob_analysis['custom'].values())
            percentage_custom = (total_custom / analysis['total_outcomes']) * 100
            
            report.append(f"### Custom OBs: {total_custom} occurrences ({percentage_custom:.1f}%)")
            report.append("**Most Common Custom OBs:**")
            for ob, count in ob_analysis['custom'].most_common(10):
                ob_percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Single vs Multi-portfolio comparison
        if 'single_portfolio_ob_analysis' in analysis and 'multi_portfolio_ob_analysis' in analysis:
            report.append("## Single vs Multi-Portfolio Grants OB Comparison")
            
            single_custom = len(analysis['single_portfolio_ob_analysis']['custom'])
            multi_custom = len(analysis['multi_portfolio_ob_analysis']['custom'])
            single_total = analysis['single_portfolio_ob_analysis']['total_outcomes']
            multi_total = analysis['multi_portfolio_ob_analysis']['total_outcomes']
            
            report.append(f"- **Single Portfolio Grants:** {single_total} outcomes, {single_custom} custom OB types")
            report.append(f"- **Multi-Portfolio Grants:** {multi_total} outcomes, {multi_custom} custom OB types")
            report.append("")
        
        # Geographic distribution
        if analysis['geographic_distribution']:
            report.append("## Geographic Distribution")
            for country, count in analysis['geographic_distribution'].most_common(10):
                percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- **{country}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Significance distribution
        if analysis['significance_distribution']:
            report.append("## Significance Level Distribution")
            for level, count in analysis['significance_distribution'].items():
                percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- **{level}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Special cases
        if analysis['special_cases']:
            report.append("## Special Case Outcomes")
            for case_type, count in analysis['special_cases'].items():
                percentage = (count / analysis['total_outcomes']) * 100
                clean_type = case_type.replace('Special case - ', '')
                report.append(f"- **{clean_type}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        return "\n".join(report)
    
    def generate_theme_report(self, analysis):
        """Generate markdown report for theme-wide analysis"""
        report = []
        
        # Header
        report.append("# Digital Theme Comprehensive Analysis")
        report.append(f"*Analysis generated on {self.timestamp.replace('.', '/')}*")
        report.append("")
        
        # Overview
        report.append("## Theme Overview")
        report.append(f"- **Total Grants Analyzed:** {analysis['total_grants']}")
        report.append(f"- **Total Outcomes:** {analysis['total_outcomes']}")
        if analysis['total_theme_budget'] > 0:
            report.append(f"- **Total Theme Budget (CHF):** {analysis['total_theme_budget']:,}")
            report.append(f"- **Average Grant Budget (CHF):** {analysis['avg_grant_budget']:,.0f}")
        report.append("")
        
        # Sub-portfolio distribution
        report.append("## Sub-Portfolio Distribution")
        for sp, count in analysis['subportfolio_distribution'].items():
            sp_name = sp.replace(' sub-portfolio', '')
            percentage = (count / analysis['total_grants']) * 100
            budget = analysis['budget_by_subportfolio'].get(sp, 0)
            if budget > 0:
                report.append(f"- **{sp_name}:** {count} grants ({percentage:.1f}%) - Budget: {budget:,} CHF")
            else:
                report.append(f"- **{sp_name}:** {count} grants ({percentage:.1f}%)")
        report.append("")
        
        # Domain distribution theme-wide
        report.append("## Theme-Wide Domain Distribution")
        total_outcomes = sum(analysis['domain_distribution'].values())
        for domain, count in analysis['domain_distribution'].items():
            percentage = (count / total_outcomes) * 100
            report.append(f"- **{domain}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Comprehensive OB Analysis
        ob_analysis = analysis['ob_analysis']
        report.append("## Theme-Wide Outcome Basket Analysis")
        report.append("")
        
        # Default OBs usage summary
        if ob_analysis['default_assigned']:
            total_default_assigned = sum(ob_analysis['default_assigned'].values())
            percentage_default_assigned = (total_default_assigned / analysis['total_outcomes']) * 100
            
            report.append(f"### Default OBs Usage: {total_default_assigned} occurrences ({percentage_default_assigned:.1f}%)")
            report.append("**Top 15 Most Used Default OBs:**")
            for ob, count in ob_analysis['default_assigned'].most_common(15):
                ob_percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Comprehensive unused defaults analysis
        if ob_analysis['unused_defaults']:
            report.append("### Unused Default Outcome Baskets (Potential Gaps)")
            total_unused = sum(len(unused_set) for unused_set in ob_analysis['unused_defaults'].values())
            report.append(f"**Total unused default OBs across all domains: {total_unused}**")
            report.append("")
            
            for domain, unused_obs in ob_analysis['unused_defaults'].items():
                if unused_obs:
                    total_defaults = len(self.default_obs_by_domain.get(domain, []))
                    used_defaults = total_defaults - len(unused_obs)
                    usage_rate = (used_defaults / total_defaults) * 100 if total_defaults > 0 else 0
                    
                    report.append(f"**{domain} (Usage rate: {usage_rate:.1f}% - {len(unused_obs)} unused):**")
                    for ob in sorted(unused_obs):
                        report.append(f"  - \"{ob}\"")
                    report.append("")
        
        # Cross-domain OB usage
        if ob_analysis['default_other']:
            total_default_other = sum(ob_analysis['default_other'].values())
            percentage_default_other = (total_default_other / analysis['total_outcomes']) * 100
            
            report.append(f"### Cross-Domain Default OBs: {total_default_other} occurrences ({percentage_default_other:.1f}%)")
            report.append("**Most Common Cross-Domain Usage:**")
            for ob, count in ob_analysis['default_other'].most_common(15):
                ob_percentage = (count / analysis['total_outcomes']) * 100
                # Find source domain
                source_domain = "Unknown"
                for domain, obs_list in self.default_obs_by_domain.items():
                    if ob in obs_list:
                        source_domain = domain
                        break
                report.append(f"- \"{ob}\" (default for {source_domain}): {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Custom OBs - candidates for standardization
        if analysis['top_custom_obs']:
            total_custom = sum(ob_analysis['custom'].values())
            percentage_custom = (total_custom / analysis['total_outcomes']) * 100
            
            report.append(f"### Custom OBs - Standardization Candidates: {total_custom} occurrences ({percentage_custom:.1f}%)")
            report.append("**Top Custom OBs (Potential for Standardization):**")
            for ob, count in analysis['top_custom_obs'].items():
                ob_percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- \"{ob}\": {count} ({ob_percentage:.1f}%)")
            report.append("")
        
        # Cross-portfolio OB patterns
        if analysis['ob_patterns_by_subportfolio']:
            report.append("## OB Usage Patterns by Sub-Portfolio")
            for sp, sp_ob_analysis in analysis['ob_patterns_by_subportfolio'].items():
                sp_name = sp.replace(' sub-portfolio', '')
                
                # Calculate percentages for this subportfolio
                sp_total = sp_ob_analysis['total_outcomes']
                sp_default_assigned = sum(sp_ob_analysis['default_assigned'].values())
                sp_custom = sum(sp_ob_analysis['custom'].values())
                
                default_pct = (sp_default_assigned / sp_total) * 100 if sp_total > 0 else 0
                custom_pct = (sp_custom / sp_total) * 100 if sp_total > 0 else 0
                
                report.append(f"### {sp_name}")
                report.append(f"- **Total Outcomes:** {sp_total}")
                report.append(f"- **Default OB Usage:** {sp_default_assigned} ({default_pct:.1f}%)")
                report.append(f"- **Custom OB Usage:** {sp_custom} ({custom_pct:.1f}%)")
                report.append(f"- **Unique Custom OBs:** {len(sp_ob_analysis['custom'])}")
                report.append("")
        
        # Geographic distribution theme-wide
        if analysis['geographic_distribution']:
            report.append("## Theme-Wide Geographic Distribution")
            total_geo_outcomes = sum(analysis['geographic_distribution'].values())
            report.append(f"**Total geographic outcome instances:** {total_geo_outcomes}")
            report.append("")
            for country, count in analysis['geographic_distribution'].most_common(15):
                percentage = (count / total_geo_outcomes) * 100
                report.append(f"- **{country}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Significance patterns
        if analysis['significance_distribution']:
            report.append("## Theme-Wide Significance Patterns")
            for level, count in analysis['significance_distribution'].items():
                percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- **{level}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        # Special cases
        if analysis['special_cases']:
            report.append("## Theme-Wide Special Cases")
            for case_type, count in analysis['special_cases'].items():
                percentage = (count / analysis['total_outcomes']) * 100
                clean_type = case_type.replace('Special case - ', '')
                report.append(f"- **{clean_type}:** {count} ({percentage:.1f}%)")
            report.append("")
        
        return "\n".join(report)
    
    def generate_all_reports(self):
        """Generate all analysis reports"""
        print("Generating analysis reports...")
        
        # Create output directories
        individual_dir = self.output_dir / f"{self.timestamp}_individual_grant_analyses"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        subportfolio_dir = self.output_dir / f"{self.timestamp}_subportfolio_analyses"
        subportfolio_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual grant reports
        for gams_code, analysis in self.individual_analyses.items():
            short_name = analysis['short_name'].replace(' ', '_').replace('/', '_')
            filename = f"{gams_code}_{short_name}_analysis.md"
            filepath = individual_dir / filename
            
            report_content = self.generate_individual_grant_report(gams_code, analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        print(f"Generated {len(self.individual_analyses)} individual grant reports")
        
        # Generate sub-portfolio reports
        for sp_name, analysis in self.subportfolio_analyses.items():
            clean_name = sp_name.replace(' sub-portfolio', '').replace(' ', '_').replace('&', 'and')
            filename = f"{clean_name}_analysis.md"
            filepath = subportfolio_dir / filename
            
            report_content = self.generate_subportfolio_report(sp_name, analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        print(f"Generated {len(self.subportfolio_analyses)} sub-portfolio reports")
        
        # Generate theme-wide report
        theme_filename = f"{self.timestamp}_digital_theme_comprehensive_analysis.md"
        theme_filepath = self.output_dir / theme_filename
        
        theme_report_content = self.generate_theme_report(self.theme_analysis)
        
        with open(theme_filepath, 'w', encoding='utf-8') as f:
            f.write(theme_report_content)
        
        print("Generated comprehensive theme-wide analysis report")

if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = DigitalGrantsOutcomesAnalyzer()
    analyzer.run_analysis()
    analyzer.generate_all_reports()
    
    print("Analysis completed successfully!")