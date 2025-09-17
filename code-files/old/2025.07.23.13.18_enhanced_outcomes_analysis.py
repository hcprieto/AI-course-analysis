#!/usr/bin/env python3
"""
Enhanced Digital Grants Outcomes Analysis with Phase Consolidation
Created: 2025.07.23.13.18

This script performs comprehensive analysis of Digital theme grant outcomes with:
1. Two-phase grant consolidation 
2. Domain-organized Outcome Basket analysis
3. Analysis at three levels: Individual Grant, Sub-Portfolio, Theme-Wide

Key enhancements:
- Consolidates grant phases into single analytical units
- Organizes OB analysis by Domain of Change rather than aggregated
- Provides detailed cross-domain usage patterns
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

class EnhancedDigitalGrantsAnalyzer:
    """
    Enhanced analyzer with phase consolidation and domain-centric OB analysis
    """
    
    def __init__(self, source_dir="source-files", output_dir="analysis-output-files"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.timestamp = "2025.07.23.13.34"
        
        # Data containers
        self.outcomes_df = None
        self.grants_df = None
        self.key_concepts_df = None
        self.portfolio_descriptions_df = None
        
        # Consolidated data
        self.consolidated_grants = {}
        self.phase_relationships = {}
        
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
        
        # Comprehensive sub-portfolio value detection
        self.valid_portfolio_values = [
            'X', 'x', 'YES', '1', 
            'PRIMARY', 'SECONDARY', 'TERTIARY', 'SECONDARY (WEAK)',
            'Primary', 'Secondary', 'Tertiary'
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
    
    def build_phase_relationships(self):
        """Build mapping of phase 1 to phase 2 grant relationships"""
        print("Building phase relationships...")
        
        self.phase_relationships = {}
        for _, grant in self.grants_df.iterrows():
            if pd.notna(grant['GAMS code - next phase']):
                phase1_code = grant['GAMS Code']
                phase2_code = grant['GAMS code - next phase']
                self.phase_relationships[phase1_code] = phase2_code
        
        print(f"Found {len(self.phase_relationships)} grant pairs with phase relationships")
        for p1, p2 in self.phase_relationships.items():
            print(f"  {p1} → {p2}")
    
    def consolidate_grants(self):
        """Consolidate two-phase grants into single analytical units"""
        print("Consolidating grants...")
        
        processed_codes = set()
        
        for _, grant in self.grants_df.iterrows():
            gams_code = grant['GAMS Code']
            
            if gams_code in processed_codes:
                continue
            
            if gams_code in self.phase_relationships:
                # This is Phase 1 - consolidate with Phase 2
                phase2_code = self.phase_relationships[gams_code]
                
                # Check if Phase 2 exists in grants data
                phase2_grants = self.grants_df[self.grants_df['GAMS Code'] == phase2_code]
                if len(phase2_grants) == 0:
                    # Phase 2 doesn't exist, treat as single grant
                    consolidated_grant = grant.copy()
                    consolidated_grant['consolidated_gams_codes'] = [gams_code]
                    consolidated_grant['is_consolidated'] = False
                    self.consolidated_grants[gams_code] = consolidated_grant
                    processed_codes.add(gams_code)
                    continue
                
                phase2_grant = phase2_grants.iloc[0]
                
                # Use Phase 2 as primary record, merge phase 1 data where needed
                consolidated_grant = phase2_grant.copy()
                
                # Track consolidation info
                consolidated_grant['consolidated_gams_codes'] = [gams_code, phase2_code]
                consolidated_grant['phase_1_code'] = gams_code
                consolidated_grant['phase_2_code'] = phase2_code
                consolidated_grant['is_consolidated'] = True
                
                # Clean up name (remove phase indicators)
                base_name = str(consolidated_grant['Grant\'s short name'])
                base_name = base_name.replace(', phase 2', '').replace(', phase 1', '')
                base_name = base_name.replace(' phase 2', '').replace(' phase 1', '')
                consolidated_grant['Grant\'s short name'] = base_name
                
                # Use consolidated identifier (phase 2 code for consistency)
                self.consolidated_grants[phase2_code] = consolidated_grant
                processed_codes.add(gams_code)
                processed_codes.add(phase2_code)
                
            elif gams_code not in [p2 for p2 in self.phase_relationships.values()]:
                # Single-phase grant
                consolidated_grant = grant.copy()
                consolidated_grant['consolidated_gams_codes'] = [gams_code]
                consolidated_grant['is_consolidated'] = False
                self.consolidated_grants[gams_code] = consolidated_grant
                processed_codes.add(gams_code)
        
        print(f"Consolidated into {len(self.consolidated_grants)} analytical units")
        
        # Show consolidation summary
        consolidated_count = sum(1 for g in self.consolidated_grants.values() if g['is_consolidated'])
        single_count = len(self.consolidated_grants) - consolidated_count
        print(f"  {consolidated_count} consolidated grants (from {consolidated_count * 2} phase records)")
        print(f"  {single_count} single-phase grants")
    
    def parse_outcome_baskets(self, ob_string):
        """Parse semi-colon separated outcome baskets"""
        if pd.isna(ob_string) or not str(ob_string).strip():
            return []
        
        # Split by semicolon and clean
        obs = [ob.strip() for ob in str(ob_string).split(';') if ob.strip()]
        return obs
    
    def classify_outcome_basket(self, ob, assigned_domain):
        """
        Classify an outcome basket and return detailed classification info
        """
        # Extract domain key from full domain string
        domain_key = assigned_domain.split(' - ')[0] if ' - ' in assigned_domain else assigned_domain
        
        # Check if it's a default for the assigned domain
        if ob in self.default_obs_by_domain.get(domain_key, []):
            return {
                'category': 'default_assigned',
                'source_domain': domain_key,
                'ob': ob
            }
        
        # Check if it's a default for another domain
        for other_domain, obs_list in self.default_obs_by_domain.items():
            if ob in obs_list:
                return {
                    'category': 'cross_domain',
                    'source_domain': other_domain,
                    'ob': ob
                }
        
        # Must be custom
        return {
            'category': 'custom',
            'source_domain': None,
            'ob': ob
        }
    
    def analyze_outcome_baskets_by_domain(self, outcomes_subset):
        """
        Enhanced domain-centric analysis of outcome baskets
        """
        analysis = {
            'by_domain': {},
            'cross_domain_summary': {
                'most_borrowed_obs': Counter(),
                'most_lending_domains': Counter(),
                'most_borrowing_domains': Counter()
            },
            'total_outcomes': len(outcomes_subset)
        }
        
        # Initialize domain analysis
        domains_in_subset = set(outcomes_subset['Domain of Change'].dropna())
        for domain_full in domains_in_subset:
            domain_key = domain_full.split(' - ')[0] if ' - ' in domain_full else domain_full
            
            analysis['by_domain'][domain_key] = {
                'domain_full_name': domain_full,
                'total_outcomes': 0,
                'default_obs_for_domain': {
                    'used': Counter(),
                    'unused': set(self.default_obs_by_domain.get(domain_key, []))
                },
                'domain_obs_used_by_others': Counter(),
                'cross_domain_obs_used': Counter(),
                'custom_obs': Counter()
            }
        
        # Process each outcome
        for _, outcome in outcomes_subset.iterrows():
            assigned_domain = outcome['Domain of Change']
            if pd.isna(assigned_domain):
                continue
                
            domain_key = assigned_domain.split(' - ')[0] if ' - ' in assigned_domain else assigned_domain
            
            if domain_key not in analysis['by_domain']:
                continue
            
            analysis['by_domain'][domain_key]['total_outcomes'] += 1
            
            ob_string = outcome['Outcome Basket names']
            obs = self.parse_outcome_baskets(ob_string)
            
            for ob in obs:
                classification = self.classify_outcome_basket(ob, assigned_domain)
                
                if classification['category'] == 'default_assigned':
                    # This domain's default OB used correctly
                    analysis['by_domain'][domain_key]['default_obs_for_domain']['used'][ob] += 1
                    analysis['by_domain'][domain_key]['default_obs_for_domain']['unused'].discard(ob)
                    
                elif classification['category'] == 'cross_domain':
                    # Another domain's default OB used here
                    source_domain = classification['source_domain']
                    ob_label = f"{ob} ({source_domain})"
                    analysis['by_domain'][domain_key]['cross_domain_obs_used'][ob_label] += 1
                    
                    # Track for cross-domain summary
                    analysis['cross_domain_summary']['most_borrowed_obs'][ob] += 1
                    analysis['cross_domain_summary']['most_lending_domains'][source_domain] += 1
                    analysis['cross_domain_summary']['most_borrowing_domains'][domain_key] += 1
                    
                    # Track that this source domain's OB was used by others
                    if source_domain in analysis['by_domain']:
                        analysis['by_domain'][source_domain]['domain_obs_used_by_others'][f"{ob} → {domain_key}"] += 1
                    
                elif classification['category'] == 'custom':
                    # Custom OB
                    analysis['by_domain'][domain_key]['custom_obs'][ob] += 1
        
        return analysis
    
    def is_in_subportfolio(self, grant, subportfolio_name):
        """Check if grant belongs to subportfolio with enhanced value detection"""
        if subportfolio_name not in grant:
            return False
        
        value = grant[subportfolio_name]
        if pd.isna(value):
            return False
        
        return str(value).strip() in self.valid_portfolio_values
    
    def create_individual_grant_analysis(self, consolidated_grant_id):
        """Create detailed analysis for a consolidated grant"""
        consolidated_grant = self.consolidated_grants[consolidated_grant_id]
        
        # Get all outcomes for this consolidated grant (from all phases)
        gams_codes = consolidated_grant['consolidated_gams_codes']
        grant_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].isin(gams_codes)]
        
        if len(grant_outcomes) == 0:
            return None  # No outcomes for this grant
        
        # Basic grant metadata
        analysis = {
            'consolidated_grant_id': consolidated_grant_id,
            'gams_codes': gams_codes,
            'is_consolidated': consolidated_grant['is_consolidated'],
            'grant_name': consolidated_grant.get('Grant\'s Name', 'N/A'),
            'short_name': consolidated_grant.get('Grant\'s short name', 'N/A'),
            'budget': consolidated_grant.get('Phase 1 + 2 Budget (CHF)' if consolidated_grant['is_consolidated'] else 'Budget (CHF)', 'N/A'),
            'duration': consolidated_grant.get('Phase 1 + 2 Duration' if consolidated_grant['is_consolidated'] else 'Duration (months)', 'N/A'),
            'countries': consolidated_grant.get('Countries', 'N/A'),
            'outcomes_count': len(grant_outcomes),
            'outcomes_data': grant_outcomes
        }
        
        # Sub-portfolio assignments
        analysis['subportfolios'] = []
        for sp in self.subportfolios:
            if self.is_in_subportfolio(consolidated_grant, sp):
                analysis['subportfolios'].append(sp)
        
        # Performance ratings
        performance_cols = ['Implementation', 'Contribution to strategic intent', 'Learning', 
                          'Sustainability of changes', 'Stakeholder engagement and collaboration', 
                          'Meaningful youth participation']
        analysis['performance'] = {}
        for col in performance_cols:
            if col in consolidated_grant:
                analysis['performance'][col] = consolidated_grant.get(col, 'N/A')
        
        # Domain distribution
        domain_counts = grant_outcomes['Domain of Change'].value_counts()
        analysis['domain_distribution'] = domain_counts.to_dict()
        
        # Significance distribution (including null/empty values)
        significance_counts = grant_outcomes['Significance Level'].value_counts(dropna=False)
        # Replace NaN key with 'Not specified' for better readability
        significance_dict = significance_counts.to_dict()
        if pd.isna(list(significance_dict.keys())).any():
            nan_count = significance_dict.pop(np.nan, 0)
            if nan_count > 0:
                significance_dict['Not specified'] = nan_count
        analysis['significance_distribution'] = significance_dict
        
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
        
        # Enhanced domain-centric OB analysis
        analysis['ob_analysis'] = self.analyze_outcome_baskets_by_domain(grant_outcomes)
        
        return analysis
    
    def create_subportfolio_analysis(self, subportfolio_name):
        """Create detailed analysis for a sub-portfolio with consolidated grants"""
        # Find consolidated grants in this sub-portfolio
        subportfolio_grants = []
        subportfolio_grant_ids = []
        
        for grant_id, consolidated_grant in self.consolidated_grants.items():
            if self.is_in_subportfolio(consolidated_grant, subportfolio_name):
                if grant_id in self.individual_analyses:
                    subportfolio_grants.append(consolidated_grant)
                    subportfolio_grant_ids.append(grant_id)
        
        if not subportfolio_grants:
            return None
        
        # Get all outcomes for grants in this sub-portfolio
        all_gams_codes = []
        for grant_id in subportfolio_grant_ids:
            all_gams_codes.extend(self.consolidated_grants[grant_id]['consolidated_gams_codes'])
        
        subportfolio_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].isin(all_gams_codes)]
        
        # Basic sub-portfolio info
        analysis = {
            'name': subportfolio_name,
            'grants_count': len(subportfolio_grants),
            'consolidated_grants': [(g_id, self.consolidated_grants[g_id].get('Grant\'s short name', 'N/A'), 
                                   self.consolidated_grants[g_id]['is_consolidated']) for g_id in subportfolio_grant_ids],
            'total_outcomes': len(subportfolio_outcomes),
            'outcomes_data': subportfolio_outcomes
        }
        
        # Get sub-portfolio description
        sp_clean = subportfolio_name.replace(' sub-portfolio', '').replace('&', 'and')
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
        
        # Budget and duration aggregation from consolidated grants
        total_budget = 0
        valid_budgets = 0
        total_duration = 0
        valid_durations = 0
        
        for grant in subportfolio_grants:
            # Use consolidated budget if available
            budget_col = 'Phase 1 + 2 Budget (CHF)' if grant['is_consolidated'] else 'Budget (CHF)'
            budget = grant.get(budget_col)
            if pd.notna(budget) and str(budget).replace(',', '').replace(' ', '').isdigit():
                total_budget += int(str(budget).replace(',', '').replace(' ', ''))
                valid_budgets += 1
            
            # Use consolidated duration if available
            duration_col = 'Phase 1 + 2 Duration' if grant['is_consolidated'] else 'Duration (months)'
            duration = grant.get(duration_col)
            if pd.notna(duration):
                try:
                    total_duration += float(duration)
                    valid_durations += 1
                except:
                    pass
        
        analysis['total_budget'] = total_budget if valid_budgets > 0 else 'N/A'
        analysis['avg_budget'] = total_budget / valid_budgets if valid_budgets > 0 else 'N/A'
        analysis['avg_duration'] = total_duration / valid_durations if valid_durations > 0 else 'N/A'
        
        # Geographic distribution
        countries_list = []
        for country_str in subportfolio_outcomes['Country(ies) / Global'].dropna():
            countries_list.extend([c.strip() for c in str(country_str).split(',') if c.strip()])
        analysis['geographic_distribution'] = Counter(countries_list)
        
        # Significance analysis (including null/empty values)
        significance_counts = subportfolio_outcomes['Significance Level'].value_counts(dropna=False)
        # Replace NaN key with 'Not specified' for better readability
        significance_dict = significance_counts.to_dict()
        if pd.isna(list(significance_dict.keys())).any():
            nan_count = significance_dict.pop(np.nan, 0)
            if nan_count > 0:
                significance_dict['Not specified'] = nan_count
        analysis['significance_distribution'] = significance_dict
        
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
        
        # Enhanced domain-centric OB analysis for sub-portfolio
        analysis['ob_analysis'] = self.analyze_outcome_baskets_by_domain(subportfolio_outcomes)
        
        # Add consolidated grants info for outcome records section
        analysis['consolidated_grants_info'] = {}
        for grant_id in subportfolio_grant_ids:
            analysis['consolidated_grants_info'][grant_id] = self.consolidated_grants[grant_id]
        
        return analysis
    
    def create_theme_analysis(self):
        """Create comprehensive theme-wide analysis with consolidated grants"""
        # Get all outcomes from all consolidated grants with outcomes
        all_gams_codes = []
        for grant_id in self.individual_analyses.keys():
            all_gams_codes.extend(self.consolidated_grants[grant_id]['consolidated_gams_codes'])
        
        theme_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].isin(all_gams_codes)]
        
        analysis = {
            'total_consolidated_grants': len(self.individual_analyses),
            'total_outcomes': len(theme_outcomes),
            'outcomes_data': theme_outcomes
        }
        
        # Overall domain distribution
        domain_counts = theme_outcomes['Domain of Change'].value_counts()
        analysis['domain_distribution'] = domain_counts.to_dict()
        
        # Sub-portfolio distribution analysis (consolidated grants)
        subportfolio_grant_counts = {}
        for sp in self.subportfolios:
            count = 0
            for consolidated_grant in self.consolidated_grants.values():
                if self.is_in_subportfolio(consolidated_grant, sp):
                    count += 1
            subportfolio_grant_counts[sp] = count
        analysis['subportfolio_distribution'] = subportfolio_grant_counts
        
        # Budget analysis across theme (using consolidated budgets)
        total_budget = 0
        valid_budgets = 0
        budget_by_portfolio = {sp: 0 for sp in self.subportfolios}
        
        for consolidated_grant in self.consolidated_grants.values():
            # Use consolidated budget if available
            budget_col = 'Phase 1 + 2 Budget (CHF)' if consolidated_grant['is_consolidated'] else 'Budget (CHF)'
            budget = consolidated_grant.get(budget_col)
            if pd.notna(budget) and str(budget).replace(',', '').replace(' ', '').isdigit():
                budget_val = int(str(budget).replace(',', '').replace(' ', ''))
                total_budget += budget_val
                valid_budgets += 1
                
                # Add to relevant sub-portfolios
                for sp in self.subportfolios:
                    if self.is_in_subportfolio(consolidated_grant, sp):
                        budget_by_portfolio[sp] += budget_val
        
        analysis['total_theme_budget'] = total_budget
        analysis['avg_grant_budget'] = total_budget / valid_budgets if valid_budgets > 0 else 0
        analysis['budget_by_subportfolio'] = budget_by_portfolio
        
        # Geographic spread across theme
        countries_list = []
        for country_str in theme_outcomes['Country(ies) / Global'].dropna():
            countries_list.extend([c.strip() for c in str(country_str).split(',') if c.strip()])
        analysis['geographic_distribution'] = Counter(countries_list)
        
        # Theme-wide significance patterns (including null/empty values)
        significance_counts = theme_outcomes['Significance Level'].value_counts(dropna=False)
        # Replace NaN key with 'Not specified' for better readability
        significance_dict = significance_counts.to_dict()
        if pd.isna(list(significance_dict.keys())).any():
            nan_count = significance_dict.pop(np.nan, 0)
            if nan_count > 0:
                significance_dict['Not specified'] = nan_count
        analysis['significance_distribution'] = significance_dict
        
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
            if col in theme_outcomes.columns:
                count = (theme_outcomes[col] == 1).sum()
                if count > 0:
                    analysis['special_cases'][col] = count
        
        # Comprehensive theme-wide domain-centric OB analysis
        analysis['ob_analysis'] = self.analyze_outcome_baskets_by_domain(theme_outcomes)
        
        # Cross-portfolio OB usage patterns
        analysis['ob_patterns_by_subportfolio'] = {}
        for sp_name in self.subportfolios:
            sp_analysis = self.subportfolio_analyses.get(sp_name)
            if sp_analysis:
                analysis['ob_patterns_by_subportfolio'][sp_name] = sp_analysis['ob_analysis']
        
        # Identify most common custom OBs theme-wide (candidates for standardization)
        all_custom_obs = Counter()
        for domain_data in analysis['ob_analysis']['by_domain'].values():
            all_custom_obs.update(domain_data['custom_obs'])
        analysis['top_custom_obs'] = dict(all_custom_obs.most_common(10))
        
        return analysis
    
    def generate_domain_ob_report_section(self, ob_analysis, title_level="##"):
        """Generate domain-organized OB analysis section for reports"""
        report = []
        
        report.append(f"{title_level} Domain-Specific Outcome Basket Analysis")
        report.append("")
        
        if not ob_analysis['by_domain']:
            report.append("No domain-specific analysis available.")
            return report
        
        # Sort domains by total outcomes (descending)
        sorted_domains = sorted(ob_analysis['by_domain'].items(), 
                              key=lambda x: x[1]['total_outcomes'], reverse=True)
        
        for domain_key, domain_data in sorted_domains:
            domain_name = domain_data['domain_full_name']
            total_outcomes = domain_data['total_outcomes']
            
            report.append(f"### {domain_name} ({total_outcomes} outcomes)")
            report.append("")
            
            # Default OBs for this domain
            used_defaults = domain_data['default_obs_for_domain']['used']
            unused_defaults = domain_data['default_obs_for_domain']['unused']
            total_domain_defaults = len(self.default_obs_by_domain.get(domain_key, []))
            used_count = len(used_defaults)
            
            if total_domain_defaults > 0:
                usage_rate = (used_count / total_domain_defaults) * 100
                report.append(f"#### Default OBs for {domain_key} Domain (Usage: {used_count} of {total_domain_defaults} - {usage_rate:.1f}%)")
                
                if used_defaults:
                    report.append("**Used Default OBs:**")
                    for ob, count in used_defaults.most_common():
                        percentage = (count / total_outcomes) * 100
                        report.append(f"- \"{ob}\": {count} occurrences ({percentage:.1f}% of {domain_key} outcomes)")
                    report.append("")
                
                if unused_defaults:
                    report.append(f"**Unused Default OBs ({len(unused_defaults)} of {total_domain_defaults}):**")
                    for ob in sorted(unused_defaults):
                        report.append(f"- \"{ob}\"")
                    report.append("")
                
                # Cross-references: this domain's OBs used by others
                domain_obs_used_by_others = domain_data['domain_obs_used_by_others']
                if domain_obs_used_by_others:
                    report.append(f"**{domain_key} Default OBs Borrowed by Other Domains:**")
                    for ob_usage, count in domain_obs_used_by_others.most_common():
                        report.append(f"- {ob_usage}: {count} times")
                    report.append("")
            
            # Cross-domain OBs used in this domain
            cross_domain_obs = domain_data['cross_domain_obs_used']
            if cross_domain_obs:
                total_cross_domain = sum(cross_domain_obs.values())
                cross_domain_percentage = (total_cross_domain / total_outcomes) * 100
                
                report.append(f"#### Cross-Domain OBs Used in {domain_key} Outcomes ({cross_domain_percentage:.1f}% cross-domain usage)")
                for ob_label, count in cross_domain_obs.most_common():
                    percentage = (count / total_outcomes) * 100
                    report.append(f"- \"{ob_label}\": {count} occurrences ({percentage:.1f}%)")
                report.append("")
            
            # Custom OBs in this domain
            custom_obs = domain_data['custom_obs']
            if custom_obs:
                total_custom = sum(custom_obs.values())
                custom_percentage = (total_custom / total_outcomes) * 100
                
                report.append(f"#### Custom OBs in {domain_key} Outcomes ({custom_percentage:.1f}% custom usage)")
                for ob, count in custom_obs.most_common():
                    percentage = (count / total_outcomes) * 100
                    report.append(f"- \"{ob}\": {count} occurrences ({percentage:.1f}%)")
                report.append("")
            
            report.append("---")  # Separator between domains
            report.append("")
        
        # Cross-domain summary
        if ob_analysis['cross_domain_summary']['most_borrowed_obs']:
            report.append("### Cross-Domain Usage Summary")
            report.append("")
            
            report.append("**Most Borrowed OBs (across all domains):**")
            for ob, count in ob_analysis['cross_domain_summary']['most_borrowed_obs'].most_common(10):
                report.append(f"- \"{ob}\": {count} cross-domain uses")
            report.append("")
            
            if ob_analysis['cross_domain_summary']['most_lending_domains']:
                report.append("**Domains Most Borrowed From:**")
                for domain, count in ob_analysis['cross_domain_summary']['most_lending_domains'].most_common():
                    report.append(f"- **{domain}**: {count} times")
                report.append("")
            
            if ob_analysis['cross_domain_summary']['most_borrowing_domains']:
                report.append("**Domains That Borrow Most:**")
                for domain, count in ob_analysis['cross_domain_summary']['most_borrowing_domains'].most_common():
                    report.append(f"- **{domain}**: {count} cross-domain uses")
                report.append("")
        
        return report
    
    def generate_individual_grant_report(self, grant_id, analysis):
        """Generate markdown report for individual consolidated grant"""
        report = []
        
        # Header
        phase_info = " (Consolidated)" if analysis['is_consolidated'] else ""
        report.append(f"# Grant Analysis: {analysis['short_name']}{phase_info}")
        report.append(f"**Primary Grant ID:** {grant_id}")
        if analysis['is_consolidated']:
            report.append(f"**Consolidated GAMS Codes:** {', '.join(analysis['gams_codes'])}")
        report.append(f"**Full Grant Name:** {analysis['grant_name']}")
        report.append("")
        
        # Basic metadata
        report.append("## Grant Metadata")
        report.append(f"- **Budget (CHF):** {analysis['budget']}")
        report.append(f"- **Duration (months):** {analysis['duration']}")
        report.append(f"- **Countries:** {analysis['countries']}")
        report.append(f"- **Total Outcomes:** {analysis['outcomes_count']}")
        if analysis['is_consolidated']:
            report.append(f"- **Grant Type:** Two-phase consolidated grant")
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
                if pd.notna(rating) and str(rating) != 'nan':
                    report.append(f"- **{metric}:** {rating}")
            report.append("")
        
        # Domain distribution
        report.append("## Outcomes by Domain of Change")
        total_outcomes = sum(analysis['domain_distribution'].values())
        for domain, count in analysis['domain_distribution'].items():
            percentage = (count / total_outcomes) * 100
            report.append(f"- **{domain}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Enhanced domain-centric OB Analysis
        domain_report = self.generate_domain_ob_report_section(analysis['ob_analysis'])
        report.extend(domain_report)
        
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
        
        # All outcome records
        report.append("## All Outcome Records")
        report.append("")
        
        # Create a detailed list of all outcomes
        outcomes_data = analysis['outcomes_data'].copy()
        key_columns = ['Description of the Reported Outcome', 'Domain of Change', 
                      'Outcome Basket names', 'Significance Level', 'Country(ies) / Global',
                      'Additional details on the Outcome', 'Report name', 'Report submission date']
        
        for i, (_, outcome) in enumerate(outcomes_data.iterrows(), 1):
            report.append(f"### Outcome {i}")
            for col in key_columns:
                if col in outcome and pd.notna(outcome[col]) and str(outcome[col]).strip() not in ['', 'nan']:
                    report.append(f"**{col}:** {outcome[col]}")
            
            # Add special case indicators if any are true
            special_indicators = []
            special_cases_cols = [
                'Special case - Unexpected Outcome',
                'Special case - Negative Outcome', 
                'Special case - Signal towards Outcome',
                'Special case - Indicator Based Outcome',
                'Special case - Keep on Radar',
                'Special case - Programme Outcome'
            ]
            for col in special_cases_cols:
                if col in outcome and outcome[col] == 1:
                    clean_name = col.replace('Special case - ', '')
                    special_indicators.append(clean_name)
            
            if special_indicators:
                report.append(f"**Special Cases:** {', '.join(special_indicators)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def generate_subportfolio_report(self, analysis):
        """Generate markdown report for sub-portfolio"""
        report = []
        
        # Header
        clean_name = analysis['name'].replace(' sub-portfolio', '')
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
        report.append(f"- **Total Consolidated Grants:** {analysis['grants_count']}")
        report.append(f"- **Total Outcomes:** {analysis['total_outcomes']}")
        if analysis['total_budget'] != 'N/A':
            report.append(f"- **Total Budget (CHF):** {analysis['total_budget']:,}")
            report.append(f"- **Average Grant Budget (CHF):** {analysis['avg_budget']:,.0f}")
        if analysis['avg_duration'] != 'N/A':
            report.append(f"- **Average Grant Duration (months):** {analysis['avg_duration']:.1f}")
        report.append("")
        
        # Consolidated grants list
        report.append("## Consolidated Grants in Sub-Portfolio")
        for grant_id, short_name, is_consolidated in analysis['consolidated_grants']:
            phase_info = " (2-phase)" if is_consolidated else " (single-phase)"
            report.append(f"- **{grant_id}:** {short_name}{phase_info}")
        report.append("")
        
        # Domain distribution
        report.append("## Outcomes by Domain of Change")
        total_outcomes = sum(analysis['domain_distribution'].values())
        for domain, count in analysis['domain_distribution'].items():
            percentage = (count / total_outcomes) * 100
            report.append(f"- **{domain}:** {count} ({percentage:.1f}%)")
        report.append("")
        
        # Enhanced domain-centric OB Analysis for sub-portfolio
        domain_report = self.generate_domain_ob_report_section(analysis['ob_analysis'])
        report.extend(domain_report)
        
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
        
        # All outcome records for the sub-portfolio
        report.append("## All Outcome Records from Sub-Portfolio Grants")
        report.append("")
        
        # Sort outcomes by grant and then by order
        outcomes_data = analysis['outcomes_data'].copy()
        outcomes_data = outcomes_data.sort_values(['GAMS code']).reset_index(drop=True)
        
        key_columns = ['Description of the Reported Outcome', 'Domain of Change', 
                      'Outcome Basket names', 'Significance Level', 'Country(ies) / Global',
                      'Additional details on the Outcome', 'Report name', 'Report submission date']
        
        current_grant = None
        outcome_counter = 1
        
        for _, outcome in outcomes_data.iterrows():
            gams_code = outcome['GAMS code']
            grant_name = outcome.get('Grant Name', 'Unknown Grant')
            
            # Add grant header when we encounter a new grant
            if current_grant != gams_code:
                if current_grant is not None:
                    report.append("---")  # Separator between grants
                    report.append("")
                
                # Find consolidated grant info
                consolidated_info = "Unknown Grant"
                for grant_id, consolidated_grant in analysis.get('consolidated_grants_info', {}).items():
                    if gams_code in consolidated_grant.get('consolidated_gams_codes', []):
                        phase_info = " (2-phase)" if consolidated_grant.get('is_consolidated', False) else " (single-phase)"
                        consolidated_info = f"{consolidated_grant.get('short_name', grant_name)}{phase_info}"
                        break
                
                report.append(f"#### Grant: {consolidated_info} (GAMS: {gams_code})")
                report.append("")
                current_grant = gams_code
            
            report.append(f"### Outcome {outcome_counter}")
            for col in key_columns:
                if col in outcome and pd.notna(outcome[col]) and str(outcome[col]).strip() not in ['', 'nan']:
                    report.append(f"**{col}:** {outcome[col]}")
            
            # Add special case indicators if any are true
            special_indicators = []
            special_cases_cols = [
                'Special case - Unexpected Outcome',
                'Special case - Negative Outcome', 
                'Special case - Signal towards Outcome',
                'Special case - Indicator Based Outcome',
                'Special case - Keep on Radar',
                'Special case - Programme Outcome'
            ]
            for col in special_cases_cols:
                if col in outcome and outcome[col] == 1:
                    clean_name = col.replace('Special case - ', '')
                    special_indicators.append(clean_name)
            
            if special_indicators:
                report.append(f"**Special Cases:** {', '.join(special_indicators)}")
            
            report.append("")
            outcome_counter += 1
        
        return "\n".join(report)
    
    def generate_theme_report(self, analysis):
        """Generate markdown report for theme-wide analysis"""
        report = []
        
        # Header
        report.append("# Digital Theme Comprehensive Analysis (Enhanced)")
        report.append(f"*Enhanced analysis with phase consolidation generated on {self.timestamp.replace('.', '/')}*")
        report.append("")
        
        # Overview
        report.append("## Theme Overview")
        report.append(f"- **Total Consolidated Grants Analyzed:** {analysis['total_consolidated_grants']}")
        report.append(f"- **Total Outcomes:** {analysis['total_outcomes']}")
        if analysis['total_theme_budget'] > 0:
            report.append(f"- **Total Theme Budget (CHF):** {analysis['total_theme_budget']:,}")
            report.append(f"- **Average Grant Budget (CHF):** {analysis['avg_grant_budget']:,.0f}")
        report.append("")
        
        # Sub-portfolio distribution
        report.append("## Sub-Portfolio Distribution (Consolidated Grants)")
        for sp, count in analysis['subportfolio_distribution'].items():
            sp_name = sp.replace(' sub-portfolio', '')
            percentage = (count / analysis['total_consolidated_grants']) * 100
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
        
        # Comprehensive domain-centric OB Analysis
        domain_report = self.generate_domain_ob_report_section(analysis['ob_analysis'], title_level="##")
        report.extend(domain_report)
        
        # Custom OBs - candidates for standardization
        if analysis['top_custom_obs']:
            report.append("## Custom OBs - Standardization Candidates")
            report.append("**Top Custom OBs (Potential for Adding to Default Framework):**")
            for ob, count in analysis['top_custom_obs'].items():
                ob_percentage = (count / analysis['total_outcomes']) * 100
                report.append(f"- \"{ob}\": {count} occurrences ({ob_percentage:.1f}%)")
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
        """Generate all enhanced analysis reports"""
        print("Generating enhanced analysis reports...")
        
        # Create output directories
        individual_dir = self.output_dir / f"{self.timestamp}_individual_grant_analyses_with_outcomes"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        subportfolio_dir = self.output_dir / f"{self.timestamp}_subportfolio_analyses_with_outcomes"
        subportfolio_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate individual grant reports
        for grant_id, analysis in self.individual_analyses.items():
            short_name = analysis['short_name'].replace(' ', '_').replace('/', '_')
            phase_suffix = "_consolidated" if analysis['is_consolidated'] else "_single"
            filename = f"{grant_id}_{short_name}{phase_suffix}_analysis.md"
            filepath = individual_dir / filename
            
            report_content = self.generate_individual_grant_report(grant_id, analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        print(f"Generated {len(self.individual_analyses)} individual grant reports")
        
        # Generate sub-portfolio reports
        for sp_name, analysis in self.subportfolio_analyses.items():
            clean_name = sp_name.replace(' sub-portfolio', '').replace(' ', '_').replace('&', 'and')
            filename = f"{clean_name}_enhanced_analysis.md"
            filepath = subportfolio_dir / filename
            
            report_content = self.generate_subportfolio_report(analysis)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        print(f"Generated {len(self.subportfolio_analyses)} sub-portfolio reports")
        
        # Generate theme-wide report
        theme_filename = f"{self.timestamp}_digital_theme_enhanced_with_outcomes.md"
        theme_filepath = self.output_dir / theme_filename
        
        theme_report_content = self.generate_theme_report(self.theme_analysis)
        
        with open(theme_filepath, 'w', encoding='utf-8') as f:
            f.write(theme_report_content)
        
        print("Generated comprehensive enhanced theme-wide analysis report")
    
    def run_analysis(self):
        """Run the complete enhanced analysis workflow"""
        print("Starting enhanced outcomes analysis with phase consolidation...")
        
        # Load data
        self.load_data()
        
        # Extract default outcome baskets
        self.extract_default_outcome_baskets()
        
        # Build phase relationships and consolidate grants
        self.build_phase_relationships()
        self.consolidate_grants()
        
        # Individual grant analyses (consolidated)
        print("Analyzing consolidated grants...")
        
        for consolidated_grant_id in self.consolidated_grants.keys():
            analysis = self.create_individual_grant_analysis(consolidated_grant_id)
            if analysis:
                self.individual_analyses[consolidated_grant_id] = analysis
        
        print(f"Completed analysis for {len(self.individual_analyses)} consolidated grants")
        
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
        
        # Generate all reports
        self.generate_all_reports()

if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = EnhancedDigitalGrantsAnalyzer()
    analyzer.run_analysis()
    
    print("Enhanced analysis completed successfully!")