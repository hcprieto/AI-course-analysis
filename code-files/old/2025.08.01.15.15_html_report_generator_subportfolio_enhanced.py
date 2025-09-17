#!/usr/bin/env python3
"""
Enhanced HTML Report Generator for Digital Grants Analysis
Created: 2025.07.24.11.00

This enhanced version generates HTML reports directly from CSV data, 
eliminating the need for intermediate markdown files and providing:
1. Clean, consistent formatting from the source
2. Professional CSS styling
3. Interactive navigation system
4. Grants vs Sub-portfolios matrix table
5. Direct CSV-to-HTML conversion for all content
"""


import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

class EnhancedHTMLReportGenerator:
    def __init__(self, base_path, theme_prefix="FB-Digital-"):
        self.base_path = Path(base_path)
        self.source_dir = self.base_path / "source-files"
        self.analysis_path = self.base_path / "analysis-output-files"
        self.html_output_path = self.analysis_path / "html_reports_enhanced"
        self.theme_prefix = theme_prefix  # e.g., "FB-Digital-" or "FB-Health-"
        
        # Load data directly from CSV files using dynamic file discovery
        self.grants_df = pd.read_csv(self.find_latest_file("Grants-Data"))
        self.outcomes_df = pd.read_csv(self.find_latest_file("Grants-Outcomes-Data"))
        
        # Load additional reference data
        try:
            self.key_concepts_df = pd.read_csv(self.find_latest_file("Key-Concepts-Descriptors"))
        except FileNotFoundError:
            print("Warning: Key-Concepts-Descriptors file not found")
            self.key_concepts_df = None
        
        try:
            self.portfolio_descriptions_df = pd.read_csv(self.find_latest_file("Portfolio-descriptions"))
        except FileNotFoundError:
            print("Warning: Portfolio-descriptions file not found")
            self.portfolio_descriptions_df = None
        
        # Initialize domain mappings and subportfolios - will be loaded from CSV
        self.domain_mapping = {}
        self.subportfolios = []
        
        # Initialize default_obs as empty - will be loaded from CSV
        self.default_obs = {}
        
        # Load all metadata from source files
        self.load_domain_definitions()
        self.load_subportfolio_definitions()
        self.load_default_outcome_baskets()
        self.load_additional_source_data()
        
        # Validate that source data was loaded correctly
        self.validate_source_data_loading()

    def find_latest_file(self, file_type):
        """
        Find the latest version of a file with the theme prefix.
        
        Args:
            file_type (str): The type of file to find (e.g., "Grants-Data", "Portfolio-descriptions")
        
        Returns:
            Path: Path to the latest version of the file
        
        Raises:
            FileNotFoundError: If no matching file is found
        """
        pattern = f"{self.theme_prefix}{file_type}*.csv"
        matching_files = list(self.source_dir.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        # Sort by version number if present, otherwise by filename
        def extract_version(filepath):
            """Extract version number from filename like 'FB-Digital-Grants-Data.v5.3.csv'"""
            import re
            filename = filepath.name
            version_match = re.search(r'\.v(\d+)(?:\.(\d+))?\.csv$', filename)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2)) if version_match.group(2) else 0
                return (major, minor)
            return (0, 0)  # Default for files without version numbers
        
        # Get the file with the highest version number
        latest_file = max(matching_files, key=extract_version)
        
        print(f"📁 Using latest {file_type} file: {latest_file.name}")
        return latest_file
    
    def validate_source_data_loading(self):
        """
        Validate that all source data was loaded correctly.
        This helps identify any issues with hardcoded vs. source-file-based data.
        """
        print("\n=== Source Data Validation ===")
        
        # Check domain definitions
        print(f"✓ Loaded {len(self.domain_mapping)} domain definitions:")
        for domain, definition in list(self.domain_mapping.items())[:3]:
            print(f"  - {domain}: {definition[:50]}..." if len(definition) > 50 else f"  - {domain}: {definition}")
        if len(self.domain_mapping) > 3:
            print(f"  ... and {len(self.domain_mapping) - 3} more")
        
        # Check subportfolio definitions
        print(f"\n✓ Loaded {len(self.subportfolios)} subportfolio definitions:")
        for subportfolio in self.subportfolios[:3]:
            print(f"  - {subportfolio}")
        if len(self.subportfolios) > 3:
            print(f"  ... and {len(self.subportfolios) - 3} more")
        
        # Check default outcome baskets
        total_default_obs = sum(len(obs_list) for obs_list in self.default_obs.values())
        print(f"\n✓ Loaded {total_default_obs} total default outcome baskets across {len(self.default_obs)} domains:")
        for domain, obs_list in self.default_obs.items():
            print(f"  - {domain}: {len(obs_list)} outcome baskets")
        
        # Check additional source data
        if hasattr(self, 'performance_categories'):
            print(f"\n✓ Loaded {len(self.performance_categories)} performance categories")
        if hasattr(self, 'modes_of_operation'):
            print(f"\n✓ Loaded {len(self.modes_of_operation)} modes of operation")
        if hasattr(self, 'strategic_focus_areas'):
            print(f"\n✓ Loaded {len(self.strategic_focus_areas)} strategic focus areas")
        if hasattr(self, 'valid_portfolio_values'):
            print(f"\n✓ Loaded {len(self.valid_portfolio_values)} valid portfolio values: {self.valid_portfolio_values}")
        
        print("=== Validation Complete ===\n")

    def load_domain_definitions(self):
        """Load domain definitions from Key-Concepts-Descriptors CSV file"""
        try:
            key_concepts_path = self.find_latest_file("Key-Concepts-Descriptors")
            key_concepts_df = pd.read_csv(key_concepts_path)
            
            # Find the row with domain definitions by searching for "DoC 1" pattern
            domain_row_idx = None
            for i, row in key_concepts_df.iterrows():
                row_text = ' '.join([str(cell) for cell in row if pd.notna(cell)])
                if 'DoC 1' in row_text and 'DoC 2' in row_text:
                    domain_row_idx = i
                    break
            
            if domain_row_idx is not None:
                # Extract domain definitions from the found row
                domain_row = key_concepts_df.iloc[domain_row_idx]
                domains_text = [str(domain_row.iloc[i]).strip() for i in range(len(domain_row)) if pd.notna(domain_row.iloc[i])]
                
                # Dynamic domain detection for any theme (D1-D9, C1-C9, E1-E9, MH1-MH9, etc.)
                for domain_text in domains_text:
                    # Look for pattern like "DoC X -" or "CoC X -" or "EoC X -" etc.
                    match = re.search(r'([A-Z]+)oC (\d+) - (.+)', domain_text)
                    if match:
                        theme_prefix = match.group(1)  # D, C, E, MH, etc.
                        domain_number = match.group(2)  # 1, 2, 3, etc.
                        domain_description = match.group(3)  # Full description
                        domain_code = f"{theme_prefix}{domain_number}"
                        self.domain_mapping[domain_code] = domain_description
                    else:
                        # Fallback: try direct parsing for formats like "D1 - Description"
                        parts = domain_text.split(' - ', 1)
                        if len(parts) == 2:
                            domain_code = parts[0].strip()
                            domain_description = parts[1].strip()
                            # Validate it looks like a domain code (letters + numbers)
                            if re.match(r'^[A-Z]+\d+$', domain_code):
                                self.domain_mapping[domain_code] = domain_description
                
                print(f"Loaded {len(self.domain_mapping)} domain definitions from CSV (row {domain_row_idx + 1})")
            else:
                print("Could not find domain definitions row in Key-Concepts file")
                raise ValueError("Domain definitions not found")
            
        except Exception as e:
            print(f"Error loading domain definitions: {e}")
            # Fallback to hardcoded values
            self.domain_mapping = {
                'D1': 'Stronger agency of young people of all backgrounds to shape their digital futures',
                'D2': 'Expanded equitable access to services for young people',
                'D3': 'Collective efforts by relevant stakeholders for joint learning, knowledge exchange and strategic alignment',
                'D4': 'Increased collaborative efforts towards strengthened human rights-based digital governance frameworks'
            }
            print("Using fallback domain definitions")
    
    def load_subportfolio_definitions(self):
        """Load subportfolio definitions from grants CSV column headers"""
        try:
            # Extract sub-portfolio column names from grants CSV
            subportfolio_columns = [col for col in self.grants_df.columns if 'sub-portfolio' in col]
            
            # Extract clean subportfolio names (remove " sub-portfolio" suffix)
            self.subportfolios = [col.replace(' sub-portfolio', '') for col in subportfolio_columns]
            
            print(f"Loaded {len(self.subportfolios)} sub-portfolios from CSV: {self.subportfolios}")
            
            # Load portfolio descriptions
            self.load_portfolio_descriptions()
            
        except Exception as e:
            print(f"Error loading subportfolio definitions: {e}")
            # Fallback to hardcoded values
            self.subportfolios = ['Digital Health', 'Digital Rights & Governance', 'Digital Literacy', 'Digital Innovation']
            self.portfolio_descriptions = {}

    def load_portfolio_descriptions(self):
        """Load detailed descriptions for each sub-portfolio from Portfolio-descriptions CSV"""
        try:
            portfolio_file = self.find_latest_file("Portfolio-descriptions")
            portfolio_df = pd.read_csv(portfolio_file)
            # Filter out empty rows
            portfolio_df = portfolio_df.dropna(subset=['NAME'])
            portfolio_df = portfolio_df[portfolio_df['NAME'].str.strip() != '']
            
            # Store portfolio descriptions - no name mapping needed now
            self.portfolio_descriptions = {}
            for _, row in portfolio_df.iterrows():
                portfolio_name = row['NAME']
                
                self.portfolio_descriptions[portfolio_name] = {
                    'core_focus': row.get('CORE FOCUS', 'Description not available'),
                    'focus_details': row.get('Focus of grants', 'Details not available'),
                    'grants_included': row.get('Grants included', 'Grant details not available')
                }
            
            print(f"✓ Loaded portfolio descriptions for {len(self.portfolio_descriptions)} sub-portfolios")
            
            # Debug: show loaded portfolio names
            loaded_names = list(self.portfolio_descriptions.keys())
            print(f"✓ Portfolio descriptions loaded for: {loaded_names}")
                
        except Exception as e:
            print(f"Error loading portfolio descriptions: {e}")
            self.portfolio_descriptions = {}
    
    def load_default_outcome_baskets(self):
        """Load default outcome baskets from Key-Concepts-Descriptors CSV file"""
        try:
            key_concepts_path = self.find_latest_file("Key-Concepts-Descriptors")
            key_concepts_df = pd.read_csv(key_concepts_path)
            
            # Find the section with DEFAULT OUTCOME BASKETS
            obs_section_start = None
            for i, row in key_concepts_df.iterrows():
                if pd.notna(row.iloc[0]) and 'DEFAULT OUTCOME BASKETS' in str(row.iloc[0]):
                    obs_section_start = i
                    break
            
            if obs_section_start is None:
                print("WARNING: Could not find DEFAULT OUTCOME BASKETS section in Key-Concepts file")
                return
            
            # Find the header row with domain names (skip empty row)
            domain_header_row = obs_section_start + 2
            domain_headers = key_concepts_df.iloc[domain_header_row]
            
            # Map columns to domains
            domain_columns = {}
            for col_idx, header in enumerate(domain_headers):
                if pd.notna(header):
                    header_str = str(header).strip()
                    # Dynamic domain column detection for any theme
                    match = re.search(r'([A-Z]+)oC (\d+)', header_str)
                    if match:
                        theme_prefix = match.group(1)  # D, C, E, MH, etc.
                        domain_number = match.group(2)  # 1, 2, 3, etc.
                        domain_code = f"{theme_prefix}{domain_number}"
                        domain_columns[domain_code] = col_idx
            
            # Extract OBs for each domain
            for domain, col_idx in domain_columns.items():
                obs_list = []
                # Read OBs from subsequent rows
                for i in range(domain_header_row + 1, len(key_concepts_df)):
                    cell_value = key_concepts_df.iloc[i, col_idx]
                    if pd.notna(cell_value) and str(cell_value).strip():
                        ob = str(cell_value).strip()
                        if ob and ob not in ['', 'nan']:
                            obs_list.append(ob)
                    
                    # Stop if we've read enough rows (reasonable limit)
                    if i > domain_header_row + 15:
                        break
                
                self.default_obs[domain] = obs_list
                print(f"Loaded {len(obs_list)} default OB for {domain}: {obs_list}")
        
        except Exception as e:
            print(f"Error loading default outcome baskets: {e}")
            # Fallback to empty defaults for discovered domains
            self.default_obs = {domain: [] for domain in self.domain_mapping.keys()}
    
    def load_additional_source_data(self):
        """Load additional data from source files that should not be hardcoded"""
        try:
            # Load performance rating categories from Key-Concepts file
            self.load_performance_categories()
            
            # Load modes of operation from Key-Concepts file  
            self.load_modes_of_operation()
            
            # Load strategic focus areas from grants data
            self.load_strategic_focus_areas()
            
            # Load valid portfolio values from actual data
            self.load_valid_portfolio_values()
            
            print("✅ Additional source data loaded successfully")
            
        except Exception as e:
            print(f"Warning: Error loading additional source data: {e}")
    
    def load_performance_categories(self):
        """Extract performance rating categories from Key-Concepts file"""
        try:
            if self.key_concepts_df is None:
                return
                
            # Look for performance legend section
            performance_cols = []
            for col in self.key_concepts_df.columns:
                if any(term in str(col).lower() for term in ['implementation', 'strategic intent', 'learning', 'sustainability', 'stakeholder', 'youth participation']):
                    performance_cols.append(str(col).strip())
            
            if performance_cols:
                self.performance_categories = performance_cols
                print(f"Loaded {len(performance_cols)} performance categories from CSV")
            else:
                # Fallback from actual grants data columns  
                grants_performance_cols = [col for col in self.grants_df.columns if any(term in col.lower() for term in ['implementation', 'strategic intent', 'learning', 'sustainability', 'stakeholder', 'youth'])]
                self.performance_categories = grants_performance_cols
                print(f"Loaded {len(grants_performance_cols)} performance categories from grants data")
                
        except Exception as e:
            print(f"Error loading performance categories: {e}")
            self.performance_categories = []
    
    def load_modes_of_operation(self):
        """Extract modes of operation from Key-Concepts file"""
        try:
            if self.key_concepts_df is None:
                return
                
            # Look for modes of operation section
            modes_section_start = None
            for i, row in self.key_concepts_df.iterrows():
                if pd.notna(row.iloc[0]) and 'MODES OF OPERATIONS' in str(row.iloc[0]):
                    modes_section_start = i
                    break
            
            if modes_section_start is not None:
                # Extract modes from subsequent rows
                modes = []
                for i in range(modes_section_start + 1, min(modes_section_start + 15, len(self.key_concepts_df))):
                    cell_value = self.key_concepts_df.iloc[i, 0]
                    if pd.notna(cell_value) and str(cell_value).strip():
                        mode = str(cell_value).strip()
                        if mode and mode not in ['', 'nan'] and not mode.startswith('DEFAULT'):
                            modes.append(mode)
                
                self.modes_of_operation = modes
                print(f"Loaded {len(modes)} modes of operation from CSV")
            else:
                # Fallback from grants data columns
                grants_modes_cols = [col for col in self.grants_df.columns if any(term in col.lower() for term in ['research', 'implementation', 'policy', 'venture', 'stakeholder', 'communication'])]
                self.modes_of_operation = [col.replace('_', ' ').title() for col in grants_modes_cols]
                print(f"Loaded {len(self.modes_of_operation)} modes from grants data")
                
        except Exception as e:
            print(f"Error loading modes of operation: {e}")
            self.modes_of_operation = []
    
    def load_strategic_focus_areas(self):
        """Extract strategic focus areas from grants data columns"""
        try:
            # Look for strategic focus area columns in grants data
            focus_area_cols = []
            for col in self.grants_df.columns:
                if any(term in col.lower() for term in [
                    'promote digital rights', 'increase the evidence', 'support legal', 
                    'catalyse the digital transformation', 'facilitate the imagination'
                ]):
                    focus_area_cols.append(str(col).strip())
            
            self.strategic_focus_areas = focus_area_cols
            print(f"Loaded {len(focus_area_cols)} strategic focus areas from grants data")
            
        except Exception as e:
            print(f"Error loading strategic focus areas: {e}")
            self.strategic_focus_areas = []
    
    def load_valid_portfolio_values(self):
        """Extract actual portfolio assignment values from the data"""
        try:
            valid_values = set()
            
            # Check all sub-portfolio columns for actual values used
            subportfolio_columns = [col for col in self.grants_df.columns if 'sub-portfolio' in col]
            
            for col in subportfolio_columns:
                unique_values = self.grants_df[col].dropna().astype(str).str.strip()
                for value in unique_values:
                    if value and value.lower() not in ['nan', '', 'none', 'null']:
                        valid_values.add(value)
            
            self.valid_portfolio_values = list(valid_values)
            print(f"Loaded valid portfolio assignment values from data: {sorted(self.valid_portfolio_values)}")
            
        except Exception as e:
            print(f"Error loading valid portfolio values: {e}")
            self.valid_portfolio_values = ['X', 'x']  # Fallback based on your confirmation
        
    def create_css_stylesheet(self):
        """Generate professional CSS stylesheet"""
        return """
/* Digital Grants Analysis HTML Reports - Enhanced Professional Styling */

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Header and Navigation */
.header {
    border-bottom: 3px solid #2c5aa0;
    margin-bottom: 30px;
    padding-bottom: 20px;
    background-color: white;
    position: sticky;
    top: 0;
    z-index: 1000;
    transition: all 0.3s ease;
    cursor: pointer;
}

.header.scrolled {
    margin-bottom: 15px;
    padding-bottom: 10px;
    padding-top: 10px;
    border-bottom: 2px solid #2c5aa0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header h1 {
    color: #2c5aa0;
    font-size: 2.2em;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.header.scrolled h1 {
    font-size: 1.5em;
    margin-bottom: 5px;
}

.header .subtitle {
    transition: all 0.3s ease;
    opacity: 1;
    color: #6c757d;
    margin: 0;
    font-size: 1em;
}

.header.scrolled .subtitle {
    font-size: 0.85em;
    opacity: 0.8;
}

.current-section {
    background-color: #e3f2fd;
    color: #2c5aa0;
    padding: 5px 15px;
    border-radius: 15px;
    font-size: 0.8em;
    font-weight: 500;
    margin-top: 5px;
    display: inline-block;
    transition: all 0.3s ease;
    border: 1px solid rgba(44, 90, 160, 0.2);
    opacity: 0;
    transform: translateY(-10px);
}

.current-section.visible {
    opacity: 1;
    transform: translateY(0);
    display: inline-block;
}

.header.scrolled .current-section {
    font-size: 0.75em;
    padding: 3px 12px;
}

.breadcrumb {
    background-color: #e9ecef;
    padding: 10px 15px;
    border-radius: 5px;
    margin-bottom: 20px;
    font-size: 0.9em;
}

.breadcrumb a {
    color: #2c5aa0;
    text-decoration: none;
    font-weight: 500;
}

.breadcrumb a:hover {
    text-decoration: underline;
}

.breadcrumb span {
    margin: 0 8px;
    color: #6c757d;
}

/* Navigation Menu */
.nav-menu {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 25px;
    border-left: 4px solid #2c5aa0;
}

.nav-menu ul {
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.nav-menu a {
    color: #2c5aa0;
    text-decoration: none;
    font-weight: 500;
    padding: 5px 10px;
    border-radius: 3px;
    transition: background-color 0.3s;
}

.nav-menu a:hover {
    background-color: #e3f2fd;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #2c5aa0;
    margin-top: 25px;
    margin-bottom: 15px;
}

h1 { font-size: 2.2em; }

h2 { 
    font-size: 1.8em; 
    border-bottom: 2px solid #e9ecef; 
    padding-bottom: 10px;
    counter-reset: h3-counter;
}

h3 { 
    font-size: 1.4em;
    counter-increment: h3-counter;
    counter-reset: h4-counter;
}

h3::before {
    content: counter(h3-counter) ". ";
    font-weight: bold;
    color: #2c5aa0;
}

h4 { 
    font-size: 1.2em;
    counter-increment: h4-counter;
    counter-reset: h5-counter;
}

h4::before {
    content: counter(h3-counter) "." counter(h4-counter) " ";
    font-weight: bold;
    color: #2c5aa0;
}

h5 { 
    font-size: 1em;
    counter-increment: h5-counter;
}

h5::before {
    content: counter(h3-counter) "." counter(h4-counter) "." counter(h5-counter) " ";
    font-weight: bold;
    color: #2c5aa0;
}

/* Grants vs Sub-portfolios Matrix Table */
.grants-matrix {
    overflow-x: auto;
    margin: 25px 0;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.grants-matrix table {
    width: 100%;
    border-collapse: collapse;
    background-color: white;
}

.grants-matrix th {
    background-color: #2c5aa0;
    color: white;
    padding: 12px 8px;
    text-align: center;
    font-weight: 600;
    border: 1px solid #1e3a6f;
}

.grants-matrix th.grant-name {
    text-align: left;
    width: 250px;
    max-width: 250px;
}

.grants-matrix th a {
    color: white;
    text-decoration: none;
    display: block;
    padding: 5px;
    border-radius: 3px;
    transition: background-color 0.3s;
}

.grants-matrix th a:hover {
    background-color: rgba(255,255,255,0.2);
}

.grants-matrix td {
    padding: 10px 8px;
    text-align: center;
    border: 1px solid #dee2e6;
    vertical-align: middle;
}

.grants-matrix td.grant-info {
    text-align: left;
    background-color: #f8f9fa;
    width: 250px;
    max-width: 250px;
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.grants-matrix td.grant-info a {
    color: #2c5aa0;
    text-decoration: none;
    font-weight: 600;
    display: block;
    margin-bottom: 5px;
}

.grants-matrix td.grant-info a:hover {
    text-decoration: underline;
}

.grants-matrix td.grant-info .grant-details {
    font-size: 0.85em;
    color: #6c757d;
    line-height: 1.4;
}

.grants-matrix .portfolio-mark {
    font-size: 1.2em;
    font-weight: bold;
    color: #28a745;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    background-color: white;
    border-radius: 5px;
    overflow: hidden;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #dee2e6;
}

th {
    background-color: #f8f9fa;
    font-weight: 600;
    color: #2c5aa0;
}

tr:hover {
    background-color: #f8f9fa;
}

/* Grants Table Specific Styles */
.table-responsive {
    overflow-x: auto;
    margin: 20px 0;
}

.grants-table {
    min-width: 800px;
}

.grants-table th {
    background-color: #2c5aa0;
    color: white;
    font-weight: 600;
    text-align: center;
}

.grants-table td {
    vertical-align: top;
}

.grants-table td.number {
    text-align: right;
    font-family: 'Courier New', monospace;
}

.grants-table td:first-child {
    font-weight: 600;
}

.grants-table tr:nth-child(even) {
    background-color: #f8f9fa;
}

/* Lists */
ul, ol {
    margin: 15px 0;
    padding-left: 25px;
}

li {
    margin-bottom: 5px;
}

/* Links */
a {
    color: #2c5aa0;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Metrics boxes */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 25px 0;
}

.metric-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #2c5aa0;
    text-align: center;
}

.metric-box .metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #2c5aa0;
    display: block;
}

.metric-box .metric-label {
    color: #6c757d;
    font-size: 0.9em;
    margin-top: 5px;
}

/* Outcome records styling */
.outcome-record {
    background-color: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 15px;
    margin: 15px 0;
    border-radius: 0 5px 5px 0;
}

.outcome-record h4 {
    color: #28a745;
    margin-top: 0;
    margin-bottom: 10px;
}

.outcome-record .outcome-meta {
    font-size: 0.9em;
    color: #6c757d;
    margin-top: 10px;
}

/* Portfolio grants description styling */
.portfolio-grants-description {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 20px;
    margin: 15px 0;
}

.portfolio-grants-description h4 {
    color: #007bff;
    margin-top: 20px;
    margin-bottom: 10px;
    font-size: 1.1em;
}

.portfolio-grants-description h4:first-child {
    margin-top: 0;
}

.portfolio-grants-description p {
    margin: 8px 0;
    line-height: 1.5;
}

/* Special sections */
.domain-section {
    border: 1px solid #dee2e6;
    border-radius: 8px;
    margin: 20px 0;
    overflow: hidden;
}

.domain-header {
    background-color: #2c5aa0;
    color: white;
    padding: 15px 20px;
    font-weight: 600;
}

.domain-content {
    padding: 20px;
}

/* Responsive design */
@media (max-width: 768px) {
    body {
        padding: 10px;
    }
    
    .container {
        padding: 15px;
    }
    
    .nav-menu ul {
        flex-direction: column;
        gap: 10px;
    }
    
    .grants-matrix {
        font-size: 0.9em;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
}

/* Print styles */
@media print {
    body {
        background-color: white;
        padding: 0;
    }
    
    .container {
        box-shadow: none;
        max-width: none;
    }
    
    .nav-menu {
        display: none;
    }
    
    a {
        color: #333 !important;
    }
}

/* Info icon and modal styles */
.info-icon {
    cursor: pointer;
    color: #2c5aa0;
    font-size: 1.1em;
    margin-left: 5px;
    padding: 2px 4px;
    border-radius: 50%;
    background-color: rgba(44, 90, 160, 0.1);
    display: inline-block;
    width: 20px;
    height: 20px;
    text-align: center;
    line-height: 16px;
    transition: all 0.3s ease;
}

.info-icon:hover {
    background-color: rgba(44, 90, 160, 0.2);
    transform: scale(1.1);
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: 10000;
    display: none;
    justify-content: center;
    align-items: center;
}

.modal-content {
    background: white;
    border-radius: 8px;
    padding: 30px;
    max-width: 700px;
    max-height: 80vh;
    overflow-y: auto;
    position: relative;
    margin: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.modal-close {
    position: absolute;
    top: 15px;
    right: 20px;
    font-size: 24px;
    cursor: pointer;
    color: #999;
    background: none;
    border: none;
    padding: 0;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
}

.modal-close:hover {
    color: #333;
    background-color: #f0f0f0;
}

.modal-content h4 {
    color: #2c5aa0;
    margin-top: 0;
    margin-bottom: 20px;
    font-size: 1.3em;
}

.modal-content h5 {
    color: #2c5aa0;
    margin-top: 25px;
    margin-bottom: 10px;
    font-size: 1.1em;
}

.modal-content p {
    line-height: 1.6;
    margin-bottom: 15px;
}
"""

    # Helper methods for string manipulation
    def sanitize_filename(self, name):
        """Clean filename by replacing spaces and ampersands"""
        return name.lower().replace(' ', '_').replace('&', 'and')
    
    def clean_budget_string(self, budget_str):
        """Remove commas and spaces from budget strings"""
        return str(budget_str).replace(',', '').replace(' ', '')
    
    def format_display_name(self, name):
        """Format display name by replacing underscores with spaces"""
        return str(name).replace('_', ' ')
    
    def generate_grant_filename(self, gams_code, short_name):
        """Generate consistent filename for grant pages with both GAMS code and short name"""
        # Clean the short name for filename use
        clean_name = str(short_name).replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c in ['_', '-'])
        
        # Limit length to avoid overly long filenames
        if len(clean_name) > 30:
            clean_name = clean_name[:30]
        
        return f"{gams_code}_{clean_name}_analysis.html"
    
    def get_css_path(self, is_index_page=False):
        """Get appropriate CSS path based on page type"""
        return 'css/analysis_styles.css' if is_index_page else '../css/analysis_styles.css'
    
    def add_venn_diagram_if_exists(self):
        """Generate Venn diagram HTML if file exists"""
        venn_diagram_path = self.html_output_path / "Digital Theme Grants Sub-Portfolios.VennDiagram.png"
        if venn_diagram_path.exists():
            return """
<h3>Grants Sub-Portfolios Venn Diagram</h3>
<div style="text-align: center; margin: 20px 0;">
    <a href="Digital Theme Grants Sub-Portfolios.VennDiagram.png" target="_blank">
        <img src="Digital Theme Grants Sub-Portfolios.VennDiagram.png" 
             alt="Digital Theme Grants Sub-Portfolios Venn Diagram" 
             style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 8px; cursor: pointer;"
             title="Click to view full size">
    </a>
    <p style="font-style: italic; color: #666; margin-top: 10px;">Click image to view full size</p>
</div>
"""
        return ""

    def create_html_template(self, title, content, breadcrumb=None, nav_menu=None, is_index_page=False):
        """Create complete HTML page with template"""
        breadcrumb_html = ""
        if breadcrumb:
            breadcrumb_items = []
            for item in breadcrumb:
                if 'url' in item:
                    breadcrumb_items.append(f'<a href="{item["url"]}">{item["text"]}</a>')
                else:
                    breadcrumb_items.append(item['text'])
            breadcrumb_html = f"""
            <div class="breadcrumb">
                {' <span>→</span> '.join(breadcrumb_items)}
            </div>
            """
        
        nav_menu_html = ""
        if nav_menu:
            nav_items = []
            for item in nav_menu:
                if item.get("url"):
                    nav_items.append(f'<li><a href="{item["url"]}">{item["text"]}</a></li>')
                else:
                    nav_items.append(f'<li>{item["text"]}</li>')
            nav_menu_html = f"""
            <div class="nav-menu">
                <ul>
                    {''.join(nav_items)}
                </ul>
            </div>
            """
        
        css_path = self.get_css_path(is_index_page)
        index_link = 'index.html' if is_index_page else '../index.html'
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Digital Theme Grants' Outcome Analysis</title>
    <link rel="stylesheet" href="{css_path}">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
    // Geographic details modal functions - defined early to be available globally
    function showGeographicDetails() {{
        const modal = document.getElementById('geographicModal');
        const content = document.getElementById('geographicContent');
        content.innerHTML = window.geographicData || 'No geographic data available.';
        modal.style.display = 'flex';
    }}

    function hideGeographicDetails() {{
        document.getElementById('geographicModal').style.display = 'none';
    }}

    // Add event listeners when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {{
        // Close modal when clicking outside of it
        window.addEventListener('click', function(event) {{
            const modal = document.getElementById('geographicModal');
            if (event.target == modal) {{
                hideGeographicDetails();
            }}
        }});

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                hideGeographicDetails();
            }}
        }});
    }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="current-section" id="currentSection" style="display: none; cursor: pointer;" title="Click to navigate to section"></div>
        </div>
        
        {breadcrumb_html}
        {nav_menu_html}
        
        <div class="content">
            {content}
        </div>
        
        <div class="footer">
            <hr>
            <p style="text-align: center; color: #6c757d; margin-top: 20px;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
                <a href="{index_link}">Back to Theme Overview</a>
            </p>
        </div>
    </div>
    
    <script>
        // Sticky header with scroll behavior and section tracking
        window.addEventListener('scroll', function() {{
            const header = document.querySelector('.header');
            const scrolled = window.scrollY > 100;
            
            if (scrolled) {{
                header.classList.add('scrolled');
            }} else {{
                header.classList.remove('scrolled');
            }}
            
            // Update current section indicator
            updateCurrentSection();
        }});
        
        function updateCurrentSection() {{
            const sections = [
                {{ element: 'h2', text: 'Overview of Grants and Sub-Portfolios', display: 'Overview of Grants and Sub-Portfolios' }},
                {{ element: 'h2', text: 'Outcomes analysis', display: 'Outcomes analysis' }},
                {{ element: 'h2', text: 'Outcome Basket analysis', display: 'Outcome Basket analysis' }},
                {{ element: 'h2', text: 'Outcomes from grants in the', display: 'Outcome descriptions' }}
            ];
            
            const currentSectionElement = document.getElementById('currentSection');
            if (!currentSectionElement) return;
            
            let activeSection = null;
            let currentGrant = null;
            let currentReport = null;
            let shouldShow = false;
            const scrollPosition = window.scrollY + 150; // Offset for sticky header
            
            // Find the current section based on scroll position
            for (let i = sections.length - 1; i >= 0; i--) {{
                const section = sections[i];
                const elements = document.querySelectorAll(section.element);
                
                for (let element of elements) {{
                    // For outcomes section, use partial text matching
                    const textMatch = section.text === 'Outcomes from grants in the' 
                        ? element.textContent.includes(section.text)
                        : element.textContent.trim() === section.text;
                        
                    if (textMatch && element.offsetTop <= scrollPosition) {{
                        activeSection = section.display;
                        shouldShow = true;
                        
                        // If we're in the outcome descriptions section, find current grant
                        if (section.display === 'Outcome descriptions') {{
                            // Find the outcomes section element to constrain our search
                            const outcomesSection = element;
                            const grantHeaders = [];
                            
                            // Get all h3 elements that come after the outcomes section header
                            const allH3s = document.querySelectorAll('h3');
                            let foundOutcomesSection = false;
                            
                            allH3s.forEach(header => {{
                                if (foundOutcomesSection) {{
                                    grantHeaders.push(header);
                                }} else if (header.offsetTop >= outcomesSection.offsetTop) {{
                                    foundOutcomesSection = true;
                                    grantHeaders.push(header);
                                }}
                            }});
                            
                            // Only find current grant if we've actually reached a grant header
                            let closestGrant = null;
                            let closestGrantDistance = Infinity;
                            
                            grantHeaders.forEach(header => {{
                                const headerTop = header.offsetTop;
                                const distance = scrollPosition - headerTop;
                                
                                // Only consider grants that we've actually scrolled past
                                if (distance >= 0 && distance < closestGrantDistance) {{
                                    closestGrantDistance = distance;
                                    closestGrant = header.textContent.trim();
                                }}
                            }});
                            
                            // Only set currentGrant if we found one and we're actually past it
                            currentGrant = closestGrant;
                            
                            // If we have a current grant, also find the current report (h4 elements)
                            if (currentGrant) {{
                                const reportHeaders = [];
                                const allH4s = document.querySelectorAll('h4');
                                
                                // Get all h4 elements that come after the outcomes section header
                                allH4s.forEach(header => {{
                                    if (header.offsetTop >= outcomesSection.offsetTop) {{
                                        reportHeaders.push(header);
                                    }}
                                }});
                                
                                let closestReport = null;
                                let closestReportDistance = Infinity;
                                
                                reportHeaders.forEach(header => {{
                                    const headerTop = header.offsetTop;
                                    const distance = scrollPosition - headerTop;
                                    
                                    // Only consider reports that we've actually scrolled past
                                    if (distance >= 0 && distance < closestReportDistance) {{
                                        closestReportDistance = distance;
                                        // Extract just the report name, removing the date in parentheses if present
                                        let reportText = header.textContent.trim();
                                        // Remove date pattern like "(2024-01-01)" from the end
                                        reportText = reportText.replace(/\s*\([^)]*\)\s*$/, '');
                                        closestReport = reportText;
                                    }}
                                }});
                                
                                currentReport = closestReport;
                            }}
                        }}
                        break;
                    }}
                }}
                
                if (activeSection) break;
            }}
            
            // Show/hide the indicator based on whether we're in a main section
            if (shouldShow && activeSection) {{
                let displayText = activeSection;
                if (currentGrant && activeSection === 'Outcome descriptions') {{
                    displayText = `${{activeSection}} - ${{currentGrant}}`;
                    if (currentReport) {{
                        displayText += `  -  Source: ${{currentReport}}`;
                    }}
                }}
                
                currentSectionElement.textContent = displayText;
                currentSectionElement.style.display = 'inline-block';
                currentSectionElement.classList.add('visible');
                
                // Store current section info for click navigation
                currentSectionElement.setAttribute('data-active-section', activeSection);
                currentSectionElement.setAttribute('data-current-grant', currentGrant || '');
                currentSectionElement.setAttribute('data-current-report', currentReport || '');
            }} else {{
                currentSectionElement.classList.remove('visible');
                // Hide after transition completes
                setTimeout(() => {{
                    if (!currentSectionElement.classList.contains('visible')) {{
                        currentSectionElement.style.display = 'none';
                    }}
                }}, 300);
            }}
        }}
        
        // Handle section indicator clicks for navigation
        function handleSectionClick() {{
            const currentSectionElement = document.getElementById('currentSection');
            if (!currentSectionElement) return;
            
            const activeSection = currentSectionElement.getAttribute('data-active-section');
            const currentGrant = currentSectionElement.getAttribute('data-current-grant');
            const currentReport = currentSectionElement.getAttribute('data-current-report');
            
            let targetElement = null;
            
            // Navigate based on context - always prefer grant header when in outcomes section
            if (currentGrant && activeSection === 'Outcome descriptions') {{
                // Always navigate to grant header, not report header
                const grantHeaders = document.querySelectorAll('h3');
                grantHeaders.forEach(header => {{
                    if (header.textContent.trim() === currentGrant) {{
                        targetElement = header;
                    }}
                }});
            }} else {{
                // Find the main section header (h2)
                const sectionHeaders = document.querySelectorAll('h2');
                sectionHeaders.forEach(header => {{
                    if (activeSection === 'Outcome descriptions' && header.textContent.includes('Outcomes from grants in the')) {{
                        targetElement = header;
                    }} else if (header.textContent.trim() === activeSection) {{
                        targetElement = header;
                    }}
                }});
            }}
            
            // Smooth scroll to the target element with proper offset for sticky header
            if (targetElement) {{
                const headerHeight = 160; // Increased offset to ensure headers are fully visible
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - headerHeight;
                
                window.scrollTo({{
                    top: targetPosition,
                    behavior: 'smooth'
                }});
            }}
        }}
        
        // Initialize section tracking on page load
        document.addEventListener('DOMContentLoaded', function() {{
            updateCurrentSection();
        }});
        
        // Make header clickable to scroll to top
        document.querySelector('.header').addEventListener('click', function(e) {{
            // Don't scroll to top if clicking on section indicator
            if (e.target.id === 'currentSection') {{
                return;
            }}
            window.scrollTo({{
                top: 0,
                behavior: 'smooth'
            }});
        }});
        
        // Make section indicator clickable to navigate to section
        document.addEventListener('click', function(e) {{
            if (e.target.id === 'currentSection') {{
                e.preventDefault();
                e.stopPropagation();
                handleSectionClick();
            }}
        }});
        
        // Significance information modal functionality
        function showSignificanceInfo() {{
            document.getElementById('significanceModal').style.display = 'flex';
        }}
        
        function hideSignificanceInfo() {{
            document.getElementById('significanceModal').style.display = 'none';
        }}
        
        // Special cases information modal functionality
        function showSpecialCasesInfo() {{
            document.getElementById('specialCasesModal').style.display = 'flex';
        }}
        
        function hideSpecialCasesInfo() {{
            document.getElementById('specialCasesModal').style.display = 'none';
        }}
        
        // Close modal when clicking outside of content
        document.addEventListener('click', function(event) {{
            const significanceModal = document.getElementById('significanceModal');
            const specialCasesModal = document.getElementById('specialCasesModal');
            
            if (event.target === significanceModal) {{
                hideSignificanceInfo();
            }}
            if (event.target === specialCasesModal) {{
                hideSpecialCasesInfo();
            }}
        }});
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                hideSignificanceInfo();
                hideSpecialCasesInfo();
            }}
        }});
    </script>
    
    <!-- Significance Information Modal -->
    <div id="significanceModal" class="modal-overlay">
        <div class="modal-content">
            <button class="modal-close" onclick="hideSignificanceInfo()">&times;</button>
            <strong>Understanding Significance Levels</strong>

            <p>When assessing outcomes, "significance" refers to how meaningful a step the outcome represents towards the overall goals of an intervention. It's about whether the changes involved have the potential to lead to important, long-term, or widespread effects. What is considered significant can change over the course of a program, with early-stage outcomes being different from late-stage outcomes.</p>

            <ul>
                <li><strong>High Significance</strong></li>
                <p>These outcomes represent a meaningful step towards the overarching goals or priorities of the intervention. They show progress in key areas like advocacy or policy shifts and have the potential for consequential, long-term, systemic effects, or broad replicability, affecting multiple stakeholders or a key system.</p>

                <li><strong>Medium Significance</strong></li>
                <p>These outcomes are valuable but represent an incremental or localized step towards broader goals. They show initial progress but may affect fewer stakeholders or lack immediate systemic influence or sustainability. While they add value, they are not considered critical to the intervention's success, potentially having limited impact or sustainability.</p>

                <li><strong>Low Significance</strong></li>
                <p>These outcomes refer to positive changes with limited importance to the broader objectives of the program. They may be only partially aligned with goals or reflect discrete improvements affecting a small number of stakeholders, with limited foreseeable potential to contribute to broader impact.</p>
            </ul>
        </div>
    </div>
    
    <!-- Special Cases Information Modal -->
    <div id="specialCasesModal" class="modal-overlay">
        <div class="modal-content">
            <button class="modal-close" onclick="hideSpecialCasesInfo()">&times;</button>
            <strong>Understanding Special Case Tags</strong>

            <p>Special case tags refer to the following cases:</p>
            
            <ul>
                <li><strong>Unexpected Outcome:</strong> in case the outcome came as a surprise.</li>

                <li><strong>Negative Outcome:</strong> in case the outcome is considered negative.</li>

                <li><strong>Signal towards Outcome:</strong> in case the reported changes do not still qualify as a full-fledged outcome but could be conducive to them.</li>

                <li><strong>Based on Outcome Indicator:</strong> in case the mined outcome is based on outcome indicators established for the grant. Include information on the indicator in the "Additional details on the Outcome" field.</li>
                
                <li><strong>Keep on our radar:</strong> in case the reporter wants to keep an eye on this outcome in the future (eg: when additional reports are received, or for interactions with grantees).</li>

                <li><strong>Programme Outcome:</strong> for cases where the reported outcome refers to the overall programme, rather than its grants (esp. useful for programmes implemented by intermediaries).</li>
            </ul>
        </div>
    </div>

    <!-- Geographic Details Modal -->
    <div id="geographicModal" class="modal-overlay">
        <div class="modal-content">
            <button class="modal-close" onclick="hideGeographicDetails()">&times;</button>
            <strong>Country/Region Distribution Details</strong>
            <div id="geographicContent"></div>
        </div>
    </div>

</body>
</html>"""

    def setup_html_structure(self):
        """Create HTML directory structure and CSS file"""
        # Create directories
        self.html_output_path.mkdir(exist_ok=True)
        (self.html_output_path / "css").mkdir(exist_ok=True)
        (self.html_output_path / "subportfolios").mkdir(exist_ok=True)
        (self.html_output_path / "grants").mkdir(exist_ok=True)
        
        # Create CSS file
        css_content = self.create_css_stylesheet()
        with open(self.html_output_path / "css" / "analysis_styles.css", 'w', encoding='utf-8') as f:
            f.write(css_content)
            
        print("✅ Enhanced HTML directory structure and CSS created")

    def get_consolidated_grants_data(self):
        """Get consolidated grants data for the matrix table, using CSV as primary source"""
        consolidated_grants = {}
        
        # Use the dynamically loaded sub-portfolios
        subportfolios = self.subportfolios
        
        # Build consolidation mapping from CSV phase relationships
        consolidation_map = {}  # Maps phase 1 -> phase 2 for consolidation
        phase_info = {}  # Stores which grants are phases of others
        
        for _, row in self.grants_df.iterrows():
            primary_code = str(row['GAMS Code'])
            if pd.notna(row['GAMS Code']) and str(row['GAMS Code']) != 'nan':
                if pd.notna(row['GAMS code - next phase']) and str(row['GAMS code - next phase']) != 'nan':
                    next_phase = str(row['GAMS code - next phase'])
                    consolidation_map[primary_code] = next_phase
                    phase_info[primary_code] = 'phase1'
                    phase_info[next_phase] = 'phase2'
        
        # Get grants from CSV (primary source of truth)
        for _, row in self.grants_df.iterrows():
            primary_code = str(row['GAMS Code'])
            
            # Skip invalid GAMS codes
            if not pd.notna(row['GAMS Code']) or str(row['GAMS Code']) == 'nan':
                continue
                
            # Skip if this grant is a phase 1 and we should consolidate it
            if primary_code in consolidation_map and consolidation_map[primary_code] in [str(r['GAMS Code']) for _, r in self.grants_df.iterrows() if pd.notna(r['GAMS Code']) and str(r['GAMS Code']) != 'nan']:
                continue  # Will be handled by phase 2
            
            # Determine if this is a consolidated grant
            consolidated_gams = primary_code
            display_name = str(row['Grant\'s short name']).replace(' ', '_') if pd.notna(row['Grant\'s short name']) else f"Grant_{primary_code}"
            is_consolidated = False
            
            # Check if this is a phase 2 that should include phase 1
            phase1_code = None
            for phase1, phase2 in consolidation_map.items():
                if phase2 == primary_code:
                    phase1_code = phase1
                    consolidated_gams = f"{phase1}, {primary_code}"
                    is_consolidated = True
                    # Update display name to indicate it's consolidated
                    display_name = self.format_display_name(display_name)
                    if 'phase 2' in display_name.lower():
                        # Replace "phase 2" with "phase 1 and 2" for consolidated grants
                        display_name = display_name.replace('phase 2', 'phase 1 and 2').replace('Phase 2', 'Phase 1 and 2')
                        display_name += ' (consolidated)'
                    elif ', phase 1' not in display_name and ', phase 2' not in display_name:
                        display_name += ' (consolidated)'
                    break
            
            # Get budget and duration (sum if consolidated)
            budget = 0
            duration = 0
            
            if is_consolidated and phase1_code:
                # Sum budget and duration from both phases
                phase1_row = self.grants_df[self.grants_df['GAMS Code'] == phase1_code]
                if not phase1_row.empty:
                    phase1_budget = self.clean_budget_string(phase1_row.iloc[0]['Budget (CHF)'])
                    phase1_duration = phase1_row.iloc[0]['Duration (months)']
                    try:
                        if phase1_budget != 'N/A' and phase1_budget != 'nan':
                            budget += float(phase1_budget)
                    except:
                        pass
                    try:
                        if pd.notna(phase1_duration):
                            duration = max(duration, float(phase1_duration))
                    except:
                        pass
            
            # Add current phase budget and duration
            current_budget = self.clean_budget_string(row['Budget (CHF)'])
            current_duration = row['Duration (months)']
            try:
                if current_budget != 'N/A' and current_budget != 'nan':
                    budget += float(current_budget)
            except:
                pass
            try:
                if pd.notna(current_duration):
                    duration = max(duration, float(current_duration))
            except:
                pass
            
            # Format budget and duration for display
            budget_display = f"{budget:,.0f}" if budget > 0 else 'N/A'
            duration_display = f"{int(duration)}" if duration > 0 else 'N/A'
            
            # Get outcomes count for this grant (sum across all phases if consolidated)
            outcomes_count = 0
            if is_consolidated and phase1_code:
                # Count outcomes for both phases
                phase1_outcomes = len(self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == phase1_code])
                phase2_outcomes = len(self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == primary_code])
                outcomes_count = phase1_outcomes + phase2_outcomes
            else:
                # Count outcomes for single phase
                outcomes_count = len(self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == primary_code])
            
            # Extract other metadata
            countries = str(row['Countries']) if pd.notna(row['Countries']) else 'N/A'
            
            # Determine sub-portfolio memberships from CSV (dynamically from loaded subportfolios)
            subportfolio_memberships = {}
            portfolio_cols = {f"{sp} sub-portfolio": sp for sp in self.subportfolios}
            
            for col, portfolio_name in portfolio_cols.items():
                if col in row.index and pd.notna(row[col]):
                    # Check for any non-empty value (X, x, or any other value)
                    value = str(row[col]).strip()
                    if value and value.lower() not in ['nan', '', 'none', 'null']:
                        subportfolio_memberships[portfolio_name] = True
            
            # Get end date information
            end_date_sort = 999999
            end_date_display = 'TBD'
            try:
                end_date_str = row['Grant end']
                if pd.notna(end_date_str) and str(end_date_str).strip() and str(end_date_str) != 'nan':
                    end_date_str_clean = str(end_date_str).strip()
                    if '-' in end_date_str_clean:
                        month_str, year_str = end_date_str_clean.split('-')
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        if month_str in month_map:
                            month_num = month_map[month_str]
                            year_full = int(f"20{year_str}") if len(year_str) == 2 else int(year_str)
                            end_date_sort = year_full * 100 + month_num
                            end_date_display = f"{month_num:02d}/{year_full}"
            except:
                pass
            
            consolidated_grants[primary_code] = {
                'short_name': display_name,
                'budget': budget_display,
                'duration': duration_display,
                'countries': countries,
                'subportfolios': subportfolio_memberships,
                'consolidated_gams': consolidated_gams,
                'is_consolidated': is_consolidated,
                'outcomes_count': outcomes_count,
                'end_date_sort': end_date_sort,
                'end_date_display': end_date_display,
                'phase1_code': phase1_code
            }
        
        return consolidated_grants, subportfolios

    def generate_subportfolio_distribution(self, consolidated_grants):
        """Generate dynamic sub-portfolio distribution based on current grants data"""
        subportfolios = self.subportfolios
        total_grants = len(consolidated_grants)
        
        # Calculate total theme budget
        total_theme_budget = 0
        for grant_info in consolidated_grants.values():
            try:
                budget_str = self.clean_budget_string(grant_info['budget'])
                if budget_str != 'N/A':
                    total_theme_budget += float(budget_str)
            except:
                pass
        
        distribution_html = "<ul>\n"
        
        for sp in subportfolios:
            count = 0
            total_budget = 0
            
            for grant_info in consolidated_grants.values():
                if grant_info['subportfolios'].get(sp, False):
                    count += 1
                    try:
                        budget_str = self.clean_budget_string(grant_info['budget'])
                        if budget_str != 'N/A':
                            total_budget += float(budget_str)
                    except:
                        pass
            
            grants_percentage = (count / total_grants * 100) if total_grants > 0 else 0
            budget_percentage = (total_budget / total_theme_budget * 100) if total_theme_budget > 0 else 0
            budget_display = f"{total_budget:,.0f}" if total_budget > 0 else "0"
            
            distribution_html += f"<li><strong>{sp}:</strong> {count} grants ({grants_percentage:.1f}% of Theme's grants) - Budget: {budget_display} CHF ({budget_percentage:.1f}% of Theme's budget)</li>\n"
        
        distribution_html += "</ul>"
        return distribution_html

    def create_grants_matrix_table(self):
        """Create the interactive grants vs sub-portfolios matrix table"""
        consolidated_grants, subportfolios = self.get_consolidated_grants_data()
        
        if not consolidated_grants:
            return "<p>No grant data available for matrix table.</p>"
        
        # Sort by end date (future dates first, descending order)
        sorted_grants = sorted(consolidated_grants.items(), key=lambda x: x[1]['end_date_sort'], reverse=True)
        
        # Start table
        html = '<div class="grants-matrix">\n<table>\n<thead>\n<tr>\n'
        html += '<th class="grant-name">Grant</th>\n'
        
        # Sub-portfolio headers (clickable)
        for sp in subportfolios:
            sp_filename = self.sanitize_filename(sp)
            html += f'<th><a href="subportfolios/{sp_filename}.html">{sp}</a></th>\n'
        
        html += '<th>Budget (CHF)</th>\n<th>Duration (months)</th>\n<th>End Date</th>\n<th>Countries</th>\n<th>Outcomes in DB</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Grant rows
        for gams_code, grant_info in sorted_grants:
            html += '<tr>\n'
            
            # Grant name (clickable)
            grant_filename = self.generate_grant_filename(gams_code, grant_info["short_name"])
            display_name = self.format_display_name(grant_info["short_name"])
            html += f'<td class="grant-info">'
            html += f'<a href="grants/{grant_filename}">{display_name}</a>'
            html += f'<div class="grant-details">GAMS: {grant_info.get("consolidated_gams", gams_code)}</div>'
            html += f'</td>\n'
            
            # Sub-portfolio memberships
            for sp in subportfolios:
                if grant_info['subportfolios'].get(sp, False):
                    html += '<td class="portfolio-mark">✓</td>\n'
                else:
                    html += '<td></td>\n'
            
            # Metadata
            html += f'<td>{grant_info["budget"]}</td>\n'
            html += f'<td>{grant_info["duration"]}</td>\n'
            html += f'<td>{grant_info["end_date_display"]}</td>\n'
            html += f'<td>{grant_info["countries"]}</td>\n'
            html += f'<td>{grant_info["outcomes_count"]}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>\n</div>\n'
        return html
        
    def generate_domain_distribution_from_csv(self):
        """Generate domain distribution analysis directly from CSV data"""
        if self.outcomes_df.empty:
            return "<p>No outcomes data available.</p>"
        
        # Count outcomes by domain
        # Initialize with all possible domains to ensure consistency
        domain_counts = {}
        total_outcomes = len(self.outcomes_df)
        
        # Initialize all domains from domain mapping with 0 counts
        for domain_code, domain_description in self.domain_mapping.items():
            domain_full_text = f"{domain_code} - {domain_description}"
            domain_counts[domain_full_text] = 0
        
        # Count actual outcomes by domain
        for _, outcome in self.outcomes_df.iterrows():
            domain = outcome.get('Domain of Change')  # Correct column name
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key in domain_counts:
                    domain_counts[domain_key] += 1
        
        # Generate only pie chart visualization
        html = self.generate_domain_pie_chart(domain_counts, total_outcomes)
        
        return html
    
    def generate_domain_pie_chart(self, domain_counts, total_outcomes):
        """Generate a pie chart visualization for domain distribution using Chart.js"""
        if not domain_counts or total_outcomes == 0:
            return ""
        
        # Prepare data for the chart
        chart_data = []
        chart_labels = []
        chart_colors = ['#2c5aa0', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
        
        for i, (domain_full_text, count) in enumerate(sorted(domain_counts.items())):
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            chart_labels.append(f"{domain_full_text}")
            chart_data.append(count)
        
        # Generate Chart.js HTML
        chart_html = f"""
<div style="margin: 30px 0; text-align: center;">
    <div style="max-width: 800px; margin: 0 auto;">
        <canvas id="domainDistributionChart" width="400" height="400"></canvas>
    </div>
</div>

<script>
    const ctx = document.getElementById('domainDistributionChart').getContext('2d');
    const domainChart = new Chart(ctx, {{
        type: 'pie',
        data: {{
            labels: {chart_labels},
            datasets: [{{
                data: {chart_data},
                backgroundColor: {chart_colors[:len(chart_data)]},
                borderColor: '#ffffff',
                borderWidth: 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'right',
                    labels: {{
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {{
                            size: 11
                        }},
                        boxWidth: 12,
                        boxHeight: 12,
                        generateLabels: function(chart) {{
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {{
                                return data.labels.map((label, i) => {{
                                    const meta = chart.getDatasetMeta(0);
                                    const style = meta.controller.getStyle(i);
                                    
                                    // Function to wrap text for long labels
                                    function wrapText(text, maxLength = 40) {{
                                        if (text.length <= maxLength) return text;
                                        
                                        const words = text.split(' ');
                                        const lines = [];
                                        let currentLine = '';
                                        
                                        words.forEach(word => {{
                                            if ((currentLine + word).length <= maxLength) {{
                                                currentLine += (currentLine ? ' ' : '') + word;
                                            }} else {{
                                                if (currentLine) lines.push(currentLine);
                                                currentLine = word;
                                            }}
                                        }});
                                        
                                        if (currentLine) lines.push(currentLine);
                                        return lines;
                                    }}
                                    
                                    const wrappedText = wrapText(label);
                                    
                                    return {{
                                        text: Array.isArray(wrappedText) ? wrappedText : [wrappedText],
                                        fillStyle: style.backgroundColor,
                                        strokeStyle: style.borderColor,
                                        lineWidth: style.borderWidth,
                                        pointStyle: 'circle',
                                        hidden: isNaN(data.datasets[0].data[i]) || meta.data[i].hidden,
                                        index: i
                                    }};
                                }});
                            }}
                            return [];
                        }}
                    }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + context.parsed + ' outcomes (' + percentage + '%)';
                        }}
                    }}
                }}
            }},
            layout: {{
                padding: {{
                    right: 20,
                    left: 20,
                    top: 20,
                    bottom: 20
                }}
            }}
        }},
        plugins: []
    }});
</script>
"""
        
        return chart_html
    
    def generate_domain_analysis_from_csv(self):
        """Generate detailed domain analysis with outcome basket usage directly from CSV data"""
        return self.generate_unified_domain_analysis(
            outcomes_df=self.outcomes_df,
            analysis_level="theme"
        )
    
    def generate_subportfolio_domain_analysis(self, subportfolio_outcomes):
        """Generate domain-specific outcome basket analysis for subportfolio"""
        return self.generate_unified_domain_analysis(
            outcomes_df=subportfolio_outcomes,
            analysis_level="subportfolio"
        )
    
    def generate_unified_domain_analysis(self, outcomes_df, analysis_level="theme"):
        """
        Unified function to generate domain analysis for theme, sub-portfolio, or individual grant pages
        
        Args:
            outcomes_df: pandas DataFrame with outcomes data
            analysis_level: str - "theme", "subportfolio", or "grant" to determine header levels
        
        Returns:
            str: HTML content for domain analysis
        """
        if outcomes_df.empty:
            return "<p>No outcomes data available for detailed analysis.</p>"
        
        # Define header levels based on analysis level
        header_levels = {
            "theme": {"domain": "h4", "section": "h5"},
            "subportfolio": {"domain": "h4", "section": "h5"},  # Fixed to match theme
            "grant": {"domain": "h3", "section": "h4"}
        }
        
        domain_header = header_levels[analysis_level]["domain"]
        section_header = header_levels[analysis_level]["section"]
        
        # Group outcomes by domain
        outcomes_by_domain = {}
        for _, outcome in outcomes_df.iterrows():
            domain = outcome.get('Domain of Change')  # Correct column name
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key not in outcomes_by_domain:
                    outcomes_by_domain[domain_key] = []
                outcomes_by_domain[domain_key].append(outcome)
        
        # Calculate total outcomes for percentage calculation
        total_outcomes = len(outcomes_df)
        
        # Generate analysis for each domain
        html = ""
        for domain_full_text in sorted(outcomes_by_domain.keys()):
            domain_outcomes = outcomes_by_domain[domain_full_text]
            outcome_count = len(domain_outcomes)
            
            # Extract domain key (e.g., D1 from "D1 - Long name")
            domain_key = domain_full_text.split(' - ')[0] if ' - ' in domain_full_text else domain_full_text
            
            # Count total outcome baskets tagged in this domain
            total_obs_tagged = 0
            for outcome in domain_outcomes:
                ob_names = outcome.get('Outcome Basket names')  # Correct column name
                if pd.notna(ob_names):
                    # Split by semicolon and count each OB
                    ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                    total_obs_tagged += len(ob_list)
            
            # Domain header format consistent with theme
            html += f"<{domain_header}>Domain of Change {domain_full_text}</{domain_header}>\n"
            html += f"<ul><li><strong>Number of outcomes:</strong> {outcome_count}</li>\n"
            
            # Get default OBs for this domain
            default_obs_for_domain = self.default_obs.get(domain_key, [])
            
            # Classify all OBs used in this domain
            used_default_obs = {}
            cross_domain_obs = {}
            custom_obs = {}
            
            for outcome in domain_outcomes:
                ob_names = outcome.get('Outcome Basket names')  # Correct column name
                if pd.notna(ob_names):
                    # Split by semicolon and process each OB
                    ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                    for ob_str in ob_list:
                        # Classify this OB
                        if ob_str in default_obs_for_domain:
                            # Default OB for this domain
                            if ob_str not in used_default_obs:
                                used_default_obs[ob_str] = 0
                            used_default_obs[ob_str] += 1
                        else:
                            # Check if it's from another domain
                            source_domain = None
                            for other_domain, other_obs in self.default_obs.items():
                                if ob_str in other_obs:
                                    source_domain = other_domain
                                    break
                            
                            if source_domain:
                                # Cross-domain OB
                                ob_label = f"{ob_str} ({source_domain})"
                                if ob_label not in cross_domain_obs:
                                    cross_domain_obs[ob_label] = 0
                                cross_domain_obs[ob_label] += 1
                            else:
                                # Custom OB
                                if ob_str not in custom_obs:
                                    custom_obs[ob_str] = 0
                                custom_obs[ob_str] += 1
            
            # Calculate totals for percentages
            total_default_obs = sum(used_default_obs.values())
            total_cross_domain_obs = sum(cross_domain_obs.values())
            total_custom_obs = sum(custom_obs.values())
            
            # Add OB breakdown
            html += f"<li><strong>Number of OB:</strong> {total_obs_tagged}</li></ul>\n"
            
            # Add pie chart for OB distribution (only for theme and subportfolio)
            if analysis_level in ["theme", "subportfolio"]:
                html += self.generate_domain_ob_pie_chart(domain_key, total_default_obs, total_cross_domain_obs, total_custom_obs, total_obs_tagged)
            
            # 1. Usage of default Outcome Baskets - Display as horizontal bar chart for theme, list for others
            if default_obs_for_domain:
                html += f"<{section_header}>Default OB used</{section_header}>\n"
                
                # Create combined list of used and unused default OBs
                all_default_obs = []
                
                # Add used OBs with their counts
                for ob in default_obs_for_domain:
                    count = used_default_obs.get(ob, 0)
                    if count > 0:
                        percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                        all_default_obs.append((ob, count, f"{count} uses (tagged in {percentage:.1f}% of {domain_key} outcomes)"))
                    else:
                        all_default_obs.append((ob, 0, "0 uses"))

                # Always show all default OBs (including 0-count) using charts for all analysis levels
                if all_default_obs:
                    html += self.generate_default_ob_bar_chart(domain_key, all_default_obs, outcome_count)
                else:
                    html += "<p>No default outcome baskets are defined for this domain.</p>\n"
            
            # 2. Cross-Domain OBs Used in this Domain
            if cross_domain_obs:
                html += f"<{section_header}>Cross-domain OB used</{section_header}>\n<ul>\n"
                for ob_label, count in sorted(cross_domain_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li><strong>{ob_label}</strong>: {count} uses (tagged in {percentage:.1f}% of {domain_key} outcomes)</li>\n'
                html += "</ul>\n\n"
            
            # 3. Custom OBs in this Domain
            if custom_obs:
                html += f"<{section_header}>Custom OB used</{section_header}>\n<ul>\n"
                for ob, count in sorted(custom_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li><strong>{ob}</strong>: {count} uses (tagged in {percentage:.1f}% of {domain_key} outcomes)</li>\n'
                html += "</ul>\n\n"
        
        return html
    
    def generate_default_ob_bar_chart(self, domain_key, all_default_obs, outcome_count):
        """Generate a horizontal bar chart for default OB usage within a specific domain"""
        if not all_default_obs:
            return ""
        
        # Use ALL default OBs (including those with 0 count) to show complete picture
        chart_obs = all_default_obs
        
        if not chart_obs:
            return "<p>No default outcome baskets are defined for this domain.</p>"
        
        # Prepare data for the chart
        chart_labels = []
        chart_data = []
        
        for ob, count, _ in chart_obs:
            chart_labels.append(ob)
            chart_data.append(count)
        
        # Generate unique chart ID
        chart_id = f"defaultOBChart_{domain_key.replace(' ', '_').replace('-', '_')}"
        
        # Use consistent max value across all domain charts for visual alignment
        max_value = 70  # Fixed scale to allow comparison across domains (next multiple of 10 after 60)
        
        # Generate Chart.js HTML
        chart_html = f"""
<div style="margin: 20px 0;">
    <div style="max-width: 800px; margin: 0 auto;">
        <canvas id="{chart_id}" width="600" height="{len(chart_data) * 60 + 120}"></canvas>
    </div>
</div>
<script>
    const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
    
    const data_{chart_id} = {{
        labels: {chart_labels},
        datasets: [{{
            label: 'Occurrences',
            data: {chart_data},
            backgroundColor: '#2c5aa0',
            borderColor: '#1e3a6f',
            borderWidth: 1
        }}]
    }};
    
    const config_{chart_id} = {{
        type: 'bar',
        data: data_{chart_id},
        options: {{
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    display: false
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const count = context.parsed.x;
                            const percentage = ((count / {outcome_count}) * 100).toFixed(1);
                            return count + ' occurrences (' + percentage + '% of {domain_key} outcomes)';
                        }}
                    }}
                }},
                datalabels: {{
                    anchor: 'end',
                    align: 'right',
                    color: '#333',
                    font: {{
                        size: 11,
                        weight: 'normal'
                    }},
                    formatter: function(value, context) {{
                        const percentage = ((value / {outcome_count}) * 100).toFixed(1);
                        const firstLine = value;
                        const secondLine = '(' + percentage + '% of {domain_key} outcomes)';
                        
                        return [firstLine, secondLine];
                    }}
                }}
            }},
            scales: {{
                x: {{
                    beginAtZero: true,
                    max: {max_value},
                    title: {{
                        display: true,
                        text: 'Number of Occurrences'
                    }}
                }},
                y: {{
                    title: {{
                        display: true,
                        text: 'Outcome Baskets'
                    }},
                    ticks: {{
                        callback: function(value, index, values) {{
                            const label = this.getLabelForValue(value);
                            if (label.length <= 35) {{
                                return label;
                            }}
                            
                            // Simple text wrapping for long labels
                            const words = label.split(' ');
                            const lines = [];
                            let currentLine = '';
                            
                            for (let i = 0; i < words.length; i++) {{
                                const word = words[i];
                                if ((currentLine + ' ' + word).trim().length <= 35) {{
                                    currentLine = (currentLine + ' ' + word).trim();
                                }} else {{
                                    if (currentLine) {{
                                        lines.push(currentLine);
                                        currentLine = word;
                                    }} else {{
                                        currentLine = word;
                                    }}
                                }}
                            }}
                            if (currentLine) {{
                                lines.push(currentLine);
                            }}
                            
                            return lines;
                        }}
                    }}
                }}
            }}
        }},
        plugins: []
    }};
    
    new Chart(ctx_{chart_id}, config_{chart_id});
</script>
"""
        
        return chart_html
    
    def generate_domain_ob_pie_chart(self, domain_key, total_default_obs, total_cross_domain_obs, total_custom_obs, total_obs_tagged):
        """Generate a pie chart for outcome basket distribution within a specific domain"""
        if total_obs_tagged == 0:
            return ""
        
        # Prepare data for the chart - ALWAYS include all three categories for consistent colors
        chart_data = []
        chart_labels = []
        chart_colors = []
        
        # Define fixed color mapping for consistent visualization
        category_colors = {
            'Default OB': '#2c5aa0',      # Blue
            'Other domains OB': '#28a745',  # Green  
            'Custom OB': '#ffc107'        # Yellow
        }
        
        # Add all categories, including those with 0 values, but only show non-zero in chart
        categories = [
            ('Default OB', total_default_obs),
            ('Other domains OB', total_cross_domain_obs),
            ('Custom OB', total_custom_obs)
        ]
        
        # Only include categories with values > 0 in the actual chart data
        for category_name, count in categories:
            if count > 0:
                chart_labels.append(category_name)
                chart_data.append(count)
                chart_colors.append(category_colors[category_name])
        
        # Generate unique chart ID
        chart_id = f"domainOBChart_{domain_key.replace(' ', '_')}"
        
        # Generate Chart.js HTML
        chart_html = f"""
<div style="margin: 20px 0; text-align: center;">
    <div style="max-width: 600px; margin: 0 auto;">
        <canvas id="{chart_id}" width="300" height="300"></canvas>
    </div>
</div>
<script>
    const ctx_{domain_key.replace(' ', '_')} = document.getElementById('{chart_id}').getContext('2d');
    const domainOBChart_{domain_key.replace(' ', '_')} = new Chart(ctx_{domain_key.replace(' ', '_')}, {{
        type: 'pie',
        data: {{
            labels: {chart_labels},
            datasets: [{{
                data: {chart_data},
                backgroundColor: {chart_colors},
                borderColor: '#fff',
                borderWidth: 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'right',
                    labels: {{
                        padding: 15,
                        usePointStyle: true,
                        font: {{
                            size: 11
                        }},
                        boxWidth: 12,
                        boxHeight: 12
                    }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(0);
                            return label + ': ' + value + ' (' + percentage + '%)';
                        }}
                    }}
                }},
                datalabels: {{
                    display: true,
                    color: 'white',
                    font: {{
                        weight: 'bold',
                        size: 12
                    }},
                    anchor: 'center',
                    align: 'end',
                    offset: 8,
                    formatter: function(value, context) {{
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(0);
                        return value + '\\n(' + percentage + '%)';
                    }}
                }}
            }},
            layout: {{
                padding: {{
                    right: 20,
                    left: 20,
                    top: 20,
                    bottom: 20
                }}
            }}
        }},
        plugins: []
    }});
</script>
"""
        return chart_html
    
    def generate_cross_domain_summary(self):
        """Generate cross-domain usage summary - simplified version"""
        if self.outcomes_df.empty:
            return ""
        
        # For now, we'll show the most frequently used outcome baskets across all domains
        all_obs_usage = {}
        total_outcomes = len(self.outcomes_df)
        
        for _, outcome in self.outcomes_df.iterrows():
            ob_names = outcome.get('Outcome Basket names')
            if pd.notna(ob_names):
                # Split by semicolon and process each OB
                ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                for ob_str in ob_list:
                    if ob_str not in all_obs_usage:
                        all_obs_usage[ob_str] = 0
                    all_obs_usage[ob_str] += 1
        
        html = "<h3>Most frequently used Outcome Baskets (across all Domains of change)</h3>\n\n"
        
        # Most used OBs across all domains
        html += "<ol>\n"
        for ob_name, count in sorted(all_obs_usage.items(), key=lambda x: x[1], reverse=True)[:15]:  # Top 15
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            html += f'<li><strong>{ob_name}</strong>: {count} uses (tagged in {percentage:.1f}% of the outcomes)</li>\n'
        html += "</ol>\n\n"
        
        return html
    
    def generate_custom_obs_analysis(self):
        """Generate analysis of custom outcome baskets for standardization candidates"""
        if self.outcomes_df.empty:
            return ""
        
        # Build a comprehensive set of all default outcome baskets across all domains
        all_default_obs = set()
        for domain_obs in self.default_obs.values():
            all_default_obs.update(domain_obs)
        
        # Count only truly custom outcome baskets (not default in any domain)
        custom_obs_usage = {}
        total_outcomes = len(self.outcomes_df)
        
        for _, outcome in self.outcomes_df.iterrows():
            ob_names = outcome.get('Outcome Basket names')
            if pd.notna(ob_names):
                # Split by semicolon and process each OB
                ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                for ob_str in ob_list:
                    # Only count if it's NOT a default OB in any domain
                    if ob_str not in all_default_obs:
                        if ob_str not in custom_obs_usage:
                            custom_obs_usage[ob_str] = 0
                        custom_obs_usage[ob_str] += 1
        
        html = "<h4>Top Custom OB - with potential to become default</h4>\n"
        html += "<p>Some of these correspond to OB that used to be default for a DoC, but stopped being default after a review of the Theme's OB).</p>\n"
        
        if not custom_obs_usage:
            html += "<p>No truly custom outcome baskets found (all outcome baskets are default for at least one domain).</p>\n"
        else:
            html += "<ul>\n"
            # Show top custom OBs by frequency
            for ob_name, count in sorted(custom_obs_usage.items(), key=lambda x: x[1], reverse=True)[:6]:  # Top 6
                percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
                html += f'<li><strong>{ob_name}</strong>: {count} uses (tagged in {percentage:.1f}% of the outcomes)</li>\n'
            html += "</ul>\n"
        
        html += "\n"
        return html
    
    def generate_significance_stacked_chart(self, significance_by_domain, domain_totals, overall_significance, total_outcomes):
        """Generate a stacked bar chart for significance patterns across domains"""
        if not significance_by_domain or total_outcomes == 0:
            return ""
        
        # Define significance order and domain order
        significance_order = ['High', 'Medium', 'Low', 'Not specified']
        
        # Use all possible domains from domain mapping to ensure consistency across pages
        domain_order = sorted(self.domain_mapping.keys())
        
        # Extended color palette to support more domains (up to 12 domains)
        domain_colors = [
            '#2c5aa0',  # Blue (D1)
            '#28a745',  # Green (D2) 
            '#ffc107',  # Yellow (D3)
            '#dc3545',  # Red (D4)
            '#6f42c1',  # Purple (D5/C5/E5)
            '#fd7e14',  # Orange (D6/C6/E6)
            '#20c997',  # Teal (D7/C7/E7)
            '#e83e8c',  # Pink (D8/C8/E8)
            '#6c757d',  # Gray (D9/C9/E9)
            '#17a2b8',  # Cyan (D10/C10/E10)
            '#795548',  # Brown (D11/C11/E11)
            '#607d8b'   # Blue Gray (D12/C12/E12)
        ]
        
        # Build datasets for Chart.js (one dataset per domain)
        datasets_json = []
        for i, domain_code in enumerate(domain_order):
            domain_name = self.domain_mapping.get(domain_code, domain_code)
            dataset_data = []
            
            # Get data for each significance level for this domain (0 if domain has no data)
            for sig_level in significance_order:
                if domain_code in significance_by_domain:
                    count = significance_by_domain[domain_code].get(sig_level, 0)
                else:
                    count = 0  # Domain has no outcomes at all
                dataset_data.append(count)
            
            # Create proper JSON-formatted dataset
            domain_label = f'{domain_code}: {domain_name[:50]}...' if len(domain_name) > 50 else f'{domain_code}: {domain_name}'
            datasets_json.append(f"""{{
                label: '{domain_label}',
                data: {dataset_data},
                backgroundColor: '{domain_colors[i]}',
                borderColor: '{domain_colors[i]}',
                borderWidth: 1
            }}""")
        
        # Build labels for x-axis
        chart_labels = significance_order
        
        # Calculate totals and percentages for top labels
        top_labels = []
        for sig_level in significance_order:
            total_count = overall_significance.get(sig_level, 0)
            total_percentage = (total_count / total_outcomes * 100) if total_outcomes > 0 else 0
            top_labels.append(f'{total_count} ({total_percentage:.1f}%)')
        
        # Generate unique chart ID
        chart_id = "significanceStackedChart"
        
        # Join datasets with proper formatting
        datasets_str = ',\n        '.join(datasets_json)
        
        # Calculate maximum stack total to determine appropriate stepSize and max
        max_stack_total = 0
        for domain_code in domain_order:
            stack_total = 0
            for sig_level in significance_order:
                if domain_code in significance_by_domain:
                    stack_total += significance_by_domain[domain_code].get(sig_level, 0)
            max_stack_total = max(max_stack_total, stack_total)

        # Simple approach: add 5% padding to max value for better visibility
        suggested_max = max_stack_total * 1.05

        # Create the Chart.js stacked bar chart
        chart_html = f"""
<div style="margin: 30px 0;">
    <div style="max-width: 900px; margin: 0 auto;">
        <canvas id="{chart_id}" width="800" height="500"></canvas>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
    
    const data_{chart_id} = {{
        labels: {chart_labels},
        datasets: [{datasets_str}]
    }};
    
    const config_{chart_id} = {{
        type: 'bar',
        data: data_{chart_id},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    stacked: true,
                    title: {{
                        display: true,
                        text: 'Significance Levels'
                    }}
                }},
                y: {{
                    stacked: true,
                    suggestedMax: {suggested_max},
                    title: {{
                        display: true,
                        text: 'Number of Outcomes'
                    }}
                }}
            }},
            plugins: {{
                legend: {{
                    display: true,
                    position: 'bottom'
                }},
                tooltip: {{
                    callbacks: {{
                        afterLabel: function(context) {{
                            const domainCode = context.dataset.label.split(':')[0];
                            const domainTotals = {str(domain_totals).replace("'", '"')};
                            const domainTotal = domainTotals[domainCode] || 1;
                            const percentage = ((context.raw / domainTotal) * 100).toFixed(1);
                            return `${{percentage}}% of ${{domainCode}} outcomes`;
                        }}
                    }}
                }}
            }}
        }},
        plugins: [{{
            afterDatasetsDraw: function(chart) {{
                const ctx = chart.ctx;
                
                // Calculate totals for each significance level
                const totals = [];
                const totalOutcomes = {total_outcomes};
                
                chart.data.labels.forEach((label, index) => {{
                    let stackTotal = 0;
                    chart.data.datasets.forEach((dataset, datasetIndex) => {{
                        const meta = chart.getDatasetMeta(datasetIndex);
                        if (meta && !meta.hidden) {{
                            stackTotal += dataset.data[index];
                        }}
                    }});
                    totals.push(stackTotal);
                }});
                
                // Draw total labels on top of each bar
                chart.data.labels.forEach((label, index) => {{
                    const meta = chart.getDatasetMeta(0);
                    if (meta && meta.data[index]) {{
                        const stackTotal = totals[index];
                        const percentage = ((stackTotal / totalOutcomes) * 100).toFixed(1);
                        const labelText = `${{stackTotal}} outcomes (${{percentage}}%)`;
                        
                        // Get the pixel position for the top of the stack
                        const yPosition = chart.scales.y.getPixelForValue(stackTotal);
                        const xPosition = meta.data[index].x;
                        
                        // Draw the total label
                        ctx.save();
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'bottom';
                        ctx.fillStyle = '#333';
                        ctx.font = 'bold 12px Arial';
                        ctx.fillText(labelText, xPosition, yPosition - 8);
                        ctx.restore();
                    }}
                }});
            }}
        }}]
    }};
    
    new Chart(ctx_{chart_id}, config_{chart_id});
</script>
"""
        
        return chart_html
    
    def generate_significance_distribution(self):
        """Generate theme-wide significance patterns analysis with domain breakdown"""
        if self.outcomes_df.empty:
            return ""
        
        
        # Count significance levels by domain (including null/empty values)
        total_outcomes = len(self.outcomes_df)
        
        # Initialize data structure for significance by domain
        significance_by_domain = {}
        domain_totals = {}
        
        # Count outcomes by domain and significance level
        for _, outcome in self.outcomes_df.iterrows():
            significance = outcome.get('Significance Level')
            domain_full = outcome.get('Domain of Change')
            
            # Clean up significance level (replace NaN with 'Not specified')
            if pd.isna(significance):
                significance = 'Not specified'
            
            # Extract domain code from full domain string (e.g., "D4 - Description" -> "D4")
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    
                    # Initialize domain if not seen before
                    if domain_code not in significance_by_domain:
                        significance_by_domain[domain_code] = {}
                        domain_totals[domain_code] = 0
                    
                    # Initialize significance level if not seen before
                    if significance not in significance_by_domain[domain_code]:
                        significance_by_domain[domain_code][significance] = 0
                    
                    # Count this outcome
                    significance_by_domain[domain_code][significance] += 1
                    domain_totals[domain_code] += 1
        
        # Calculate overall significance totals
        overall_significance = {}
        for domain_data in significance_by_domain.values():
            for sig_level, count in domain_data.items():
                if sig_level not in overall_significance:
                    overall_significance[sig_level] = 0
                overall_significance[sig_level] += count
        
        html = "<h3>Theme-Wide Significance Patterns</h3>\n"
        html += "<p>Distribution of outcome significance levels across domains of change. "
        html += "<span class='info-icon' onclick='showSignificanceInfo()' title='Click for more information about significance levels'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_significance_stacked_chart(significance_by_domain, domain_totals, overall_significance, total_outcomes)
        
        # Add examples of outcomes by significance level
        html += self.generate_significance_examples(self.outcomes_df, overall_significance)
        
        return html
    
    def generate_special_cases_stacked_chart(self, special_cases_by_domain, domain_totals, overall_special_cases, total_outcomes):
        """Generate a stacked bar chart for special cases patterns across domains"""
        if not special_cases_by_domain or total_outcomes == 0:
            return ""
        
        # Use all possible domains from domain mapping to ensure consistency across pages
        domain_order = sorted(self.domain_mapping.keys())
        
        # Get all defined special case types (show even if 0 values)
        all_defined_special_cases = [
            'Unexpected Outcome',
            'Negative Outcome', 
            'Signal towards Outcome',
            'Indicator Based Outcome',
            'Keep on Radar',
            'Programme Outcome'
        ]
        special_case_types = all_defined_special_cases
        
        # Extended color palette for special cases
        special_case_colors = [
            '#dc3545',  # Red
            '#fd7e14',  # Orange
            '#ffc107',  # Yellow
            '#28a745',  # Green
            '#17a2b8',  # Cyan
            '#6f42c1',  # Purple
            '#e83e8c',  # Pink
            '#6c757d'   # Gray
        ]
        
        # Build datasets for Chart.js (one dataset per domain, matching significance chart)
        # Use same domain colors as significance chart for consistency
        domain_colors = [
            '#2c5aa0',  # Blue (D1)
            '#28a745',  # Green (D2) 
            '#ffc107',  # Yellow (D3)
            '#dc3545',  # Red (D4)
            '#6f42c1',  # Purple (D5)
            '#fd7e14',  # Orange (D6)
            '#20c997',  # Teal (D7)
            '#e83e8c',  # Pink (D8)
            '#6c757d',  # Gray (D9)
            '#17a2b8',  # Cyan (D10)
            '#795548',  # Brown (D11)
            '#607d8b'   # Blue Gray (D12)
        ]
        
        datasets_json = []
        for i, domain_code in enumerate(domain_order):
            domain_name = self.domain_mapping.get(domain_code, domain_code)
            dataset_data = []
            
            # Get data for each special case type for this domain (0 if domain has no data)
            for case_type in special_case_types:
                if domain_code in special_cases_by_domain:
                    count = special_cases_by_domain[domain_code].get(case_type, 0)
                else:
                    count = 0  # Domain has no outcomes at all
                dataset_data.append(count)
            
            # Create proper JSON-formatted dataset with full domain description
            domain_name = self.domain_mapping.get(domain_code, domain_code)
            domain_label = f'{domain_code}: {domain_name[:50]}...' if len(domain_name) > 50 else f'{domain_code}: {domain_name}'
            datasets_json.append(f"""{{
                label: '{domain_label}',
                data: {dataset_data},
                backgroundColor: '{domain_colors[i % len(domain_colors)]}',
                borderColor: '{domain_colors[i % len(domain_colors)]}',
                borderWidth: 1
            }}""")
        
        # Build labels for x-axis using special case types
        chart_labels = special_case_types
        
        # Generate unique chart ID
        import time
        chart_id = f"specialCasesChart_{int(time.time() * 1000) % 100000}"
        
        # Join datasets with proper formatting
        datasets_joined = ',\n            '.join(datasets_json)
        
        # Calculate maximum stack total for simple padding (max per special case type)
        max_stack_total = 0
        for case_type in special_case_types:
            stack_total = 0
            for domain_code in domain_order:
                if domain_code in special_cases_by_domain:
                    stack_total += special_cases_by_domain[domain_code].get(case_type, 0)
            max_stack_total = max(max_stack_total, stack_total)

        # Simple approach: add 5% padding to max value for better visibility
        suggested_max = max_stack_total * 1.05

        html = f"""<div style="margin: 30px 0;">
    <div style="max-width: 900px; margin: 0 auto;">
        <canvas id="{chart_id}" width="800" height="500"></canvas>
    </div>
</div>

<script>
(function() {{
    const data_{chart_id} = {{
        labels: {chart_labels},
        datasets: [
            {datasets_joined}
        ]
    }};

    const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
    const chart_{chart_id} = new Chart(ctx_{chart_id}, {{
        type: 'bar',
        data: data_{chart_id},
        options: {{
            responsive: true,
            scales: {{
                x: {{
                    stacked: true,
                    title: {{
                        display: true,
                        text: 'Special Case Type'
                    }}
                }},
                y: {{
                    stacked: true,
                    suggestedMax: {suggested_max},
                    title: {{
                        display: true,
                        text: 'Number of Outcomes'
                    }}
                }}
            }},
            plugins: {{
                title: {{
                    display: false
                }},
                legend: {{
                    display: true,
                    position: 'bottom'
                }},
                tooltip: {{
                    callbacks: {{
                        afterLabel: function(context) {{
                            const domainCode = context.dataset.label.split(':')[0];
                            const domainTotals = {str(domain_totals).replace("'", '"')};
                            const domainTotal = domainTotals[domainCode] || 1;
                            const percentage = ((context.raw / domainTotal) * 100).toFixed(1);
                            return `${{percentage}}% of ${{domainCode}} outcomes`;
                        }}
                    }}
                }}
            }}
        }},
        plugins: [{{
            afterDatasetsDraw: function(chart) {{
                const ctx = chart.ctx;
                
                // Calculate totals for each domain
                const totals = [];
                const totalOutcomes = {total_outcomes};
                
                chart.data.labels.forEach((label, index) => {{
                    let stackTotal = 0;
                    chart.data.datasets.forEach((dataset, datasetIndex) => {{
                        const meta = chart.getDatasetMeta(datasetIndex);
                        if (meta && !meta.hidden) {{
                            stackTotal += dataset.data[index];
                        }}
                    }});
                    totals.push(stackTotal);
                }});
                
                // Draw total labels on top of each stack
                chart.data.labels.forEach((label, index) => {{
                    const total = totals[index];
                    if (total > 0) {{
                        let lastDatasetIndex = -1;
                        for (let i = chart.data.datasets.length - 1; i >= 0; i--) {{
                            const meta = chart.getDatasetMeta(i);
                            if (meta && !meta.hidden && chart.data.datasets[i].data[index] > 0) {{
                                lastDatasetIndex = i;
                                break;
                            }}
                        }}
                        
                        if (lastDatasetIndex >= 0) {{
                            const meta = chart.getDatasetMeta(lastDatasetIndex);
                            const bar = meta.data[index];
                            
                            // Calculate percentage of total outcomes
                            const percentage = ((total / {total_outcomes}) * 100).toFixed(1);
                            const displayText = `${{total}} outcomes (${{percentage}}%)`;
                            
                            ctx.fillStyle = '#000';
                            ctx.font = 'bold 11px Arial';
                            ctx.textAlign = 'center';
                            ctx.fillText(displayText, bar.x, bar.y - 8);
                        }}
                    }}
                }});
            }}
        }}]
    }});
}})();
</script>
"""
        
        return html
    
    def generate_special_cases_analysis(self):
        """Generate theme-wide special cases analysis with domain breakdown"""
        if self.outcomes_df.empty:
            return ""
        
        # Special cases columns
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        
        # Initialize data structures
        special_cases_by_domain = {}
        domain_totals = {}
        total_outcomes = len(self.outcomes_df)
        
        # Count special cases by domain
        for _, outcome in self.outcomes_df.iterrows():
            domain_full = outcome.get('Domain of Change')
            
            # Extract domain code from full domain string (e.g., "D4 - Description" -> "D4")
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    
                    # Initialize domain if not seen before
                    if domain_code not in special_cases_by_domain:
                        special_cases_by_domain[domain_code] = {}
                        domain_totals[domain_code] = 0
                    
                    # Count total outcomes for this domain
                    domain_totals[domain_code] += 1
                    
                    # Check each special case column
                    for col in special_cases_cols:
                        if col in self.outcomes_df.columns:
                            clean_name = col.replace('Special case - ', '')
                            
                            # Initialize special case type if not seen before
                            if clean_name not in special_cases_by_domain[domain_code]:
                                special_cases_by_domain[domain_code][clean_name] = 0
                            
                            # Count if this outcome has this special case
                            if outcome.get(col) == 1:
                                special_cases_by_domain[domain_code][clean_name] += 1
        
        # Calculate overall totals for summary (include all defined special cases)
        special_cases_totals = {}
        for col in special_cases_cols:
            clean_name = col.replace('Special case - ', '')
            if col in self.outcomes_df.columns:
                count = (self.outcomes_df[col] == 1).sum()
            else:
                count = 0
            special_cases_totals[clean_name] = count
        
        # Only return empty if no special case columns exist at all
        if not special_cases_cols:
            return ""
        
        html = "<h3>Theme-wide special case outcomes</h3>\n"
        html += "<p>Distribution of special case outcomes across domains of change. "
        html += "<span class='info-icon' onclick='showSpecialCasesInfo()' title='Click for more information about special case types'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart using the dedicated method
        html += self.generate_special_cases_stacked_chart(special_cases_by_domain, domain_totals, special_cases_totals, total_outcomes)
        
        # Add examples of special case outcomes
        html += self.generate_special_cases_examples(self.outcomes_df, special_cases_totals)
        
        return html

    def generate_significance_examples(self, outcomes_df, significance_with_counts):
        """Generate examples for significance patterns, showing up to 3 examples per significance level"""
        import pandas as pd
        import html as html_module
        
        if outcomes_df.empty or not significance_with_counts:
            return ""
        
        html = "<h4>Examples of outcomes for each significance level</h4>\n"
        
        # Define the order we want to display significance levels
        significance_order = ['High', 'Medium', 'Low']
        
        # Process each significance level that has count > 0
        for significance_level in significance_order:
            if significance_level in significance_with_counts and significance_with_counts[significance_level] > 0:
                # Filter outcomes with this significance level
                relevant_outcomes = outcomes_df[outcomes_df['Significance Level'] == significance_level].copy()
                
                if not relevant_outcomes.empty:
                    # Sort by report date (most recent first) for consistency
                    relevant_outcomes = relevant_outcomes.sort_values('Report submission date', ascending=False)
                    
                    # Select up to 3 examples, prioritizing different grants for variety
                    examples = []
                    used_grants = set()
                    
                    for _, outcome in relevant_outcomes.iterrows():
                        gams_code = outcome.get('GAMS code', '')
                        # If we haven't used this grant yet, or if we need more examples and have no choice
                        if gams_code not in used_grants or len(examples) < 3:
                            examples.append(outcome)
                            used_grants.add(gams_code)
                            if len(examples) >= 3:
                                break
                    
                    # Convert back to DataFrame for consistent processing
                    if examples:
                        examples = pd.DataFrame(examples).reset_index(drop=True)
                        
                        # Generate HTML for this significance level
                        html += f"<strong>{significance_level} significance</strong>\n<ul>\n"
                        
                        for _, outcome in examples.iterrows():
                            # Main description (no bolding, longer truncation)
                            description = outcome.get('Description of the Reported Outcome', 'Not specified')
                            if pd.notna(description) and str(description).strip():
                                # Truncate long descriptions (300 characters)
                                if len(str(description)) > 300:
                                    description = str(description)[:297] + "..."
                                description = html_module.escape(str(description))
                            else:
                                description = "Not specified"
                            
                            html += f"  <li>{description}</li>\n"
                        
                        html += "</ul>\n\n"
        
        return html

    def generate_special_cases_examples(self, outcomes_df, special_cases_with_counts):
        """Generate examples of special case outcomes, prioritizing high significance outcomes"""
        import pandas as pd
        import html as html_module
        
        if outcomes_df.empty or not special_cases_with_counts:
            return ""
        
        # Define significance priority (lower number = higher priority)
        significance_priority = {
            'High': 1, 
            'Medium': 2, 
            'Low': 3, 
            'Not specified': 4
        }
        
        html = "<h4>Examples of special case outcomes</h4>\n"
        
        # Process each special case type that has count > 0
        for case_type, count in special_cases_with_counts.items():
            if count > 0:
                # Find the corresponding column name
                case_col = f'Special case - {case_type}'
                
                if case_col in outcomes_df.columns:
                    # Filter outcomes where this special case = 1
                    relevant_outcomes = outcomes_df[outcomes_df[case_col] == 1].copy()
                    
                    if not relevant_outcomes.empty:
                        # Add significance priority for sorting
                        relevant_outcomes['sig_priority'] = relevant_outcomes['Significance Level'].fillna('Not specified').map(
                            lambda x: significance_priority.get(x, 5)
                        )
                        
                        # Sort by significance priority, then by report date (most recent first)
                        relevant_outcomes = relevant_outcomes.sort_values([
                            'sig_priority', 
                            'Report submission date'
                        ], ascending=[True, False])
                        
                        # Select up to 3 examples, prioritizing different grants for variety
                        examples = []
                        used_grants = set()
                        
                        for _, outcome in relevant_outcomes.iterrows():
                            gams_code = outcome.get('GAMS code', '')
                            # If we haven't used this grant yet, or if we need more examples and have no choice
                            if gams_code not in used_grants or len(examples) < 3:
                                examples.append(outcome)
                                used_grants.add(gams_code)
                                if len(examples) >= 3:
                                    break
                        
                        # Convert back to DataFrame for consistent processing
                        if examples:
                            examples = pd.DataFrame(examples).reset_index(drop=True)
                            
                            # Generate HTML for this special case type
                            html += f"<strong>{case_type}</strong>\n<ul>\n"
                            
                            for _, outcome in examples.iterrows():
                                # Main description (no bolding, longer truncation)
                                description = outcome.get('Description of the Reported Outcome', 'Not specified')
                                if pd.notna(description) and str(description).strip():
                                    # Truncate long descriptions (50% longer: 200 -> 300)
                                    if len(str(description)) > 300:
                                        description = str(description)[:297] + "..."
                                    description = html_module.escape(str(description))
                                else:
                                    description = "Not specified"
                                
                                html += f"  <li>{description}</li>\n"
                            
                            html += "</ul>\n\n"
        
        return html

    def generate_geographic_distribution(self):
        """Generate geographic distribution analysis with regional aggregation"""
        if self.outcomes_df.empty:
            return ""
        
        # Define regions and their countries
        regions = {
            'Sub-Saharan Africa': ['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 
                                 'Chad', 'Comoros', 'Congo', 'Democratic Republic of the Congo', 'Côte d\'Ivoire', 'Djibouti', 'Equatorial Guinea', 
                                 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 
                                 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
                                 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 
                                 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'],
            'South Asia': ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'],
            'East Asia & Pacific': ['Cambodia', 'China', 'Fiji', 'Indonesia', 'Kiribati', 'Korea, Democratic People\'s Republic', 'Lao PDR', 
                                  'Malaysia', 'Marshall Islands', 'Micronesia', 'Mongolia', 'Myanmar', 'Palau', 'Papua New Guinea', 'Philippines', 
                                  'Samoa', 'Solomon Islands', 'Thailand', 'Timor-Leste', 'Tonga', 'Tuvalu', 'Vanuatu', 'Vietnam'],
            'Middle East & North Africa': ['Algeria', 'Bahrain', 'Djibouti', 'Egypt', 'Iran', 'Iraq', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 
                                         'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'],
            'Latin America & Caribbean': ['Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bolivia', 'Brazil', 'Chile', 
                                        'Colombia', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada', 
                                        'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 
                                        'Peru', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Suriname', 
                                        'Trinidad and Tobago', 'Uruguay', 'Venezuela'],
            'Europe & Central Asia': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Georgia', 
                                    'Kazakhstan', 'Kosovo', 'Kyrgyz Republic', 'Moldova', 'Montenegro', 'North Macedonia', 'Romania', 'Russia', 
                                    'Serbia', 'Tajikistan', 'Turkey', 'Turkmenistan', 'Ukraine', 'Uzbekistan']
        }
        
        # Create reverse mapping from country to region
        country_to_region = {}
        for region, countries in regions.items():
            for country in countries:
                country_to_region[country.lower()] = region
        
        # Count outcomes by country and region
        geo_counts = {}  # For countries
        region_counts = {}  # For regions
        global_count = 0
        total_geo_instances = 0
        
        for _, outcome in self.outcomes_df.iterrows():
            countries = outcome.get('Country(ies) / Global')
            if pd.notna(countries):
                # Only split by semicolon to preserve country names that contain commas
                country_list = [c.strip() for c in str(countries).split(';') if c.strip()]
                
                # Process each country
                for country in country_list:
                    # Handle global specially
                    if country.lower() == 'global':
                        global_count += 1
                        total_geo_instances += 1
                        continue
                    
                    # Skip only truly invalid entries
                    if country.lower() in ['n/a', 'na', 'none']:
                        continue
                    # Count by country
                    if country not in geo_counts:
                        geo_counts[country] = 0
                    geo_counts[country] += 1
                    total_geo_instances += 1
                    
                    # Count by region
                    country_lower = country.lower()
                    if country_lower in country_to_region:
                        region = country_to_region[country_lower]
                        if region not in region_counts:
                            region_counts[region] = 0
                        region_counts[region] += 1
        
        html = "<h3>Geographic distribution of outcomes</h3>\n"
        
        # Calculate total outcomes for percentage calculation
        total_outcomes = len(self.outcomes_df)
        
        # Show regional distribution with Global first (above the map)
        html += "<ul>\n"
        
        # Add Global as first item if it exists
        if global_count > 0:
            percentage = (global_count / total_outcomes * 100)
            html += f"<li><strong>Global:</strong> {global_count} ({percentage:.1f}% of all outcomes)</li>\n"
        
        # Add regional data
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            html += f"<li><strong>{region}:</strong> {count} ({percentage:.1f}% of all outcomes)</li>\n"
        html += "</ul>\n\n"
        
        # Generate data for the world map - create a mapping object
        country_mapping = {
            'Tanzania': 'Tanzania',
            'Kenya': 'Kenya', 
            'Indonesia': 'Indonesia',
            'Ecuador': 'Ecuador',
            'India': 'India',
            'Ghana': 'Ghana',
            'Senegal': 'Senegal',
            'Mexico': 'Mexico',
            'Philippines': 'Philippines',
            'Vietnam': 'Vietnam',
            'South Africa': 'South Africa',
            'Rwanda': 'Rwanda',
            'Argentina': 'Argentina',
            'Nigeria': 'Nigeria',
            'Cameroon': 'Cameroon',
            'Sri Lanka': 'Sri Lanka',
            'Moldova': 'Moldova',
            'Maldives': 'Maldives',
            'Kyrgyzstan': 'Kyrgyzstan',
            'Benin': 'Benin',
            'Central African Republic': 'Central African Republic',
            'Comoros': 'Comoros',
            'Congo [Republic]': 'Republic of the Congo',
            'Gabon': 'Gabon',
            'Guinea': 'Guinea',
            'Madagascar': 'Madagascar',
            'Togo': 'Togo',
            'Tunisia': 'Tunisia',
            'Zambia': 'Zambia',
            'Zimbabwe': 'Zimbabwe',
            'Belgium': 'Belgium',
            'Colombia': 'Colombia',
            'Lebanon': 'Lebanon',
            'Pakistan': 'Pakistan'
        }
        
        # Create country data as JavaScript map entries
        map_data_entries = []
        for country, count in geo_counts.items():
            if country not in ['Not relevant / Not available', 'Global']:
                mapped_country = country_mapping.get(country, country)
                map_data_entries.append(f'["{mapped_country}", {count}]')
        
        map_data_js = ',\n        '.join(map_data_entries)
        
        # Generate the world map visualization  
        global_indicator_html = ""
        if global_count > 0:
            global_indicator_html = f"""
        <div id="globalIndicator" style="
            position: absolute;
            left: 380px;
            top: 180px;
            width: 30px;
            height: 30px;
            background-color: rgba(0, 123, 255, 0.8);
            border: 2px solid #007bff;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10;
        " title="Global: {global_count} outcomes ({(global_count / total_outcomes * 100):.1f}% of all outcomes)">
            G
        </div>"""
        
        html += f"""
<div style="margin: 30px 0;">
    <div style="max-width: 1000px; margin: 0 auto; position: relative;">
        <canvas id="worldMap" width="1000" height="500"></canvas>{global_indicator_html}
    </div>
</div>

<script>
(function() {{
    // Create a completely isolated scope for the map
    let chartJsLoaded = false;
    let chartGeoLoaded = false;
    
    function loadScript(src, callback) {{
        const script = document.createElement('script');
        script.src = src;
        script.onload = callback;
        script.onerror = () => console.error('Failed to load script:', src);
        document.head.appendChild(script);
    }}
    
    function createIsolatedMap() {{
        if (!chartJsLoaded || !chartGeoLoaded) {{
            console.log('Waiting for libraries to load...');
            return;
        }}
        
        console.log('Creating isolated map with loaded libraries');
        
        // Force re-registration of ChartGeo components
        if (typeof ChartGeo !== 'undefined' && ChartGeo.ChoroplethController) {{
            Chart.register(ChartGeo.ChoroplethController, ChartGeo.GeoFeature, ChartGeo.ColorScale);
            console.log('ChartGeo components force-registered');
        }}
        
        fetch('https://unpkg.com/world-atlas/countries-50m.json')
            .then(response => {{
                if (!response.ok) throw new Error('Failed to load topology data');
                return response.json();
            }})
            .then(topology => {{
                const canvas = document.getElementById('worldMap');
                if (!canvas) throw new Error('Canvas element not found');
                
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('Could not get canvas context');
                
                const countries = ChartGeo.topojson.feature(topology, topology.objects.countries).features;
                
                // Create a map of country data for quick lookup
                const countryData = new Map([
                    {map_data_js}
                ]);

                const chart = new Chart(ctx, {{
                    type: 'choropleth',
                    data: {{
                        labels: countries.map(c => c.properties.name),
                        datasets: [{{
                            label: 'Grant Outcomes',
                            outline: countries,
                            showOutline: true,
                            backgroundColor: (context) => {{
                                if (!context.dataIndex || context.dataIndex < 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                const feature = context.chart.data.labels[context.dataIndex];
                                const value = countryData.get(feature) || 0;
                                
                                if (value === 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                // Create intensity based on value (logarithmic scale for better visualization)
                                const maxValue = Math.max(...Array.from(countryData.values()));
                                const intensity = Math.min(value / maxValue, 1);
                                const opacity = 0.2 + intensity * 0.8;
                                return `rgba(44, 90, 160, ${{opacity}})`;
                            }},
                            borderColor: 'rgba(0, 0, 0, 0.3)',
                            borderWidth: 1,
                            data: countries.map(country => {{
                                const countryName = country.properties.name;
                                const value = countryData.get(countryName) || 0;
                                return {{
                                    feature: country,
                                    value: value
                                }};
                            }})
                        }}]
                    }},
                    options: {{
                        showOutline: true,
                        showGraticule: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: (context) => {{
                                        // Get country name from the feature properties
                                        const feature = context[0].raw?.feature;
                                        if (feature && feature.properties && feature.properties.name) {{
                                            return feature.properties.name;
                                        }}
                                        // Fallback to label if feature not available
                                        return context[0].label || 'Unknown Country';
                                    }},
                                    label: (context) => {{
                                        const value = context.raw?.value || 0;
                                        const percentage = ((value / {total_outcomes}) * 100).toFixed(1);
                                        if (value === 0) {{
                                            return 'No outcomes recorded';
                                        }}
                                        return `${{value}} outcomes (${{percentage}}% of all outcomes)`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            projection: {{
                                axis: 'x',
                                projection: 'equalEarth'
                            }}
                        }}
                    }}
                }});
                
                console.log('Map created successfully');
                
                // Add interactive behavior to Global indicator after map is created
                const globalIndicator = document.getElementById('globalIndicator');
                if (globalIndicator) {{
                    // Create a custom tooltip element
                    const tooltip = document.createElement('div');
                    tooltip.style.cssText = `
                        position: absolute;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 8px 12px;
                        border-radius: 4px;
                        font-size: 12px;
                        pointer-events: none;
                        z-index: 1000;
                        display: none;
                        white-space: nowrap;
                    `;
                    document.body.appendChild(tooltip);
                    
                    globalIndicator.addEventListener('mouseenter', (e) => {{
                        globalIndicator.style.transform = 'scale(1.2)';
                        globalIndicator.style.backgroundColor = 'rgba(220, 53, 69, 1)';
                        tooltip.textContent = 'Global: {global_count} outcomes ({(global_count / total_outcomes * 100):.1f}% of all outcomes)';
                        tooltip.style.display = 'block';
                    }});
                    
                    globalIndicator.addEventListener('mousemove', (e) => {{
                        tooltip.style.left = e.pageX + 10 + 'px';
                        tooltip.style.top = e.pageY - 30 + 'px';
                    }});
                    
                    globalIndicator.addEventListener('mouseleave', () => {{
                        globalIndicator.style.transform = 'scale(1)';
                        globalIndicator.style.backgroundColor = 'rgba(220, 53, 69, 0.8)';
                        tooltip.style.display = 'none';
                    }});
                }}
            }})
            .catch(error => {{
                console.error('Error creating world map:', error);
                const canvas = document.getElementById('worldMap');
                if (canvas) {{
                    const ctx = canvas.getContext('2d');
                    if (ctx) {{
                        ctx.font = '14px Arial';
                        ctx.fillStyle = '#dc3545';
                        ctx.textAlign = 'center';
                        ctx.fillText('Error loading map visualization: ' + error.message, canvas.width / 2, canvas.height / 2);
                    }}
                }}
            }});
    }}
    
    // Load Chart.js first, then ChartGeo
    if (typeof Chart === 'undefined') {{
        loadScript('https://cdn.jsdelivr.net/npm/chart.js', () => {{
            console.log('Chart.js loaded');
            chartJsLoaded = true;
            createIsolatedMap();
        }});
    }} else {{
        chartJsLoaded = true;
    }}
    
    if (typeof ChartGeo === 'undefined') {{
        loadScript('https://cdn.jsdelivr.net/npm/chartjs-chart-geo@4.2.4/build/index.umd.min.js', () => {{
            console.log('ChartGeo loaded');
            chartGeoLoaded = true;
            createIsolatedMap();
        }});
    }} else {{
        chartGeoLoaded = true;
    }}
    
    // If both libraries are already loaded, create the map immediately
    if (chartJsLoaded && chartGeoLoaded) {{
        createIsolatedMap();
    }}
}})()
</script>
"""
        
        # Add clickable link to show country distribution details (centered)
        html += "<p style='text-align: center; margin: 15px 0;'><a href='javascript:void(0)' onclick='showGeographicDetails()' style='color: #2c5aa0; text-decoration: underline; cursor: pointer;'>[see information as a list]</a></p>\n"
        
        # Prepare data for modal
        country_data = []
        for country, count in sorted(geo_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            country_region = country_to_region.get(country.lower(), "Other")
            country_data.append(f"<li><strong>{country}</strong> ({country_region}): {count} ({percentage:.1f}% of all outcomes)</li>")
        
        # Add JavaScript to set the modal data
        html += f"""
<script>
window.geographicData = `<ul>{''.join(country_data)}</ul>`;
</script>
"""
        return html

    def analyze_domain_outcomes(self):
        """Analyze outcomes by domain directly from CSV data"""
        domain_stats = {}
        total_outcomes = len(self.outcomes_df)
        
        for domain_code, domain_name in self.domain_mapping.items():
            # Get outcomes for this domain
            domain_col = f'Domain of Change {domain_code}'
            if domain_col in self.outcomes_df.columns:
                domain_outcomes = self.outcomes_df[
                    self.outcomes_df[domain_col].notna() & 
                    (self.outcomes_df[domain_col] != '') &
                    (self.outcomes_df[domain_col].astype(str).str.strip() != '') &
                    (self.outcomes_df[domain_col].astype(str).str.lower() != 'nan')
                ]
                count = len(domain_outcomes)
                percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
                
                domain_stats[domain_code] = {
                    'name': domain_name,
                    'count': count,
                    'percentage': percentage,
                    'outcomes': domain_outcomes
                }
        
        return domain_stats, total_outcomes

    def analyze_outcome_baskets_by_domain(self, domain_stats):
        """Analyze outcome basket usage by domain"""
        domain_ob_analysis = {}
        
        for domain_code, stats in domain_stats.items():
            outcomes = stats['outcomes']
            total_domain_outcomes = len(outcomes)
            
            # Analyze default OBs usage
            default_obs = self.default_obs.get(domain_code, [])
            used_obs = []
            unused_obs = []
            
            for ob in default_obs:
                # Count occurrences of this OB in the domain outcomes
                ob_count = 0
                for _, outcome in outcomes.iterrows():
                    outcome_basket = str(outcome.get('Outcome Basket', '')).strip()
                    if outcome_basket.lower() == ob.lower():
                        ob_count += 1
                
                if ob_count > 0:
                    percentage = (ob_count / total_domain_outcomes * 100) if total_domain_outcomes > 0 else 0
                    used_obs.append({
                        'name': ob,
                        'count': ob_count,
                        'percentage': percentage
                    })
                else:
                    unused_obs.append({'name': ob, 'count': 0})
            
            # Sort used OBs by count (descending)
            used_obs.sort(key=lambda x: x['count'], reverse=True)
            
            domain_ob_analysis[domain_code] = {
                'used_obs': used_obs,
                'unused_obs': unused_obs,
                'total_outcomes': total_domain_outcomes
            }
        
        return domain_ob_analysis

    def generate_theme_index(self):
        """Generate the main theme index page directly from CSV data"""
        consolidated_grants, subportfolios = self.get_consolidated_grants_data()
        total_grants = len(consolidated_grants)
        
        # Calculate total budget
        total_budget = 0
        for grant_info in consolidated_grants.values():
            try:
                budget_str = self.clean_budget_string(grant_info['budget'])
                if budget_str != 'N/A':
                    total_budget += float(budget_str)
            except:
                pass
        
        budget_display = f"{total_budget/1000000:.1f}M" if total_budget > 0 else "N/A"
        
        # Analyze domain outcomes
        domain_stats, total_outcomes = self.analyze_domain_outcomes()
        domain_ob_analysis = self.analyze_outcome_baskets_by_domain(domain_stats)
        
        # Build HTML content
        content_html = f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <span class="metric-value">{total_grants}</span>
                <div class="metric-label">Consolidated Grants</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{budget_display}</span>
                <div class="metric-label">Total Budget (CHF)</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{len(self.subportfolios)}</span>
                <div class="metric-label">Sub-Portfolios</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{total_outcomes}</span>
                <div class="metric-label">Outcomes in DB</div>
            </div>
        </div>
        
        <div class="nav-menu">
            <ul>
                <li><strong>Sub-Portfolios:</strong></li>"""
        
        # Add dynamic sub-portfolio navigation links
        for sp in self.subportfolios:
            filename = self.sanitize_filename(sp)
            content_html += f"""
                <li><a href="subportfolios/{filename}.html">{sp}</a></li>"""
        
        content_html += """
            </ul>
        </div>
        
        <h2>Overview of Grants and Sub-Portfolios</h2>
        """
        
        content_html += "<h3>Grant and Sub-Portfolios Overview</h3><p>Click on grant or sub-portfolio names in the table below to access more granular analysis pages.</p>\n"
        
        # Add matrix table
        content_html += self.create_grants_matrix_table()
        
        # Add sub-portfolio distribution
        content_html += "<h3>Sub-Portfolio Distribution</h3>\n"
        content_html += self.generate_subportfolio_distribution(consolidated_grants)
        
        # Add Venn diagram if it exists
        content_html += self.add_venn_diagram_if_exists()
        
        # Add domain distribution and analysis
        content_html += "<h2>Outcomes analysis</h2>\n"
        content_html += "<h3>Distribution of outcomes across domains of change</h3>\n"
        content_html += self.generate_domain_distribution_from_csv()
        
        # Add significance distribution
        content_html += self.generate_significance_distribution()
        
        # Add special cases analysis
        content_html += self.generate_special_cases_analysis()
        
        # Add geographic distribution as h3 subsection within outcomes
        content_html += self.generate_geographic_distribution()
        
        # Add Outcome Basket analysis main section  
        content_html += "<h2>Outcome Basket analysis</h2>\n"
        
        # Add cross-domain summary
        content_html += self.generate_cross_domain_summary()
        
        # Add custom OB analysis for standardization candidates
        content_html += self.generate_custom_obs_analysis()
        
        # Add detailed domain analysis
        content_html += "<h3>Analysis of OB use across Domains of change</h3>\n"
        content_html += self.generate_domain_analysis_from_csv()
        
        # Create HTML page without nav_menu in template (we added it manually)
        html_content = self.create_html_template(
            title="Digital Theme Grants Analysis - Interactive Report",
            content=content_html,
            nav_menu=None,
            is_index_page=True
        )
        
        # Write file
        with open(self.html_output_path / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print("✅ Enhanced theme index page generated")

    def generate_subportfolio_pages(self):
        """Generate comprehensive HTML pages for each sub-portfolio directly from CSV data"""
        consolidated_grants, subportfolios = self.get_consolidated_grants_data()
        
        for sp in subportfolios:
            # Get grants for this sub-portfolio
            sp_grants = {k: v for k, v in consolidated_grants.items() if v['subportfolios'].get(sp, False)}
            
            if not sp_grants:
                continue
                
            # Generate comprehensive subportfolio analysis
            analysis = self.create_comprehensive_subportfolio_analysis(sp, sp_grants)
            
            # Generate content HTML using the analysis
            content_html = self.generate_subportfolio_content_html(sp, analysis)
            
            # Create breadcrumb
            breadcrumb = [
                {"text": "Theme Overview", "url": "../index.html"},
                {"text": sp}
            ]
            
            # Navigation menu - create links to other sub-portfolios
            nav_menu = [{"text": "Other Sub-Portfolios:", "url": None}]  # Header with no link
            
            # Add links to all other sub-portfolios
            for other_sp in subportfolios:
                if other_sp != sp:  # Exclude current sub-portfolio
                    other_sp_filename = self.sanitize_filename(other_sp)
                    nav_menu.append({"text": other_sp, "url": f"{other_sp_filename}.html"})
            
            # Add link back to theme if no other sub-portfolios exist
            if len(nav_menu) == 1:  # Only the header
                nav_menu.append({"text": "Back to Theme", "url": "../index.html"})
            
            # Create HTML page with proper title format
            html_page = self.create_html_template(
                title=f"Sub-Portfolio Analysis: {sp}",
                content=content_html,
                breadcrumb=breadcrumb,
                nav_menu=nav_menu
            )
            
            # Generate filename
            sp_filename = self.sanitize_filename(sp)
            
            # Write file
            output_file = self.html_output_path / "subportfolios" / f"{sp_filename}.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_page)
        
        print(f"✅ Generated {len(subportfolios)} comprehensive enhanced sub-portfolio pages")

    def create_comprehensive_subportfolio_analysis(self, subportfolio_name, sp_grants):
        """Create comprehensive analysis for a sub-portfolio matching the original script"""
        # Get all outcomes for grants in this sub-portfolio
        all_gams_codes = []
        for gams_code, grant_info in sp_grants.items():
            all_gams_codes.append(gams_code)
            if grant_info['is_consolidated'] and grant_info['phase1_code']:
                all_gams_codes.append(grant_info['phase1_code'])
        
        subportfolio_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].astype(str).isin(all_gams_codes)]
        
        # Calculate statistics
        total_budget = sum(float(g['budget'].replace(',', '')) for g in sp_grants.values() if g['budget'] != 'N/A')
        total_outcomes = sum(g['outcomes_count'] for g in sp_grants.values())
        
        # Average calculations
        avg_budget = total_budget / len(sp_grants) if len(sp_grants) > 0 else 0
        
        # Calculate average duration
        total_duration = 0
        valid_durations = 0
        for grant_info in sp_grants.values():
            try:
                duration = float(grant_info['duration']) if grant_info['duration'] != 'N/A' else 0
                if duration > 0:
                    total_duration += duration
                    valid_durations += 1
            except:
                pass
        avg_duration = total_duration / valid_durations if valid_durations > 0 else 0
        
        # Domain distribution
        domain_counts = {}
        for _, outcome in subportfolio_outcomes.iterrows():
            domain = outcome.get('Domain of Change')
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key not in domain_counts:
                    domain_counts[domain_key] = 0
                domain_counts[domain_key] += 1
        
        # Geographic distribution
        geo_counts = {}
        total_geo_instances = 0
        for _, outcome in subportfolio_outcomes.iterrows():
            countries = outcome.get('Country(ies) / Global')
            if pd.notna(countries):
                country_list = [country.strip() for country in str(countries).split(',') if country.strip()]
                for country in country_list:
                    if country not in geo_counts:
                        geo_counts[country] = 0
                    geo_counts[country] += 1
                    total_geo_instances += 1
        
        # Significance distribution
        significance_counts = subportfolio_outcomes['Significance Level'].value_counts(dropna=False)
        significance_dict = significance_counts.to_dict()
        if pd.isna(list(significance_dict.keys())).any():
            nan_count = 0
            for key in list(significance_dict.keys()):
                if pd.isna(key):
                    nan_count = significance_dict.pop(key)
                    break
            if nan_count > 0:
                significance_dict['Not specified'] = nan_count
        
        # Special cases
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        special_cases_data = {}
        for col in special_cases_cols:
            clean_name = col.replace('Special case - ', '')
            if col in subportfolio_outcomes.columns:
                count = (subportfolio_outcomes[col] == 1).sum()
            else:
                count = 0
            special_cases_data[clean_name] = count
        
        return {
            'name': subportfolio_name,
            'grants_count': len(sp_grants),
            'total_budget': total_budget,
            'avg_budget': avg_budget,
            'avg_duration': avg_duration,
            'total_outcomes': total_outcomes,
            'grants': sp_grants,
            'outcomes': subportfolio_outcomes,
            'domain_distribution': domain_counts,
            'geographic_distribution': geo_counts,
            'total_geo_instances': total_geo_instances,
            'significance_distribution': significance_dict,
            'special_cases': special_cases_data
        }

    def generate_subportfolio_content_html(self, subportfolio_name, analysis):
        """Generate comprehensive HTML content for subportfolio page"""
        
        # Get portfolio description from loaded data
        portfolio_desc = self.portfolio_descriptions.get(subportfolio_name, {})
        core_focus = portfolio_desc.get('core_focus', 'Description not available')
        focus_details = portfolio_desc.get('focus_details', 'Details not available')
        
        content_html = f"""
        <h2>Sub-Portfolio Description</h2>
        <p><strong>Core Focus:</strong> {core_focus}</p>
        <p><strong>Focus Details:</strong> {focus_details}</p>
        
        <h2>Overview Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <span class="metric-value">{analysis['grants_count']}</span>
                <div class="metric-label">Total Consolidated Grants</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['total_budget']:,.0f}</span>
                <div class="metric-label">Total Budget (CHF)</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['avg_duration']:.1f}</span>
                <div class="metric-label">Average Grant Duration (months)</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['total_outcomes']}</span>
                <div class="metric-label">Total Outcomes in DB</div>
            </div>
        </div>
        
        <h2>Consolidated Grants in the {subportfolio_name} Sub-Portfolio</h2>
        {self.generate_subportfolio_grants_table(subportfolio_name, analysis['grants'])}
        """
        
        # Add Outcomes analysis section (enhanced structure matching theme page)
        content_html += "<h2>Outcomes analysis</h2>\n"
        content_html += "<h3>Distribution of outcomes across domains of change</h3>\n"
        content_html += self.generate_subportfolio_domain_distribution(subportfolio_name, analysis['outcomes'])
        
        # Add significance distribution with stacked chart
        content_html += self.generate_subportfolio_significance_distribution(subportfolio_name)
        
        # Add special cases analysis with stacked chart  
        content_html += self.generate_subportfolio_special_cases_analysis(subportfolio_name)
        
        # Add geographic distribution with map
        content_html += self.generate_subportfolio_geographic_distribution(subportfolio_name)
        
        # Add Outcome Basket analysis section (enhanced structure matching theme page)
        content_html += "<h2>Outcome Basket analysis</h2>\n"
        
        # Add cross-domain summary with chart
        content_html += self.generate_subportfolio_cross_domain_summary(subportfolio_name)
        
        # Add custom OB analysis
        content_html += self.generate_subportfolio_custom_obs_analysis(subportfolio_name)
        
        # Add detailed domain analysis (existing method, now positioned correctly)
        content_html += "<h3>Analysis of OB use across Domains of change</h3>\n"
        content_html += self.generate_subportfolio_domain_analysis(analysis['outcomes'])
        
        # All outcome records
        content_html += self.generate_subportfolio_outcome_records(subportfolio_name, analysis['grants'], analysis['outcomes'])
        
        return content_html

    def generate_subportfolio_grants_table(self, current_subportfolio, grants):
        """Generate comprehensive grants table for sub-portfolio page matching theme-level format"""
        
        # Get consolidated grants data using the same method as theme-level
        consolidated_grants, all_subportfolios = self.get_consolidated_grants_data()
        
        # Filter to only grants that belong to this sub-portfolio
        subportfolio_grants = {}
        for gams_code, grant_info in consolidated_grants.items():
            if grant_info['subportfolios'].get(current_subportfolio, False):
                subportfolio_grants[gams_code] = grant_info
        
        if not subportfolio_grants:
            return "<p>No grants found for this sub-portfolio.</p>"
        
        # Sort by end date (future dates first, descending order) - same as theme-level
        sorted_grants = sorted(subportfolio_grants.items(), key=lambda x: x[1]['end_date_sort'], reverse=True)
        
        # Generate table HTML using grants-matrix class for consistent styling
        table_html = """
        <div class="grants-matrix">
            <table>
                <thead>
                    <tr>
                        <th class="grant-name">Grant</th>
                        <th>Budget (CHF)</th>
                        <th>Duration (months)</th>
                        <th>End Date</th>
                        <th>Countries</th>
                        <th>Outcomes in DB</th>
                        <th>Additional sub-portfolios</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Generate rows for each grant
        for gams_code, grant_info in sorted_grants:
            # Grant name with GAMS code (using consistent filename format)
            grant_filename = self.generate_grant_filename(gams_code, grant_info["short_name"])
            display_name = self.format_display_name(grant_info["short_name"])
            
            # Find other sub-portfolios this grant belongs to
            other_subportfolios = []
            for sp in all_subportfolios:
                if sp != current_subportfolio and grant_info['subportfolios'].get(sp, False):
                    other_subportfolios.append(sp)
            other_subportfolios_str = ", ".join(other_subportfolios) if other_subportfolios else "None"
            
            table_html += f"""
                    <tr>
                        <td class="grant-info">
                            <a href="../grants/{grant_filename}">{display_name}</a>
                            <div class="grant-details">GAMS: {grant_info.get("consolidated_gams", gams_code)}</div>
                        </td>
                        <td>{grant_info["budget"]}</td>
                        <td>{grant_info["duration"]}</td>
                        <td>{grant_info["end_date_display"]}</td>
                        <td>{grant_info["countries"]}</td>
                        <td>{grant_info["outcomes_count"]}</td>
                        <td>{other_subportfolios_str}</td>
                    </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        return table_html

    def generate_subportfolio_domain_distribution(self, subportfolio_name, subportfolio_outcomes):
        """Generate domain distribution analysis for sub-portfolio using same format as theme page"""
        if subportfolio_outcomes is None or (hasattr(subportfolio_outcomes, 'empty') and subportfolio_outcomes.empty) or len(subportfolio_outcomes) == 0:
            return "<p>No outcomes data available.</p>"
        
        # Count outcomes by domain for this sub-portfolio
        # Initialize with all possible domains to ensure consistency
        domain_counts = {}
        total_outcomes = len(subportfolio_outcomes)
        
        # Initialize all domains from domain mapping with 0 counts
        for domain_code, domain_description in self.domain_mapping.items():
            domain_full_text = f"{domain_code} - {domain_description}"
            domain_counts[domain_full_text] = 0
        
        # Count actual outcomes by domain
        for _, outcome in subportfolio_outcomes.iterrows():
            domain = outcome.get('Domain of Change')
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key in domain_counts:
                    domain_counts[domain_key] += 1
        
        # Generate only the pie chart visualization (no bullet point list for consistency)
        html = self.generate_subportfolio_domain_pie_chart(domain_counts, total_outcomes, subportfolio_name)
        
        return html
    
    def generate_subportfolio_domain_pie_chart(self, domain_counts, total_outcomes, subportfolio_name):
        """Generate a pie chart visualization for sub-portfolio domain distribution using Chart.js"""
        if not domain_counts or total_outcomes == 0:
            return ""
        
        # Prepare data for the chart
        chart_data = []
        chart_labels = []
        chart_colors = ['#2c5aa0', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
        
        for i, (domain_full_text, count) in enumerate(sorted(domain_counts.items())):
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            chart_labels.append(f"{domain_full_text}")
            chart_data.append(count)
        
        # Generate unique chart ID for this sub-portfolio
        chart_id = f"domainDistributionChart_{subportfolio_name.replace(' ', '_').replace('&', 'and')}"
        
        # Generate Chart.js HTML
        chart_html = f"""
<div style="margin: 30px 0; text-align: center;">
    <div style="max-width: 800px; margin: 0 auto;">
        <canvas id="{chart_id}" width="400" height="400"></canvas>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
    const domainChart_{chart_id} = new Chart(ctx_{chart_id}, {{
        type: 'pie',
        data: {{
            labels: {chart_labels},
            datasets: [{{
                data: {chart_data},
                backgroundColor: {chart_colors[:len(chart_data)]},
                borderColor: '#ffffff',
                borderWidth: 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'right',
                    labels: {{
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {{
                            size: 11
                        }},
                        boxWidth: 12,
                        boxHeight: 12,
                        generateLabels: function(chart) {{
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {{
                                return data.labels.map((label, i) => {{
                                    const meta = chart.getDatasetMeta(0);
                                    const style = meta.controller.getStyle(i);
                                    
                                    // Function to wrap text for long labels (copied exactly from theme page)
                                    function wrapText(text, maxLength = 40) {{
                                        if (text.length <= maxLength) return text;
                                        
                                        const words = text.split(' ');
                                        const lines = [];
                                        let currentLine = '';
                                        
                                        words.forEach(word => {{
                                            if ((currentLine + word).length <= maxLength) {{
                                                currentLine += (currentLine ? ' ' : '') + word;
                                            }} else {{
                                                if (currentLine) lines.push(currentLine);
                                                currentLine = word;
                                            }}
                                        }});
                                        
                                        if (currentLine) lines.push(currentLine);
                                        return lines;
                                    }}
                                    
                                    const wrappedText = wrapText(label);
                                    
                                    return {{
                                        text: Array.isArray(wrappedText) ? wrappedText : [wrappedText],
                                        fillStyle: style.backgroundColor,
                                        strokeStyle: style.borderColor,
                                        lineWidth: style.borderWidth,
                                        pointStyle: 'circle',
                                        hidden: isNaN(data.datasets[0].data[i]) || meta.data[i].hidden,
                                        index: i
                                    }};
                                }});
                            }}
                            return [];
                        }}
                    }}
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return context.label + ': ' + context.parsed + ' outcomes (' + percentage + '%)';
                        }}
                    }}
                }}
            }},
            layout: {{
                padding: {{
                    bottom: 50
                }}
            }}
        }}
    }});
</script>
"""
        
        return chart_html

    def format_grants_description(self, grants_included_text):
        """Format the grants organization description from the CSV"""
        if not grants_included_text or grants_included_text == 'Grant details not available':
            return "<p>Grant organization details not available.</p>"
        
        # Convert the text to HTML format
        formatted_text = str(grants_included_text).strip()
        
        # Replace line breaks and format sections
        lines = formatted_text.split('\n')
        html_content = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header (typically categories like "Health Systems Transformation")
            if line and not line.startswith('-') and not line.startswith('•') and not any(char in line for char in ['(', ')']):
                # Section headers
                html_content += f"<h4>{line}</h4>\n"
            elif line.startswith('-') or line.startswith('•'):
                # Individual grant items - clean up and format
                clean_line = line.lstrip('- •').strip()
                html_content += f"<p style='margin-left: 20px;'>{clean_line}</p>\n"
            else:
                # Regular paragraph
                html_content += f"<p>{line}</p>\n"
        
        return html_content if html_content else "<p>Grant organization details not available.</p>"

    def get_subportfolio_outcomes(self, subportfolio_name):
        """Get filtered outcomes DataFrame for a specific subportfolio"""
        # Get grants for this subportfolio
        consolidated_grants, _ = self.get_consolidated_grants_data()
        sp_grants = {k: v for k, v in consolidated_grants.items() 
                    if v['subportfolios'].get(subportfolio_name, False)}
        
        # Get all GAMS codes for this subportfolio (including phase 1 codes)
        all_gams_codes = []
        for gams_code, grant_info in sp_grants.items():
            all_gams_codes.append(gams_code)
            if grant_info['is_consolidated'] and grant_info['phase1_code']:
                all_gams_codes.append(grant_info['phase1_code'])
        
        # Return filtered outcomes DataFrame
        return self.outcomes_df[self.outcomes_df['GAMS code'].astype(str).isin(all_gams_codes)]

    def generate_subportfolio_significance_distribution(self, subportfolio_name):
        """Generate subportfolio-wide significance patterns analysis with domain breakdown"""
        # Use helper method to get filtered outcomes for this subportfolio
        filtered_outcomes = self.get_subportfolio_outcomes(subportfolio_name)
        
        if filtered_outcomes.empty:
            return ""
        
        # Count significance levels by domain (including null/empty values)
        total_outcomes = len(filtered_outcomes)
        
        # Initialize data structure for significance by domain
        significance_by_domain = {}
        domain_totals = {}
        
        # Count outcomes by domain and significance level
        for _, outcome in filtered_outcomes.iterrows():
            significance = outcome.get('Significance Level')
            domain_full = outcome.get('Domain of Change')
            
            # Clean up significance level (replace NaN with 'Not specified')
            if pd.isna(significance):
                significance = 'Not specified'
            
            # Extract domain code from full domain string (e.g., "D4 - Description" -> "D4")
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    
                    # Initialize domain if not seen before
                    if domain_code not in significance_by_domain:
                        significance_by_domain[domain_code] = {}
                        domain_totals[domain_code] = 0
                    
                    # Initialize significance level if not seen before
                    if significance not in significance_by_domain[domain_code]:
                        significance_by_domain[domain_code][significance] = 0
                    
                    # Count this outcome
                    significance_by_domain[domain_code][significance] += 1
                    domain_totals[domain_code] += 1
        
        # Calculate overall significance totals
        overall_significance = {}
        for domain_data in significance_by_domain.values():
            for sig_level, count in domain_data.items():
                if sig_level not in overall_significance:
                    overall_significance[sig_level] = 0
                overall_significance[sig_level] += count
        
        html = "<h3>Sub-Portfolio Significance Patterns</h3>\n"
        html += "<p>Distribution of outcomes' significance levels across Domains of change within this sub-portfolio. "
        html += "<span class='info-icon' onclick='showSignificanceInfo()' title='Click for more information about significance levels'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_significance_stacked_chart(significance_by_domain, domain_totals, overall_significance, total_outcomes)
        
        # Add examples of outcomes by significance level (reuse filtered_outcomes data)
        html += self.generate_significance_examples(filtered_outcomes, overall_significance)
        
        return html

    def generate_subportfolio_special_cases_analysis(self, subportfolio_name):
        """Generate subportfolio-wide special cases analysis with domain breakdown"""
        # Use helper method to get filtered outcomes for this subportfolio
        filtered_outcomes = self.get_subportfolio_outcomes(subportfolio_name)
        
        if filtered_outcomes.empty:
            return ""
        
        # Special cases columns
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        
        # Initialize data structures
        special_cases_by_domain = {}
        domain_totals = {}
        total_outcomes = len(filtered_outcomes)
        
        # Count special cases by domain
        for _, outcome in filtered_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            
            # Extract domain code from full domain string (e.g., "D4 - Description" -> "D4")
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    
                    # Initialize domain if not seen before
                    if domain_code not in special_cases_by_domain:
                        special_cases_by_domain[domain_code] = {}
                        domain_totals[domain_code] = 0
                    
                    domain_totals[domain_code] += 1
                    
                    # Check each special case column
                    for case_col in special_cases_cols:
                        case_value = outcome.get(case_col)
                        # Handle both 'X' values and numeric 1 values
                        is_special_case = False
                        if pd.notna(case_value):
                            case_str = str(case_value).strip().lower()
                            is_special_case = (case_str == 'x' or case_str == '1')
                        
                        if is_special_case:
                            # Clean up the case name (remove "Special case - " prefix)
                            case_name = case_col.replace('Special case - ', '')
                            
                            # Initialize case if not seen before
                            if case_name not in special_cases_by_domain[domain_code]:
                                special_cases_by_domain[domain_code][case_name] = 0
                            
                            # Count this special case
                            special_cases_by_domain[domain_code][case_name] += 1
        
        # Calculate overall special cases totals (include all defined special cases)
        overall_special_cases = {}
        
        # Initialize all defined special cases with 0
        all_defined_special_cases = [
            'Unexpected Outcome',
            'Negative Outcome', 
            'Signal towards Outcome',
            'Indicator Based Outcome',
            'Keep on Radar',
            'Programme Outcome'
        ]
        for case_name in all_defined_special_cases:
            overall_special_cases[case_name] = 0
        
        # Count actual occurrences
        for domain_data in special_cases_by_domain.values():
            for case_name, count in domain_data.items():
                if case_name in overall_special_cases:
                    overall_special_cases[case_name] += count
        
        # Always show section (don't hide if all are 0)
        # if not overall_special_cases:
        #     return ""
        
        html = "<h3>Portfolio-wide special case outcomes</h3>\n"
        html += "<p>Distribution of special case outcomes across domains within this sub-portfolio. "
        html += "<span class='info-icon' onclick='showSpecialCasesInfo()' title='Click for more information about special case types'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_special_cases_stacked_chart(special_cases_by_domain, domain_totals, overall_special_cases, total_outcomes)
        
        # Add examples of special case outcomes (reuse filtered_outcomes data)
        html += self.generate_special_cases_examples(filtered_outcomes, overall_special_cases)
        
        return html

    def generate_subportfolio_geographic_distribution(self, subportfolio_name):
        """Generate geographic distribution analysis for subportfolio with regional aggregation and world map"""
        # Use helper method to get filtered outcomes for this subportfolio
        filtered_outcomes = self.get_subportfolio_outcomes(subportfolio_name)
        
        if filtered_outcomes.empty:
            return ""
        
        # Define regions and their countries (same as theme-level)
        regions = {
            'Sub-Saharan Africa': ['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 
                                 'Chad', 'Comoros', 'Congo', 'Democratic Republic of the Congo', 'Côte d\'Ivoire', 'Djibouti', 'Equatorial Guinea', 
                                 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 
                                 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
                                 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 
                                 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'],
            'South Asia': ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'],
            'East Asia & Pacific': ['Cambodia', 'China', 'Fiji', 'Indonesia', 'Kiribati', 'Korea, Democratic People\'s Republic', 'Lao PDR', 
                                  'Malaysia', 'Marshall Islands', 'Micronesia', 'Mongolia', 'Myanmar', 'Palau', 'Papua New Guinea', 'Philippines', 
                                  'Samoa', 'Solomon Islands', 'Thailand', 'Timor-Leste', 'Tonga', 'Tuvalu', 'Vanuatu', 'Vietnam'],
            'Middle East & North Africa': ['Algeria', 'Bahrain', 'Djibouti', 'Egypt', 'Iran', 'Iraq', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 
                                         'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'],
            'Latin America & Caribbean': ['Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bolivia', 'Brazil', 'Chile', 
                                        'Colombia', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada', 
                                        'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 
                                        'Peru', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Suriname', 
                                        'Trinidad and Tobago', 'Uruguay', 'Venezuela'],
            'Europe & Central Asia': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Georgia', 
                                    'Kazakhstan', 'Kosovo', 'Kyrgyz Republic', 'Moldova', 'Montenegro', 'North Macedonia', 'Romania', 'Russia', 
                                    'Serbia', 'Tajikistan', 'Turkey', 'Turkmenistan', 'Ukraine', 'Uzbekistan']
        }
        
        # Create reverse mapping from country to region
        country_to_region = {}
        for region, countries in regions.items():
            for country in countries:
                country_to_region[country.lower()] = region
        
        # Count outcomes by country and region
        geo_counts = {}  # For countries
        region_counts = {}  # For regions
        global_count = 0
        total_geo_instances = 0
        
        for _, outcome in filtered_outcomes.iterrows():
            countries = outcome.get('Country(ies) / Global')
            if pd.notna(countries):
                # Only split by semicolon to preserve country names that contain commas
                country_list = [c.strip() for c in str(countries).split(';') if c.strip()]
                
                # Process each country
                for country in country_list:
                    # Handle global specially
                    if country.lower() == 'global':
                        global_count += 1
                        total_geo_instances += 1
                        continue
                    
                    # Skip only truly invalid entries
                    if country.lower() in ['n/a', 'na', 'none']:
                        continue
                    # Count by country
                    if country not in geo_counts:
                        geo_counts[country] = 0
                    geo_counts[country] += 1
                    total_geo_instances += 1
                    
                    # Count by region
                    country_lower = country.lower()
                    if country_lower in country_to_region:
                        region = country_to_region[country_lower]
                        if region not in region_counts:
                            region_counts[region] = 0
                        region_counts[region] += 1
        
        html = "<h3>Geographic distribution of outcomes</h3>\n"
        
        # Calculate total outcomes for percentage calculation
        total_outcomes = len(filtered_outcomes)
        
        # Show regional distribution with Global first (above the map)
        html += "<ul>\n"
        
        # Add Global as first item if it exists
        if global_count > 0:
            percentage = (global_count / total_outcomes * 100)
            html += f"<li><strong>Global:</strong> {global_count} ({percentage:.1f}% of all outcomes)</li>\n"
        
        # Add regional data
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            html += f"<li><strong>{region}:</strong> {count} ({percentage:.1f}% of all outcomes)</li>\n"
        html += "</ul>\n\n"
        
        # Generate data for the world map - create a mapping object
        country_mapping = {
            'Tanzania': 'Tanzania',
            'Kenya': 'Kenya', 
            'Indonesia': 'Indonesia',
            'Ecuador': 'Ecuador',
            'India': 'India',
            'Ghana': 'Ghana',
            'Senegal': 'Senegal',
            'South Africa': 'South Africa',
            'Uganda': 'Uganda',
            'Bangladesh': 'Bangladesh',
            'Nigeria': 'Nigeria',
            'Rwanda': 'Rwanda',
            'Myanmar': 'Myanmar',
            'Nepal': 'Nepal',
            'Brazil': 'Brazil',
            'Colombia': 'Colombia',
            'Peru': 'Peru',
            'Philippines': 'Philippines',
            'Vietnam': 'Vietnam',
            'Thailand': 'Thailand',
            'Pakistan': 'Pakistan',
            'Ethiopia': 'Ethiopia',
            'Democratic Republic of the Congo': 'Dem. Rep. Congo',
            'United States': 'United States',
            'United Kingdom': 'United Kingdom',
            'Switzerland': 'Switzerland',
            'Germany': 'Germany',
            'France': 'France',
            'Netherlands': 'Netherlands',
            'Canada': 'Canada',
            'Australia': 'Australia',
            'Argentina': 'Argentina',
            'Chile': 'Chile',
            'Mexico': 'Mexico',
            'China': 'China',
            'Japan': 'Japan',
            'South Korea': 'South Korea',
            'Russia': 'Russia',
            'Turkey': 'Turkey',
            'Egypt': 'Egypt',
            'Morocco': 'Morocco',
            'Iran': 'Iran',
            'Iraq': 'Iraq',
            'Afghanistan': 'Afghanistan',
            'Jordan': 'Jordan',
            'Lebanon': 'Lebanon',
            'Syria': 'Syria',
            'Yemen': 'Yemen',
            'Saudi Arabia': 'Saudi Arabia',
            'United Arab Emirates': 'United Arab Emirates',
            'Israel': 'Israel',
            'Palestine': 'Palestine',
            'Libya': 'Libya',
            'Tunisia': 'Tunisia',
            'Algeria': 'Algeria',
            'Sudan': 'Sudan',
            'South Sudan': 'South Sudan',
            'Chad': 'Chad',
            'Niger': 'Niger',
            'Mali': 'Mali',
            'Burkina Faso': 'Burkina Faso',
            'Mauritania': 'Mauritania',
            'Guinea': 'Guinea',
            'Sierra Leone': 'Sierra Leone',
            'Liberia': 'Liberia',
            'Ivory Coast': 'Côte d\'Ivoire',
            'Côte d\'Ivoire': 'Côte d\'Ivoire',
            'Togo': 'Togo',
            'Benin': 'Benin',
            'Cameroon': 'Cameroon',
            'Central African Republic': 'Central African Rep.',
            'Gabon': 'Gabon',
            'Republic of the Congo': 'Congo',
            'Congo': 'Congo',
            'Angola': 'Angola',
            'Zambia': 'Zambia',
            'Zimbabwe': 'Zimbabwe',
            'Botswana': 'Botswana',
            'Namibia': 'Namibia',
            'Lesotho': 'Lesotho',
            'Swaziland': 'Swaziland',
            'Mozambique': 'Mozambique',
            'Madagascar': 'Madagascar',
            'Mauritius': 'Mauritius',
            'Malawi': 'Malawi',
            'Burundi': 'Burundi',
            'Somalia': 'Somalia',
            'Djibouti': 'Djibouti',
            'Eritrea': 'Eritrea',
            'Gambia': 'Gambia',
            'Guinea-Bissau': 'Guinea-Bissau',
            'Cape Verde': 'Cape Verde',
            'São Tomé and Príncipe': 'São Tomé and Príncipe',
            'Comoros': 'Comoros',
            'Seychelles': 'Seychelles',
            'Sri Lanka': 'Sri Lanka',
            'Bhutan': 'Bhutan',
            'Maldives': 'Maldives',
            'Cambodia': 'Cambodia',
            'Laos': 'Laos',
            'Lao PDR': 'Laos',
            'Malaysia': 'Malaysia',
            'Singapore': 'Singapore',
            'Brunei': 'Brunei',
            'East Timor': 'Timor-Leste',
            'Timor-Leste': 'Timor-Leste',
            'Papua New Guinea': 'Papua New Guinea',
            'Fiji': 'Fiji',
            'Solomon Islands': 'Solomon Is.',
            'Vanuatu': 'Vanuatu',
            'New Caledonia': 'New Caledonia',
            'Samoa': 'Samoa',
            'Tonga': 'Tonga',
            'Kiribati': 'Kiribati',
            'Tuvalu': 'Tuvalu',
            'Palau': 'Palau',
            'Marshall Islands': 'Marshall Is.',
            'Micronesia': 'Micronesia',
            'Mongolia': 'Mongolia',
            'North Korea': 'North Korea',
            'South Korea': 'South Korea',
            'Taiwan': 'Taiwan',
            'Hong Kong': 'Hong Kong',
            'Macau': 'Macau'
        }
        
        # Create country data as JavaScript map entries (matching theme page format exactly)
        map_data_entries = []
        for country, count in geo_counts.items():
            if country not in ['Not relevant / Not available', 'Global']:
                mapped_country = country_mapping.get(country, country)
                map_data_entries.append(f'["{mapped_country}", {count}]')
        
        map_data_js = ',\n        '.join(map_data_entries)
        
        # Generate global indicator overlay HTML with proper positioning and tooltip
        global_indicator_html = ""
        if global_count > 0:
            global_indicator_html = f"""
        <div id="globalIndicator" style="
            position: absolute;
            left: 380px;
            top: 180px;
            width: 30px;
            height: 30px;
            background-color: rgba(0, 123, 255, 0.8);
            border: 2px solid #007bff;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10;
        " title="Global: {global_count} outcomes ({(global_count / total_outcomes * 100):.1f}% of all outcomes)">
            G
        </div>"""
        
        # Generate unique canvas ID for this subportfolio
        import hashlib
        canvas_id = f"worldMapSubportfolio_{hashlib.md5(str(hash(tuple(sorted(geo_counts.items())))).encode()).hexdigest()[:8]}"
        
        html += f"""
<div style="margin: 30px 0;">
    <div style="width: 100%; margin: 0 auto; position: relative;">
        <canvas id="{canvas_id}" width="1000" height="500" style="width: 100%; height: auto;"></canvas>{global_indicator_html}
    </div>
</div>

<script>
(function() {{
    // Create a completely isolated scope for the subportfolio map
    let chartJsLoaded = false;
    let chartGeoLoaded = false;
    
    function loadScript(src, callback) {{
        const script = document.createElement('script');
        script.src = src;
        script.onload = callback;
        script.onerror = () => console.error('Failed to load script:', src);
        document.head.appendChild(script);
    }}
    
    function createIsolatedSubportfolioMap() {{
        if (!chartJsLoaded || !chartGeoLoaded) {{
            console.log('Waiting for libraries to load...');
            return;
        }}
        
        console.log('Creating isolated subportfolio map with loaded libraries');
        
        // Force re-registration of ChartGeo components
        if (typeof ChartGeo !== 'undefined' && ChartGeo.ChoroplethController) {{
            Chart.register(ChartGeo.ChoroplethController, ChartGeo.GeoFeature, ChartGeo.ColorScale);
            console.log('ChartGeo components force-registered for subportfolio');
        }}
        
        fetch('https://unpkg.com/world-atlas/countries-50m.json')
            .then(response => {{
                if (!response.ok) throw new Error('Failed to load topology data');
                return response.json();
            }})
            .then(topology => {{
                const canvas = document.getElementById('{canvas_id}');
                if (!canvas) throw new Error('Canvas element not found');
                
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('Could not get canvas context');
                
                const countries = ChartGeo.topojson.feature(topology, topology.objects.countries).features;
                
                // Create a map of country data for quick lookup
                const countryData = new Map([
                    {map_data_js}
                ]);

                const chart = new Chart(ctx, {{
                    type: 'choropleth',
                    data: {{
                        labels: countries.map(c => c.properties.name),
                        datasets: [{{
                            label: 'Grant Outcomes',
                            outline: countries,
                            showOutline: true,
                            backgroundColor: (context) => {{
                                if (!context.dataIndex || context.dataIndex < 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                const feature = context.chart.data.datasets[0].outline[context.dataIndex];
                                if (!feature || !feature.properties) {{
                                    return 'rgba(211, 211, 211, 0.2)';
                                }}
                                
                                const countryName = feature.properties.name;
                                const count = countryData.get(countryName) || 0;
                                
                                if (count === 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                // Create intensity based on value (matching theme page)
                                const maxCount = Math.max(...Array.from(countryData.values()));
                                const intensity = Math.min(count / maxCount, 1);
                                const opacity = 0.2 + intensity * 0.8;
                                return `rgba(44, 90, 160, ${{opacity}})`;
                            }},
                            borderColor: 'rgba(0, 0, 0, 0.3)',
                            borderWidth: 1,
                            data: countries.map(c => {{
                                const count = countryData.get(c.properties.name) || 0;
                                return {{
                                    feature: c,
                                    value: count
                                }};
                            }})
                        }}]
                    }},
                    options: {{
                        responsive: false,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(context) {{
                                        if (context && context.length > 0 && context[0].raw && context[0].raw.feature) {{
                                            return context[0].raw.feature.properties.name || 'Unknown';
                                        }}
                                        return 'Unknown';
                                    }},
                                    label: function(context) {{
                                        if (context && context.raw && context.raw.feature) {{
                                            const countryName = context.raw.feature.properties.name;
                                            const count = countryData.get(countryName) || 0;
                                            return `Outcomes: ${{count}}`;
                                        }}
                                        return 'Outcomes: 0';
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            projection: {{
                                axis: 'x',
                                projection: 'naturalEarth1'
                            }}
                        }}
                    }}
                }});
                
                console.log('Subportfolio map created successfully');
            }})
            .catch(error => {{
                console.error('Error creating subportfolio map:', error);
                const canvas = document.getElementById('{canvas_id}');
                if (canvas) {{
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#f8f9fa';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = '#6c757d';
                    ctx.font = '16px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('Map could not be loaded', canvas.width / 2, canvas.height / 2);
                }}
            }});
    }}
    
    // Load Chart.js first, then ChartGeo (matching theme page exactly)
    if (typeof Chart === 'undefined') {{
        loadScript('https://cdn.jsdelivr.net/npm/chart.js', () => {{
            console.log('Chart.js loaded');
            chartJsLoaded = true;
            createIsolatedSubportfolioMap();
        }});
    }} else {{
        chartJsLoaded = true;
    }}
    
    if (typeof ChartGeo === 'undefined') {{
        loadScript('https://cdn.jsdelivr.net/npm/chartjs-chart-geo@4.2.4/build/index.umd.min.js', () => {{
            console.log('ChartGeo loaded');
            chartGeoLoaded = true;
            createIsolatedSubportfolioMap();
        }});
    }} else {{
        chartGeoLoaded = true;
    }}
    
    // If both are already loaded, create the map immediately
    if (chartJsLoaded && chartGeoLoaded) {{
        createIsolatedSubportfolioMap();
    }}
}})();
</script>
"""
        
        # Add clickable link to show country distribution details (centered)
        html += "<p style='text-align: center; margin: 15px 0;'><a href='javascript:void(0)' onclick='showGeographicDetails()' style='color: #2c5aa0; text-decoration: underline; cursor: pointer;'>[see information as a list]</a></p>\n"
        
        # Prepare data for modal
        country_data = []
        for country, count in sorted(geo_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            country_region = country_to_region.get(country.lower(), "Other")
            country_data.append(f"<li><strong>{country}</strong> ({country_region}): {count} ({percentage:.1f}% of all outcomes)</li>")
        
        # Add JavaScript to set the modal data
        html += f"""
<script>
window.geographicData = `<ul>{''.join(country_data)}</ul>`;
</script>
"""
        
        return html

    def generate_subportfolio_cross_domain_summary(self, subportfolio_name):
        """Generate cross-domain OB usage summary for subportfolio"""
        # Use helper method to get filtered outcomes for this subportfolio
        filtered_outcomes = self.get_subportfolio_outcomes(subportfolio_name)
        
        if filtered_outcomes.empty:
            return ""
        
        # Count OB usage across all domains in this subportfolio
        all_obs_usage = {}
        total_outcomes = len(filtered_outcomes)
        
        for _, outcome in filtered_outcomes.iterrows():
            ob_names = outcome.get('Outcome Basket names')
            if pd.notna(ob_names):
                # Split by semicolon and process each OB
                ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                for ob_str in ob_list:
                    if ob_str not in all_obs_usage:
                        all_obs_usage[ob_str] = 0
                    all_obs_usage[ob_str] += 1
        
        html = "<h3>Most frequently used Outcome Baskets (across all Domains of change)</h3>\n\n"
        
        # Most used OBs across all domains in this subportfolio
        html += "<ol>\n"
        for ob_name, count in sorted(all_obs_usage.items(), key=lambda x: x[1], reverse=True)[:15]:  # Top 15
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            html += f'<li><strong>{ob_name}</strong>: {count} uses (tagged in {percentage:.1f}% of the outcomes)</li>\n'
        html += "</ol>\n\n"
        
        return html

    def generate_subportfolio_custom_obs_analysis(self, subportfolio_name):
        """Generate analysis of custom outcome baskets for standardization candidates within subportfolio"""
        # Use helper method to get filtered outcomes for this subportfolio
        filtered_outcomes = self.get_subportfolio_outcomes(subportfolio_name)
        
        if filtered_outcomes.empty:
            return ""
        
        # Build a comprehensive set of all default outcome baskets across all domains
        all_default_obs = set()
        for domain_obs in self.default_obs.values():
            all_default_obs.update(domain_obs)
        
        # Count only truly custom outcome baskets (not default in any domain)
        custom_obs_usage = {}
        total_outcomes = len(filtered_outcomes)
        
        for _, outcome in filtered_outcomes.iterrows():
            ob_names = outcome.get('Outcome Basket names')
            if pd.notna(ob_names):
                # Split by semicolon and process each OB
                ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                for ob_str in ob_list:
                    # Only count if it's NOT a default OB in any domain
                    if ob_str not in all_default_obs:
                        if ob_str not in custom_obs_usage:
                            custom_obs_usage[ob_str] = 0
                        custom_obs_usage[ob_str] += 1
        
        if not custom_obs_usage:
            return ""

        html = "<h4>Top Custom OB - with potential to become default</h4>\n"
        html += "<p>This sub-portfolio analysis shows custom outcome baskets that could potentially become default baskets in the theme-wide categorization.</p>\n"
        html += "<ul>\n"
        
        # Show top custom OBs by frequency
        for ob_name, count in sorted(custom_obs_usage.items(), key=lambda x: x[1], reverse=True)[:6]:  # Top 6
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            html += f'<li><strong>{ob_name}</strong>: {count} uses (tagged in {percentage:.1f}% of the outcomes)</li>\n'
        html += "</ul>\n"
        
        html += "\n"
        return html

    def generate_subportfolio_outcome_records(self, subportfolio_name, sp_grants, subportfolio_outcomes):
        """Generate all outcome records section for subportfolio, organized by grant with proper hierarchy"""
        html = f"<h2>Outcomes from grants in the {subportfolio_name} sub-portfolio</h2>\n"
        
        # Group outcomes by grant
        outcomes_by_grant = {}
        for _, outcome in subportfolio_outcomes.iterrows():
            gams_code = str(outcome.get('GAMS code', ''))
            if gams_code not in outcomes_by_grant:
                outcomes_by_grant[gams_code] = []
            outcomes_by_grant[gams_code].append(outcome)
        
        # Sort grants by end date (later dates first) - same logic as used in grants table
        sorted_grants = sorted(sp_grants.items(), key=lambda x: x[1]['end_date_sort'], reverse=True)
        
        # Process each grant in sorted order
        for gams_code, grant_info in sorted_grants:
            # Check both main grant and phase 1 if consolidated
            grant_gams_codes = [gams_code]
            if grant_info['is_consolidated'] and grant_info['phase1_code']:
                grant_gams_codes.append(grant_info['phase1_code'])
            
            # Get outcomes for this grant (including phase 1)
            grant_outcomes = []
            for code in grant_gams_codes:
                if code in outcomes_by_grant:
                    grant_outcomes.extend(outcomes_by_grant[code])
            
            # Only show grants that have outcomes
            if not grant_outcomes:
                continue
                
            # Add h3 header for this grant (simplified to only show name and phase info)
            grant_display_name = self.format_display_name(grant_info['short_name'])
            phase_info = "(2-phase)" if grant_info['is_consolidated'] else "(single-phase)"
            
            html += f"<h3>{grant_display_name} {phase_info}</h3>\n"
            
            # Get additional grant details from CSV
            grant_row = self.grants_df[self.grants_df['GAMS Code'].astype(str) == gams_code]
            if not grant_row.empty:
                grant_data = grant_row.iloc[0]
                
                # Get grant summary (prefer Brief Summary, fallback to GAMS summary)
                brief_summary = grant_data.get('Brief Summary', '')
                gams_summary = grant_data.get("Grant's Summary (from GAMS)", '')
                grant_summary = brief_summary if pd.notna(brief_summary) and str(brief_summary).strip() else gams_summary
                grant_summary = str(grant_summary).strip() if pd.notna(grant_summary) else 'No summary available'
                
                # Get full grant name
                full_name = grant_data.get("Grant's Name", '')
                full_name = str(full_name).strip() if pd.notna(full_name) else 'N/A'
                
                # Get subportfolio memberships for display
                subportfolio_list = []
                for portfolio, is_member in grant_info['subportfolios'].items():
                    if is_member:
                        subportfolio_list.append(portfolio)
                subportfolio_display = ', '.join(subportfolio_list) if subportfolio_list else 'None'
                
                # Get consolidated GAMS code and end date for the info
                consolidated_gams = grant_info.get('consolidated_gams', gams_code)
                grant_end_date = grant_info['end_date_display']
                
                # Add comprehensive grant information as bullet points
                html += f"""
<ul>
    <li><strong>Full Name:</strong> {full_name}</li>
    <li><strong>Summary:</strong> {grant_summary}</li>
    <li><strong>GAMS Code:</strong> {consolidated_gams}</li>
    <li><strong>Budget:</strong> CHF {grant_info['budget']}</li>
    <li><strong>Duration:</strong> {grant_info['duration']} months</li>
    <li><strong>End Date:</strong> {grant_end_date}</li>
    <li><strong>Countries:</strong> {grant_info['countries']}</li>
    <li><strong>Sub-portfolios:</strong> {subportfolio_display}</li>
    <li><strong>Outcomes in DB:</strong> {grant_info['outcomes_count']}</li>
</ul>
"""
            
            # Group outcomes by report name first
            outcomes_by_report = {}
            for outcome in grant_outcomes:
                report_name = outcome.get('Report name', 'N/A')
                report_date = outcome.get('Report submission date', 'N/A')
                report_key = f"{report_name} ({report_date})" if report_date != 'N/A' else report_name
                
                if report_key not in outcomes_by_report:
                    outcomes_by_report[report_key] = []
                outcomes_by_report[report_key].append(outcome)
            
            # Sort reports by date (most recent first)
            sorted_reports = sorted(outcomes_by_report.items(), 
                                  key=lambda x: x[1][0].get('Report submission date', ''), 
                                  reverse=True)
            
            # Process each report section
            for report_key, report_outcomes in sorted_reports:
                html += f"<h4>{report_key}</h4>\n"
                
                # Process each outcome within this report
                for outcome in report_outcomes:
                    # Get full description for the title
                    description = outcome.get('Description of the Reported Outcome', 'N/A')
                    
                    # Add bold title with full description (no numbering)
                    html += f"<p><strong>{description}</strong></p>\n"
                    
                    # Add detailed information as bullet points in requested order
                    html += '<ul>\n'
                    
                    # 1. Additional details (if available)
                    additional_details = outcome.get('Additional details on the Outcome')
                    if pd.notna(additional_details) and str(additional_details).strip():
                        html += f"<li><strong>Additional details:</strong> {additional_details}</li>\n"
                    
                    # 2. Significance level (if available)
                    significance = outcome.get('Significance Level', 'N/A')
                    if pd.notna(significance):
                        html += f"<li><strong>Significance level:</strong> {significance}</li>\n"
                    
                    # 3. Special case tags
                    special_cases = []
                    special_cases_cols = [
                        'Special case - Unexpected Outcome',
                        'Special case - Negative Outcome', 
                        'Special case - Signal towards Outcome',
                        'Special case - Indicator Based Outcome',
                        'Special case - Keep on Radar',
                        'Special case - Programme Outcome'
                    ]
                    
                    for col in special_cases_cols:
                        if col in outcome.index and outcome[col] == 1:
                            clean_name = col.replace('Special case - ', '')
                            special_cases.append(clean_name)
                    
                    if special_cases:
                        html += f"<li><strong>Special case tags:</strong> {', '.join(special_cases)}</li>\n"
                    
                    # 4. Location
                    countries = outcome.get('Country(ies) / Global', 'N/A')
                    html += f"<li><strong>Location:</strong> {countries}</li>\n"
                    
                    # 5. Domain of Change
                    domain = outcome.get('Domain of Change', 'N/A')
                    html += f"<li><strong>Domain of Change:</strong> {domain}</li>\n"
                    
                    # 6. Outcome Baskets
                    ob_names = outcome.get('Outcome Basket names', 'N/A')
                    html += f"<li><strong>Outcome Baskets:</strong> {ob_names}</li>\n"
                    
                    html += '</ul>\n'
                    
                    # Add separator line after each outcome for better readability
                    html += '<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 20px 0;">\n'
        
        return html
    
    def crop_outcome_title(self, description, max_chars=120):
        """Crop outcome description to approximately 2 lines for use as h4 title"""
        if not description or description == 'N/A':
            return 'Outcome'
        
        # Clean the description
        description = str(description).strip()
        
        # If it's already short enough, return as is
        if len(description) <= max_chars:
            return description
        
        # Find the best break point (preferably at word boundary)
        if len(description) > max_chars:
            # Try to break at a word boundary near the max_chars limit
            break_point = max_chars
            # Look backwards for a space
            for i in range(max_chars, max(0, max_chars - 20), -1):
                if description[i] == ' ':
                    break_point = i
                    break
            
            cropped = description[:break_point].strip()
            # Add ellipsis if we actually cropped something
            if break_point < len(description):
                cropped += '...'
            
            return cropped
        
        return description

    def generate_grant_pages(self):
        """Generate HTML pages for individual grants directly from CSV data"""
        consolidated_grants, _ = self.get_consolidated_grants_data()
        
        for gams_code, grant_info in consolidated_grants.items():
            # Get grant details from CSV
            grant_row = self.grants_df[self.grants_df['GAMS Code'] == gams_code].iloc[0]
            
            # Get outcomes for this grant (including phase 1 if consolidated)
            grant_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == gams_code]
            if grant_info['is_consolidated'] and grant_info['phase1_code']:
                phase1_outcomes = self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == grant_info['phase1_code']]
                grant_outcomes = pd.concat([phase1_outcomes, grant_outcomes])
            
            # Generate content
            title = self.format_display_name(grant_info["short_name"])
            
            grant_name = grant_row.get("Grant's Name", 'N/A')
            # Use the formatted title for consolidated grants, otherwise use the raw short name
            display_short_name = title if grant_info['is_consolidated'] else grant_row.get("Grant's short name", 'N/A')
            
            content_html = f"""
            <h2>Grant Overview</h2>
            <ul>
                <li><strong>Primary Grant ID:</strong> {grant_info.get('consolidated_gams', gams_code)}</li>
                <li><strong>Grant Name:</strong> {grant_name}</li>
                <li><strong>Short Name:</strong> {display_short_name}</li>
                <li><strong>Budget (CHF):</strong> {grant_info['budget']}</li>
                <li><strong>Duration (months):</strong> {grant_info['duration']}</li>
                <li><strong>Closure Date:</strong> {grant_info['end_date_display']}</li>
                <li><strong>Countries:</strong> {grant_info['countries']}</li>
            </ul>
            
            <h2>Sub-Portfolio Memberships</h2>
            <ul>
            """
            
            for sp, is_member in grant_info['subportfolios'].items():
                if is_member:
                    sp_filename = self.sanitize_filename(sp)
                    content_html += f"<li><a href=\"../subportfolios/{sp_filename}.html\">{sp}</a></li>\n"
            
            content_html += "</ul>\n"
            
            # Add grant summary if available
            grant_summary = grant_row.get("Grant's Summary (from GAMS)")
            if pd.notna(grant_summary):
                content_html += f"""
                <h2>Grant Summary</h2>
                <p>{grant_summary}</p>
                """
            
            # Add outcomes section
            if len(grant_outcomes) > 0:
                content_html += f"""
                <h2>Outcomes Reported ({len(grant_outcomes)} total)</h2>
                """
                
                # Group outcomes by domain
                domain_outcomes = {}
                for _, outcome in grant_outcomes.iterrows():
                    # Use dynamic domain codes from discovered domains
                    for domain_code in self.domain_mapping.keys():
                        domain_col = f'Domain of Change {domain_code}'
                        if domain_col in outcome.index and pd.notna(outcome[domain_col]):
                            value = str(outcome[domain_col]).strip()
                            if value and value.lower() not in ['nan', '', 'none', 'null']:
                                if domain_code not in domain_outcomes:
                                    domain_outcomes[domain_code] = []
                                domain_outcomes[domain_code].append(outcome)
                
                # Display outcomes by domain
                for domain_code in sorted(domain_outcomes.keys()):
                    domain_name = self.domain_mapping[domain_code]
                    outcomes = domain_outcomes[domain_code]
                    
                    content_html += f"""
                    <h3>{domain_code} - {domain_name} ({len(outcomes)} outcomes)</h3>
                    """
                    
                    for outcome in outcomes:
                        ob = outcome.get('Outcome Basket', 'N/A')
                        description = outcome.get('Description of the Reported Outcome', 'N/A')
                        significance = outcome.get('Significance of the Outcome', 'N/A')
                        
                        content_html += f"""
                        <div class="outcome-record">
                            <h4>Outcome Basket: {ob}</h4>
                            <p><strong>Description:</strong> {description}</p>
                            <div class="outcome-meta">
                                <strong>Significance:</strong> {significance}
                            </div>
                        </div>
                        """
            else:
                content_html += "<h2>No outcomes reported yet</h2>"
            
            # Add detailed analysis for grants with outcomes (matching portfolio page structure)
            if len(grant_outcomes) > 0:
                # Add Outcomes analysis section
                content_html += "<h2>Outcomes analysis</h2>\n"
                content_html += "<h3>Distribution of outcomes across domains of change</h3>\n"
                content_html += self.generate_grant_domain_distribution(grant_outcomes)
                content_html += "<h3>Significance patterns</h3>\n"
                content_html += self.generate_grant_significance_patterns(grant_outcomes)
                content_html += "<h3>Special case outcomes</h3>\n"
                content_html += self.generate_grant_special_cases(grant_outcomes)
                content_html += self.generate_grant_geographic_distribution(grant_outcomes)
                
                # Add Outcome Basket analysis section
                content_html += "<h2>Outcome Basket analysis</h2>\n"
                
                # Add cross-domain summary for the grant
                content_html += self.generate_grant_cross_domain_summary(grant_outcomes)
                
                # Add custom OB analysis for the grant  
                content_html += self.generate_grant_custom_obs_analysis(grant_outcomes)
                
                # Add detailed domain analysis
                content_html += "<h3>Analysis of OB use across Domains of change</h3>\n"
                content_html += self.generate_unified_domain_analysis(
                    outcomes_df=grant_outcomes,
                    analysis_level="grant"
                )
            
            # Add performance ratings if available
            perf_columns = [
                ('Implementation.1', 'Implementation'),
                ('Contribution to strategic intent ', 'Contribution to strategic intent'),
                ('Learning ', 'Learning'), 
                ('Sustainability of changes', 'Sustainability of changes'),
                ('Stakeholder engagement and collaboration', 'Stakeholder engagement and collaboration'),
                ('Meaningful youth participation', 'Meaningful youth participation')
            ]
            
            content_html += "<h2>Performance Ratings</h2>\n<ul>\n"
            for csv_col, display_name in perf_columns:
                if csv_col in grant_row.index:
                    value = grant_row[csv_col]
                    if pd.notna(value) and str(value).strip() and str(value).strip().lower() != 'nan':
                        content_html += f"<li><strong>{display_name}:</strong> {str(value).strip()}</li>\n"
                    else:
                        content_html += f"<li><strong>{display_name}:</strong> N/A</li>\n"
                else:
                    content_html += f"<li><strong>{display_name}:</strong> N/A</li>\n"
            content_html += "</ul>\n"
            
            # Create breadcrumb
            breadcrumb = [
                {"text": "Theme Overview", "url": "../index.html"},
                {"text": title}
            ]
            
            # No navigation menu for grant pages (redundant with breadcrumb)
            
            # Create HTML page
            html_page = self.create_html_template(
                title=title,
                content=content_html,
                breadcrumb=breadcrumb,
                nav_menu=[]
            )
            
            # Generate filename with consistent format including grant name
            output_filename = self.generate_grant_filename(gams_code, grant_info["short_name"])
            
            # Write file
            output_file = self.html_output_path / "grants" / output_filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_page)
        
        print(f"✅ Generated {len(consolidated_grants)} enhanced individual grant pages")

    def generate_grant_domain_distribution(self, grant_outcomes):
        """Generate domain distribution analysis for individual grant page"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for domain analysis.</p>\n"
        
        # Count outcomes by domain
        domain_counts = {}
        total_outcomes = len(grant_outcomes)
        
        for _, outcome in grant_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    domain_counts[domain_code] = domain_counts.get(domain_code, 0) + 1
        
        if not domain_counts:
            return "<p>No domain information available.</p>\n"
        
        # Generate chart and summary
        html = f"<p>Total outcomes analyzed: <strong>{total_outcomes}</strong></p>\n"
        
        # Create pie chart data - using same colors as portfolio pages for consistency
        chart_data = []
        colors = ['#2c5aa0', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14']
        
        for i, (domain, count) in enumerate(sorted(domain_counts.items())):
            percentage = (count / total_outcomes) * 100
            domain_desc = self.domain_mapping.get(domain, domain)
            chart_data.append({
                'label': f"{domain}: {domain_desc}",
                'value': count,
                'percentage': percentage,
                'color': colors[i % len(colors)]
            })
        
        # Generate chart
        chart_id = f"grantDomainChart_{hash(str(sorted(domain_counts.items())))}"
        chart_labels = [item['label'] for item in chart_data]
        chart_values = [item['value'] for item in chart_data]
        chart_colors = [item['color'] for item in chart_data]
        
        html += f"""
<div style="margin: 30px 0; text-align: center;">
    <div style="max-width: 800px; margin: 0 auto;">
        <canvas id="{chart_id}" width="400" height="400"></canvas>
    </div>
</div>
<script>
document.addEventListener('DOMContentLoaded', function() {{
    const ctx = document.getElementById('{chart_id}').getContext('2d');
    new Chart(ctx, {{
        type: 'pie',
        data: {{
            labels: {chart_labels},
            datasets: [{{
                data: {chart_values},
                backgroundColor: {chart_colors},
                borderColor: '#ffffff',
                borderWidth: 2
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    position: 'right',
                    labels: {{
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {{
                            size: 11
                        }},
                        boxWidth: 12,
                        boxHeight: 12,
                        generateLabels: function(chart) {{
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {{
                                return data.labels.map((label, i) => {{
                                    const meta = chart.getDatasetMeta(0);
                                    const style = meta.controller.getStyle(i);
                                    
                                    // Function to wrap text for long labels
                                    function wrapText(text, maxLength = 40) {{
                                        if (text.length <= maxLength) return text;
                                        
                                        const words = text.split(' ');
                                        const lines = [];
                                        let currentLine = '';
                                        
                                        words.forEach(word => {{
                                            if ((currentLine + word).length <= maxLength) {{
                                                currentLine += (currentLine ? ' ' : '') + word;
                                            }} else {{
                                                if (currentLine) lines.push(currentLine);
                                                currentLine = word;
                                            }}
                                        }});
                                        
                                        if (currentLine) lines.push(currentLine);
                                        return lines;
                                    }}
                                    
                                    const wrappedText = wrapText(label);
                                    
                                    return {{
                                        text: wrappedText,
                                        fillStyle: style.backgroundColor,
                                        strokeStyle: style.borderColor,
                                        lineWidth: style.borderWidth,
                                        pointStyle: 'circle',
                                        rotation: 0,
                                        datasetIndex: 0,
                                        index: i
                                    }};
                                }});
                            }}
                            return [];
                        }}
                    }}
                }}
            }}
        }}
    }});
}});
</script>
"""
        
        return html

    def generate_grant_significance_patterns(self, grant_outcomes):
        """Generate grant significance patterns analysis matching portfolio format"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for significance analysis.</p>\n"
        
        # Build significance data by domain (reusing portfolio logic)
        significance_by_domain = {}
        domain_totals = {}
        overall_significance = {'High': 0, 'Medium': 0, 'Low': 0, 'Not specified': 0}
        total_outcomes = len(grant_outcomes)
        
        # Initialize with all domains present in this grant's outcomes
        for _, outcome in grant_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    if domain_code not in significance_by_domain:
                        significance_by_domain[domain_code] = {'High': 0, 'Medium': 0, 'Low': 0, 'Not specified': 0}
                        domain_totals[domain_code] = 0
        
        # Count outcomes by domain and significance
        for _, outcome in grant_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    if domain_code in significance_by_domain:
                        significance = outcome.get('Significance Level', 'Not specified')
                        if pd.notna(significance) and str(significance).strip():
                            sig_level = str(significance).strip()
                            if sig_level not in ['High', 'Medium', 'Low']:
                                sig_level = 'Not specified'
                        else:
                            sig_level = 'Not specified'
                        
                        significance_by_domain[domain_code][sig_level] += 1
                        domain_totals[domain_code] += 1
                        overall_significance[sig_level] += 1
        
        # Generate HTML with same format as portfolio pages
        html = "<p>Distribution of the outcomes' significance levels across Domains of change. "
        html += "<span class='info-icon' onclick='showSignificanceInfo()' title='Click for more information about significance levels'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart (reusing existing method)
        html += self.generate_significance_stacked_chart(significance_by_domain, domain_totals, overall_significance, total_outcomes)
        
        # Add examples of outcomes by significance level (reusing existing method)
        html += self.generate_significance_examples(grant_outcomes, overall_significance)
        
        return html

    def generate_grant_special_cases(self, grant_outcomes):
        """Generate grant special case outcomes analysis matching portfolio format"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for special cases analysis.</p>\n"
        
        # Special cases columns
        special_cases_cols = [
            'Special case - Unexpected Outcome',
            'Special case - Negative Outcome', 
            'Special case - Signal towards Outcome',
            'Special case - Indicator Based Outcome',
            'Special case - Keep on Radar',
            'Special case - Programme Outcome'
        ]
        
        # Build special cases data by domain (reusing portfolio logic)
        special_cases_by_domain = {}
        domain_totals = {}
        overall_special_cases = {}
        total_outcomes = len(grant_outcomes)
        
        # Initialize with all domains present in this grant's outcomes
        for _, outcome in grant_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    if domain_code not in special_cases_by_domain:
                        special_cases_by_domain[domain_code] = {}
                        domain_totals[domain_code] = 0
        
        # Count special cases by domain
        for _, outcome in grant_outcomes.iterrows():
            domain_full = outcome.get('Domain of Change')
            if pd.notna(domain_full) and str(domain_full).strip():
                domain_parts = str(domain_full).split(' - ', 1)
                if len(domain_parts) >= 1:
                    domain_code = domain_parts[0].strip()
                    if domain_code in special_cases_by_domain:
                        # Count total outcomes for this domain
                        domain_totals[domain_code] += 1
                        
                        # Check each special case column
                        for col in special_cases_cols:
                            if col in outcome.index:
                                value = outcome[col]
                                if pd.notna(value):
                                    value_str = str(value).strip().lower()
                                    is_special_case = (value_str == 'x' or value_str == '1')
                                    
                                    if is_special_case:
                                        case_name = col.replace('Special case - ', '')
                                        if case_name not in special_cases_by_domain[domain_code]:
                                            special_cases_by_domain[domain_code][case_name] = 0
                                        if case_name not in overall_special_cases:
                                            overall_special_cases[case_name] = 0
                                        
                                        special_cases_by_domain[domain_code][case_name] += 1
                                        overall_special_cases[case_name] += 1
        
        # Generate HTML with same format as portfolio pages
        html = "<p>Distribution of special case outcomes across domains of change. "
        html += "<span class='info-icon' onclick='showSpecialCasesInfo()' title='Click for more information about special case types'>ℹ️</span></p>\n"
        
        # Generate the stacked bar chart (reusing existing method)
        html += self.generate_special_cases_stacked_chart(special_cases_by_domain, domain_totals, overall_special_cases, total_outcomes)
        
        # Add examples of special case outcomes (reusing existing method)
        html += self.generate_special_cases_examples(grant_outcomes, overall_special_cases)
        
        return html

    def generate_geographic_chart_html(self, geo_counts, global_count, total_outcomes, chart_type="grant"):
        """Generate reusable geographic chart HTML that works for both portfolio and grant pages"""
        if not geo_counts and global_count == 0:
            return "<p>No geographic information available.</p>\n"
        
        # Use existing portfolio logic to generate the complete country mapping
        country_mapping = {
            'Tanzania': 'Tanzania',
            'Kenya': 'Kenya',
            'Indonesia': 'Indonesia',
            'Ecuador': 'Ecuador',
            'India': 'India',
            'Ghana': 'Ghana',
            'Senegal': 'Senegal',
            'Mexico': 'Mexico',
            'Philippines': 'Philippines',
            'Vietnam': 'Vietnam',
            'South Africa': 'South Africa',
            'Rwanda': 'Rwanda',
            'Argentina': 'Argentina',
            'Nigeria': 'Nigeria',
            'Cameroon': 'Cameroon',
            'Sri Lanka': 'Sri Lanka',
            'Moldova': 'Moldova',
            'Maldives': 'Maldives',
            'Kyrgyzstan': 'Kyrgyzstan',
            'Benin': 'Benin',
            'Central African Republic': 'Central African Rep.',
            'Comoros': 'Comoros',
            'Congo [Republic]': 'Congo',
            'Gabon': 'Gabon',
            'Guinea': 'Guinea',
            'Madagascar': 'Madagascar', 
            'Togo': 'Togo',
            'Tunisia': 'Tunisia',
            'Zambia': 'Zambia',
            'Myanmar': 'Myanmar',
            'Nepal': 'Nepal',
            'Brazil': 'Brazil',
            'Colombia': 'Colombia',
            'Peru': 'Peru',
            'Thailand': 'Thailand',
            'Pakistan': 'Pakistan',
            'Ethiopia': 'Ethiopia',
            'Democratic Republic of the Congo': 'Dem. Rep. Congo',
        }
        
        # Create mapping for ChartGeo
        map_data_pairs = []
        for country, count in geo_counts.items():
            mapped_country = country_mapping.get(country, country)
            map_data_pairs.append(f'["{mapped_country}", {count}]')
        
        # Join the map data pairs for JavaScript
        map_data_js = ', '.join(map_data_pairs)
        
        # Generate Global indicator if needed (matches portfolio format exactly)
        global_indicator_html = ""
        if global_count > 0:
            global_indicator_html = f"""
        <div style="
            position: absolute;
            left: 50%;
            top: 180px;
            width: 30px;
            height: 30px;
            background-color: rgba(0, 123, 255, 0.8);
            border: 2px solid #007bff;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            color: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10;
        " title="Global: {global_count} outcomes ({(global_count / total_outcomes * 100):.1f}% of all outcomes)">
            G
        </div>"""
        
        # Generate unique canvas ID
        import hashlib
        canvas_id = f"worldMap{chart_type.title()}_{hashlib.md5(str(hash(tuple(sorted(geo_counts.items())))).encode()).hexdigest()[:8]}"
        
        # Return exact portfolio chart format
        html = f"""
<div style="margin: 30px 0;">
    <div style="width: 100%; margin: 0 auto; position: relative;">
        <canvas id="{canvas_id}" width="1000" height="500" style="width: 100%; height: auto;"></canvas>{global_indicator_html}
    </div>
</div>

<script>
(function() {{
    // Create a completely isolated scope for the {chart_type} map
    let chartJsLoaded = false;
    let chartGeoLoaded = false;
    
    function loadScript(src, callback) {{
        const script = document.createElement('script');
        script.src = src;
        script.onload = callback;
        script.onerror = () => console.error('Failed to load script:', src);
        document.head.appendChild(script);
    }}
    
    function createIsolated{chart_type.title()}Map() {{
        if (!chartJsLoaded || !chartGeoLoaded) {{
            console.log('Waiting for libraries to load...');
            return;
        }}
        
        console.log('Creating isolated {chart_type} map with loaded libraries');
        
        // Force re-registration of ChartGeo components
        if (typeof ChartGeo !== 'undefined' && ChartGeo.ChoroplethController) {{
            Chart.register(ChartGeo.ChoroplethController, ChartGeo.GeoFeature, ChartGeo.ColorScale);
            console.log('ChartGeo components force-registered for {chart_type}');
        }}
        
        fetch('https://unpkg.com/world-atlas/countries-50m.json')
            .then(response => {{
                if (!response.ok) throw new Error('Failed to load topology data');
                return response.json();
            }})
            .then(topology => {{
                const canvas = document.getElementById('{canvas_id}');
                if (!canvas) throw new Error('Canvas element not found');
                
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('Could not get canvas context');
                
                const countries = ChartGeo.topojson.feature(topology, topology.objects.countries).features;
                
                // Create a map of country data for quick lookup
                const countryData = new Map([
                    {map_data_js}
                ]);

                const chart = new Chart(ctx, {{
                    type: 'choropleth',
                    data: {{
                        labels: countries.map(c => c.properties.name),
                        datasets: [{{
                            label: '{chart_type.title()} Outcomes',
                            outline: countries,
                            showOutline: true,
                            backgroundColor: (context) => {{
                                if (!context.dataIndex || context.dataIndex < 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                const feature = context.chart.data.datasets[0].outline[context.dataIndex];
                                if (!feature || !feature.properties) {{
                                    return 'rgba(211, 211, 211, 0.2)';
                                }}
                                
                                const countryName = feature.properties.name;
                                const count = countryData.get(countryName) || 0;
                                
                                if (count === 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                // Create intensity based on value (matching theme page)
                                const maxCount = Math.max(...Array.from(countryData.values()));
                                const intensity = Math.min(count / maxCount, 1);
                                const opacity = 0.2 + intensity * 0.8;
                                return `rgba(44, 90, 160, ${{opacity}})`;
                            }},
                            borderColor: 'rgba(0, 0, 0, 0.3)',
                            borderWidth: 1,
                            data: countries.map(c => {{
                                const count = countryData.get(c.properties.name) || 0;
                                return {{
                                    feature: c,
                                    value: count
                                }};
                            }})
                        }}]
                    }},
                    options: {{
                        responsive: false,
                        maintainAspectRatio: false,
                        showOutline: true,
                        showGraticule: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(context) {{
                                        return context[0].parsed.feature.properties.name;
                                    }},
                                    label: function(context) {{
                                        const count = context.parsed.value;
                                        if (count === 0) {{
                                            return 'No outcomes reported';
                                        }}
                                        const percentage = ((count / {total_outcomes}) * 100).toFixed(1);
                                        return `${{count}} outcomes (${{percentage}}% of {chart_type} outcomes)`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            projection: {{
                                axis: 'x',
                                projection: 'naturalEarth1'
                            }}
                        }}
                    }}
                }});
            }})
            .catch(error => {{
                console.error('Error creating {chart_type} map:', error);
            }});
    }}
    
    // Check if Chart.js is already loaded
    if (typeof Chart !== 'undefined') {{
        console.log('Chart.js already loaded for {chart_type}');
        chartJsLoaded = true;
    }} else {{
        loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js', () => {{
            console.log('Chart.js loaded for {chart_type}');
            chartJsLoaded = true;
            createIsolated{chart_type.title()}Map();
        }});
    }}
    
    // Check if ChartGeo is already loaded
    if (typeof ChartGeo !== 'undefined') {{
        console.log('ChartGeo already loaded for {chart_type}');
        chartGeoLoaded = true;
    }} else {{
        loadScript('https://cdn.jsdelivr.net/npm/chartjs-chart-geo@4', () => {{
            console.log('ChartGeo loaded for {chart_type}');
            chartGeoLoaded = true;
            createIsolated{chart_type.title()}Map();
        }});
    }}
    
    // If both libraries are already loaded, create the map immediately
    if (chartJsLoaded && chartGeoLoaded) {{
        createIsolated{chart_type.title()}Map();
    }}
}})();
</script>
"""
        return html

    def generate_grant_geographic_distribution(self, grant_outcomes):
        """Generate geographic distribution analysis for individual grant page - reuses portfolio function"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for geographic analysis.</p>\n"
        
        # Save original outcomes and temporarily replace with grant outcomes
        original_outcomes = self.outcomes_df
        self.outcomes_df = grant_outcomes
        
        # Create a temporary method to get grant outcomes (like get_subportfolio_outcomes)
        def temp_get_grant_outcomes(unused_param):
            return grant_outcomes
        
        # Temporarily replace get_subportfolio_outcomes method
        original_method = self.get_subportfolio_outcomes
        self.get_subportfolio_outcomes = temp_get_grant_outcomes
        
        # Call the exact portfolio function
        result = self.generate_subportfolio_geographic_distribution("temp")
        
        # Restore original methods and data
        self.get_subportfolio_outcomes = original_method
        self.outcomes_df = original_outcomes
        
        return result
        
        # OLD CODE BELOW - keeping for backup but this function now calls portfolio function above
        # Define regions and their countries (reusing portfolio logic)
        regions = {
            'Sub-Saharan Africa': ['Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 
                                 'Chad', 'Comoros', 'Congo', 'Democratic Republic of the Congo', 'Côte d\'Ivoire', 'Djibouti', 'Equatorial Guinea', 
                                 'Eritrea', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 
                                 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
                                 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 
                                 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'],
            'South Asia': ['Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'],
            'East Asia & Pacific': ['Cambodia', 'China', 'Fiji', 'Indonesia', 'Kiribati', 'Korea, Democratic People\'s Republic', 'Lao PDR', 
                                  'Malaysia', 'Marshall Islands', 'Micronesia', 'Mongolia', 'Myanmar', 'Palau', 'Papua New Guinea', 'Philippines', 
                                  'Samoa', 'Solomon Islands', 'Thailand', 'Timor-Leste', 'Tonga', 'Tuvalu', 'Vanuatu', 'Vietnam'],
            'Middle East & North Africa': ['Algeria', 'Bahrain', 'Djibouti', 'Egypt', 'Iran', 'Iraq', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 
                                         'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Yemen'],
            'Latin America & Caribbean': ['Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bolivia', 'Brazil', 'Chile', 
                                        'Colombia', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Grenada', 
                                        'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 
                                        'Peru', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Suriname', 
                                        'Trinidad and Tobago', 'Uruguay', 'Venezuela'],
            'Europe & Central Asia': ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Georgia', 
                                    'Kazakhstan', 'Kosovo', 'Kyrgyz Republic', 'Moldova', 'Montenegro', 'North Macedonia', 'Romania', 'Russia', 
                                    'Serbia', 'Tajikistan', 'Turkey', 'Turkmenistan', 'Ukraine', 'Uzbekistan']
        }
        
        # Create reverse mapping from country to region
        country_to_region = {}
        for region, countries in regions.items():
            for country in countries:
                country_to_region[country.lower()] = region
        
        # Count outcomes by country and region
        geo_counts = {}  # For countries
        region_counts = {}  # For regions
        global_count = 0
        total_outcomes = len(grant_outcomes)
        
        # Process each outcome
        for _, outcome in grant_outcomes.iterrows():
            countries = outcome.get('Country(ies) / Global')
            if pd.notna(countries) and str(countries).strip():
                country_list = [c.strip() for c in str(countries).split(';') if c.strip()]
                for country in country_list:
                    country_clean = country.strip()
                    if country_clean.lower() in ['global', 'worldwide']:
                        global_count += 1
                    elif country_clean.lower() not in ['n/a', 'na', 'none', '']:
                        geo_counts[country_clean] = geo_counts.get(country_clean, 0) + 1
                        
                        # Map to region
                        region = country_to_region.get(country_clean.lower(), "Other")
                        if region != "Other":
                            region_counts[region] = region_counts.get(region, 0) + 1
        
        if not geo_counts and global_count == 0:
            return "<p>No geographic information available.</p>\n"
        
        html = "<h3>Geographic distribution of outcomes</h3>\n"
        
        # Show regional distribution with Global first (matching portfolio format)
        html += "<ul>\n"
        
        # Add Global as first item if it exists
        if global_count > 0:
            percentage = (global_count / total_outcomes * 100)
            html += f"<li><strong>Global:</strong> {global_count} ({percentage:.1f}% of all outcomes)</li>\n"
        
        # Add regional data
        for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            html += f"<li><strong>{region}:</strong> {count} ({percentage:.1f}% of all outcomes)</li>\n"
        html += "</ul>\n\n"
        
        # Generate the world map (reusing portfolio structure)
        if geo_counts:  # Only show map if there are specific countries
            # Generate data for the world map - create a mapping object
            country_mapping = {
                'Tanzania': 'Tanzania',
                'Kenya': 'Kenya', 
                'Indonesia': 'Indonesia',
                'Ecuador': 'Ecuador',
                'India': 'India',
                'Ghana': 'Ghana',
                'Senegal': 'Senegal',
                'Mexico': 'Mexico',
                'Philippines': 'Philippines',
                'Vietnam': 'Vietnam',
                'South Africa': 'South Africa',
                'Rwanda': 'Rwanda',
                'Argentina': 'Argentina',
                'Nigeria': 'Nigeria',
                'Cameroon': 'Cameroon',
                'Sri Lanka': 'Sri Lanka',
                'Moldova': 'Moldova',
                'Maldives': 'Maldives',
                'Kyrgyzstan': 'Kyrgyzstan',
                'Benin': 'Benin',
                'Central African Republic': 'Central African Republic',
                'Comoros': 'Comoros',
                'Congo [Republic]': 'Republic of the Congo',
                'Gabon': 'Gabon',
                'Guinea': 'Guinea',
                'Madagascar': 'Madagascar',
                'Togo': 'Togo',
                'Tunisia': 'Tunisia',
                'Zambia': 'Zambia',
            }
            
            # Create mapping for ChartGeo
            map_data_pairs = []
            for country, count in geo_counts.items():
                mapped_country = country_mapping.get(country, country)
                percentage = (count / total_outcomes * 100)
                map_data_pairs.append(f'["{mapped_country}", {count}]')
            
            # Join the map data pairs for JavaScript
            map_data_js = ', '.join(map_data_pairs)            
            # Generate the world map canvas and script
            chart_id = f"worldMapChart_{hash(str(sorted(geo_counts.items())))}"
            html += f"""
<div style="margin: 30px 0; text-align: center;">
    <div style="max-width: 900px; margin: 0 auto;">
        <canvas id="{chart_id}" width="900" height="500" style="max-width: 100%; height: auto;"></canvas>
    </div>
</div>

<script>
// Load Chart.js first, then ChartGeo (matching theme page exactly)
if (typeof Chart !== 'undefined') {{
    console.log('Chart.js loaded');
    
    // Load ChartGeo from CDN
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/chartjs-chart-geo@4';
    script.onload = function() {{
        console.log('ChartGeo loaded');
        
        // Fetch world data and create the map
        fetch('https://unpkg.com/world-atlas/countries-50m.json')
            .then(response => response.json())
            .then(topology => {{
                console.log('World data loaded');
                
                const countries = ChartGeo.topojson.feature(topology, topology.objects.countries).features;
                
                // Create a map of country data for quick lookup
                const countryData = new Map([
                    {map_data_js}
                ]);
                
                const ctx = document.getElementById('{chart_id}').getContext('2d');
                new Chart(ctx, {{
                    type: 'choropleth',
                    data: {{
                        labels: countries.map(d => d.properties.name),
                        datasets: [{{
                            label: 'Outcomes',
                            data: countries.map(country => {{
                                const countryName = country.properties.name;
                                const value = countryData.get(countryName) || 0;
                                return {{
                                    feature: country,
                                    value: value
                                }};
                            }}),
                            backgroundColor: (ctx) => {{
                                if (!ctx.dataIndex || ctx.dataIndex < 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                const feature = ctx.chart.data.labels[ctx.dataIndex];
                                const value = countryData.get(feature) || 0;
                                
                                if (value === 0) {{
                                    return 'rgba(211, 211, 211, 0.2)'; // Light gray for no data
                                }}
                                
                                // Create intensity based on value (logarithmic scale for better visualization)
                                const maxValue = Math.max(...Array.from(countryData.values()));
                                const intensity = Math.min(value / maxValue, 1);
                                const opacity = 0.2 + intensity * 0.8;
                                return `rgba(44, 90, 160, ${{opacity}})`;
                            }},
                            borderColor: 'rgba(0, 0, 0, 0.2)',
                            borderWidth: 0.5
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                display: false
                            }},
                            tooltip: {{
                                callbacks: {{
                                    title: function(context) {{
                                        return context[0].parsed.feature.properties.name;
                                    }},
                                    label: function(context) {{
                                        const value = context.parsed.value;
                                        if (value === 0) {{
                                            return 'No outcomes reported';
                                        }}
                                        const total = {total_outcomes};
                                        const percentage = ((value / total) * 100).toFixed(1);
                                        return `${{value}} outcomes (${{percentage}}% of grant outcomes)`;
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            projection: {{
                                axis: 'x',
                                projection: 'equalEarth'
                            }}
                        }}
                    }}
                }});
            }})
            .catch(error => {{
                console.error('Error loading world map data:', error);
                document.getElementById('{chart_id}').style.display = 'none';
            }});
    }};
    script.onerror = function() {{
        console.error('Failed to load ChartGeo');
        document.getElementById('{chart_id}').style.display = 'none';
    }};
    document.head.appendChild(script);
}} else {{
    console.error('Chart.js not available');
    document.getElementById('{chart_id}').style.display = 'none';
}}
</script>
"""
        
            # Add clickable link to show country distribution details (centered)
            html += "<p style='text-align: center; margin: 15px 0;'><a href='javascript:void(0)' onclick='showGeographicDetails()' style='color: #2c5aa0; text-decoration: underline; cursor: pointer;'>[see information as a list]</a></p>\n"
            
            # Prepare data for modal
            country_data = []
            for country, count in sorted(geo_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_outcomes * 100)
                country_region = country_to_region.get(country.lower(), "Other")
                country_data.append(f"<li><strong>{country}</strong> ({country_region}): {count} ({percentage:.1f}% of all outcomes)</li>")
            
            # Add JavaScript to set the modal data
            html += f"""
<script>
window.geographicData = `<ul>{''.join(country_data)}</ul>`;
</script>
"""
        
        return html

    def generate_grant_cross_domain_summary(self, grant_outcomes):
        """Generate cross-domain OB usage summary for individual grant"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for cross-domain analysis.</p>\n"
        
        # Count outcome baskets across all domains
        ob_counts = {}
        
        for _, outcome in grant_outcomes.iterrows():
            ob_names = outcome.get('Outcome Basket names', '')
            if pd.notna(ob_names) and str(ob_names).strip():
                obs = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                for ob in obs:
                    ob_counts[ob] = ob_counts.get(ob, 0) + 1
        
        if not ob_counts:
            return "<p>No outcome basket information available.</p>\n"
        
        # Sort by frequency
        sorted_obs = sorted(ob_counts.items(), key=lambda x: x[1], reverse=True)
        
        html = "<h3>Most frequently used Outcome Baskets (across domains)</h3>\n"
        html += f"<p>Total unique outcome baskets used: <strong>{len(ob_counts)}</strong></p>\n"
        html += "<div class='summary-stats'>\n<ul>\n"
        
        for ob, count in sorted_obs[:10]:  # Show top 10
            html += f"<li><strong>{ob}:</strong> {count} time(s)</li>\n"
        
        if len(sorted_obs) > 10:
            html += f"<li><em>... and {len(sorted_obs) - 10} more</em></li>\n"
        
        html += "</ul>\n</div>\n"
        
        return html

    def generate_grant_custom_obs_analysis(self, grant_outcomes):
        """Generate custom OB analysis for individual grant"""
        if grant_outcomes.empty:
            return "<p>No outcomes available for custom OB analysis.</p>\n"
        
        # Collect all custom (non-default) OBs used in this grant
        custom_obs = set()
        
        for _, outcome in grant_outcomes.iterrows():
            non_default_obs = outcome.get('SPECIAL COLUMN (generated) - Non Default OB Tags', '')
            if pd.notna(non_default_obs) and str(non_default_obs).strip():
                obs = [ob.strip() for ob in str(non_default_obs).split(';') if ob.strip()]
                custom_obs.update(obs)
        
        if not custom_obs:
            return "<h3>Custom Outcome Baskets</h3>\n<p>This grant uses only standard outcome baskets.</p>\n"
        
        html = "<h3>Custom Outcome Baskets</h3>\n"
        html += f"<p>This grant uses <strong>{len(custom_obs)}</strong> custom outcome basket(s):</p>\n"
        html += "<div class='summary-stats'>\n<ul>\n"
        
        for ob in sorted(custom_obs):
            html += f"<li>{ob}</li>\n"
        
        html += "</ul>\n</div>\n"
        
        return html

    def generate_all_reports(self):
        """Generate all enhanced HTML reports directly from CSV data"""
        print("Starting Enhanced HTML report generation...")
        
        # Setup directory structure
        self.setup_html_structure()
        
        # Generate theme index page
        self.generate_theme_index()
        
        # Generate sub-portfolio pages
        self.generate_subportfolio_pages()
        
        # Generate individual grant pages
        self.generate_grant_pages()
        
        print("✅ All Enhanced HTML reports generated successfully!")
        print(f"📁 Enhanced reports available at: {self.html_output_path}")
        print(f"🌐 Open in browser: file://{self.html_output_path / 'index.html'}")

if __name__ == "__main__":
    base_path = "/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev"
    
    # For Digital theme, use FB-Digital- prefix (default)
    # For future Health theme, use: generator = EnhancedHTMLReportGenerator(base_path, theme_prefix="FB-Health-")
    generator = EnhancedHTMLReportGenerator(base_path, theme_prefix="FB-Digital-")
    generator.generate_all_reports()
