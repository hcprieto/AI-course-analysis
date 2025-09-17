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
    
    def get_css_path(self, is_index_page=False):
        """Get appropriate CSS path based on page type"""
        return 'css/analysis_styles.css' if is_index_page else '../css/analysis_styles.css'
    
    def add_venn_diagram_if_exists(self):
        """Generate Venn diagram HTML if file exists"""
        venn_diagram_path = self.html_output_path / "Digital Theme Grants Sub-Portfolios.VennDiagram.png"
        if venn_diagram_path.exists():
            return """
<h3>Digital Theme Grants Sub-Portfolios Venn Diagram</h3>
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
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="current-section" id="currentSection" style="display: none;"></div>
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
                {{ element: 'h2', text: 'Outcome Basket analysis', display: 'Outcome Basket analysis' }}
            ];
            
            const currentSectionElement = document.getElementById('currentSection');
            if (!currentSectionElement) return;
            
            let activeSection = null;
            let shouldShow = false;
            const scrollPosition = window.scrollY + 150; // Offset for sticky header
            
            // Find the current section based on scroll position
            for (let i = sections.length - 1; i >= 0; i--) {{
                const section = sections[i];
                const elements = document.querySelectorAll(section.element);
                
                for (let element of elements) {{
                    if (element.textContent.trim() === section.text && element.offsetTop <= scrollPosition) {{
                        activeSection = section.display;
                        shouldShow = true;
                        break;
                    }}
                }}
                
                if (activeSection) break;
            }}
            
            // Show/hide the indicator based on whether we're in a main section
            if (shouldShow && activeSection) {{
                currentSectionElement.textContent = activeSection;
                currentSectionElement.style.display = 'inline-block';
                currentSectionElement.classList.add('visible');
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
        
        // Initialize section tracking on page load
        document.addEventListener('DOMContentLoaded', function() {{
            updateCurrentSection();
        }});
    </script>
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
                    if ', phase 1' not in display_name and ', phase 2' not in display_name:
                        display_name = self.format_display_name(display_name) + ' (consolidated)'
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
        
        html += '<th>Budget (CHF)</th>\n<th>Duration (months)</th>\n<th>End Date</th>\n<th>Countries</th>\n<th>Outcomes gathered</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Grant rows
        for gams_code, grant_info in sorted_grants:
            html += '<tr>\n'
            
            # Grant name (clickable)
            grant_filename = f"{gams_code}_analysis.html"
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
        if self.outcomes_df.empty:
            return "<p>No outcomes data available for detailed analysis.</p>"
        
        # Group outcomes by domain
        outcomes_by_domain = {}
        for _, outcome in self.outcomes_df.iterrows():
            domain = outcome.get('Domain of Change')  # Correct column name
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key not in outcomes_by_domain:
                    outcomes_by_domain[domain_key] = []
                outcomes_by_domain[domain_key].append(outcome)
        
        # Calculate total outcomes for percentage calculation
        total_outcomes = len(self.outcomes_df)
        
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
            
            # New domain header format
            html += f"<h4>Domain of Change {domain_full_text}</h4>\n"
            html += f"<lu><li><strong>Number of outcomes:</strong> {outcome_count}</li>\n"
            
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
            default_percentage = (total_default_obs / total_obs_tagged * 100) if total_obs_tagged > 0 else 0
            cross_domain_percentage = (total_cross_domain_obs / total_obs_tagged * 100) if total_obs_tagged > 0 else 0
            custom_percentage = (total_custom_obs / total_obs_tagged * 100) if total_obs_tagged > 0 else 0
            
            html += f"<li><strong>Number of OB:</strong> {total_obs_tagged}</li></ul>\n"
            
            # Add pie chart for OB distribution
            html += self.generate_domain_ob_pie_chart(domain_key, total_default_obs, total_cross_domain_obs, total_custom_obs, total_obs_tagged)
            
            # 1. Usage of default Outcome Baskets - Display as horizontal bar chart
            if default_obs_for_domain:
                html += "<h5>Default OB used</h5>\n"
                
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

                # Keep original CSV order (don't sort)
                # Filter to only include OBs with count > 0 but maintain original order
                used_default_obs_in_order = [(ob, count, description) for ob, count, description in all_default_obs if count > 0]
                
                # Generate horizontal bar chart for default OBs
                html += self.generate_default_ob_bar_chart(domain_key, used_default_obs_in_order, outcome_count)
            
            # 2. Cross-Domain OBs Used in this Domain
            if cross_domain_obs:
                html += f"<h5>Cross-domain OB used</h5>\n<ul>\n"
                for ob_label, count in sorted(cross_domain_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li><strong>{ob_label}</strong>: {count} uses (tagged in {percentage:.1f}% of {domain_key} outcomes)</li>\n'
                html += "</ul>\n\n"
            
            # 3. Custom OBs in this Domain
            if custom_obs:
                html += f"<h5>Custom OB used</h5>\n<ul>\n"
                for ob, count in sorted(custom_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li><strong>{ob}</strong>: {count} uses (tagged in {percentage:.1f}% of {domain_key} outcomes)</li>\n'
                html += "</ul>\n\n"
        
        return html
    
    def generate_default_ob_bar_chart(self, domain_key, all_default_obs, outcome_count):
        """Generate a horizontal bar chart for default OB usage within a specific domain"""
        if not all_default_obs:
            return ""
        
        # Filter and prepare data for only OBs with count > 0
        used_obs = [(ob, count, description) for ob, count, description in all_default_obs if count > 0]
        
        if not used_obs:
            return "<p>No default outcome baskets were used for this domain.</p>"
        
        # Prepare data for the chart
        chart_labels = []
        chart_data = []
        
        for ob, count, _ in used_obs:
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
        
        # Prepare data for the chart
        chart_data = []
        chart_labels = []
        chart_colors = ['#2c5aa0', '#28a745', '#ffc107']  # Blue, Green, Yellow
        
        # Add data only for categories that have values > 0
        if total_default_obs > 0:
            default_percentage = (total_default_obs / total_obs_tagged * 100)
            chart_labels.append(f'Default OB')
            chart_data.append(total_default_obs)
        
        if total_cross_domain_obs > 0:
            cross_domain_percentage = (total_cross_domain_obs / total_obs_tagged * 100)
            chart_labels.append(f'Other domains OB')
            chart_data.append(total_cross_domain_obs)
        
        if total_custom_obs > 0:
            custom_percentage = (total_custom_obs / total_obs_tagged * 100)
            chart_labels.append(f'Custom OB')
            chart_data.append(total_custom_obs)
        
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
                backgroundColor: {chart_colors[:len(chart_data)]},
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
        
        # Determine appropriate stepSize based on maximum value
        if max_stack_total <= 16:
            step_size = 2
        elif max_stack_total <= 24:
            step_size = 3
        elif max_stack_total <= 32:
            step_size = 4
        elif max_stack_total <= 40:
            step_size = 5
        elif max_stack_total <= 80:
            step_size = 10
        elif max_stack_total <= 160:
            step_size = 20
        elif max_stack_total <= 200:
            step_size = 25
        else:
            step_size = 30

        # Calculate y-axis max: round up to next step interval and add one step for better visibility
        y_axis_max = (((max_stack_total // step_size) + 1) * step_size) + step_size

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
                    min: 0,
                    max: {y_axis_max},
                    title: {{
                        display: true,
                        text: 'Number of Outcomes'
                    }},
                    ticks: {{
                        stepSize: {step_size},
                        callback: function(value) {{
                            return Number.isInteger(value) ? value : '';
                        }}
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
        html += "<p>To see examples of high significance outcomes, please refer to the pages for the sub-portfolios and grants.</p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_significance_stacked_chart(significance_by_domain, domain_totals, overall_significance, total_outcomes)
        
        return html
    
    def generate_special_cases_stacked_chart(self, special_cases_by_domain, domain_totals, overall_special_cases, total_outcomes):
        """Generate a stacked bar chart for special cases patterns across domains"""
        if not special_cases_by_domain or total_outcomes == 0:
            return ""
        
        # Use all possible domains from domain mapping to ensure consistency across pages
        domain_order = sorted(self.domain_mapping.keys())
        
        # Get all special case types
        special_case_types = sorted(overall_special_cases.keys())
        
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
        
        # Build datasets for Chart.js (one dataset per special case type)
        datasets_json = []
        for i, case_type in enumerate(special_case_types):
            dataset_data = []
            
            # Get data for each domain for this special case type (0 if domain has no data)
            for domain_code in domain_order:
                if domain_code in special_cases_by_domain:
                    count = special_cases_by_domain[domain_code].get(case_type, 0)
                else:
                    count = 0  # Domain has no outcomes at all
                dataset_data.append(count)
            
            # Create proper JSON-formatted dataset
            datasets_json.append(f"""{{
                label: '{case_type}',
                data: {dataset_data},
                backgroundColor: '{special_case_colors[i % len(special_case_colors)]}',
                borderColor: '{special_case_colors[i % len(special_case_colors)]}',
                borderWidth: 1
            }}""")
        
        # Build labels for x-axis using full domain names
        chart_labels = []
        for domain_code in domain_order:
            domain_name = self.domain_mapping.get(domain_code, domain_code)
            short_label = f'{domain_code}' if len(domain_name) > 30 else f'{domain_code}: {domain_name[:20]}...'
            chart_labels.append(short_label)
        
        # Generate unique chart ID
        import time
        chart_id = f"specialCasesChart_{int(time.time() * 1000) % 100000}"
        
        # Join datasets with proper formatting
        datasets_joined = ',\n            '.join(datasets_json)
        
        # Calculate maximum stack total to determine appropriate stepSize and max
        max_stack_total = 0
        for domain_code in domain_order:
            stack_total = 0
            for case_type in special_case_types:
                if domain_code in special_cases_by_domain:
                    stack_total += special_cases_by_domain[domain_code].get(case_type, 0)
            max_stack_total = max(max_stack_total, stack_total)
        
        # Determine appropriate stepSize based on maximum value
        if max_stack_total <= 16:
            step_size = 2
        elif max_stack_total <= 24:
            step_size = 3
        elif max_stack_total <= 32:
            step_size = 4
        elif max_stack_total <= 40:
            step_size = 5
        elif max_stack_total <= 80:
            step_size = 10
        elif max_stack_total <= 160:
            step_size = 20
        elif max_stack_total <= 200:
            step_size = 25
        else:
            step_size = 30
        
        # Calculate y-axis max: round up to next step interval and add one step for better visibility
        y_axis_max = (((max_stack_total // step_size) + 1) * step_size) + step_size
                
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
                        text: 'Domain of Change'
                    }}
                }},
                y: {{
                    stacked: true,
                    min: 0,
                    max: {y_axis_max},
                    title: {{
                        display: true,
                        text: 'Number of Outcomes'
                    }},
                    ticks: {{
                        stepSize: {step_size},
                        callback: function(value) {{
                            return Number.isInteger(value) ? value : '';
                        }}
                    }}
                }}
            }},
            plugins: {{
                title: {{
                    display: true,
                    text: 'Special Case Outcomes by Domain of Change'
                }},
                legend: {{
                    display: true,
                    position: 'bottom'
                }},
                tooltip: {{
                    callbacks: {{
                        afterLabel: function(context) {{
                            const domainCode = context.label.split(':')[0];
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
        
        # Calculate overall totals for summary
        special_cases_totals = {}
        for col in special_cases_cols:
            if col in self.outcomes_df.columns:
                count = (self.outcomes_df[col] == 1).sum()
                if count > 0:
                    clean_name = col.replace('Special case - ', '')
                    special_cases_totals[clean_name] = count
        
        if not special_cases_totals:
            return ""
        
        html = "<h3>Theme-Wide Special Cases</h3>\n"
        
        # Overall summary list
        html += "<ul>\n"
        for case_type, count in sorted(special_cases_totals.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            html += f"<li><strong>{case_type}:</strong> {count} ({percentage:.1f}%)</li>\n"
        html += "</ul>\n\n"
        
        # Generate the stacked bar chart using the dedicated method
        html += self.generate_special_cases_stacked_chart(special_cases_by_domain, domain_totals, special_cases_totals, total_outcomes)
        
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
            background-color: rgba(220, 53, 69, 0.8);
            border: 2px solid #dc3545;
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
        
        # Show country distribution (without header, directly under the map)
        html += "\n<ul>\n"
        for country, count in sorted(geo_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes * 100)
            country_region = country_to_region.get(country.lower(), "Other")
            html += f"<li><strong>{country}</strong> ({country_region}): {count} ({percentage:.1f}% of all outcomes)</li>\n"
        
        html += "</ul>\n\n"
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
                <div class="metric-label">Outcomes Gathered</div>
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
        
        # Add Venn diagram if it exists
        content_html += self.add_venn_diagram_if_exists()
        content_html += "<h3>Grant Analysis Overview</h3><p>Click on grant or sub-portfolio names in the table below to access more granular analysis pages.</p>\n"
        
        # Add matrix table
        content_html += self.create_grants_matrix_table()
        
        # Add sub-portfolio distribution
        content_html += "<h3>Sub-Portfolio Distribution</h3>\n"
        content_html += self.generate_subportfolio_distribution(consolidated_grants)
        
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
            
            # Navigation menu
            nav_menu = [
                {"text": "Back to Theme", "url": "../index.html"},
                {"text": "All Sub-Portfolios", "url": "../index.html#subportfolios"}
            ]
            
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
            if col in subportfolio_outcomes.columns:
                count = (subportfolio_outcomes[col] == 1).sum()
                if count > 0:
                    clean_name = col.replace('Special case - ', '')
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
                <span class="metric-value">{analysis['total_outcomes']}</span>
                <div class="metric-label">Total Outcomes</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['total_budget']:,.0f}</span>
                <div class="metric-label">Total Budget (CHF)</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['avg_budget']:,.0f}</span>
                <div class="metric-label">Average Grant Budget (CHF)</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">{analysis['avg_duration']:.1f}</span>
                <div class="metric-label">Average Grant Duration (months)</div>
            </div>
        </div>
        
        <h2>Consolidated Grants in Sub-Portfolio</h2>
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
        content_html += "<h3>Analysis of OB use across Sub-Portfolio's domains of change</h3>\n"
        content_html += self.generate_subportfolio_domain_analysis(analysis['outcomes'])
        
        # All outcome records
        content_html += self.generate_subportfolio_outcome_records(analysis['grants'], analysis['outcomes'])
        
        # Original simple grants table at the end
        content_html += """
        <h2>Grants Summary Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Grant Name</th>
                    <th>GAMS Code</th>
                    <th>Budget (CHF)</th>
                    <th>Duration</th>
                    <th>Countries</th>
                    <th>Outcomes</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Sort grants by name
        sorted_sp_grants = sorted(analysis['grants'].items(), key=lambda x: x[1]['short_name'])
        
        for gams_code, grant_info in sorted_sp_grants:
            grant_filename = f"{gams_code}_analysis.html"
            display_name = self.format_display_name(grant_info["short_name"])
            content_html += f"""
            <tr>
                <td><a href="../grants/{grant_filename}">{display_name}</a></td>
                <td>{grant_info.get('consolidated_gams', gams_code)}</td>
                <td>{grant_info['budget']}</td>
                <td>{grant_info['duration']} months</td>
                <td>{grant_info['countries']}</td>
                <td>{grant_info['outcomes_count']}</td>
            </tr>
            """
        
        content_html += """
            </tbody>
        </table>
        """
        
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
                        <th>Outcomes gathered</th>
                        <th>Additional sub-portfolios</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Generate rows for each grant
        for gams_code, grant_info in sorted_grants:
            # Grant name with GAMS code (matching theme format)
            grant_filename = f"{gams_code}_single_analysis.html"
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
                            <a href="{grant_filename}">{display_name}</a>
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

    def generate_subportfolio_domain_analysis(self, subportfolio_outcomes):
        """Generate domain-specific outcome basket analysis for subportfolio"""
        if subportfolio_outcomes.empty:
            return ""
        
        # Group outcomes by domain
        outcomes_by_domain = {}
        for _, outcome in subportfolio_outcomes.iterrows():
            domain = outcome.get('Domain of Change')
            if pd.notna(domain) and str(domain).strip():
                domain_key = str(domain).strip()
                if domain_key not in outcomes_by_domain:
                    outcomes_by_domain[domain_key] = []
                outcomes_by_domain[domain_key].append(outcome)
        
        html = "<h2>Domain-Specific Outcome Basket Analysis</h2>\n"
        
        # Generate analysis for each domain
        for domain_full_text in sorted(outcomes_by_domain.keys(), key=lambda x: len(outcomes_by_domain[x]), reverse=True):
            domain_outcomes = outcomes_by_domain[domain_full_text]
            outcome_count = len(domain_outcomes)
            
            # Extract domain key (e.g., D1 from "D1 - Long name")
            domain_key = domain_full_text.split(' - ')[0] if ' - ' in domain_full_text else domain_full_text
            
            html += f"<h3>{domain_full_text} ({outcome_count} outcomes)</h3>\n"
            
            # Get default OBs for this domain
            default_obs_for_domain = self.default_obs.get(domain_key, [])
            
            # Classify all OBs used in this domain
            used_default_obs = {}
            cross_domain_obs = {}
            custom_obs = {}
            
            for outcome in domain_outcomes:
                ob_names = outcome.get('Outcome Basket names')
                if pd.notna(ob_names):
                    ob_list = [ob.strip() for ob in str(ob_names).split(';') if ob.strip()]
                    for ob_str in ob_list:
                        if ob_str in default_obs_for_domain:
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
                                ob_label = f"{ob_str} ({source_domain})"
                                if ob_label not in cross_domain_obs:
                                    cross_domain_obs[ob_label] = 0
                                cross_domain_obs[ob_label] += 1
                            else:
                                if ob_str not in custom_obs:
                                    custom_obs[ob_str] = 0
                                custom_obs[ob_str] += 1
            
            # Usage of default Outcome Baskets
            if default_obs_for_domain:
                used_count = len(used_default_obs)
                total_count = len(default_obs_for_domain)
                usage_percentage = (used_count / total_count * 100) if total_count > 0 else 0
                
                html += f"<h4>Default OB for {domain_key} Domain (Usage: {used_count} of {total_count} - {usage_percentage:.1f}%)</h4>\n"
                
                if used_default_obs:
                    html += "<strong>Used Default OB:</strong>\n<ul>\n"
                    for ob, count in sorted(used_default_obs.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                        html += f'<li>"{ob}": {count} uses ({percentage:.1f}% of {domain_key} outcomes)</li>\n'
                    html += "</ul>\n"
                
                # Unused default OBs
                unused_obs = [ob for ob in default_obs_for_domain if ob not in used_default_obs]
                if unused_obs:
                    html += f"<strong>Unused Default OB ({len(unused_obs)} of {total_count}):</strong>\n<ul>\n"
                    for ob in unused_obs:
                        html += f'<li>"{ob}"</li>\n'
                    html += "</ul>\n"
            
            # Cross-Domain OBs
            if cross_domain_obs:
                total_cross_domain = sum(cross_domain_obs.values())
                cross_domain_percentage = (total_cross_domain / outcome_count * 100) if outcome_count > 0 else 0
                
                html += f"<h4>Cross-Domain OB Used in {domain_key} Outcomes ({cross_domain_percentage:.1f}% cross-domain usage)</h4>\n<ul>\n"
                for ob_label, count in sorted(cross_domain_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li>"{ob_label}": {count} uses ({percentage:.1f}%)</li>\n'
                html += "</ul>\n"
            
            # Custom OBs
            if custom_obs:
                total_custom = sum(custom_obs.values())
                custom_percentage = (total_custom / outcome_count * 100) if outcome_count > 0 else 0
                
                html += f"<h4>Custom OB in {domain_key} Outcomes ({custom_percentage:.1f}% custom usage)</h4>\n<ul>\n"
                for ob, count in sorted(custom_obs.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / outcome_count * 100) if outcome_count > 0 else 0
                    html += f'<li>"{ob}": {count} uses ({percentage:.1f}% of {domain_key} outcomes)</li>\n'
                html += "</ul>\n"
            
            html += "<hr>\n"
        
        return html

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
        html += "<p>Distribution of outcome significance levels across domains within this sub-portfolio.</p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_significance_stacked_chart(significance_by_domain, domain_totals, overall_significance, total_outcomes)
        
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
                        if pd.notna(case_value) and str(case_value).strip().lower() == 'x':
                            # Clean up the case name (remove "Special case - " prefix)
                            case_name = case_col.replace('Special case - ', '')
                            
                            # Initialize case if not seen before
                            if case_name not in special_cases_by_domain[domain_code]:
                                special_cases_by_domain[domain_code][case_name] = 0
                            
                            # Count this special case
                            special_cases_by_domain[domain_code][case_name] += 1
        
        # Calculate overall special cases totals
        overall_special_cases = {}
        for domain_data in special_cases_by_domain.values():
            for case_name, count in domain_data.items():
                if case_name not in overall_special_cases:
                    overall_special_cases[case_name] = 0
                overall_special_cases[case_name] += count
        
        # Only show section if there are special cases
        if not overall_special_cases:
            return ""
        
        html = "<h3>Sub-Portfolio Special Cases</h3>\n"
        html += "<p>Distribution of special case outcomes across domains within this sub-portfolio.</p>\n"
        
        # Generate the stacked bar chart
        html += self.generate_special_cases_stacked_chart(special_cases_by_domain, domain_totals, overall_special_cases, total_outcomes)
        
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
            background-color: rgba(220, 53, 69, 0.8);
            border: 2px solid #dc3545;
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
    <div style="max-width: 1000px; margin: 0 auto; position: relative;">
        <canvas id="{canvas_id}" width="1000" height="500"></canvas>{global_indicator_html}
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
        
        html = "<h3>Top Custom OB - with potential to become default</h3>\n"
        html += "<p>This sub-portfolio analysis shows custom outcome baskets that could potentially become default baskets in the theme-wide categorization.</p>\n"
        html += "<ul>\n"
        
        # Show top custom OBs by frequency
        for ob_name, count in sorted(custom_obs_usage.items(), key=lambda x: x[1], reverse=True)[:6]:  # Top 6
            percentage = (count / total_outcomes * 100) if total_outcomes > 0 else 0
            html += f'<li><strong>{ob_name}</strong>: {count} uses (tagged in {percentage:.1f}% of the outcomes)</li>\n'
        html += "</ul>\n"
        
        html += "\n"
        return html

    def generate_subportfolio_outcome_records(self, sp_grants, subportfolio_outcomes):
        """Generate all outcome records section for subportfolio"""
        html = "<h2>All Outcome Records from Sub-Portfolio Grants</h2>\n"
        
        # Group outcomes by grant
        outcomes_by_grant = {}
        for _, outcome in subportfolio_outcomes.iterrows():
            gams_code = str(outcome.get('GAMS code', ''))
            if gams_code not in outcomes_by_grant:
                outcomes_by_grant[gams_code] = []
            outcomes_by_grant[gams_code].append(outcome)
        
        outcome_counter = 1
        
        # Process each grant
        for gams_code, grant_info in sp_grants.items():
            # Check both main grant and phase 1 if consolidated
            grant_gams_codes = [gams_code]
            if grant_info['is_consolidated'] and grant_info['phase1_code']:
                grant_gams_codes.append(grant_info['phase1_code'])
            
            grant_display_name = self.format_display_name(grant_info['short_name'])
            phase_info = "(2-phase)" if grant_info['is_consolidated'] else "(single-phase)"
            consolidated_gams = grant_info.get('consolidated_gams', gams_code)
            
            html += f"<h4>Grant: {grant_display_name} {phase_info} (GAMS: {consolidated_gams})</h4>\n"
            
            # Get outcomes for this grant (including phase 1)
            grant_outcomes = []
            for code in grant_gams_codes:
                if code in outcomes_by_grant:
                    grant_outcomes.extend(outcomes_by_grant[code])
            
            # Sort outcomes by report submission date
            grant_outcomes.sort(key=lambda x: x.get('Report submission date', ''), reverse=True)
            
            for outcome in grant_outcomes:
                html += f"<h3>Outcome {outcome_counter}</h3>\n"
                html += '<div class="outcome-record">\n'
                
                # Basic outcome information
                description = outcome.get('Description of the Reported Outcome', 'N/A')
                html += f"<p><strong>Description of the Reported Outcome:</strong> {description}</p>\n"
                
                domain = outcome.get('Domain of Change', 'N/A')
                html += f"<p><strong>Domain of Change:</strong> {domain}</p>\n"
                
                ob_names = outcome.get('Outcome Basket names', 'N/A')
                html += f"<p><strong>Outcome Basket names:</strong> {ob_names}</p>\n"
                
                significance = outcome.get('Significance Level', 'N/A')
                if pd.notna(significance):
                    html += f"<p><strong>Significance Level:</strong> {significance}</p>\n"
                
                countries = outcome.get('Country(ies) / Global', 'N/A')
                html += f"<p><strong>Country(ies) / Global:</strong> {countries}</p>\n"
                
                # Additional details if available
                additional_details = outcome.get('Additional details on the Outcome')
                if pd.notna(additional_details) and str(additional_details).strip():
                    html += f"<p><strong>Additional details on the Outcome:</strong> {additional_details}</p>\n"
                
                # Report information
                report_name = outcome.get('Report name', 'N/A')
                html += f"<p><strong>Report name:</strong> {report_name}</p>\n"
                
                report_date = outcome.get('Report submission date', 'N/A')
                html += f"<p><strong>Report submission date:</strong> {report_date}</p>\n"
                
                # Special cases
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
                    html += f"<p><strong>Special Cases:</strong> {', '.join(special_cases)}</p>\n"
                
                html += '</div>\n'
                outcome_counter += 1
        
        return html

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
            short_name = grant_row.get("Grant's short name", 'N/A')
            
            content_html = f"""
            <h2>Grant Overview</h2>
            <ul>
                <li><strong>Primary Grant ID:</strong> {gams_code}</li>
                <li><strong>Grant Name:</strong> {grant_name}</li>
                <li><strong>Short Name:</strong> {short_name}</li>
                <li><strong>Budget (CHF):</strong> {grant_info['budget']}</li>
                <li><strong>Duration (months):</strong> {grant_info['duration']}</li>
                <li><strong>Closure Date:</strong> {grant_info['end_date_display']}</li>
                <li><strong>Countries:</strong> {grant_info['countries']}</li>
                <li><strong>Consolidated Grant:</strong> {'Yes' if grant_info['is_consolidated'] else 'No'}</li>
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
            
            # Navigation menu
            nav_menu = [
                {"text": "Back to Theme", "url": "../index.html"},
                {"text": "All Grants", "url": "../index.html#grants"}
            ]
            
            # Create HTML page
            html_page = self.create_html_template(
                title=title,
                content=content_html,
                breadcrumb=breadcrumb,
                nav_menu=nav_menu
            )
            
            # Generate filename
            output_filename = f"{gams_code}_analysis.html"
            
            # Write file
            output_file = self.html_output_path / "grants" / output_filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_page)
        
        print(f"✅ Generated {len(consolidated_grants)} enhanced individual grant pages")

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
