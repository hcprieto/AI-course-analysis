#!/usr/bin/env python3
"""
HTML Report Generator for Digital Grants Analysis
Created: 2025.07.23.19.20

This script converts markdown analysis reports to interactive HTML format with:
1. Professional CSS styling
2. Interactive navigation system
3. Grants vs Sub-portfolios matrix table
4. Cross-linking between reports
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from pathlib import Path

class HTMLReportGenerator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.analysis_path = self.base_path / "analysis-output-files"
        self.html_output_path = self.analysis_path / "html_reports"
        
        # Load data for matrix table generation
        self.grants_df = pd.read_csv(self.base_path / "source-files/FBs-Digital-Grants-Data.v5.3.csv")
        self.outcomes_df = pd.read_csv(self.base_path / "source-files/FB-Digital-Grants-Outcomes-Data.v2.csv")
        
        # Find latest analysis files
        self.find_latest_analysis_files()
        
    def find_latest_analysis_files(self):
        """Find the most recent analysis files"""
        # Find theme file
        theme_files = list(self.analysis_path.glob("*digital_theme_enhanced_with_outcomes.md"))
        self.theme_file = max(theme_files, key=os.path.getctime) if theme_files else None
        
        # Find sub-portfolio directory
        subportfolio_dirs = [p for p in self.analysis_path.iterdir() if p.is_dir() and "subportfolio" in p.name]
        self.subportfolio_dir = max(subportfolio_dirs, key=os.path.getctime) if subportfolio_dirs else None
        
        # Find individual grants directory
        grants_dirs = [p for p in self.analysis_path.iterdir() if p.is_dir() and "individual_grant" in p.name]
        self.grants_dir = max(grants_dirs, key=os.path.getctime) if grants_dirs else None
        
    def create_css_stylesheet(self):
        """Generate professional CSS stylesheet"""
        css_content = """
/* Digital Grants Analysis HTML Reports - Professional Styling */

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
}

.header h1 {
    color: #2c5aa0;
    font-size: 2.2em;
    margin-bottom: 10px;
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
h2 { font-size: 1.8em; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }
h3 { font-size: 1.4em; }
h4 { font-size: 1.2em; }

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
        return css_content
    
    def markdown_to_html(self, markdown_content):
        """Convert markdown content to HTML"""
        html = markdown_content
        
        # Fix escaped apostrophes first
        html = html.replace(r"Theme\'s", "Theme's")
        
        # Convert headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
        
        # Convert bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Convert lists
        lines = html.split('\n')
        in_list = False
        html_lines = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                content = line.strip()[2:]  # Remove '- '
                html_lines.append(f'<li>{content}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        
        if in_list:
            html_lines.append('</ul>')
        
        html = '\n'.join(html_lines)
        
        # Convert paragraphs
        paragraphs = html.split('\n\n')
        html_paragraphs = []
        
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith('<'):
                p = f'<p>{p}</p>'
            html_paragraphs.append(p)
        
        return '\n\n'.join(html_paragraphs)
    
    def create_html_template(self, title, content, breadcrumb=None, nav_menu=None):
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
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Digital Grants Analysis</title>
    <link rel="stylesheet" href="../css/analysis_styles.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="subtitle">Digital Theme Grants Analysis - Interactive Report</p>
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
                <a href="../index.html">Back to Theme Overview</a>
            </p>
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
            
        print("✅ HTML directory structure and CSS created")

    def get_consolidated_grants_data(self):
        """Get consolidated grants data for the matrix table, using CSV as primary source"""
        consolidated_grants = {}
        
        # Sub-portfolios mapping
        subportfolios = ['Digital Health', 'Digital Rights & Governance', 'Digital Literacy', 'Digital Innovation']
        
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
        processed_grants = set()
        
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
                        display_name = display_name.replace('_', ' ') + ' (consolidated)'
                    break
            
            # Get budget and duration (sum if consolidated)
            budget = 0
            duration = 0
            
            if is_consolidated and phase1_code:
                # Sum budget and duration from both phases
                phase1_row = self.grants_df[self.grants_df['GAMS Code'] == phase1_code]
                if not phase1_row.empty:
                    phase1_budget = str(phase1_row.iloc[0]['Budget (CHF)']).replace(',', '').replace(' ', '')
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
            current_budget = str(row['Budget (CHF)']).replace(',', '').replace(' ', '')
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
            duration_display = f"{int(duration)}" if duration > 0 else 'N/A'  # Remove decimals
            
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
            
            # Determine sub-portfolio memberships from CSV
            subportfolio_memberships = {}
            portfolio_cols = {
                'Digital Health sub-portfolio': 'Digital Health',
                'Digital Rights & Governance sub-portfolio': 'Digital Rights & Governance',
                'Digital Literacy sub-portfolio': 'Digital Literacy', 
                'Digital Innovation sub-portfolio': 'Digital Innovation'
            }
            
            for col, portfolio_name in portfolio_cols.items():
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip().upper() == 'X':
                    subportfolio_memberships[portfolio_name] = True
            
            # Find corresponding analysis file
            filename = None
            if self.grants_dir and self.grants_dir.exists():
                # Look for files starting with this GAMS code
                for grant_file in self.grants_dir.glob(f"{primary_code}_*.md"):
                    filename = grant_file.name
                    break
                
                # If not found and this is consolidated, look for phase 2 file
                if not filename and is_consolidated:
                    for grant_file in self.grants_dir.glob(f"{primary_code}_*consolidated*.md"):
                        filename = grant_file.name
                        break
            
            if not filename:
                # Generate a default filename if no analysis file exists
                analysis_type = 'consolidated_analysis' if is_consolidated else 'single_analysis'
                filename = f"{primary_code}_{display_name}_{analysis_type}.md"
            
            consolidated_grants[primary_code] = {
                'short_name': display_name,
                'budget': budget_display,
                'duration': duration_display,
                'countries': countries,
                'subportfolios': subportfolio_memberships,
                'filename': filename,
                'consolidated_gams': consolidated_gams,
                'is_consolidated': is_consolidated,
                'outcomes_count': outcomes_count
            }
            
            processed_grants.add(primary_code)
        
        return consolidated_grants, subportfolios
        
    def process_domain_distribution(self, domain_content):
        """Process domain distribution content to sort alphabetically and update percentage format"""
        lines = domain_content.strip().split('\n')
        domain_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('- **D') and ':' in line:
                # Extract domain and update percentage text, removing backslash before apostrophe
                updated_line = re.sub(r'\((\d+\.?\d*)% of Theme\\\'s outcomes\)', r'(\1% of Theme\'s outcomes)', line)
                updated_line = re.sub(r'\((\d+\.?\d*)% of Theme\'s outcomes\)', r'(\1% of Theme\'s outcomes)', updated_line)
                # Also handle simple percentages like "(18.0%)" to add "of Theme's outcomes"
                updated_line = re.sub(r'\((\d+\.?\d*)%\)(?! of)', r'(\1% of Theme\'s outcomes)', updated_line)
                domain_lines.append(updated_line)
            elif line:  # Keep non-domain lines as is
                # Also fix escaped apostrophes in other content
                fixed_line = line.replace(r"Theme\'s", "Theme's")
                domain_lines.append(fixed_line)
        
        # Sort domain lines alphabetically by domain number (D1, D2, D3, D4)
        domain_entries = []
        other_lines = []
        
        for line in domain_lines:
            if line.startswith('- **D'):
                # Extract domain number for sorting
                domain_match = re.match(r'- \*\*D(\d+)', line)
                if domain_match:
                    domain_num = int(domain_match.group(1))
                    domain_entries.append((domain_num, line))
                else:
                    other_lines.append(line)
            else:
                other_lines.append(line)
        
        # Sort by domain number
        domain_entries.sort(key=lambda x: x[0])
        
        # Reconstruct content
        sorted_lines = other_lines + [entry[1] for entry in domain_entries]
        return self.markdown_to_html('\n'.join(sorted_lines))
    
    def process_domain_analysis_sections(self, analysis_content):
        """Process domain analysis sections to sort them alphabetically by domain"""
        # Split into sections by ### headers
        sections = re.split(r'\n### (D\d+.*?)\n', analysis_content)
        
        if len(sections) < 2:
            return self.markdown_to_html(analysis_content)
        
        # First element is content before first section (if any)
        pre_content = sections[0].strip()
        
        # Group sections (header, content) pairs
        domain_sections = []
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i]
                content = sections[i + 1]
                
                # Extract domain number for sorting
                domain_match = re.match(r'D(\d+)', header)
                if domain_match:
                    domain_num = int(domain_match.group(1))
                    domain_sections.append((domain_num, header, content))
        
        # Sort by domain number
        domain_sections.sort(key=lambda x: x[0])
        
        # Reconstruct HTML - demote headers by one level (### becomes ####)
        html_content = ""
        if pre_content:
            html_content += self.markdown_to_html(pre_content) + "\n"
        
        for _, header, content in domain_sections:
            html_content += f"<h4>{header}</h4>\n"
            
            # Process content to remove percentages from domain headers and restructure OB lists
            processed_content = self.process_domain_ob_content(content)
            html_content += self.markdown_to_html(processed_content) + "\n"
        
        return html_content
    
    def process_domain_ob_content(self, content):
        """Process domain OB content to remove percentages and restructure lists"""
        # Change headers from various formats to "Usage of default Outcome Baskets"
        # Handle both with and without percentages
        content = re.sub(
            r'#### Default OBs for (D\d+) Domain \(Usage: (\d+ of \d+)(?: - [0-9.]+%)?\)',
            r'#### Usage of default Outcome Baskets',
            content
        )
        
        # Also handle the already processed format from previous runs
        content = re.sub(
            r'#### Default OBs for (D\d+) Domain',
            r'#### Usage of default Outcome Baskets',
            content
        )
        
        # Process the OB lists - remove bold headers and combine used/unused
        lines = content.split('\n')
        processed_lines = []
        in_ob_section = False
        used_obs = []
        unused_obs = []
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if we're entering an OB section
            if '#### Usage of default Outcome Baskets' in line:
                in_ob_section = True
                processed_lines.append(line)
                continue
            
            # Check for section headers within OB sections
            if in_ob_section and line_stripped.startswith('**Used Default OBs'):
                current_section = 'used'
                continue
            elif in_ob_section and line_stripped.startswith('**Unused Default OBs'):
                current_section = 'unused'
                continue
            
            # Process OB list items
            if in_ob_section and line_stripped.startswith('- "'):
                if current_section == 'used':
                    # Remove quotes and make OB name bold
                    updated_line = re.sub(r'- "([^"]+)": (.+)', r'- **\1**: \2', line)
                    used_obs.append(updated_line)
                elif current_section == 'unused':
                    # Convert unused items to show ": 0 occurrences" and remove quotes, make bold
                    if '": No occurrences' in line or '": 0 occurrences' in line:
                        # Already formatted correctly or close
                        updated_line = re.sub(r'- "([^"]+)": No occurrences', r'- **\1**: 0 occurrences', line)
                        updated_line = re.sub(r'- "([^"]+)": 0 occurrences', r'- **\1**: 0 occurrences', updated_line)
                        unused_obs.append(updated_line)
                    else:
                        # Add ": 0 occurrences" if not present and remove quotes, make bold
                        updated_line = re.sub(r'- "([^"]+)".*', r'- **\1**: 0 occurrences', line)
                        unused_obs.append(updated_line)
                continue
            
            # Check if we're leaving an OB section
            if in_ob_section and (line_stripped.startswith('####') or line_stripped.startswith('###') or line_stripped.startswith('##')):
                # Output combined list before starting new section
                if used_obs or unused_obs:
                    processed_lines.extend(used_obs)
                    processed_lines.extend(unused_obs)
                    used_obs = []
                    unused_obs = []
                in_ob_section = False
                current_section = None
                processed_lines.append(line)
                continue
            
            # Regular lines
            if not in_ob_section:
                processed_lines.append(line)
        
        # Handle any remaining OBs at the end
        if used_obs or unused_obs:
            processed_lines.extend(used_obs)
            processed_lines.extend(unused_obs)
        
        return '\n'.join(processed_lines)

    def generate_subportfolio_distribution(self, consolidated_grants):
        """Generate dynamic sub-portfolio distribution based on current grants data"""
        subportfolios = ['Digital Health', 'Digital Rights & Governance', 'Digital Literacy', 'Digital Innovation']
        total_grants = len(consolidated_grants)
        
        # Calculate total theme budget
        total_theme_budget = 0
        for grant_info in consolidated_grants.values():
            try:
                budget_str = grant_info['budget'].replace(',', '').replace(' ', '')
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
                        budget_str = grant_info['budget'].replace(',', '').replace(' ', '')
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
        
        # Enrich grant data with end dates from CSV for sorting
        enriched_grants = []
        for gams_code, grant_info in consolidated_grants.items():
            try:
                grant_row = self.grants_df[self.grants_df['GAMS Code'] == gams_code]
                if not grant_row.empty:
                    end_date_str = grant_row['Grant end'].iloc[0]
                    if pd.notna(end_date_str) and str(end_date_str).strip() and str(end_date_str) != 'nan':
                        # Parse end date for sorting (format like "Jan-28", "May-26", etc.)
                        try:
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
                                    # Create sortable date value (YYYYMM format for sorting)
                                    sort_date = year_full * 100 + month_num
                                    # Format for display
                                    display_date = f"{month_num:02d}/{year_full}"
                                    grant_info['end_date_sort'] = sort_date
                                    grant_info['end_date_display'] = display_date
                                else:
                                    grant_info['end_date_sort'] = 999999  # Put unknown dates at end
                                    grant_info['end_date_display'] = 'TBD'
                            else:
                                grant_info['end_date_sort'] = 999999
                                grant_info['end_date_display'] = 'TBD'
                        except:
                            grant_info['end_date_sort'] = 999999
                            grant_info['end_date_display'] = 'TBD'
                    else:
                        grant_info['end_date_sort'] = 999999
                        grant_info['end_date_display'] = 'TBD'
                else:
                    grant_info['end_date_sort'] = 999999
                    grant_info['end_date_display'] = 'TBD'
            except:
                grant_info['end_date_sort'] = 999999
                grant_info['end_date_display'] = 'TBD'
            
            enriched_grants.append((gams_code, grant_info))
        
        # Sort by end date (future dates first, descending order)
        enriched_grants.sort(key=lambda x: x[1]['end_date_sort'], reverse=True)
        
        # Start table
        html = '<div class="grants-matrix">\n<table>\n<thead>\n<tr>\n'
        html += '<th class="grant-name">Grant</th>\n'
        
        # Sub-portfolio headers (clickable)
        for sp in subportfolios:
            sp_filename = sp.lower().replace(' ', '_').replace('&', 'and')
            html += f'<th><a href="subportfolios/{sp_filename}.html">{sp}</a></th>\n'
        
        html += '<th>Budget (CHF)</th>\n<th>Duration (months)</th>\n<th>End Date</th>\n<th>Countries</th>\n<th>Outcomes gathered</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Grant rows
        total_budgets_by_portfolio = {sp: 0 for sp in subportfolios}
        total_budget_all = 0
        
        for gams_code, grant_info in enriched_grants:
            html += '<tr>\n'
            
            # Grant name (clickable)
            grant_filename = grant_info['filename'].replace('.md', '.html')
            # Replace underscores with spaces for display
            display_name = grant_info["short_name"].replace('_', ' ')
            html += f'<td class="grant-info">'
            html += f'<a href="grants/{grant_filename}">{display_name}</a>'
            html += f'<div class="grant-details">GAMS: {grant_info.get("consolidated_gams", gams_code)}</div>'
            html += f'</td>\n'
            
            # Sub-portfolio memberships and budget tracking
            for sp in subportfolios:
                if grant_info['subportfolios'].get(sp, False):
                    html += '<td class="portfolio-mark">✓</td>\n'
                    # Add budget to portfolio total (only once per grant, using first matching portfolio)
                    try:
                        budget_value = float(grant_info["budget"].replace(',', '').replace(' ', ''))
                        if sp == list(grant_info['subportfolios'].keys())[0]:  # Only count once per grant
                            total_budget_all += budget_value
                        if grant_info['subportfolios'].get(sp, False):
                            total_budgets_by_portfolio[sp] += budget_value
                    except (ValueError, IndexError):
                        pass
                else:
                    html += '<td></td>\n'
            
            # Metadata
            html += f'<td>{grant_info["budget"]}</td>\n'
            html += f'<td>{grant_info["duration"]}</td>\n'
            html += f'<td>{grant_info["end_date_display"]}</td>\n'
            html += f'<td>{grant_info["countries"]}</td>\n'
            html += f'<td>{grant_info["outcomes_count"]}</td>\n'
            html += '</tr>\n'
        
        # Totals row removed - budget information is provided in the sub-portfolio distribution section below
        
        html += '</tbody>\n</table>\n</div>\n'
        return html
        
    def generate_theme_index(self):
        """Generate the main theme index page with matrix table"""
        if not self.theme_file or not self.theme_file.exists():
            print("Warning: Theme file not found")
            return
            
        # Read theme analysis content
        with open(self.theme_file, 'r', encoding='utf-8') as f:
            theme_content = f.read()
        
        # Extract key sections
        overview_match = re.search(r'## Theme Overview\n(.*?)(?=\n##)', theme_content, re.DOTALL)
        subportfolio_dist_match = re.search(r'## Sub-Portfolio Distribution.*?\n(.*?)(?=\n##)', theme_content, re.DOTALL)
        domain_dist_match = re.search(r'## Theme-Wide Domain Distribution\n(.*?)(?=\n##)', theme_content, re.DOTALL)
        
        # Get current grant data for accurate metrics
        consolidated_grants, _ = self.get_consolidated_grants_data()
        total_grants = len(consolidated_grants)
        
        # Calculate total budget from consolidated grants
        total_budget = 0
        for grant_info in consolidated_grants.values():
            try:
                budget_str = grant_info['budget'].replace(',', '').replace(' ', '')
                if budget_str != 'N/A':
                    total_budget += float(budget_str)
            except:
                pass
        
        budget_display = f"{total_budget/1000000:.1f}M" if total_budget > 0 else "77.5M"  # fallback
        
        # Build HTML content - First show metrics, then navigation
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
                <span class="metric-value">4</span>
                <div class="metric-label">Sub-Portfolios</div>
            </div>
            <div class="metric-box">
                <span class="metric-value">250</span>
                <div class="metric-label">outcomes gathered</div>
            </div>
        </div>
        
        <h2>Overview of Grants and Sub-Portfolios</h2>
        """
        
        # Add Venn diagram visualization before the table description
        venn_diagram_path = self.html_output_path / "Digital Theme Grants Sub-Portfolios.VennDiagram.png"
        if venn_diagram_path.exists():
            content_html += """
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
        
        content_html += "<p>Click on grant names to view detailed analysis, or sub-portfolio headers to view portfolio summaries.</p>"
        
        # Add matrix table
        content_html += self.create_grants_matrix_table()
        
        # Add dynamic sub-portfolio distribution
        content_html += "<h2>Sub-Portfolio Distribution</h2>\n"
        content_html += self.generate_subportfolio_distribution(consolidated_grants)
            
        # Add domain distribution
        if domain_dist_match:
            content_html += "<h2>Outcomes gathered for the Theme's grants</h2>\n"
            content_html += "<h3>Distribution of outcomes across Theme's Domains of Change</h3>\n"
            content_html += self.process_domain_distribution(domain_dist_match.group(1))
            
            # Add detailed domain analysis sections as subsection
            domain_analysis_match = re.search(r'## Domain-Specific Outcome Basket Analysis(.*?)(?=## Custom OBs - Standardization Candidates|## Theme-Wide Geographic Distribution|$)', theme_content, re.DOTALL)
            if domain_analysis_match:
                content_html += "<h3>Analysis of Theme's Outcome Basket tagging by Domain of Change</h3>\n"
                content_html += self.process_domain_analysis_sections(domain_analysis_match.group(1))
            
        # Add custom OBs section
        custom_obs_match = re.search(r'## Custom OBs - Standardization Candidates(.*?)(?=## Theme-Wide Geographic Distribution|$)', theme_content, re.DOTALL)
        if custom_obs_match:
            content_html += "<h2>Custom OBs - Standardization Candidates</h2>\n"
            content_html += self.markdown_to_html(custom_obs_match.group(1))
            
        # Add geographic distribution
        geo_dist_match = re.search(r'## Theme-Wide Geographic Distribution(.*?)(?=## Theme-Wide Significance Patterns|$)', theme_content, re.DOTALL)
        if geo_dist_match:
            content_html += "<h2>Theme-Wide Geographic Distribution</h2>\n"
            content_html += self.markdown_to_html(geo_dist_match.group(1))
            
        # Add significance patterns
        sig_patterns_match = re.search(r'## Theme-Wide Significance Patterns(.*?)(?=## Theme-Wide Special Cases|$)', theme_content, re.DOTALL)
        if sig_patterns_match:
            content_html += "<h2>Theme-Wide Significance Patterns</h2>\n"
            content_html += self.markdown_to_html(sig_patterns_match.group(1))
            
        # Add special cases
        special_cases_match = re.search(r'## Theme-Wide Special Cases(.*?)$', theme_content, re.DOTALL)
        if special_cases_match:
            content_html += "<h2>Theme-Wide Special Cases</h2>\n"
            content_html += self.markdown_to_html(special_cases_match.group(1))
        
        # Navigation menu (Sub-Portfolios as bold text, not link) - appears after metrics grid, before Overview section
        nav_menu_html = """
        <div class="nav-menu">
            <ul>
                <li><strong>Sub-Portfolios:</strong></li>
                <li><a href="subportfolios/digital_health.html">Digital Health</a></li>
                <li><a href="subportfolios/digital_rights_and_governance.html">Digital Rights & Governance</a></li>
                <li><a href="subportfolios/digital_literacy.html">Digital Literacy</a></li>
                <li><a href="subportfolios/digital_innovation.html">Digital Innovation</a></li>
            </ul>
        </div>
        """
        
        # Insert navigation menu after the complete metrics-grid section but before the Overview heading
        overview_start = content_html.find('<h2>Overview of Grants and Sub-Portfolios</h2>')
        if overview_start != -1:
            content_html = content_html[:overview_start] + nav_menu_html + '\n        ' + content_html[overview_start:]
        
        # Create HTML page without nav_menu in template (we added it manually)
        html_content = self.create_html_template(
            title="Digital Theme Comprehensive Analysis",
            content=content_html,
            nav_menu=None
        )
        
        # Special handling for index page CSS path
        html_content = html_content.replace('../css/analysis_styles.css', 'css/analysis_styles.css')
        html_content = html_content.replace('../index.html', 'index.html')
        
        # Write file
        with open(self.html_output_path / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print("✅ Theme index page generated")
        
    def generate_subportfolio_pages(self):
        """Generate HTML pages for each sub-portfolio"""
        if not self.subportfolio_dir or not self.subportfolio_dir.exists():
            print("Warning: Sub-portfolio directory not found")
            return
        
        subportfolio_files = list(self.subportfolio_dir.glob("*.md"))
        
        for sp_file in subportfolio_files:
            # Read content
            with open(sp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title
            title_match = re.search(r'^# (.+)', content, re.MULTILINE)
            title = title_match.group(1) if title_match else sp_file.stem
            
            # Convert content to HTML
            html_content_body = self.markdown_to_html(content)
            
            # Create breadcrumb
            breadcrumb = [
                {"text": "Theme Overview", "url": "../index.html"},
                {"text": title}
            ]
            
            # Navigation menu
            nav_menu = [
                {"text": "Back to Theme", "url": "../index.html"},
                {"text": "All Sub-Portfolios", "url": "../index.html#subportfolios"}
            ]
            
            # Create HTML page
            html_page = self.create_html_template(
                title=title,
                content=html_content_body,
                breadcrumb=breadcrumb,
                nav_menu=nav_menu
            )
            
            # Generate filename
            sp_filename = sp_file.stem.lower().replace('_enhanced_analysis', '')
            
            # Write file
            output_file = self.html_output_path / "subportfolios" / f"{sp_filename}.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_page)
        
        print(f"✅ Generated {len(subportfolio_files)} sub-portfolio pages")
        
    def process_grant_content(self, content):
        """Process grant content to apply required formatting changes"""
        from datetime import datetime, timedelta
        import re
        
        # 1. Add closure date from Grant end column
        # Extract GAMS code to look up end date
        gams_match = re.search(r'\*\*Primary Grant ID:\*\* ([A-Z]+-[0-9]+-[0-9]+)', content)
        duration_match = re.search(r'- \*\*Duration \(months\):\*\* ([0-9.]+)', content)
        
        if duration_match and gams_match:
            gams_code = gams_match.group(1)
            
            # Look up end date in grants CSV
            closure_date_str = "TBD"  # Default
            
            try:
                # Find the grant in the CSV data
                grant_row = self.grants_df[self.grants_df['GAMS Code'] == gams_code]
                if not grant_row.empty:
                    end_date_str = grant_row['Grant end'].iloc[0]
                    if pd.notna(end_date_str) and str(end_date_str).strip() and str(end_date_str) != 'nan':
                        # Parse end date (format like "Jan-28", "May-26", etc.)
                        try:
                            # Convert format like "Jan-28" to "01/2028"
                            end_date_str = str(end_date_str).strip()
                            if '-' in end_date_str:
                                month_str, year_str = end_date_str.split('-')
                                
                                # Convert month name to number
                                month_map = {
                                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                                }
                                
                                if month_str in month_map:
                                    month_num = month_map[month_str]
                                    # Handle 2-digit years (assume 20xx)
                                    if len(year_str) == 2:
                                        year_full = f"20{year_str}"
                                    else:
                                        year_full = year_str
                                    
                                    closure_date_str = f"{month_num}/{year_full}"
                        except Exception as e:
                            print(f"Warning: Could not parse end date '{end_date_str}' for {gams_code}: {e}")
                            closure_date_str = "TBD"
            except Exception as e:
                print(f"Warning: Could not lookup end date for {gams_code}: {e}")
                closure_date_str = "TBD"
            
            closure_line = f"- **Closure Date:** {closure_date_str}"
            
            # Insert closure date after duration line
            content = re.sub(
                r'(- \*\*Duration \(months\):\*\* [0-9.]+)',
                r'\1\n' + closure_line,
                content
            )
        
        # 2. Sort Domain sections alphabetically (D1, D2, D3, D4)
        # Extract all domain sections
        domain_sections = []
        remaining_content = content
        
        # Find all domain sections
        domain_pattern = r'(### D[1-4] - [^#]+?(?=### D[1-4] -|### Cross-Domain Usage Summary|## Significance Level Distribution|$))'
        matches = list(re.finditer(domain_pattern, content, re.DOTALL))
        
        if matches:
            # Extract sections and sort them
            for match in matches:
                section_content = match.group(1)
                domain_num = int(re.search(r'D([1-4])', section_content).group(1))
                domain_sections.append((domain_num, section_content))
            
            # Sort by domain number
            domain_sections.sort(key=lambda x: x[0])
            
            # Replace the domain sections in order
            if domain_sections:
                # Find the start of domain sections
                first_domain_start = matches[0].start()
                last_domain_end = matches[-1].end()
                
                # Reconstruct content with sorted domains
                before_domains = content[:first_domain_start]
                after_domains = content[last_domain_end:]
                
                sorted_domains_text = '\n'.join([section[1] for section in domain_sections])
                content = before_domains + sorted_domains_text + after_domains
        
        # 3. Remove usage percentage from domain headers
        content = re.sub(
            r'#### Default OBs for D([1-4]) Domain \(Usage: \d+ of \d+ - [0-9.]+%\)',
            r'#### Default OBs for D\1 Domain',
            content
        )
        
        # 4. Update "Used Default OBs" format
        # Find patterns like "**Used Default OBs:**" and add count information
        def replace_used_obs_header(match):
            domain_section = match.group(0)
            
            # Count the number of used OBs (lines starting with "- ")
            used_obs_lines = re.findall(r'^- "[^"]+": \d+ occurrences', domain_section, re.MULTILINE)
            used_count = len(used_obs_lines)
            
            # Count total outcome baskets tagged by summing all occurrences
            total_tagged = 0
            for line in used_obs_lines:
                occurrence_match = re.search(r': (\d+) occurrences', line)
                if occurrence_match:
                    total_tagged += int(occurrence_match.group(1))
            
            # Find the "Unused Default OBs" line to get total available
            unused_match = re.search(r'\*\*Unused Default OBs \((\d+) of (\d+)\)\:\*\*', domain_section)
            if unused_match:
                total_available = int(unused_match.group(2))
            else:
                # Try to infer from context or use default
                total_available = used_count  # fallback
            
            # Replace the header
            new_header = f"**Used Default OBs ({used_count} of {total_available}, total of {total_tagged} Outcome Basket tagged):**"
            
            return domain_section.replace("**Used Default OBs:**", new_header)
        
        # Apply to each domain section
        content = re.sub(
            r'#### Default OBs for D[1-4] Domain.*?(?=#### |### |## |$)',
            replace_used_obs_header,
            content,
            flags=re.DOTALL
        )
        
        # 5. Change "X% of DX outcomes" to "tagged in X% of DX outcomes"
        content = re.sub(
            r'(\d+(?:\.\d+)?% of D[1-4] outcomes)',
            r'tagged in \1',
            content
        )
        
        # 6. Fix Performance Ratings section
        gams_match = re.search(r'\*\*Primary Grant ID:\*\* ([A-Z]+-[0-9]+-[0-9]+)', content)
        if gams_match:
            gams_code = gams_match.group(1)
            
            try:
                # Find the grant in the CSV data
                grant_row = self.grants_df[self.grants_df['GAMS Code'] == gams_code]
                if not grant_row.empty:
                    # Performance rating columns (correct ones)
                    perf_columns = [
                        ('Implementation.1', 'Implementation'),
                        ('Contribution to strategic intent ', 'Contribution to strategic intent'),
                        ('Learning ', 'Learning'), 
                        ('Sustainability of changes', 'Sustainability of changes'),
                        ('Stakeholder engagement and collaboration', 'Stakeholder engagement and collaboration'),
                        ('Meaningful youth participation', 'Meaningful youth participation')
                    ]
                    
                    # Build new performance ratings section
                    perf_lines = ["## Performance Ratings"]
                    for csv_col, display_name in perf_columns:
                        if csv_col in self.grants_df.columns:
                            value = grant_row[csv_col].iloc[0]
                            if pd.notna(value) and str(value).strip() and str(value).strip().lower() != 'nan':
                                perf_lines.append(f"- **{display_name}:** {str(value).strip()}")
                            else:
                                perf_lines.append(f"- **{display_name}:** N/A")
                        else:
                            perf_lines.append(f"- **{display_name}:** N/A")
                    
                    new_perf_section = '\n'.join(perf_lines)
                    
                    # Replace the existing Performance Ratings section
                    perf_pattern = r'## Performance Ratings\n.*?(?=\n## |$)'
                    content = re.sub(perf_pattern, new_perf_section, content, flags=re.DOTALL)
                    
            except Exception as e:
                print(f"Warning: Could not update performance ratings for {gams_code}: {e}")
        
        return content

    def generate_grant_pages(self):
        """Generate HTML pages for individual grants"""
        if not self.grants_dir or not self.grants_dir.exists():
            print("Warning: Grants directory not found")
            return
            
        grant_files = list(self.grants_dir.glob("*.md"))
        
        # Filter out phase 1 grants that have corresponding phase 2 consolidated grants
        # Build consolidation mapping from CSV
        consolidation_map = {}  # Maps phase 1 -> phase 2
        for _, row in self.grants_df.iterrows():
            primary_code = str(row['GAMS Code'])
            if pd.notna(row['GAMS Code']) and str(row['GAMS Code']) != 'nan':
                if pd.notna(row['GAMS code - next phase']) and str(row['GAMS code - next phase']) != 'nan':
                    next_phase = str(row['GAMS code - next phase'])
                    consolidation_map[primary_code] = next_phase
        
        # Filter grant files to exclude phase 1 grants that are consolidated
        filtered_grant_files = []
        excluded_count = 0
        
        for grant_file in grant_files:
            grant_name = grant_file.stem
            parts = grant_name.split('_')
            if parts:
                gams_code = parts[0]
                
                # Skip if this is a phase 1 that has a phase 2 (i.e., is consolidated)
                if gams_code in consolidation_map:
                    # Check if the phase 2 actually exists in our files
                    phase2_code = consolidation_map[gams_code]
                    phase2_exists = any(f.stem.startswith(phase2_code) for f in grant_files)
                    
                    if phase2_exists:
                        print(f"  Excluding phase 1 grant: {grant_file.name} (consolidated into {phase2_code})")
                        excluded_count += 1
                        continue
                
            filtered_grant_files.append(grant_file)
        
        print(f"Generating individual grant pages for {len(filtered_grant_files)} grants ({excluded_count} phase 1 grants excluded)...")
        
        for grant_file in filtered_grant_files:
            # Read content
            with open(grant_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process content with required changes
            processed_content = self.process_grant_content(content)
            
            # Extract title
            title_match = re.search(r'^# (.+)', processed_content, re.MULTILINE)
            title = title_match.group(1) if title_match else grant_file.stem
            
            # Convert content to HTML
            html_content_body = self.markdown_to_html(processed_content)
            
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
                content=html_content_body,
                breadcrumb=breadcrumb,
                nav_menu=nav_menu
            )
            
            # Generate filename
            output_filename = grant_file.name.replace('.md', '.html')
            
            # Write file
            output_file = self.html_output_path / "grants" / output_filename
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_page)
        
        print(f"✅ Generated {len(filtered_grant_files)} individual grant pages")
        if excluded_count > 0:
            print(f"ℹ️  Excluded {excluded_count} phase 1 grant pages (consolidated into phase 2 grants)")

    def generate_all_reports(self):
        """Generate all HTML reports"""
        print("Starting HTML report generation...")
        
        # Setup directory structure
        self.setup_html_structure()
        
        # Generate theme index page
        self.generate_theme_index()
        
        # Generate sub-portfolio pages
        self.generate_subportfolio_pages()
        
        # Generate individual grant pages
        self.generate_grant_pages()
        
        print("✅ All HTML reports generated successfully!")
        print(f"📁 Reports available at: {self.html_output_path}")
        print(f"🌐 Open in browser: file://{self.html_output_path / 'index.html'}")

if __name__ == "__main__":
    base_path = "/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev"
    
    generator = HTMLReportGenerator(base_path)
    generator.generate_all_reports()