#!/usr/bin/env python3
"""
Missing Grants Analysis File Generator

This script identifies grants from the CSV that don't have corresponding analysis files
and generates basic analysis files for them so they can be included in the HTML reports.
"""

import pandas as pd
from pathlib import Path
import re
from datetime import datetime

class MissingGrantsGenerator:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.source_files_path = self.base_path / "source-files"
        self.grants_csv_path = self.source_files_path / "FBs-Digital-Grants-Data.v5.3.csv"
        self.outcomes_csv_path = self.source_files_path / "FB-Digital-Grants-Outcomes-Data.v2.csv"
        self.existing_grants_dir = self.base_path / "analysis-output-files" / "2025.07.23.13.34_individual_grant_analyses_with_outcomes"
        
        # Load data
        self.grants_df = pd.read_csv(self.grants_csv_path)
        self.outcomes_df = pd.read_csv(self.outcomes_csv_path)
        
    def analyze_missing_grants(self):
        """Analyze which grants are missing from the analysis files"""
        print("🔍 Analyzing missing grants...")
        
        # Get all grants from CSV
        csv_grants = set(self.grants_df['GAMS Code'].dropna().astype(str))
        print(f"📊 Total grants in CSV: {len(csv_grants)}")
        
        # Get existing analysis files
        existing_files = list(self.existing_grants_dir.glob("*.md"))
        existing_grants = set()
        
        for file in existing_files:
            # Extract GAMS code from filename
            parts = file.stem.split('_')
            if parts:
                gams_code = parts[0]
                existing_grants.add(gams_code)
        
        print(f"📝 Existing grant analysis files: {len(existing_grants)}")
        
        # Find missing grants
        missing_grants = csv_grants - existing_grants
        print(f"❌ Missing grants: {len(missing_grants)}")
        
        if missing_grants:
            print("Missing GAMS codes:")
            for code in sorted(missing_grants):
                print(f"  - {code}")
        
        return missing_grants
    
    def identify_consolidation_opportunities(self, missing_grants):
        """Identify grants that should be consolidated with existing ones"""
        consolidations = {}
        
        # Check for phase relationships in the CSV
        for _, row in self.grants_df.iterrows():
            primary_code = str(row['GAMS Code'])
            if pd.notna(row['GAMS code - next phase']):
                next_phase = str(row['GAMS code - next phase'])
                
                # If we have the next phase but not the primary, or vice versa
                if primary_code in missing_grants and next_phase not in missing_grants:
                    consolidations[primary_code] = next_phase
                elif next_phase in missing_grants and primary_code not in missing_grants:
                    consolidations[next_phase] = primary_code
        
        return consolidations
    
    def generate_missing_grant_file(self, gams_code):
        """Generate a basic analysis file for a missing grant"""
        # Get grant data from CSV
        grant_row = self.grants_df[self.grants_df['GAMS Code'] == gams_code]
        if grant_row.empty:
            print(f"⚠️  No data found for GAMS code: {gams_code}")
            return None
            
        row = grant_row.iloc[0]
        
        # Extract basic information
        short_name = str(row['Grant\'s short name']).replace(' ', '_') if pd.notna(row['Grant\'s short name']) else f"Grant_{gams_code}"
        full_name = str(row['Grant\'s Name']) if pd.notna(row['Grant\'s Name']) else "N/A"
        brief_summary = str(row['Brief Summary']) if pd.notna(row['Brief Summary']) else "N/A"
        countries = str(row['Countries']) if pd.notna(row['Countries']) else "N/A"
        budget = str(row['Budget (CHF)']).strip() if pd.notna(row['Budget (CHF)']) else "N/A"
        duration = str(row['Duration (months)']) if pd.notna(row['Duration (months)']) else "N/A"
        
        # Sub-portfolio assignments
        subportfolios = []
        portfolio_cols = [
            'Digital Health sub-portfolio',
            'Digital Rights & Governance sub-portfolio', 
            'Digital Literacy sub-portfolio',
            'Digital Innovation sub-portfolio'
        ]
        
        for col in portfolio_cols:
            if pd.notna(row[col]) and str(row[col]).strip().upper() == 'X':
                portfolio_name = col.replace(' sub-portfolio', '').replace('&', '& ')
                subportfolios.append(f"- {portfolio_name}")
        
        # Performance ratings
        performance_cols = [
            'Implementation', 'Contribution to strategic intent', 'Learning',
            'Sustainability of changes', 'Stakeholder engagement and collaboration',
            'Meaningful youth participation'
        ]
        
        performance_ratings = []
        for col in performance_cols:
            if col in row.index and pd.notna(row[col]):
                rating = str(row[col]).strip()
                if rating and rating != 'nan':
                    performance_ratings.append(f"- **{col}:** {rating}")
        
        # Check if grant has outcomes in the outcomes dataset
        has_outcomes = gams_code in self.outcomes_df['GAMS code'].astype(str).values
        outcomes_count = len(self.outcomes_df[self.outcomes_df['GAMS code'].astype(str) == gams_code]) if has_outcomes else 0
        
        # Generate markdown content
        content = f"""# Grant Analysis: {short_name.replace('_', ' ')}
**Primary Grant ID:** {gams_code}
**Full Grant Name:** {full_name}

## Grant Metadata
- **Budget (CHF):** {budget}
- **Duration (months):** {duration}
- **Countries:** {countries}
- **Total Outcomes:** {outcomes_count}

## Brief Summary
{brief_summary}

## Sub-Portfolio Assignments
{chr(10).join(subportfolios) if subportfolios else "- No sub-portfolio assignments found"}

## Performance Ratings
{chr(10).join(performance_ratings) if performance_ratings else "- No performance ratings available"}

## Outcomes Analysis
{"**Note:** This grant currently has no outcomes data gathered in the analysis system. This may indicate:" if not has_outcomes else f"**Note:** This grant has {outcomes_count} outcomes recorded."}

{"- The grant is still in early implementation phases" if not has_outcomes else ""}
{"- No reports have been submitted or analyzed yet" if not has_outcomes else ""}
{"- The grant may be completed but outcomes were not captured in the current dataset" if not has_outcomes else ""}

*This analysis file was automatically generated from CSV data on {datetime.now().strftime('%Y-%m-%d %H:%M')} to ensure complete grant coverage in the Digital Theme analysis.*
"""
        
        # Create filename
        filename = f"{gams_code}_{short_name}_single_analysis.md"
        file_path = self.existing_grants_dir / filename
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Generated analysis file: {filename}")
        return file_path
    
    def run(self):
        """Main execution method"""
        print("🚀 Starting Missing Grants Generator...")
        print(f"📁 Base path: {self.base_path}")
        print(f"📄 Grants CSV: {self.grants_csv_path}")
        print(f"📄 Outcomes CSV: {self.outcomes_csv_path}")
        print(f"📁 Analysis files directory: {self.existing_grants_dir}")
        
        # Analyze missing grants
        missing_grants = self.analyze_missing_grants()
        
        if not missing_grants:
            print("✅ No missing grants found! All grants have analysis files.")
            return
        
        # Check for consolidation opportunities
        consolidations = self.identify_consolidation_opportunities(missing_grants)
        if consolidations:
            print(f"\n🔗 Found {len(consolidations)} potential consolidations:")
            for missing, existing in consolidations.items():
                print(f"  - {missing} could be consolidated with existing {existing}")
        
        # Generate missing files
        print(f"\n📝 Generating {len(missing_grants)} missing analysis files...")
        generated_files = []
        
        for gams_code in sorted(missing_grants):
            try:
                file_path = self.generate_missing_grant_file(gams_code)
                if file_path:
                    generated_files.append(file_path)
            except Exception as e:
                print(f"❌ Error generating file for {gams_code}: {e}")
        
        print(f"\n✅ Successfully generated {len(generated_files)} analysis files")
        print(f"🎯 Total analysis files now available: {len(list(self.existing_grants_dir.glob('*.md')))}")

if __name__ == "__main__":
    base_path = "/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev"
    
    generator = MissingGrantsGenerator(base_path)
    generator.run()