#!/usr/bin/env python3
"""
Refined script to identify and anonymize actual personal data (PII) in FB-Digital-Grants-Outcomes-Data.csv file.
Creates a new CSV file with PII replaced by "[PII removed]" for review.

Focus on columns:
- Description of the Reported Outcome
- Additional details on the Outcome  
- Notes/insights from the miner

Uses refined patterns to reduce false positives while catching actual names.
"""

import pandas as pd
import re
import csv
import os
from datetime import datetime

def create_refined_anonymized_csv():
    """Create an anonymized version of the CSV file with actual PII replaced."""
    
    # File paths
    input_file = '../source-files/FB-Digital-Grants-Outcomes-Data.csv'
    output_file = '../analysis-output-files/FB-Digital-Grants-Outcomes-Data-ANONYMIZED-REFINED.csv'
    changes_file = '../analysis-output-files/PII_Anonymization_Changes_Refined.csv'
    summary_file = '../analysis-output-files/PII_Anonymization_Summary_Refined.md'
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Processing {len(df)} rows for refined PII anonymization...")
    
    # Columns to check for PII
    target_columns = [
        'Description of the Reported Outcome',
        'Additional details on the Outcome', 
        'Notes/insights from the miner'
    ]
    
    # Available columns in the file
    available_columns = [col for col in target_columns if col in df.columns]
    print(f"Checking columns: {available_columns}")
    
    # Known personal names from the existing analysis that should be anonymized
    known_personal_names = {
        'Ilona Kickbush', 'Hassan Mshinda', 'Faustine Ndugulile', 'Christoph Alberts', 
        'Mia Tonogbanua', 'Paola Garcia Rey', 'Gabriela Aguinaga Romero', 'María Gabriela Aguinaga Romero',
        'Emmanuel Reyes Carmona', 'Shannon Thom', 'Gina Romero', 'Deputy Director'
    }
    
    # Define refined PII patterns with stricter matching
    pii_patterns = {
        'email': {
            'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'description': 'Email address'
        },
        'phone': {
            'pattern': r'(\+\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,}',
            'description': 'Phone number'
        },
        'personal_id': {
            'pattern': r'\\b(?:ID|SSN|passport|license|student\\s*id|employee\\s*id|member\\s*id)\\s*[:#]?\\s*[A-Z0-9]{6,}\\b',
            'description': 'Personal identifier'
        },
        'social_media': {
            'pattern': r'@[A-Za-z0-9_]{3,}',
            'description': 'Social media handle'
        }
    }
    
    # Extensive exclusion list to prevent false positives
    exclusions = {
        # Organizations and institutions
        'Digital Health', 'Transform Health', 'Smart Africa', 'Africa CDC', 'World Bank', 
        'World Health', 'Global Fund', 'UN Special', 'Special Rapporteur', 'European Union',
        'United Nations', 'United States', 'South Africa', 'Central African', 'Ministry Health',
        'Health Ministry', 'Youth Council', 'Medical Students', 'International Federation',
        'Medical Association', 'Health Association', 'Research Institute', 'University College',
        'Public Health', 'Global Health', 'Digital Rights', 'Human Rights', 'Young Leaders',
        'Leadership Network', 'Innovation Hub', 'Policy Hub', 'Health Center', 'Community Health',
        'Digital Platform', 'Health Platform', 'Digital Initiative', 'Health Initiative',
        'Digital Future', 'Health Future', 'Digital Transformation', 'Health Transformation',
        'Digital Innovation', 'Health Innovation', 'Digital System', 'Health System',
        'Digital Technology', 'Health Technology', 'Digital Service', 'Health Service',
        'Digital Framework', 'Health Framework', 'Digital Strategy', 'Health Strategy',
        'Digital Policy', 'Health Policy', 'Digital Governance', 'Health Governance',
        'Digital Leadership', 'Health Leadership', 'Digital Education', 'Health Education',
        'Digital Training', 'Health Training', 'Digital Capacity', 'Health Capacity',
        'Digital Rights', 'Health Rights', 'Digital Ethics', 'Health Ethics',
        'Digital Standards', 'Health Standards', 'Digital Guidelines', 'Health Guidelines',
        'Digital Blueprint', 'Health Blueprint', 'Digital Summit', 'Health Summit',
        'Digital Forum', 'Health Forum', 'Digital Conference', 'Health Conference',
        'Digital Workshop', 'Health Workshop', 'Digital Report', 'Health Report',
        'Digital Research', 'Health Research', 'Digital Data', 'Health Data',
        'Digital Learning', 'Health Learning', 'Digital Network', 'Health Network',
        'Digital Alliance', 'Health Alliance', 'Digital Coalition', 'Health Coalition',
        'Digital Partnership', 'Health Partnership', 'Digital Collaboration', 'Health Collaboration',
        
        # Common organizational terms that might be mistaken for names
        'Vice President', 'Deputy Director', 'Project Manager', 'Program Manager',
        'Technical Director', 'Executive Director', 'Senior Manager', 'Principal Secretary',
        'Ministry ICT', 'Digital Champion', 'Youth Champion', 'Regional Champion',
        'Program Officer', 'Project Officer', 'Research Officer', 'Policy Officer',
        'Health Officer', 'Communications Officer', 'Data Officer', 'Innovation Officer',
        'Program Coordinator', 'Project Coordinator', 'Research Coordinator',
        'Training Coordinator', 'Youth Coordinator', 'Health Coordinator',
        'Technical Lead', 'Program Lead', 'Project Lead', 'Research Lead',
        'Innovation Lead', 'Digital Lead', 'Health Lead', 'Youth Lead',
        'Team Leader', 'Project Leader', 'Research Leader', 'Innovation Leader',
        'Program Director', 'Research Director', 'Innovation Director',
        'Health Director', 'Digital Director', 'Youth Director',
        
        # Countries and places
        'United States', 'United Kingdom', 'South Africa', 'Central African',
        'European Union', 'European Commission', 'African Union', 'East Africa',
        'West Africa', 'Southern Africa', 'North Africa', 'Sub Saharan',
        'Middle East', 'Asia Pacific', 'Latin America', 'North America',
        
        # Project and program names
        'Rights Click', 'Transform Health', 'Digital Health', 'Smart Africa',
        'Next Gen', 'Future Health', 'Health Tech', 'Tech Health',
        'Innovation Hub', 'Digital Hub', 'Health Hub', 'Youth Hub',
        'Digital First', 'Health First', 'Youth First', 'Innovation First',
        'Strategic Partnership', 'Global Partnership', 'Health Partnership',
        'Digital Partnership', 'Youth Partnership', 'Innovation Partnership',
        
        # False positive patterns from the data
        'The Commission', 'Lancet Comm', 'World Programme', 'Task Force',
        'Political Declaration', 'Legislative Assembly', 'General Assembly',
        'Quarterly Convening', 'Safer Internet Day', 'Privacy First Workshops',
        'Communications Authority', 'Broadband Commission', 'Short Courses',
        'Artificial Intelligence', 'Air Conditioners', 'Quality Assurance',
        'Family Planning Agency', 'Kisumu County Department', 'The Head',
        'Key Populations', 'Nairobi Counties', 'Tharaka Nithi', 'The Roadmap',
        'Key Asks', 'Engagement Toolkit', 'Resource Portal', 'Indian Presidency',
        'Outcome Indicator', 'Kisumu County Executive', 'The Indonesian',
        'Governance Principles', 'Eastern Mediterranean', 'Curious Minds',
        'State Violence Against', 'Communications Networks', 'Are Not Just',
        'Challenges Faced', 'Peaceful Assembly', 'Facebook Workspace',
        'Keeping Children Safe', 'Feel Exposed', 'Privacy First', 'Ley Olimpia',
        'The February', 'Zanzibar President', 'Transformation Agenda',
        'Presidential Delivery Bureau', 'Principal Secretaries', 'Delivery Bureau',
        'Transition Plan', 'Other Ministries', 'Malaria Elimination',
        'Zanzibar Outcome', 'Afya Target', 'The Presidential Delivery',
        'Jamini Afya', 'Jamii ni Afya', 'Afya Activities', 'Central African Republic',
        'The Union', 'Bertlesmann Stiftung', 'Healthy Societies', 'Empowering Young People',
        'Create Tools', 'Young Leaders', 'Although Instagram', 'Lancet Reports',
        'Special Rapporteur', 'Fondation Botnar', 'World Bank', 'Amnesty Movement',
        'Argentina Progress', 'Amnesty Argentina', 'Amnesty Philippines',
        'Amnesty International Board', 'Vice Chair'
    }
    
    # Convert exclusions to lowercase for case-insensitive matching
    exclusions_lower = {item.lower() for item in exclusions}
    
    # Track changes made
    changes_made = []
    anonymized_df = df.copy()
    
    # Process each target column
    for column in available_columns:
        if column not in df.columns:
            continue
            
        print(f"\nProcessing column: {column}")
        
        for idx, cell_value in enumerate(df[column]):
            if pd.isna(cell_value) or not isinstance(cell_value, str):
                continue
                
            original_text = str(cell_value)
            modified_text = original_text
            row_changes = []
            
            # First, check for known personal names (exact matches)
            for name in known_personal_names:
                if name in original_text:
                    modified_text = modified_text.replace(name, '[PII removed]')
                    row_changes.append({
                        'type': 'known_personal_name',
                        'description': 'Known personal name',
                        'original': name,
                        'context': original_text[max(0, original_text.find(name)-50):original_text.find(name)+len(name)+50]
                    })
            
            # Then check for other PII patterns (excluding name patterns to avoid false positives)
            for pattern_name, pattern_info in pii_patterns.items():
                pattern = pattern_info['pattern']
                description = pattern_info['description']
                
                # Find all matches
                matches = re.finditer(pattern, original_text, re.IGNORECASE)
                
                for match in matches:
                    matched_text = match.group()
                    
                    # Skip if it's in our exclusion list
                    if matched_text.lower() in exclusions_lower:
                        continue
                    
                    # Replace it
                    modified_text = modified_text.replace(matched_text, '[PII removed]')
                    row_changes.append({
                        'type': pattern_name,
                        'description': description,
                        'original': matched_text,
                        'context': original_text[max(0, match.start()-50):match.end()+50]
                    })
            
            # If changes were made, update the dataframe and record the changes
            if modified_text != original_text:
                anonymized_df.iloc[idx, anonymized_df.columns.get_loc(column)] = modified_text
                
                for change in row_changes:
                    changes_made.append({
                        'row_number': idx + 2,  # +2 because Excel/CSV rows start at 1 and we have header
                        'column': column,
                        'pii_type': change['type'],
                        'description': change['description'],
                        'original_text': change['original'],
                        'context': change['context']
                    })
    
    # Save the anonymized CSV
    anonymized_df.to_csv(output_file, index=False)
    print(f"\nAnonymized CSV saved to: {output_file}")
    
    # Save the changes report
    if changes_made:
        with open(changes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Row_Number', 'Column', 'PII_Type', 'Description', 'Original_Text', 'Context'])
            
            for change in changes_made:
                writer.writerow([
                    change['row_number'],
                    change['column'],
                    change['pii_type'],
                    change['description'],
                    change['original_text'],
                    change['context']
                ])
        
        print(f"Changes report saved to: {changes_file}")
    
    # Create summary report
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Refined PII Anonymization Summary\n\n")
        f.write(f"**Processing Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Original File:** {input_file}\n")
        f.write(f"**Anonymized File:** {output_file}\n")
        f.write(f"**Changes Report:** {changes_file}\n\n")
        f.write(f"## Processing Statistics\n\n")
        f.write(f"- **Total Rows Processed:** {len(df)}\n")
        f.write(f"- **Columns Checked:** {', '.join(available_columns)}\n")
        f.write(f"- **Total PII Instances Found:** {len(changes_made)}\n")
        f.write(f"- **Rows with PII:** {len(set(change['row_number'] for change in changes_made)) if changes_made else 0}\n\n")
        
        if changes_made:
            # Group by PII type
            type_counts = {}
            for change in changes_made:
                type_counts[change['pii_type']] = type_counts.get(change['pii_type'], 0) + 1
            
            f.write("## PII Types Found\n\n")
            for pii_type, count in sorted(type_counts.items()):
                f.write(f"- **{pii_type}:** {count} instances\n")
            
            # Group by column
            column_counts = {}
            for change in changes_made:
                column_counts[change['column']] = column_counts.get(change['column'], 0) + 1
            
            f.write("\n## Distribution by Column\n\n")
            for column, count in sorted(column_counts.items()):
                f.write(f"- **{column}:** {count} instances\n")
            
            f.write("\n## Specific Changes Made\n\n")
            for change in sorted(changes_made, key=lambda x: x['row_number']):
                f.write(f"- **Row {change['row_number']}:** {change['pii_type']} - '{change['original_text']}' → '[PII removed]'\n")
        else:
            f.write("## No PII Found\n\n")
            f.write("No personally identifiable information was detected in the specified columns.\n")
            f.write("The original file can be used as-is.\n")
        
        f.write("\n## Methodology\n\n")
        f.write("This refined analysis used the following approach:\n")
        f.write("1. **Known Personal Names:** Direct matching of names identified in previous analysis\n")
        f.write("2. **Email Addresses:** Pattern matching for valid email formats\n")
        f.write("3. **Phone Numbers:** Pattern matching for phone number formats\n")
        f.write("4. **Personal IDs:** Pattern matching for ID numbers and identifiers\n")
        f.write("5. **Social Media:** Pattern matching for social media handles\n")
        f.write("6. **Extensive Filtering:** Excluded organizational names, project names, and common false positives\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the anonymized file to ensure all PII has been properly removed\n")
        f.write("2. Check the changes report to verify that legitimate PII was replaced and no false positives occurred\n")
        f.write("3. Manually review any edge cases or questionable replacements\n")
        f.write("4. Use the anonymized file for further analysis or sharing\n\n")
        f.write("---\n")
        f.write("*This refined anonymization focused on actual personal names and identifiers while minimizing false positives.*\n")
    
    print(f"Summary report saved to: {summary_file}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("REFINED ANONYMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows processed: {len(df)}")
    print(f"PII instances found: {len(changes_made)}")
    print(f"Rows modified: {len(set(change['row_number'] for change in changes_made)) if changes_made else 0}")
    print(f"Anonymized file: {output_file}")
    print(f"Changes report: {changes_file}")
    print(f"Summary report: {summary_file}")
    
    if changes_made:
        print(f"\nPII Types Found:")
        type_counts = {}
        for change in changes_made:
            type_counts[change['pii_type']] = type_counts.get(change['pii_type'], 0) + 1
        for pii_type, count in sorted(type_counts.items()):
            print(f"  - {pii_type}: {count} instances")
        
        print(f"\nSpecific PII Found:")
        for change in sorted(changes_made, key=lambda x: x['row_number']):
            print(f"  Row {change['row_number']}: {change['original_text']}")
    
    return len(changes_made)

if __name__ == "__main__":
    changes_count = create_refined_anonymized_csv()
    print(f"\nProcess completed with {changes_count} PII instances anonymized.")
