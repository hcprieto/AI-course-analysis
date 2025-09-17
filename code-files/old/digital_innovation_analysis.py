import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev/source-files/FBs-Digital-Grants-Data.v5.csv')

# Clean column names (remove extra spaces and normalize)
df.columns = df.columns.str.strip()

# Key columns for analysis
key_columns = ["Grant's short name", "Grant's Name", "Brief Summary", "Grant's Summary (from GAMS)", 
               "Digital Innovation sub-portfolio", "Digital Health sub-portfolio", "Digital Rights sub-portfolio", 
               "Digital Competencies sub-portfolio", "Digital Futures sub-portfolio"]

# Print column names to verify
print("Column names in the dataset:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

print("\n" + "="*80)
print("DIGITAL INNOVATION SUB-PORTFOLIO ANALYSIS")
print("="*80)

# Check for Digital Innovation assignments
innovation_col = 'Digital Innovation sub-portfolio'
if innovation_col in df.columns:
    innovation_assignments = df[df[innovation_col].notna() & (df[innovation_col] != '')]
    print(f"\nCurrently assigned to Digital Innovation sub-portfolio: {len(innovation_assignments)} grants")
    
    if len(innovation_assignments) > 0:
        print("\nGrants currently assigned to Digital Innovation:")
        print("-" * 50)
        for idx, row in innovation_assignments.iterrows():
            short_name = row["Grant's short name"]
            full_name = row["Grant's Name"]
            brief = row["Brief Summary"]
            assignment = row[innovation_col]
            print(f"• {short_name}: {full_name}")
            print(f"  Assignment: {assignment}")
            print(f"  Brief: {brief}")
            print()
else:
    print(f"Column '{innovation_col}' not found in dataset")

print("\n" + "="*80)
print("ALL GRANTS ANALYSIS")
print("="*80)

# Analyze all grants
print(f"\nTotal grants in dataset: {len(df)}")

# Create comprehensive grant analysis
grant_analysis = []

for idx, row in df.iterrows():
    grant_info = {
        'short_name': row["Grant's short name"],
        'full_name': row["Grant's Name"],
        'brief_summary': row["Brief Summary"],
        'detailed_summary': row["Grant's Summary (from GAMS)"],
        'digital_innovation': row.get("Digital Innovation sub-portfolio", ""),
        'digital_health': row.get("Digital Health sub-portfolio", ""),
        'digital_rights': row.get("Digital Rights sub-portfolio", ""),
        'digital_competencies': row.get("Digital Competencies sub-portfolio", ""),
        'digital_futures': row.get("Digital Futures sub-portfolio", "")
    }
    
    # Analyze for digital innovation potential
    text_to_analyze = f"{grant_info['brief_summary']} {grant_info['detailed_summary']}".lower()
    
    # Keywords indicating digital innovation potential
    innovation_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'algorithm', 'digital platform',
        'mobile app', 'technology platform', 'digital tool', 'software', 'digital solution',
        'innovation', 'digital transformation', 'blockchain', 'iot', 'internet of things',
        'big data', 'data analytics', 'predictive analytics', 'automation', 'robotics',
        'virtual reality', 'vr', 'augmented reality', 'ar', 'digital innovation',
        'tech innovation', 'emerging technology', 'cutting-edge', 'novel technology',
        'digital infrastructure', 'api', 'integration', 'interoperability', 'scalable',
        'cloud', 'biometric', 'sensors', 'real-time', 'automated', 'intelligent',
        'smart', 'adaptive', 'personalized', 'recommendation', 'predictive'
    ]
    
    found_keywords = [kw for kw in innovation_keywords if kw in text_to_analyze]
    grant_info['innovation_keywords'] = found_keywords
    grant_info['innovation_potential'] = len(found_keywords)
    
    grant_analysis.append(grant_info)

# Sort by innovation potential
grant_analysis.sort(key=lambda x: x['innovation_potential'], reverse=True)

print("\n" + "="*80)
print("DIGITAL INNOVATION POTENTIAL ANALYSIS")
print("="*80)

print("\nTop candidates for Digital Innovation sub-portfolio (ranked by digital innovation keywords):")
print("-" * 80)

for i, grant in enumerate(grant_analysis[:20]):  # Top 20
    if grant['innovation_potential'] > 0:
        print(f"\n{i+1}. {grant['short_name']}: {grant['full_name']}")
        print(f"   Innovation Score: {grant['innovation_potential']} keywords")
        print(f"   Current Assignment: DI={grant['digital_innovation']}, DH={grant['digital_health']}, DR={grant['digital_rights']}")
        print(f"   Brief: {grant['brief_summary']}")
        print(f"   Keywords found: {', '.join(grant['innovation_keywords'])}")

print("\n" + "="*80)
print("DETAILED GRANT INFORMATION")
print("="*80)

print("\nAll grants with key information:")
print("-" * 50)

for grant in grant_analysis:
    print(f"\n• {grant['short_name']}: {grant['full_name']}")
    print(f"  Brief Summary: {grant['brief_summary']}")
    
    # Show current sub-portfolio assignments
    assignments = []
    if grant['digital_innovation']: assignments.append(f"Digital Innovation: {grant['digital_innovation']}")
    if grant['digital_health']: assignments.append(f"Digital Health: {grant['digital_health']}")
    if grant['digital_rights']: assignments.append(f"Digital Rights: {grant['digital_rights']}")
    if grant['digital_competencies']: assignments.append(f"Digital Competencies: {grant['digital_competencies']}")
    if grant['digital_futures']: assignments.append(f"Digital Futures: {grant['digital_futures']}")
    
    if assignments:
        print(f"  Current Assignments: {'; '.join(assignments)}")
    else:
        print(f"  Current Assignments: None")
    
    if grant['innovation_potential'] > 0:
        print(f"  Innovation Potential: {grant['innovation_potential']} keywords ({', '.join(grant['innovation_keywords'])})")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Current assignments summary
innovation_current = len([g for g in grant_analysis if g['digital_innovation']])
health_current = len([g for g in grant_analysis if g['digital_health']])
rights_current = len([g for g in grant_analysis if g['digital_rights']])
competencies_current = len([g for g in grant_analysis if g['digital_competencies']])
futures_current = len([g for g in grant_analysis if g['digital_futures']])

print(f"\nCurrent sub-portfolio assignments:")
print(f"• Digital Innovation: {innovation_current} grants")
print(f"• Digital Health: {health_current} grants")
print(f"• Digital Rights: {rights_current} grants")
print(f"• Digital Competencies: {competencies_current} grants")
print(f"• Digital Futures: {futures_current} grants")

# Innovation potential analysis
high_potential = len([g for g in grant_analysis if g['innovation_potential'] >= 5])
medium_potential = len([g for g in grant_analysis if 2 <= g['innovation_potential'] < 5])
low_potential = len([g for g in grant_analysis if 0 < g['innovation_potential'] < 2])
no_potential = len([g for g in grant_analysis if g['innovation_potential'] == 0])

print(f"\nDigital Innovation potential distribution:")
print(f"• High potential (5+ keywords): {high_potential} grants")
print(f"• Medium potential (2-4 keywords): {medium_potential} grants")
print(f"• Low potential (1 keyword): {low_potential} grants")
print(f"• No clear innovation focus: {no_potential} grants")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\nBased on the analysis, here are recommendations for Digital Innovation sub-portfolio assignments:")

# Find grants with high innovation potential but not assigned to Digital Innovation
candidates = [g for g in grant_analysis if g['innovation_potential'] >= 3 and not g['digital_innovation']]
candidates.sort(key=lambda x: x['innovation_potential'], reverse=True)

print(f"\nTop candidates for Digital Innovation assignment (high innovation potential, not currently assigned):")
for i, grant in enumerate(candidates[:10]):
    print(f"{i+1}. {grant['short_name']} (Score: {grant['innovation_potential']})")
    print(f"   Keywords: {', '.join(grant['innovation_keywords'])}")
    print(f"   Brief: {grant['brief_summary']}")
    print()

# Find grants currently assigned to Digital Innovation with low scores
current_innovation = [g for g in grant_analysis if g['digital_innovation']]
low_scoring_current = [g for g in current_innovation if g['innovation_potential'] < 2]

if low_scoring_current:
    print(f"\nCurrently assigned to Digital Innovation but with low innovation scores:")
    for grant in low_scoring_current:
        print(f"• {grant['short_name']} (Score: {grant['innovation_potential']})")
        print(f"  May need review of assignment rationale")
        print()

print("\nAnalysis complete!")