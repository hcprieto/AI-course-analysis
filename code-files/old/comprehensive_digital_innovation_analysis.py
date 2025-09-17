import pandas as pd
import numpy as np
import re

# Read the CSV file with proper header handling
df = pd.read_csv('/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev/source-files/FBs-Digital-Grants-Data.v5.csv', 
                 header=1)

# Clean column names
df.columns = df.columns.str.strip()

print("="*100)
print("DIGITAL INNOVATION SUB-PORTFOLIO ANALYSIS")
print("="*100)

print(f"\nTotal grants in dataset: {len(df)}")

# Extract key information for each grant
grant_analysis = []

for idx, row in df.iterrows():
    if pd.isna(row['Grant\'s short name']) or row['Grant\'s short name'] == '':
        continue
        
    grant_info = {
        'gams_code': row['GAMS Code'],
        'short_name': row['Grant\'s short name'],
        'full_name': row['Grant\'s Name'],
        'brief_summary': str(row['Brief Summary']) if pd.notna(row['Brief Summary']) else '',
        'detailed_summary': str(row['Grant\'s Summary (from GAMS)']) if pd.notna(row['Grant\'s Summary (from GAMS)']) else '',
        'countries': str(row['Countries']) if pd.notna(row['Countries']) else '',
        'digital_health': str(row['Digital Health sub-portfolio']) if pd.notna(row['Digital Health sub-portfolio']) else '',
        'digital_rights': str(row['Digital Rights sub-portfolio']) if pd.notna(row['Digital Rights sub-portfolio']) else '',
        'digital_competencies': str(row['Digital Competencies sub-portfolio']) if pd.notna(row['Digital Competencies sub-portfolio']) else '',
        'digital_innovation': str(row['Digital Innovation sub-portfolio']) if pd.notna(row['Digital Innovation sub-portfolio']) else '',
        'digital_futures': str(row['Digital Futures sub-portfolio']) if pd.notna(row['Digital Futures sub-portfolio']) else '',
        'main_grantee': str(row['Main Grantee Name']) if pd.notna(row['Main Grantee Name']) else '',
        'budget': str(row['Budget (CHF)']) if pd.notna(row['Budget (CHF)']) else ''
    }
    
    # Analyze for digital innovation potential
    text_to_analyze = f"{grant_info['brief_summary']} {grant_info['detailed_summary']}".lower()
    
    # Enhanced keywords indicating digital innovation potential
    innovation_keywords = [
        # AI and Machine Learning
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural network',
        'algorithm', 'predictive analytics', 'natural language processing', 'nlp', 'computer vision',
        
        # Digital Platforms and Tools
        'digital platform', 'mobile app', 'web application', 'software', 'digital tool', 'digital solution',
        'platform', 'application', 'app', 'system', 'technology platform', 'digital infrastructure',
        
        # Innovation and Technology
        'innovation', 'digital transformation', 'digital innovation', 'tech innovation', 'technological innovation',
        'emerging technology', 'cutting-edge', 'novel technology', 'breakthrough', 'disruptive',
        
        # Data and Analytics
        'big data', 'data analytics', 'data science', 'analytics', 'data-driven', 'data mining',
        'business intelligence', 'predictive modeling', 'data visualization', 'dashboard',
        
        # Emerging Technologies
        'blockchain', 'iot', 'internet of things', 'cloud computing', 'edge computing',
        'virtual reality', 'vr', 'augmented reality', 'ar', 'mixed reality', 'metaverse',
        
        # Automation and Smart Systems
        'automation', 'automated', 'robotics', 'smart system', 'intelligent system',
        'adaptive system', 'personalized', 'recommendation system', 'decision support',
        
        # Technical Implementation
        'api', 'integration', 'interoperability', 'scalable', 'cloud-based', 'saas',
        'microservices', 'digital architecture', 'tech stack', 'framework',
        
        # Sensors and Hardware
        'sensor', 'iot device', 'wearable', 'biometric', 'fingerprint', 'facial recognition',
        'real-time monitoring', 'tracking system', 'gps', 'location-based',
        
        # Digital Health Specific
        'telemedicine', 'telehealth', 'digital health', 'mhealth', 'electronic health record',
        'ehr', 'health information system', 'clinical decision support', 'diagnostic tool',
        
        # Development and Programming
        'software development', 'coding', 'programming', 'development', 'prototype',
        'mvp', 'minimum viable product', 'beta testing', 'user testing',
        
        # Digital Transformation
        'digitization', 'digitalization', 'digital-first', 'digital-enabled', 'tech-enabled',
        'digital ecosystem', 'digital workflow', 'paperless', 'electronic'
    ]
    
    found_keywords = []
    for keyword in innovation_keywords:
        if keyword in text_to_analyze:
            found_keywords.append(keyword)
    
    # Remove duplicates while preserving order
    found_keywords = list(dict.fromkeys(found_keywords))
    
    grant_info['innovation_keywords'] = found_keywords
    grant_info['innovation_score'] = len(found_keywords)
    
    # Additional scoring based on context
    high_value_terms = ['artificial intelligence', 'machine learning', 'blockchain', 'virtual reality', 'augmented reality']
    bonus_score = sum(2 for term in high_value_terms if term in text_to_analyze)
    grant_info['innovation_score'] += bonus_score
    
    grant_analysis.append(grant_info)

# Sort by innovation score
grant_analysis.sort(key=lambda x: x['innovation_score'], reverse=True)

print("\n" + "="*100)
print("CURRENT DIGITAL INNOVATION SUB-PORTFOLIO ASSIGNMENTS")
print("="*100)

innovation_assigned = [g for g in grant_analysis if g['digital_innovation'] not in ['', 'nan']]
print(f"\nCurrently assigned to Digital Innovation sub-portfolio: {len(innovation_assigned)} grants")

if innovation_assigned:
    print("\nGrants currently assigned to Digital Innovation:")
    print("-" * 80)
    for grant in innovation_assigned:
        print(f"\n• {grant['short_name']}: {grant['full_name']}")
        print(f"  Assignment Level: {grant['digital_innovation']}")
        print(f"  Innovation Score: {grant['innovation_score']}")
        print(f"  Countries: {grant['countries']}")
        print(f"  Brief: {grant['brief_summary']}")
        if grant['innovation_keywords']:
            print(f"  Innovation Keywords: {', '.join(grant['innovation_keywords'][:10])}")  # Show first 10
else:
    print("\nNo grants currently assigned to Digital Innovation sub-portfolio")

print("\n" + "="*100)
print("ALL GRANTS WITH KEY INFORMATION")
print("="*100)

print(f"\nDetailed information for all {len(grant_analysis)} grants:")
print("-" * 80)

for i, grant in enumerate(grant_analysis, 1):
    print(f"\n{i}. {grant['short_name']}: {grant['full_name']}")
    print(f"   GAMS Code: {grant['gams_code']}")
    print(f"   Countries: {grant['countries']}")
    print(f"   Main Grantee: {grant['main_grantee']}")
    print(f"   Brief Summary: {grant['brief_summary']}")
    
    # Current sub-portfolio assignments
    assignments = []
    if grant['digital_health'] not in ['', 'nan']: assignments.append(f"Health: {grant['digital_health']}")
    if grant['digital_rights'] not in ['', 'nan']: assignments.append(f"Rights: {grant['digital_rights']}")
    if grant['digital_competencies'] not in ['', 'nan']: assignments.append(f"Competencies: {grant['digital_competencies']}")
    if grant['digital_innovation'] not in ['', 'nan']: assignments.append(f"Innovation: {grant['digital_innovation']}")
    if grant['digital_futures'] not in ['', 'nan']: assignments.append(f"Futures: {grant['digital_futures']}")
    
    if assignments:
        print(f"   Current Sub-portfolio Assignments: {'; '.join(assignments)}")
    else:
        print(f"   Current Sub-portfolio Assignments: None")
    
    if grant['innovation_score'] > 0:
        print(f"   Innovation Score: {grant['innovation_score']}")
        print(f"   Innovation Keywords: {', '.join(grant['innovation_keywords'])}")

print("\n" + "="*100)
print("DIGITAL INNOVATION POTENTIAL ANALYSIS")
print("="*100)

print("\nTop candidates for Digital Innovation sub-portfolio (ranked by innovation score):")
print("-" * 80)

# Show top 15 candidates
for i, grant in enumerate(grant_analysis[:15], 1):
    if grant['innovation_score'] > 0:
        print(f"\n{i}. {grant['short_name']}: {grant['full_name']}")
        print(f"   Innovation Score: {grant['innovation_score']}")
        print(f"   Current Innovation Assignment: {grant['digital_innovation'] if grant['digital_innovation'] not in ['', 'nan'] else 'None'}")
        print(f"   Countries: {grant['countries']}")
        print(f"   Brief: {grant['brief_summary']}")
        print(f"   Innovation Keywords: {', '.join(grant['innovation_keywords'])}")

print("\n" + "="*100)
print("POTENTIAL REASSIGNMENT CANDIDATES")
print("="*100)

# Find grants with high innovation potential but not assigned to Digital Innovation
high_potential_unassigned = [g for g in grant_analysis 
                           if g['innovation_score'] >= 5 and g['digital_innovation'] in ['', 'nan']]

print(f"\nHigh potential grants (score ≥5) NOT currently assigned to Digital Innovation: {len(high_potential_unassigned)}")

if high_potential_unassigned:
    print("\nTop candidates for Digital Innovation assignment:")
    print("-" * 60)
    for i, grant in enumerate(high_potential_unassigned[:10], 1):
        print(f"\n{i}. {grant['short_name']} (Score: {grant['innovation_score']})")
        print(f"   Current assignments: ", end="")
        assignments = []
        if grant['digital_health'] not in ['', 'nan']: assignments.append(f"Health: {grant['digital_health']}")
        if grant['digital_rights'] not in ['', 'nan']: assignments.append(f"Rights: {grant['digital_rights']}")
        if grant['digital_competencies'] not in ['', 'nan']: assignments.append(f"Competencies: {grant['digital_competencies']}")
        if grant['digital_futures'] not in ['', 'nan']: assignments.append(f"Futures: {grant['digital_futures']}")
        print('; '.join(assignments) if assignments else "None")
        print(f"   Brief: {grant['brief_summary']}")
        print(f"   Key innovation aspects: {', '.join(grant['innovation_keywords'][:8])}")

# Find grants currently assigned to Digital Innovation with lower scores
current_innovation_low_score = [g for g in grant_analysis 
                              if g['digital_innovation'] not in ['', 'nan'] and g['innovation_score'] < 3]

if current_innovation_low_score:
    print(f"\n\nCurrently assigned to Digital Innovation but with low innovation scores (<3): {len(current_innovation_low_score)}")
    print("-" * 60)
    for grant in current_innovation_low_score:
        print(f"\n• {grant['short_name']} (Score: {grant['innovation_score']})")
        print(f"  Assignment Level: {grant['digital_innovation']}")
        print(f"  Brief: {grant['brief_summary']}")
        print(f"  Consider reviewing assignment rationale")

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

# Current assignments summary
health_current = len([g for g in grant_analysis if g['digital_health'] not in ['', 'nan']])
rights_current = len([g for g in grant_analysis if g['digital_rights'] not in ['', 'nan']])
competencies_current = len([g for g in grant_analysis if g['digital_competencies'] not in ['', 'nan']])
innovation_current = len([g for g in grant_analysis if g['digital_innovation'] not in ['', 'nan']])
futures_current = len([g for g in grant_analysis if g['digital_futures'] not in ['', 'nan']])

print(f"\nCurrent sub-portfolio assignments:")
print(f"• Digital Health: {health_current} grants")
print(f"• Digital Rights: {rights_current} grants")
print(f"• Digital Competencies: {competencies_current} grants")
print(f"• Digital Innovation: {innovation_current} grants")
print(f"• Digital Futures: {futures_current} grants")

# Innovation potential distribution
high_potential = len([g for g in grant_analysis if g['innovation_score'] >= 8])
medium_high_potential = len([g for g in grant_analysis if 5 <= g['innovation_score'] < 8])
medium_potential = len([g for g in grant_analysis if 2 <= g['innovation_score'] < 5])
low_potential = len([g for g in grant_analysis if 0 < g['innovation_score'] < 2])
no_potential = len([g for g in grant_analysis if g['innovation_score'] == 0])

print(f"\nDigital Innovation potential distribution:")
print(f"• High potential (8+ keywords): {high_potential} grants")
print(f"• Medium-high potential (5-7 keywords): {medium_high_potential} grants")
print(f"• Medium potential (2-4 keywords): {medium_potential} grants")
print(f"• Low potential (1 keyword): {low_potential} grants")
print(f"• No clear innovation indicators: {no_potential} grants")

print("\n" + "="*100)
print("RECOMMENDATIONS FOR DIGITAL INNOVATION SUB-PORTFOLIO")
print("="*100)

print("\n1. STRONG CANDIDATES FOR DIGITAL INNOVATION ASSIGNMENT:")
print("   (High innovation scores, innovative technologies, not currently assigned)")
print("-" * 70)

strong_candidates = [g for g in grant_analysis 
                    if g['innovation_score'] >= 6 and g['digital_innovation'] in ['', 'nan']]

for i, grant in enumerate(strong_candidates[:8], 1):
    print(f"\n{i}. {grant['short_name']} (Score: {grant['innovation_score']})")
    print(f"   Rationale: Uses {', '.join(grant['innovation_keywords'][:5])} and similar technologies")
    print(f"   Brief: {grant['brief_summary']}")

print("\n\n2. GRANTS TO REVIEW:")
print("   (Currently assigned to Digital Innovation but may need assessment)")
print("-" * 70)

review_candidates = [g for g in grant_analysis 
                    if g['digital_innovation'] not in ['', 'nan'] and g['innovation_score'] < 4]

if review_candidates:
    for grant in review_candidates:
        print(f"\n• {grant['short_name']} (Score: {grant['innovation_score']})")
        print(f"  Current assignment: {grant['digital_innovation']}")
        print(f"  Recommendation: Review to ensure alignment with innovation focus")
else:
    print("\n• No grants currently assigned to Digital Innovation appear to need review")

print("\n\n3. BORDERLINE CASES:")
print("   (Medium innovation scores, may warrant consideration)")
print("-" * 70)

borderline = [g for g in grant_analysis 
             if 3 <= g['innovation_score'] < 6 and g['digital_innovation'] in ['', 'nan']]

for i, grant in enumerate(borderline[:5], 1):
    print(f"\n{i}. {grant['short_name']} (Score: {grant['innovation_score']})")
    print(f"   Innovation aspects: {', '.join(grant['innovation_keywords'][:5])}")
    print(f"   Brief: {grant['brief_summary']}")

print("\n\nAnalysis complete! This comprehensive analysis should help inform Digital Innovation sub-portfolio assignment decisions.")