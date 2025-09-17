#!/usr/bin/env python3
"""
Analyze non-default outcome baskets in FB-Digital-Grants-Outcomes-Data.csv
Compare with default outcome baskets from Key-Concepts-Descriptors.v5.csv
"""

import pandas as pd
import re
import csv
from collections import defaultdict, Counter
from datetime import datetime

def extract_default_outcome_baskets():
    """Extract default outcome baskets from Key-Concepts-Descriptors.v5.csv"""
    
    print("Extracting default outcome baskets...")
    df = pd.read_csv('../source-files/Key-Concepts-Descriptors.v5.csv')
    
    # Find the section starting from row 28 (DEFAULT OUTCOME BASKETS...)
    default_baskets = {
        'DoC 1': [],  # Stronger agency of young people
        'DoC 2': [],  # Expanded equitable access to services
        'DoC 3': [],  # Collective efforts for learning/alignment
        'DoC 4': []   # Human rights-based digital governance frameworks
    }
    
    # The default baskets start at row 23 (after the headers at row 22)
    for idx in range(23, len(df)):  # Start from row 23 and go to end of file
        row = df.iloc[idx]
        
        # Extract baskets from each domain column
        for i, doc in enumerate(['DoC 1', 'DoC 2', 'DoC 3', 'DoC 4']):
            if not pd.isna(row.iloc[i]) and str(row.iloc[i]).strip():
                basket = str(row.iloc[i]).strip()
                if basket and basket not in default_baskets[doc]:
                    default_baskets[doc].append(basket)
    
    # Create flat list of all default baskets
    all_default_baskets = []
    for doc, baskets in default_baskets.items():
        all_default_baskets.extend(baskets)
    
    print(f"Found default baskets:")
    for doc, baskets in default_baskets.items():
        print(f"  {doc}: {len(baskets)} baskets")
    print(f"Total unique default baskets: {len(all_default_baskets)}")
    
    return default_baskets, all_default_baskets

def extract_non_default_baskets():
    """Extract and analyze non-default outcome baskets from outcomes data"""
    
    print("\nProcessing outcomes data...")
    df = pd.read_csv('../source-files/FB-Digital-Grants-Outcomes-Data.csv')
    
    non_default_data = []
    non_default_counter = Counter()
    
    for idx, row in df.iterrows():
        # Check if there's content in the non-default column
        non_default_col = row.get('SPECIAL COLUMN (generated) - Non Default OB Tags', '')
        
        if pd.notna(non_default_col) and str(non_default_col).strip():
            # Parse semicolon-separated baskets
            non_default_baskets = [b.strip() for b in str(non_default_col).split(';') if b.strip()]
            
            # Also get default baskets for comparison
            default_col = row.get('Outcome Basket names', '')
            default_baskets = []
            if pd.notna(default_col):
                default_baskets = [b.strip() for b in str(default_col).split(';') if b.strip()]
            
            for basket in non_default_baskets:
                non_default_counter[basket] += 1
                
                non_default_data.append({
                    'row_number': idx + 2,  # +2 for 1-indexed and header
                    'grant_name': row.get('Grant Name', ''),
                    'domain': row.get('Domain of Change', ''),
                    'outcome_description': str(row.get('Description of the Reported Outcome', ''))[:200],
                    'default_baskets': '; '.join(default_baskets),
                    'non_default_basket': basket,
                    'all_non_default': non_default_col
                })
    
    print(f"Found {len(non_default_data)} non-default basket instances")
    print(f"Found {len(non_default_counter)} unique non-default baskets")
    
    return non_default_data, non_default_counter

def semantic_similarity_score(text1, text2):
    """Calculate simple semantic similarity between two outcome basket names"""
    
    # Normalize texts
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()
    
    # Exact match
    if text1_norm == text2_norm:
        return 1.0
    
    # Extract key words (remove common words)
    stop_words = {'and', 'or', 'of', 'to', 'for', 'in', 'with', 'by', 'from', 'the', 'a', 'an'}
    
    def extract_keywords(text):
        # Split on punctuation and spaces, filter stop words
        words = re.findall(r'\b\w+\b', text.lower())
        return set(word for word in words if word not in stop_words and len(word) > 2)
    
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard similarity
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0

def find_best_matches(non_default_baskets, default_baskets):
    """Find best semantic matches between non-default and default baskets"""
    
    print("\nFinding semantic matches...")
    matches = []
    
    for non_default in non_default_baskets:
        best_match = None
        best_score = 0.0
        
        for default in default_baskets:
            score = semantic_similarity_score(non_default, default)
            if score > best_score:
                best_score = score
                best_match = default
        
        match_type = 'exact' if best_score == 1.0 else 'high' if best_score > 0.7 else 'medium' if best_score > 0.4 else 'low'
        
        matches.append({
            'non_default_basket': non_default,
            'best_match': best_match,
            'similarity_score': best_score,
            'match_type': match_type
        })
    
    return matches

def analyze_consolidation_opportunities(non_default_data, matches, non_default_counter):
    """Analyze consolidation opportunities"""
    
    print("\nAnalyzing consolidation opportunities...")
    
    consolidation_recommendations = {
        'direct_matches': [],      # Exact matches (score = 1.0)
        'semantic_matches': [],    # High similarity (score > 0.7)
        'potential_matches': [],   # Medium similarity (0.4 < score <= 0.7)
        'unique_concepts': [],     # Low similarity (score <= 0.4)
        'subset_selections': []    # Cases where non-default is subset of default
    }
    
    for match in matches:
        frequency = non_default_counter[match['non_default_basket']]
        
        recommendation = {
            'non_default_basket': match['non_default_basket'],
            'frequency': frequency,
            'best_match': match['best_match'],
            'similarity_score': match['similarity_score'],
            'recommendation': '',
            'impact': ''
        }
        
        if match['similarity_score'] == 1.0:
            recommendation['recommendation'] = 'Consolidate - exact match'
            recommendation['impact'] = f'Affects {frequency} outcomes'
            consolidation_recommendations['direct_matches'].append(recommendation)
        
        elif match['similarity_score'] > 0.7:
            recommendation['recommendation'] = 'Consider consolidation - high similarity'
            recommendation['impact'] = f'Affects {frequency} outcomes'
            consolidation_recommendations['semantic_matches'].append(recommendation)
        
        elif match['similarity_score'] > 0.4:
            recommendation['recommendation'] = 'Review manually - moderate similarity'
            recommendation['impact'] = f'Affects {frequency} outcomes'
            consolidation_recommendations['potential_matches'].append(recommendation)
        
        else:
            recommendation['recommendation'] = 'Keep separate - unique concept'
            recommendation['impact'] = f'Affects {frequency} outcomes'
            consolidation_recommendations['unique_concepts'].append(recommendation)
    
    # Analyze subset patterns
    for data_point in non_default_data:
        default_baskets_set = set(b.strip() for b in data_point['default_baskets'].split(';') if b.strip())
        non_default_basket = data_point['non_default_basket']
        
        if non_default_basket in default_baskets_set and len(default_baskets_set) > 1:
            consolidation_recommendations['subset_selections'].append({
                'row_number': data_point['row_number'],
                'grant_name': data_point['grant_name'],
                'domain': data_point['domain'],
                'non_default_basket': non_default_basket,
                'full_default_baskets': data_point['default_baskets'],
                'recommendation': 'Subset selection - consider keeping for specificity'
            })
    
    return consolidation_recommendations

def generate_reports(default_baskets, non_default_data, non_default_counter, matches, consolidation_recommendations):
    """Generate comprehensive analysis reports"""
    
    print("\nGenerating reports...")
    
    # 1. Detailed CSV analysis
    output_file = '../analysis-output-files/Non_Default_Outcome_Baskets_Analysis.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Non_Default_Basket', 'Frequency', 'Best_Semantic_Match', 'Similarity_Score', 
            'Match_Type', 'Sample_Context', 'Sample_Domain', 'Consolidation_Recommendation'
        ])
        
        for match in sorted(matches, key=lambda x: x['similarity_score'], reverse=True):
            basket = match['non_default_basket']
            frequency = non_default_counter[basket]
            
            # Find a sample context
            sample_data = next((d for d in non_default_data if d['non_default_basket'] == basket), {})
            sample_context = sample_data.get('outcome_description', '')[:150] + '...' if sample_data else ''
            sample_domain = sample_data.get('domain', '') if sample_data else ''
            
            # Get recommendation
            rec = 'Keep separate'
            if match['similarity_score'] == 1.0:
                rec = 'Consolidate - exact match'
            elif match['similarity_score'] > 0.7:
                rec = 'Consider consolidation'
            elif match['similarity_score'] > 0.4:
                rec = 'Review manually'
            
            writer.writerow([
                basket,
                frequency,
                match['best_match'],
                f"{match['similarity_score']:.3f}",
                match['match_type'],
                sample_context,
                sample_domain,
                rec
            ])
    
    print(f"Detailed analysis saved to: {output_file}")
    
    # 2. Consolidation recommendations
    recommendations_file = '../analysis-output-files/Consolidation_Recommendations.md'
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        f.write("# Non-Default Outcome Baskets Consolidation Recommendations\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total non-default basket instances**: {len(non_default_data)}\n")
        f.write(f"- **Unique non-default baskets**: {len(non_default_counter)}\n")
        f.write(f"- **Direct matches for consolidation**: {len(consolidation_recommendations['direct_matches'])}\n")
        f.write(f"- **High similarity matches**: {len(consolidation_recommendations['semantic_matches'])}\n")
        f.write(f"- **Unique concepts to preserve**: {len(consolidation_recommendations['unique_concepts'])}\n")
        f.write(f"- **Subset selections**: {len(consolidation_recommendations['subset_selections'])}\n\n")
        
        # Direct matches
        if consolidation_recommendations['direct_matches']:
            f.write("## 🔗 Direct Matches - Recommended for Consolidation\n\n")
            for rec in consolidation_recommendations['direct_matches']:
                f.write(f"**{rec['non_default_basket']}** (used {rec['frequency']} times)\n")
                f.write(f"- Exact match with: *{rec['best_match']}*\n")
                f.write(f"- Impact: {rec['impact']}\n\n")
        
        # Semantic matches
        if consolidation_recommendations['semantic_matches']:
            f.write("## 🤔 High Similarity Matches - Consider for Consolidation\n\n")
            for rec in consolidation_recommendations['semantic_matches']:
                f.write(f"**{rec['non_default_basket']}** (used {rec['frequency']} times)\n")
                f.write(f"- Similar to: *{rec['best_match']}* (similarity: {rec['similarity_score']:.3f})\n")
                f.write(f"- Impact: {rec['impact']}\n\n")
        
        # Unique concepts
        if consolidation_recommendations['unique_concepts']:
            f.write("## 💡 Unique Concepts - Keep Separate\n\n")
            for rec in consolidation_recommendations['unique_concepts']:
                f.write(f"**{rec['non_default_basket']}** (used {rec['frequency']} times)\n")
                f.write(f"- Best match: *{rec['best_match']}* (similarity: {rec['similarity_score']:.3f})\n")
                f.write(f"- Recommendation: Keep as unique concept\n\n")
        
        # Subset selections
        if consolidation_recommendations['subset_selections']:
            f.write("## 🎯 Subset Selections - Review for Specificity\n\n")
            f.write("These cases use a single outcome basket from a larger set of defaults, likely for emphasis:\n\n")
            
            # Group by non-default basket
            subset_groups = defaultdict(list)
            for item in consolidation_recommendations['subset_selections']:
                subset_groups[item['non_default_basket']].append(item)
            
            for basket, items in subset_groups.items():
                f.write(f"**{basket}** (used {len(items)} times as subset selection)\n")
                f.write(f"- Appears as focused selection from broader default basket sets\n")
                f.write(f"- Consider: Keep for specificity or consolidate with full default set\n\n")
    
    print(f"Consolidation recommendations saved to: {recommendations_file}")
    
    # 3. Executive summary
    summary_file = '../analysis-output-files/Non_Default_OB_Summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Non-Default Outcome Baskets Analysis Summary\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"### Usage Statistics\n")
        f.write(f"- **{len(non_default_data)} total instances** of non-default outcome basket usage\n")
        f.write(f"- **{len(non_default_counter)} unique non-default baskets** identified\n")
        f.write(f"- **{sum(len(baskets) for baskets in default_baskets.values())} default baskets** available across 4 domains\n\n")
        
        f.write("### Most Frequently Used Non-Default Baskets\n")
        for basket, count in non_default_counter.most_common(10):
            f.write(f"- **{basket}**: {count} uses\n")
        f.write("\n")
        
        f.write("### Consolidation Potential\n")
        f.write(f"- **{len(consolidation_recommendations['direct_matches'])} exact matches** ready for immediate consolidation\n")
        f.write(f"- **{len(consolidation_recommendations['semantic_matches'])} high similarity matches** for consideration\n")
        f.write(f"- **{len(consolidation_recommendations['unique_concepts'])} unique concepts** should remain separate\n")
        f.write(f"- **{len(set(item['non_default_basket'] for item in consolidation_recommendations['subset_selections']))} baskets** used as subset selections\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Immediate consolidation**: Process exact matches to reduce redundancy\n")
        f.write("2. **Review high similarity matches**: Evaluate semantic matches for potential consolidation\n")
        f.write("3. **Preserve unique concepts**: Keep genuinely new outcome types as separate baskets\n")
        f.write("4. **Standardize subset usage**: Decide policy on using individual vs. complete basket sets\n\n")
        
        f.write("---\n")
        f.write("*See detailed analysis files for complete data and recommendations*\n")
    
    print(f"Executive summary saved to: {summary_file}")

def main():
    """Main analysis function"""
    
    print("=== Non-Default Outcome Baskets Analysis ===")
    
    # Step 1: Extract default outcome baskets
    default_baskets, all_default_baskets = extract_default_outcome_baskets()
    
    # Step 2: Extract non-default baskets from outcomes data
    non_default_data, non_default_counter = extract_non_default_baskets()
    
    if not non_default_counter:
        print("No non-default outcome baskets found.")
        return
    
    # Step 3: Find semantic matches
    unique_non_default_baskets = list(non_default_counter.keys())
    matches = find_best_matches(unique_non_default_baskets, all_default_baskets)
    
    # Step 4: Analyze consolidation opportunities
    consolidation_recommendations = analyze_consolidation_opportunities(
        non_default_data, matches, non_default_counter
    )
    
    # Step 5: Generate reports
    generate_reports(
        default_baskets, non_default_data, non_default_counter, 
        matches, consolidation_recommendations
    )
    
    print("\n=== Analysis Complete ===")
    print("Check the analysis-output-files directory for detailed reports.")

if __name__ == "__main__":
    main()