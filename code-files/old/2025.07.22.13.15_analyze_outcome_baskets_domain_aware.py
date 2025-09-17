#!/usr/bin/env python3
"""
Domain-aware analysis of non-default outcome baskets in FB-Digital-Grants-Outcomes-Data.csv
Compare with default outcome baskets from Key-Concepts-Descriptors.v5.csv by Domain of Change
Created: 2025.07.22.13.15
"""

import pandas as pd
import re
import csv
from collections import defaultdict, Counter
from datetime import datetime

def extract_default_outcome_baskets_by_domain():
    """Extract default outcome baskets organized by Domain of Change"""
    
    print("Extracting default outcome baskets by domain...")
    df = pd.read_csv('../source-files/Key-Concepts-Descriptors.v5.csv')
    
    # Default baskets organized by domain
    default_baskets_by_domain = {
        'D1': [],  # DoC 1 - Stronger agency of young people
        'D2': [],  # DoC 2 - Expanded equitable access to services
        'D3': [],  # DoC 3 - Collective efforts for learning/alignment
        'D4': []   # DoC 4 - Human rights-based digital governance frameworks
    }
    
    # The default baskets start at row 23 (after the headers at row 22)
    for idx in range(23, len(df)):
        row = df.iloc[idx]
        
        # Extract baskets from each domain column
        domain_keys = ['D1', 'D2', 'D3', 'D4']
        for i, domain in enumerate(domain_keys):
            if not pd.isna(row.iloc[i]) and str(row.iloc[i]).strip():
                basket = str(row.iloc[i]).strip()
                if basket and basket not in default_baskets_by_domain[domain]:
                    default_baskets_by_domain[domain].append(basket)
    
    print(f"Default baskets by domain:")
    for domain, baskets in default_baskets_by_domain.items():
        print(f"  {domain}: {len(baskets)} baskets")
    
    # Create flat list for cross-domain analysis
    all_default_baskets = []
    for domain, baskets in default_baskets_by_domain.items():
        all_default_baskets.extend(baskets)
    
    print(f"Total unique default baskets: {len(all_default_baskets)}")
    
    return default_baskets_by_domain, all_default_baskets

def extract_non_default_baskets_with_domain():
    """Extract non-default outcome baskets with their domain context"""
    
    print("\nProcessing outcomes data with domain context...")
    df = pd.read_csv('../source-files/FB-Digital-Grants-Outcomes-Data.csv')
    
    non_default_data = []
    non_default_counter = Counter()
    domain_usage = defaultdict(list)
    
    for idx, row in df.iterrows():
        non_default_col = row.get('SPECIAL COLUMN (generated) - Non Default OB Tags', '')
        
        if pd.notna(non_default_col) and str(non_default_col).strip():
            # Get domain for this outcome
            domain_full = row.get('Domain of Change', '')
            domain = 'D1' if 'D1' in domain_full else 'D2' if 'D2' in domain_full else 'D3' if 'D3' in domain_full else 'D4' if 'D4' in domain_full else 'Unknown'
            
            # Parse non-default baskets
            non_default_baskets = [b.strip() for b in str(non_default_col).split(';') if b.strip()]
            
            # Get default baskets for comparison
            default_col = row.get('Outcome Basket names', '')
            default_baskets = []
            if pd.notna(default_col):
                default_baskets = [b.strip() for b in str(default_col).split(';') if b.strip()]
            
            for basket in non_default_baskets:
                non_default_counter[basket] += 1
                domain_usage[basket].append(domain)
                
                non_default_data.append({
                    'row_number': idx + 2,
                    'grant_name': row.get('Grant Name', ''),
                    'domain': domain,
                    'domain_full': domain_full,
                    'outcome_description': str(row.get('Description of the Reported Outcome', ''))[:200],
                    'default_baskets': '; '.join(default_baskets),
                    'non_default_basket': basket,
                    'all_non_default': non_default_col
                })
    
    print(f"Found {len(non_default_data)} non-default basket instances")
    print(f"Found {len(non_default_counter)} unique non-default baskets")
    
    return non_default_data, non_default_counter, domain_usage

def semantic_similarity_score(text1, text2):
    """Calculate semantic similarity between two outcome basket names"""
    
    # Normalize texts
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()
    
    # Exact match
    if text1_norm == text2_norm:
        return 1.0
    
    # Extract key words (remove common words)
    stop_words = {'and', 'or', 'of', 'to', 'for', 'in', 'with', 'by', 'from', 'the', 'a', 'an', 'among', 'diverse'}
    
    def extract_keywords(text):
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

def find_domain_aware_matches(non_default_baskets, non_default_data, default_baskets_by_domain, all_default_baskets):
    """Find domain-aware semantic matches"""
    
    print("\nFinding domain-aware semantic matches...")
    matches = []
    
    # Group non-default baskets by the domains they appear in
    basket_domain_context = defaultdict(set)
    for data_point in non_default_data:
        basket_domain_context[data_point['non_default_basket']].add(data_point['domain'])
    
    for non_default in non_default_baskets:
        domains_used = list(basket_domain_context[non_default])
        
        # Find best matches within each domain this basket is used
        domain_matches = {}
        cross_domain_matches = {}
        
        for domain in domains_used:
            if domain in default_baskets_by_domain:
                domain_defaults = default_baskets_by_domain[domain]
                
                best_match = None
                best_score = 0.0
                
                # Find best match within this domain
                for default in domain_defaults:
                    score = semantic_similarity_score(non_default, default)
                    if score > best_score:
                        best_score = score
                        best_match = default
                
                domain_matches[domain] = {
                    'best_match': best_match,
                    'score': best_score,
                    'match_type': 'exact' if best_score == 1.0 else 'high' if best_score > 0.7 else 'medium' if best_score > 0.4 else 'low'
                }
        
        # Also find best match across ALL domains (for cross-domain analysis)
        best_cross_domain = None
        best_cross_score = 0.0
        best_cross_domain_name = None
        
        for domain, defaults in default_baskets_by_domain.items():
            for default in defaults:
                score = semantic_similarity_score(non_default, default)
                if score > best_cross_score:
                    best_cross_score = score
                    best_cross_domain = default
                    best_cross_domain_name = domain
        
        cross_domain_matches = {
            'best_match': best_cross_domain,
            'score': best_cross_score,
            'domain': best_cross_domain_name,
            'match_type': 'exact' if best_cross_score == 1.0 else 'high' if best_cross_score > 0.7 else 'medium' if best_cross_score > 0.4 else 'low'
        }
        
        matches.append({
            'non_default_basket': non_default,
            'domains_used': domains_used,
            'domain_matches': domain_matches,
            'cross_domain_match': cross_domain_matches
        })
    
    return matches

def get_top_similar_baskets(non_default_basket, domain, default_baskets_by_domain, top_n=3):
    """Get top N most similar default baskets for a domain (for manual review guidance)"""
    
    if domain not in default_baskets_by_domain:
        return []
    
    similarities = []
    for default_basket in default_baskets_by_domain[domain]:
        score = semantic_similarity_score(non_default_basket, default_basket)
        similarities.append({
            'basket': default_basket,
            'score': score
        })
    
    # Sort by similarity and return top N
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_n]

def analyze_domain_aware_consolidation(non_default_data, matches, non_default_counter, default_baskets_by_domain):
    """Analyze consolidation opportunities with domain awareness"""
    
    print("\nAnalyzing domain-aware consolidation opportunities...")
    
    consolidation_analysis = {
        'domain_exact_matches': [],      # Exact matches within correct domain
        'cross_domain_exact_matches': [],# Exact matches but in wrong domain
        'domain_high_similarity': [],    # High similarity within domain
        'cross_domain_high_similarity': [],# High similarity across domains
        'manual_review_needed': [],      # Medium similarity - needs review
        'unique_concepts': []            # Low similarity - unique concepts
    }
    
    for match in matches:
        basket = match['non_default_basket']
        frequency = non_default_counter[basket]
        domains_used = match['domains_used']
        
        # Analyze each domain this basket appears in
        for domain in domains_used:
            domain_match = match['domain_matches'].get(domain, {})
            cross_domain = match['cross_domain_match']
            
            analysis_item = {
                'non_default_basket': basket,
                'frequency': frequency,
                'domain': domain,
                'domain_match': domain_match.get('best_match'),
                'domain_score': domain_match.get('score', 0.0),
                'cross_domain_match': cross_domain['best_match'],
                'cross_domain_score': cross_domain['score'],
                'cross_domain_domain': cross_domain['domain']
            }
            
            # Categorize based on domain-aware matching
            if domain_match.get('score', 0.0) == 1.0:
                analysis_item['recommendation'] = 'Consolidate within domain - exact match'
                consolidation_analysis['domain_exact_matches'].append(analysis_item)
            
            elif cross_domain['score'] == 1.0 and domain_match.get('score', 0.0) < 1.0:
                analysis_item['recommendation'] = f'Review domain assignment - exact match exists in {cross_domain["domain"]}'
                consolidation_analysis['cross_domain_exact_matches'].append(analysis_item)
            
            elif domain_match.get('score', 0.0) > 0.7:
                analysis_item['recommendation'] = 'Consider consolidation within domain - high similarity'
                # Add similar baskets for guidance
                analysis_item['similar_baskets'] = get_top_similar_baskets(basket, domain, default_baskets_by_domain)
                consolidation_analysis['domain_high_similarity'].append(analysis_item)
            
            elif cross_domain['score'] > 0.7 and domain_match.get('score', 0.0) <= 0.7:
                analysis_item['recommendation'] = f'Consider domain reassignment - high similarity with {cross_domain["domain"]}'
                consolidation_analysis['cross_domain_high_similarity'].append(analysis_item)
            
            elif domain_match.get('score', 0.0) > 0.4 or cross_domain['score'] > 0.4:
                analysis_item['recommendation'] = 'Manual review needed - moderate similarity'
                analysis_item['similar_baskets'] = get_top_similar_baskets(basket, domain, default_baskets_by_domain)
                consolidation_analysis['manual_review_needed'].append(analysis_item)
            
            else:
                analysis_item['recommendation'] = 'Keep as unique concept - low similarity'
                consolidation_analysis['unique_concepts'].append(analysis_item)
    
    return consolidation_analysis

def generate_domain_aware_reports(default_baskets_by_domain, non_default_data, non_default_counter, matches, consolidation_analysis, domain_usage):
    """Generate comprehensive domain-aware analysis reports"""
    
    print("\nGenerating domain-aware reports...")
    timestamp = datetime.now().strftime('%Y.%m.%d.%H.%M')
    
    # 1. Detailed CSV analysis
    output_file = f'../analysis-output-files/{timestamp}_Non_Default_Outcome_Baskets_Domain_Analysis.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Non_Default_Basket', 'Frequency', 'Domains_Used', 
            'Domain_Best_Match', 'Domain_Score', 'Cross_Domain_Best_Match', 
            'Cross_Domain_Score', 'Cross_Domain_Domain', 'Recommendation',
            'Similar_Baskets_For_Review', 'Sample_Context'
        ])
        
        for match in matches:
            basket = match['non_default_basket']
            frequency = non_default_counter[basket]
            domains_used = ', '.join(match['domains_used'])
            
            # Get sample context
            sample_data = next((d for d in non_default_data if d['non_default_basket'] == basket), {})
            sample_context = sample_data.get('outcome_description', '')[:150] + '...' if sample_data else ''
            
            # For simplicity, show analysis for the first domain used
            first_domain = match['domains_used'][0] if match['domains_used'] else ''
            domain_match = match['domain_matches'].get(first_domain, {})
            cross_domain = match['cross_domain_match']
            
            # Determine recommendation
            if domain_match.get('score', 0.0) == 1.0:
                rec = 'Consolidate within domain - exact match'
                similar_baskets = ''
            elif cross_domain['score'] == 1.0 and domain_match.get('score', 0.0) < 1.0:
                rec = f'Review domain assignment - exact match in {cross_domain["domain"]}'
                similar_baskets = ''
            elif domain_match.get('score', 0.0) > 0.7:
                rec = 'Consider consolidation - high similarity'
                similar = get_top_similar_baskets(basket, first_domain, default_baskets_by_domain, 3)
                similar_baskets = '; '.join([f"{s['basket']} ({s['score']:.3f})" for s in similar])
            elif domain_match.get('score', 0.0) > 0.4:
                rec = 'Manual review needed - moderate similarity'
                similar = get_top_similar_baskets(basket, first_domain, default_baskets_by_domain, 3)
                similar_baskets = '; '.join([f"{s['basket']} ({s['score']:.3f})" for s in similar])
            else:
                rec = 'Keep as unique concept'
                similar_baskets = ''
            
            writer.writerow([
                basket,
                frequency,
                domains_used,
                domain_match.get('best_match', ''),
                f"{domain_match.get('score', 0.0):.3f}",
                cross_domain['best_match'],
                f"{cross_domain['score']:.3f}",
                cross_domain['domain'],
                rec,
                similar_baskets,
                sample_context
            ])
    
    print(f"Domain-aware analysis saved to: {output_file}")
    
    # 2. Consolidation recommendations
    recommendations_file = f'../analysis-output-files/{timestamp}_Consolidation_Recommendations_Domain_Aware.md'
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        f.write("# Domain-Aware Non-Default Outcome Baskets Consolidation Recommendations\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total non-default basket instances**: {len(non_default_data)}\n")
        f.write(f"- **Unique non-default baskets**: {len(non_default_counter)}\n")
        f.write(f"- **Domain-specific exact matches**: {len(consolidation_analysis['domain_exact_matches'])}\n")
        f.write(f"- **Cross-domain exact matches**: {len(consolidation_analysis['cross_domain_exact_matches'])}\n")
        f.write(f"- **Domain-specific high similarity**: {len(consolidation_analysis['domain_high_similarity'])}\n")
        f.write(f"- **Cross-domain high similarity**: {len(consolidation_analysis['cross_domain_high_similarity'])}\n")
        f.write(f"- **Manual review needed**: {len(consolidation_analysis['manual_review_needed'])}\n")
        f.write(f"- **Unique concepts**: {len(consolidation_analysis['unique_concepts'])}\n\n")
        
        # Domain-specific exact matches
        if consolidation_analysis['domain_exact_matches']:
            f.write("## ✅ Domain-Specific Exact Matches - Immediate Consolidation\n\n")
            for item in consolidation_analysis['domain_exact_matches']:
                f.write(f"**{item['non_default_basket']}** (used {item['frequency']} times in {item['domain']})\n")
                f.write(f"- Exact match with domain default: *{item['domain_match']}*\n")
                f.write(f"- Recommendation: {item['recommendation']}\n\n")
        
        # Cross-domain exact matches
        if consolidation_analysis['cross_domain_exact_matches']:
            f.write("## 🔄 Cross-Domain Exact Matches - Review Domain Assignment\n\n")
            for item in consolidation_analysis['cross_domain_exact_matches']:
                f.write(f"**{item['non_default_basket']}** (used {item['frequency']} times in {item['domain']})\n")
                f.write(f"- No match in {item['domain']}, but exact match in {item['cross_domain_domain']}: *{item['cross_domain_match']}*\n")
                f.write(f"- Recommendation: {item['recommendation']}\n\n")
        
        # Domain-specific high similarity
        if consolidation_analysis['domain_high_similarity']:
            f.write("## 🤔 Domain-Specific High Similarity - Consider Consolidation\n\n")
            for item in consolidation_analysis['domain_high_similarity']:
                f.write(f"**{item['non_default_basket']}** (used {item['frequency']} times in {item['domain']})\n")
                f.write(f"- Similar to: *{item['domain_match']}* (similarity: {item['domain_score']:.3f})\n")
                if 'similar_baskets' in item and item['similar_baskets']:
                    f.write("- Other similar baskets in this domain:\n")
                    for similar in item['similar_baskets']:
                        f.write(f"  - {similar['basket']} (similarity: {similar['score']:.3f})\n")
                f.write(f"- Recommendation: {item['recommendation']}\n\n")
        
        # Manual review needed
        if consolidation_analysis['manual_review_needed']:
            f.write("## 📋 Manual Review Needed - Moderate Similarity\n\n")
            for item in consolidation_analysis['manual_review_needed']:
                f.write(f"**{item['non_default_basket']}** (used {item['frequency']} times in {item['domain']})\n")
                f.write(f"- Best domain match: *{item['domain_match']}* (similarity: {item['domain_score']:.3f})\n")
                if 'similar_baskets' in item and item['similar_baskets']:
                    f.write("- Similar baskets in this domain for consideration:\n")
                    for similar in item['similar_baskets']:
                        f.write(f"  - {similar['basket']} (similarity: {similar['score']:.3f})\n")
                f.write(f"- Recommendation: {item['recommendation']}\n\n")
        
        # Cross-domain high similarity
        if consolidation_analysis['cross_domain_high_similarity']:
            f.write("## 🔀 Cross-Domain High Similarity - Consider Domain Reassignment\n\n")
            for item in consolidation_analysis['cross_domain_high_similarity']:
                f.write(f"**{item['non_default_basket']}** (used {item['frequency']} times in {item['domain']})\n")
                f.write(f"- Low similarity in {item['domain']} but high similarity in {item['cross_domain_domain']}: *{item['cross_domain_match']}* (similarity: {item['cross_domain_score']:.3f})\n")
                f.write(f"- Recommendation: {item['recommendation']}\n\n")
        
        # Unique concepts
        f.write("## 💡 Unique Concepts - Keep Separate\n\n")
        unique_baskets = {}
        for item in consolidation_analysis['unique_concepts']:
            basket = item['non_default_basket']
            if basket not in unique_baskets:
                unique_baskets[basket] = item['frequency']
        
        for basket, frequency in sorted(unique_baskets.items(), key=lambda x: x[1], reverse=True):
            f.write(f"**{basket}** (used {frequency} times) - Keep as unique concept\n\n")
    
    print(f"Domain-aware consolidation recommendations saved to: {recommendations_file}")
    
    # 3. Executive summary
    summary_file = f'../analysis-output-files/{timestamp}_Non_Default_OB_Domain_Summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Domain-Aware Non-Default Outcome Baskets Analysis Summary\n\n")
        f.write(f"**Analysis Date**: {timestamp}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Domain Context Analysis\n")
        f.write("This analysis considers Domain of Change context - a basket is 'non-default' only if it doesn't match the default baskets defined for that specific domain.\n\n")
        
        f.write(f"### Usage Statistics\n")
        f.write(f"- **{len(non_default_data)} total instances** of non-default outcome basket usage\n")
        f.write(f"- **{len(non_default_counter)} unique non-default baskets** identified\n")
        
        domain_totals = {domain: len(baskets) for domain, baskets in default_baskets_by_domain.items()}
        f.write(f"- **Default baskets by domain**: {domain_totals}\n\n")
        
        f.write("### Most Frequently Used Non-Default Baskets\n")
        for basket, count in non_default_counter.most_common(10):
            domains = ', '.join(set(domain_usage[basket]))
            f.write(f"- **{basket}**: {count} uses (in {domains})\n")
        f.write("\n")
        
        f.write("### Domain-Aware Consolidation Potential\n")
        f.write(f"- **{len(consolidation_analysis['domain_exact_matches'])} exact matches within domain** - ready for consolidation\n")
        f.write(f"- **{len(consolidation_analysis['cross_domain_exact_matches'])} exact matches in wrong domain** - review domain assignment\n")
        f.write(f"- **{len(consolidation_analysis['domain_high_similarity'])} high similarity within domain** - consider consolidation\n")
        f.write(f"- **{len(consolidation_analysis['cross_domain_high_similarity'])} high similarity across domains** - consider domain reassignment\n")
        f.write(f"- **{len(consolidation_analysis['manual_review_needed'])} moderate similarity** - manual review needed\n")
        f.write(f"- **{len(consolidation_analysis['unique_concepts'])} unique concepts** - keep separate\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("1. **Domain Context Matters**: Some baskets marked as 'non-default' are actually default baskets used in wrong domains\n")
        f.write("2. **Cross-Domain Usage**: Several baskets appear across multiple domains, suggesting either flexible concepts or potential domain misclassification\n")
        f.write("3. **Manual Review Guidance**: For baskets needing review, similar default baskets are provided to aid decision-making\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Process domain-specific exact matches**: Immediate consolidation opportunities\n")
        f.write("2. **Review cross-domain exact matches**: May indicate domain classification issues\n")
        f.write("3. **Evaluate high similarity matches**: Consider consolidation with guidance provided\n")
        f.write("4. **Preserve genuine innovations**: Keep unique concepts that represent new outcome types\n\n")
        
        f.write("---\n")
        f.write("*This analysis uses domain-aware matching to provide more accurate consolidation recommendations*\n")
    
    print(f"Domain-aware executive summary saved to: {summary_file}")

def main():
    """Main analysis function"""
    
    print("=== Domain-Aware Non-Default Outcome Baskets Analysis ===")
    
    # Step 1: Extract default outcome baskets by domain
    default_baskets_by_domain, all_default_baskets = extract_default_outcome_baskets_by_domain()
    
    # Step 2: Extract non-default baskets with domain context
    non_default_data, non_default_counter, domain_usage = extract_non_default_baskets_with_domain()
    
    if not non_default_counter:
        print("No non-default outcome baskets found.")
        return
    
    # Step 3: Find domain-aware semantic matches
    unique_non_default_baskets = list(non_default_counter.keys())
    matches = find_domain_aware_matches(unique_non_default_baskets, non_default_data, default_baskets_by_domain, all_default_baskets)
    
    # Step 4: Analyze consolidation opportunities with domain awareness
    consolidation_analysis = analyze_domain_aware_consolidation(
        non_default_data, matches, non_default_counter, default_baskets_by_domain
    )
    
    # Step 5: Generate reports
    generate_domain_aware_reports(
        default_baskets_by_domain, non_default_data, non_default_counter, 
        matches, consolidation_analysis, domain_usage
    )
    
    print("\n=== Domain-Aware Analysis Complete ===")
    print("Check the analysis-output-files directory for timestamped reports.")

if __name__ == "__main__":
    main()