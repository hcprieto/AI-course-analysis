import csv

file_path = '/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev/source-files/FB-Digital-Grants-Outcomes-Data.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    count = 0
    print("=== ANALYSIS OF NON-DEFAULT OUTCOME BASKETS ===\n")
    
    for i, row in enumerate(reader, start=2):
        if len(row) >= 19 and row[18].strip():
            default_ob = row[17].strip() if len(row) > 17 else ''
            non_default_ob = row[18].strip()
            
            # Check if non-default contains different content
            if non_default_ob and non_default_ob != default_ob:
                print(f'Row {i}:')
                print(f'  Grant: {row[0]}')
                print(f'  Domain: {row[16]}')
                print(f'  Default OB: {default_ob}')
                print(f'  Non-Default OB: {non_default_ob}')
                print(f'  Outcome Description: {row[7][:150]}...')
                print()
                count += 1
                if count >= 15:
                    break
    
    print(f"Found {count} rows with truly different non-default outcome baskets.")