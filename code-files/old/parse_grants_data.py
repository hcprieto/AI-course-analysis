import pandas as pd
import numpy as np

# Read the CSV file with proper header handling
df = pd.read_csv('/Users/pedro/Library/CloudStorage/OneDrive-FONDATIONBOTNAR/SLE - Digital - SLE/Theme accompaniment/2025.07 - Digital theme Grants analysis/digital-grants-analysis-dev/source-files/FBs-Digital-Grants-Data.v5.csv', 
                 header=1)  # Use row 1 as header (0-indexed)

# Clean column names
df.columns = df.columns.str.strip()

print("Column names after using row 1 as header:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")

# Check the first few rows to understand the structure
print("\nFirst 5 rows:")
print(df.head())

# Try to identify the key columns by looking at the data
print("\nSample data from key columns:")
key_cols = ['GAMS Code', 'Grant\'s short name', 'Grant\'s Name', 'Brief Summary', 'Grant\'s Summary (from GAMS)']
for col in key_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].head(3).values)
    else:
        print(f"\n{col}: NOT FOUND")

# Look for sub-portfolio columns
subportfolio_cols = [col for col in df.columns if 'sub-portfolio' in col.lower()]
print(f"\nSub-portfolio columns found: {subportfolio_cols}")