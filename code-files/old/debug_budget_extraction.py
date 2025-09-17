import pandas as pd
import numpy as np
import re

# Read the CSV file with proper header
df = pd.read_csv('Data from Projects Digital Portfolio.v2.csv', header=1)

# Clean column names
df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Filter out rows without grant codes
grant_data = df[df["Grant's GANT Code - Current phase"].notna()].copy()

print("Column names:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

print(f"\nChecking budget columns:")
for col in df.columns:
    if 'Budget' in col:
        print(f"Budget column: {col}")
        print(f"Sample values:")
        sample_values = grant_data[col].head(10)
        for idx, val in sample_values.items():
            print(f"  Row {idx}: {repr(val)}")
        print()

# Let's also check the raw data around the budget columns
print("Raw data inspection for budget columns:")
budget_cols = [col for col in df.columns if 'Budget' in col]
for col in budget_cols:
    print(f"\n{col}:")
    col_index = df.columns.get_loc(col)
    print(f"Column index: {col_index}")
    for i in range(2, min(10, len(df))):  # Start from row 2 (first actual data row)
        val = df.iloc[i, col_index]
        print(f"  Row {i}: {repr(val)}")

# Check if there's a different pattern in the data
print("\nChecking for numeric patterns in budget-related columns:")
for col in budget_cols:
    print(f"\n{col}:")
    for idx, val in grant_data[col].items():
        if pd.notna(val):
            grant_name = grant_data.loc[idx, "Grant's Name"]
            print(f"  {grant_name[:50]}...")
            print(f"    Value: {repr(val)}")
            # Try to extract any numbers
            if isinstance(val, str):
                numbers = re.findall(r'[\d,]+', val)
                if numbers:
                    print(f"    Numbers found: {numbers}")
            print()