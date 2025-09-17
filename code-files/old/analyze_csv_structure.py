import pandas as pd
import numpy as np

# Read the CSV file without header to understand the structure
df = pd.read_csv('Data from Projects Digital Portfolio.v2.csv', header=None)

print("Dataset shape:", df.shape)
print("\nFirst 10 rows:")
for i in range(min(10, len(df))):
    print(f"Row {i}: {df.iloc[i, :10].tolist()}")  # Show first 10 columns

# Look at specific rows that might contain headers
print("\nRow 0 (potential header 1):")
print(df.iloc[0, :20].tolist())

print("\nRow 1 (potential header 2):")
print(df.iloc[1, :20].tolist())

print("\nRow 2 (potential header 3):")
print(df.iloc[2, :20].tolist())

# Find the actual data rows by looking for grant codes
print("\nLooking for grant codes in first column:")
for i in range(min(15, len(df))):
    val = df.iloc[i, 0]
    if pd.notna(val) and val != "Grant's GANT code - Previous phase":
        print(f"Row {i}: {val}")

# Let's examine what looks like the actual header row
print("\nExamining row 1 (seems to be the main header):")
header_row = df.iloc[1, :].tolist()
for i, val in enumerate(header_row[:25]):  # First 25 columns
    if pd.notna(val):
        print(f"Column {i}: {val}")