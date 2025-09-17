import pandas as pd
import numpy as np
from datetime import datetime, date
import re

# Read the CSV file
df = pd.read_csv('Data from Projects Digital Portfolio.v2.csv')

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Display first few rows to understand the structure
print("\nFirst 5 rows:")
print(df.head())

# Look for the specific columns we need
budget_col = None
duration_col = None
start_date_col = None
end_date_col = None
prev_phase_col = None
current_phase_col = None

for col in df.columns:
    if 'Current Budget (CHF)' in col:
        budget_col = col
    elif 'Duration' in col:
        duration_col = col
    elif 'Current phase start' in col:
        start_date_col = col
    elif 'Current phase close' in col:
        end_date_col = col
    elif 'Previous phase' in col:
        prev_phase_col = col
    elif 'Current phase' in col and 'Code' in col:
        current_phase_col = col

print(f"\nFound columns:")
print(f"Budget: {budget_col}")
print(f"Duration: {duration_col}")
print(f"Start Date: {start_date_col}")
print(f"End Date: {end_date_col}")
print(f"Previous Phase: {prev_phase_col}")
print(f"Current Phase: {current_phase_col}")

# Extract the relevant data
if budget_col and duration_col and start_date_col and end_date_col and prev_phase_col and current_phase_col:
    # Create a subset with only the columns we need
    relevant_cols = [
        current_phase_col,
        prev_phase_col,
        "Grant's Name",
        budget_col,
        duration_col,
        start_date_col,
        end_date_col
    ]
    
    # Filter out rows that don't have grant information
    grant_data = df[relevant_cols].dropna(subset=[current_phase_col])
    
    print(f"\nNumber of grants found: {len(grant_data)}")
    print(f"Sample of grant data:")
    print(grant_data.head(10))
    
    # Clean up the data and save for further analysis
    grant_data.to_csv('cleaned_grant_data.csv', index=False)
    print("\nData saved to 'cleaned_grant_data.csv'")