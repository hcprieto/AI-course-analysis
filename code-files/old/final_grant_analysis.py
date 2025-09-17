import pandas as pd
import numpy as np
import re
from datetime import datetime

# Read the CSV file with proper header
df = pd.read_csv('Data from Projects Digital Portfolio.v2.csv', header=1)

# Clean column names
df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Filter out rows without grant codes
grant_data = df[df["Grant's GANT Code - Current phase"].notna()].copy()

print(f"Total grants found: {len(grant_data)}")

# Extract the required information
extracted_data = []

for idx, row in grant_data.iterrows():
    # Basic grant information
    prev_phase = row["Grant's GANT code - Previous phase"] if pd.notna(row["Grant's GANT code - Previous phase"]) else None
    current_phase = row["Grant's GANT Code - Current phase"]
    grant_name = row["Grant's Name"]
    
    # Budget information - use the correct column "Current Budget (CHF).1"
    current_budget_raw = row["Current Budget (CHF).1"]
    current_budget = None
    if pd.notna(current_budget_raw):
        # Extract numeric value from budget string
        budget_str = str(current_budget_raw).replace(" ", "").replace(",", "")
        try:
            current_budget = int(budget_str)
        except:
            # Try to extract number from string
            budget_numbers = re.findall(r'[\d,]+', str(current_budget_raw))
            if budget_numbers:
                clean_number = budget_numbers[0].replace(",", "")
                current_budget = int(clean_number)
    
    # Previous phase budget
    prev_budget_raw = row["Budget from previous phase (CHF)"]
    prev_budget = None
    if pd.notna(prev_budget_raw):
        try:
            prev_budget_str = str(prev_budget_raw).replace(" ", "").replace(",", "")
            prev_budget = int(prev_budget_str)
        except:
            budget_numbers = re.findall(r'[\d,]+', str(prev_budget_raw))
            if budget_numbers:
                clean_number = budget_numbers[0].replace(",", "")
                prev_budget = int(clean_number)
    
    # Duration information
    duration_raw = row["Duration"]
    duration_months = None
    if pd.notna(duration_raw):
        # Extract number from duration string
        duration_numbers = re.findall(r'\d+', str(duration_raw))
        if duration_numbers:
            duration_months = int(duration_numbers[0])
    
    # Date information
    start_date = row["Current phase start"] if pd.notna(row["Current phase start"]) else None
    end_date = row["Current phase close"] if pd.notna(row["Current phase close"]) else None
    
    # Categorize budget
    budget_category = "Unknown"
    if current_budget:
        if current_budget < 1000000:  # Less than 1M CHF
            budget_category = "Small"
        elif current_budget < 5000000:  # Less than 5M CHF
            budget_category = "Medium"
        else:  # 5M+ CHF
            budget_category = "Large"
    
    # Categorize duration
    duration_category = "Unknown"
    if duration_months:
        if duration_months < 24:  # Less than 2 years
            duration_category = "Short"
        elif duration_months < 48:  # Less than 4 years
            duration_category = "Medium"
        else:  # 4+ years
            duration_category = "Long"
    
    # Check if it's a continuation (has previous phase)
    is_continuation = "Yes" if prev_phase else "No"
    
    # Check current status (Active/Closed)
    current_status = "Active"  # Default to active
    if end_date:
        try:
            end_date_clean = str(end_date).strip()
            if end_date_clean and end_date_clean.lower() != "nan":
                # Try to parse different date formats
                today = datetime.now()
                
                # Handle formats like "Jan-28", "Dec-24", "Mar-25"
                if "-" in end_date_clean:
                    parts = end_date_clean.split("-")
                    if len(parts) == 2:
                        month_str, year_str = parts
                        # Convert month abbreviation to number
                        month_map = {
                            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                        }
                        if month_str in month_map:
                            month = month_map[month_str]
                            year = int("20" + year_str) if len(year_str) == 2 else int(year_str)
                            end_date_parsed = datetime(year, month, 1)
                            current_status = "Closed" if end_date_parsed < today else "Active"
        except:
            current_status = "Active"  # Default to active if can't parse
    
    extracted_data.append({
        "Grant Name": grant_name,
        "Current Phase Code": current_phase,
        "Previous Phase Code": prev_phase,
        "Current Budget (CHF)": current_budget,
        "Previous Budget (CHF)": prev_budget,
        "Budget Category": budget_category,
        "Duration (months)": duration_months,
        "Duration Category": duration_category,
        "Start Date": start_date,
        "End Date": end_date,
        "Is Continuation": is_continuation,
        "Current Status": current_status
    })

# Create DataFrame and save
result_df = pd.DataFrame(extracted_data)

# Sort by budget (descending)
result_df = result_df.sort_values("Current Budget (CHF)", ascending=False, na_position='last')

print("\nGRANT BUDGET AND DURATION ANALYSIS")
print("=" * 80)

# Display summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"Total grants analyzed: {len(result_df)}")
print(f"Grants with current budget data: {result_df['Current Budget (CHF)'].notna().sum()}")
print(f"Grants with previous budget data: {result_df['Previous Budget (CHF)'].notna().sum()}")
print(f"Grants with duration data: {result_df['Duration (months)'].notna().sum()}")

print(f"\nBUDGET CATEGORIES:")
budget_counts = result_df['Budget Category'].value_counts()
for category, count in budget_counts.items():
    print(f"  {category}: {count}")

print(f"\nDURATION CATEGORIES:")
duration_counts = result_df['Duration Category'].value_counts()
for category, count in duration_counts.items():
    print(f"  {category}: {count}")

print(f"\nCONTINUATION STATUS:")
continuation_counts = result_df['Is Continuation'].value_counts()
for category, count in continuation_counts.items():
    print(f"  {category}: {count}")

print(f"\nCURRENT STATUS:")
status_counts = result_df['Current Status'].value_counts()
for category, count in status_counts.items():
    print(f"  {category}: {count}")

# Display detailed results
print(f"\nDETAILED GRANT INFORMATION:")
print("=" * 80)

for idx, row in result_df.iterrows():
    print(f"\n{row['Grant Name']}")
    print(f"  Current Phase: {row['Current Phase Code']}")
    if row['Previous Phase Code']:
        print(f"  Previous Phase: {row['Previous Phase Code']}")
    
    # Budget information
    if row['Current Budget (CHF)']:
        print(f"  Current Budget: {row['Current Budget (CHF)']:,} CHF ({row['Budget Category']})")
    else:
        print(f"  Current Budget: Not available")
    
    if row['Previous Budget (CHF)']:
        print(f"  Previous Budget: {row['Previous Budget (CHF)']:,} CHF")
    
    # Duration and dates
    if row['Duration (months)']:
        print(f"  Duration: {row['Duration (months)']} months ({row['Duration Category']})")
    else:
        print(f"  Duration: Not available")
    
    print(f"  Period: {row['Start Date']} to {row['End Date']}")
    print(f"  Continuation: {row['Is Continuation']}")
    print(f"  Status: {row['Current Status']}")

# Save to CSV
result_df.to_csv('grant_budget_duration_analysis.csv', index=False)
print(f"\n\nData saved to 'grant_budget_duration_analysis.csv'")

# Additional analysis
print(f"\nADDITIONAL ANALYSIS:")
print("=" * 80)

# Budget statistics for current phase
budget_data = result_df[result_df['Current Budget (CHF)'].notna()]['Current Budget (CHF)']
if len(budget_data) > 0:
    print(f"Current Budget Statistics (CHF):")
    print(f"  Total grants with budget data: {len(budget_data)}")
    print(f"  Mean: {budget_data.mean():,.0f}")
    print(f"  Median: {budget_data.median():,.0f}")
    print(f"  Min: {budget_data.min():,.0f}")
    print(f"  Max: {budget_data.max():,.0f}")
    print(f"  Total portfolio value: {budget_data.sum():,.0f}")

# Previous budget statistics
prev_budget_data = result_df[result_df['Previous Budget (CHF)'].notna()]['Previous Budget (CHF)']
if len(prev_budget_data) > 0:
    print(f"\nPrevious Budget Statistics (CHF):")
    print(f"  Total grants with previous budget data: {len(prev_budget_data)}")
    print(f"  Mean: {prev_budget_data.mean():,.0f}")
    print(f"  Median: {prev_budget_data.median():,.0f}")
    print(f"  Min: {prev_budget_data.min():,.0f}")
    print(f"  Max: {prev_budget_data.max():,.0f}")
    print(f"  Total previous portfolio value: {prev_budget_data.sum():,.0f}")

# Duration statistics
duration_data = result_df[result_df['Duration (months)'].notna()]['Duration (months)']
if len(duration_data) > 0:
    print(f"\nDuration Statistics (months):")
    print(f"  Mean: {duration_data.mean():.1f}")
    print(f"  Median: {duration_data.median():.1f}")
    print(f"  Min: {duration_data.min()}")
    print(f"  Max: {duration_data.max()}")

# Cross-tabulation analysis
print(f"\nCROSS-TABULATION ANALYSIS:")
print("=" * 80)

# Budget vs Duration
budget_duration_cross = pd.crosstab(result_df['Budget Category'], result_df['Duration Category'])
print(f"\nBudget Category vs Duration Category:")
print(budget_duration_cross)

# Continuation vs Budget
continuation_budget_cross = pd.crosstab(result_df['Is Continuation'], result_df['Budget Category'])
print(f"\nContinuation vs Budget Category:")
print(continuation_budget_cross)

# Status vs Budget
status_budget_cross = pd.crosstab(result_df['Current Status'], result_df['Budget Category'])
print(f"\nCurrent Status vs Budget Category:")
print(status_budget_cross)

print(f"\nEnd of Analysis")
print("=" * 80)