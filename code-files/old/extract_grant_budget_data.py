import pandas as pd
import numpy as np
import re
from datetime import datetime

# Read the CSV file with proper header
df = pd.read_csv('Data from Projects Digital Portfolio.v2.csv', header=1)

# Clean column names
df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Filter out rows without grant codes (the actual grant data starts from row 1 after headers)
grant_data = df[df["Grant's GANT Code - Current phase"].notna()].copy()

print(f"Total grants found: {len(grant_data)}")

# Extract the required information
extracted_data = []

for idx, row in grant_data.iterrows():
    # Basic grant information
    prev_phase = row["Grant's GANT code - Previous phase"] if pd.notna(row["Grant's GANT code - Previous phase"]) else None
    current_phase = row["Grant's GANT Code - Current phase"]
    grant_name = row["Grant's Name"]
    
    # Budget information - clean the budget values
    current_budget_raw = row["Current Budget (CHF)"]
    current_budget = None
    if pd.notna(current_budget_raw):
        # Clean budget: remove spaces, commas, and extract numeric value
        budget_clean = str(current_budget_raw).replace(" ", "").replace(",", "").replace("█", "").replace("■", "")
        try:
            current_budget = int(budget_clean)
        except:
            # Try to extract number from string
            budget_numbers = re.findall(r'\d+', str(current_budget_raw))
            if budget_numbers:
                current_budget = int(''.join(budget_numbers))
    
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
    current_status = "Unknown"
    if end_date:
        try:
            # Try to parse the end date
            end_date_clean = str(end_date).strip()
            if end_date_clean and end_date_clean != "nan":
                # Various date formats to try
                date_formats = ["%b-%y", "%b-%Y", "%Y-%m", "%m-%Y", "%Y-%m-%d", "%d-%m-%Y"]
                end_date_parsed = None
                for fmt in date_formats:
                    try:
                        end_date_parsed = datetime.strptime(end_date_clean, fmt)
                        break
                    except:
                        continue
                
                if end_date_parsed:
                    current_status = "Closed" if end_date_parsed < datetime.now() else "Active"
                else:
                    current_status = "Active"  # Default to active if can't parse
        except:
            current_status = "Active"  # Default to active if can't parse
    
    extracted_data.append({
        "Grant Name": grant_name,
        "Current Phase Code": current_phase,
        "Previous Phase Code": prev_phase,
        "Current Budget (CHF)": current_budget,
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

print("\nExtracted Grant Data:")
print("=" * 80)

# Display summary statistics
print(f"\nSUMMARY STATISTICS:")
print(f"Total grants analyzed: {len(result_df)}")
print(f"Grants with budget data: {result_df['Current Budget (CHF)'].notna().sum()}")
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
    print(f"  Budget: {row['Current Budget (CHF)']:,} CHF ({row['Budget Category']})" if row['Current Budget (CHF)'] else "  Budget: Not available")
    print(f"  Duration: {row['Duration (months)']} months ({row['Duration Category']})" if row['Duration (months)'] else "  Duration: Not available")
    print(f"  Period: {row['Start Date']} to {row['End Date']}")
    print(f"  Continuation: {row['Is Continuation']}")
    print(f"  Status: {row['Current Status']}")

# Save to CSV
result_df.to_csv('grant_budget_duration_analysis.csv', index=False)
print(f"\n\nData saved to 'grant_budget_duration_analysis.csv'")

# Additional analysis
print(f"\nADDITIONAL ANALYSIS:")
print("=" * 80)

# Budget statistics
budget_data = result_df[result_df['Current Budget (CHF)'].notna()]['Current Budget (CHF)']
if len(budget_data) > 0:
    print(f"Budget Statistics (CHF):")
    print(f"  Mean: {budget_data.mean():,.0f}")
    print(f"  Median: {budget_data.median():,.0f}")
    print(f"  Min: {budget_data.min():,.0f}")
    print(f"  Max: {budget_data.max():,.0f}")
    print(f"  Total: {budget_data.sum():,.0f}")

# Duration statistics
duration_data = result_df[result_df['Duration (months)'].notna()]['Duration (months)']
if len(duration_data) > 0:
    print(f"\nDuration Statistics (months):")
    print(f"  Mean: {duration_data.mean():.1f}")
    print(f"  Median: {duration_data.median():.1f}")
    print(f"  Min: {duration_data.min()}")
    print(f"  Max: {duration_data.max()}")