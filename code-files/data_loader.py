"""
Data Loading and Cleaning Module for Global Thriving Analysis

This module handles the loading and cleaning of the World Data 2.0-Data.csv file,
addressing CSV parsing issues and preparing data for analysis.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

class DataLoader:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.clean_data = None

    def load_raw_data(self):
        """Load raw CSV data with error handling for parsing issues"""
        try:
            # Try standard CSV loading first
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.raw_data)} rows")
            return True
        except Exception as e:
            print(f"Standard CSV loading failed: {e}")

            # Try with different parameters for problematic CSV
            try:
                self.raw_data = pd.read_csv(
                    self.data_path,
                    encoding='utf-8',
                    quotechar='"',
                    skipinitialspace=True
                )
                print(f"Loaded {len(self.raw_data)} rows with modified parameters")
                return True
            except Exception as e2:
                print(f"Modified CSV loading also failed: {e2}")
                return False

    def clean_numeric_columns(self):
        """Clean numeric columns that may have formatting issues"""
        if self.raw_data is None:
            return False

        # Create a copy for cleaning
        self.clean_data = self.raw_data.copy()

        # Define numeric columns and their expected names
        numeric_columns = {
            'Total population (2018)': 'population_2018',
            'GDP per Capita (2018)': 'gdp_per_capita_2018',
            'Unemployment Rate (2018)': 'unemployment_rate_2018',
            'Life satisfaction in Cantril Ladder (2018)': 'life_satisfaction_2018',
            'Life expectancy at birth (years)(2018)': 'life_expectancy_2018',
            'Deaths Due to Self-Harm (2019)': 'self_harm_deaths_2019',
            'Rate of Deaths Due to Self-Harm, per 100,000 people (2019)': 'self_harm_rate_2019',
            'Number of people with access to electricity (2020)': 'electricity_access_num_2020',
            '% of population with access to electricty (2020)': 'electricity_access_pct_2020',
            'Learning-Adjusted Years of School (2020)': 'education_years_2020'
        }

        # Clean and rename columns
        for old_name, new_name in numeric_columns.items():
            if old_name in self.clean_data.columns:
                # Remove quotes, commas, and extra spaces
                self.clean_data[new_name] = (
                    self.clean_data[old_name]
                    .astype(str)
                    .str.replace('"', '')
                    .str.replace(',', '')
                    .str.strip()
                    .replace('Not Avalable', np.nan)
                    .replace('', np.nan)
                )

                # Convert to numeric
                self.clean_data[new_name] = pd.to_numeric(
                    self.clean_data[new_name],
                    errors='coerce'
                )

        # Clean categorical columns
        categorical_renames = {
            'Country': 'country',
            'ISO3': 'iso3',
            'Continent': 'continent',
            'Most Common Cause of Death (2018)': 'death_cause_1',
            'Second Most Common Cause of Death (2018)': 'death_cause_2',
            'Third Most Common Cause of Death (2018)': 'death_cause_3',
            'Text': 'un_speech_text'
        }

        for old_name, new_name in categorical_renames.items():
            if old_name in self.clean_data.columns:
                self.clean_data[new_name] = self.clean_data[old_name]

        return True

    def remove_duplicates_and_filter(self):
        """Remove duplicate countries and filter to valid country data"""
        if self.clean_data is None:
            return False

        # Remove rows where country name appears to be UN speech text
        # Keep only rows with proper country names and continent data
        valid_continents = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania']

        self.clean_data = self.clean_data[
            self.clean_data['continent'].isin(valid_continents)
        ].copy()

        # Remove duplicates based on country name, keeping first occurrence
        self.clean_data = self.clean_data.drop_duplicates(subset=['country'], keep='first')

        # Reset index
        self.clean_data = self.clean_data.reset_index(drop=True)

        print(f"After cleaning: {len(self.clean_data)} unique countries")
        return True

    def get_health_indicators(self):
        """Extract health-related indicators for analysis"""
        if self.clean_data is None:
            return None

        health_cols = [
            'country', 'continent', 'iso3',
            'life_satisfaction_2018',
            'life_expectancy_2018',
            'self_harm_deaths_2019',
            'self_harm_rate_2019',
            'death_cause_1',
            'death_cause_2',
            'death_cause_3'
        ]

        return self.clean_data[health_cols].copy()

    def get_data_summary(self):
        """Generate summary statistics for loaded data"""
        if self.clean_data is None:
            return "No clean data available"

        summary = {
            'total_countries': len(self.clean_data),
            'continents': self.clean_data['continent'].value_counts().to_dict(),
            'missing_data': self.clean_data.isnull().sum().to_dict()
        }

        return summary

def load_and_clean_data(source_path):
    """Main function to load and clean the dataset"""
    loader = DataLoader(source_path)

    if not loader.load_raw_data():
        return None, None

    if not loader.clean_numeric_columns():
        return None, None

    if not loader.remove_duplicates_and_filter():
        return None, None

    return loader, loader.clean_data

if __name__ == "__main__":
    # Test the data loading
    source_path = "../source-files/World Data 2.0-Data.csv"
    loader, clean_data = load_and_clean_data(source_path)

    if loader:
        print("Data loading successful!")
        print(f"Shape: {clean_data.shape}")
        print("\nSummary:")
        summary = loader.get_data_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("Data loading failed!")