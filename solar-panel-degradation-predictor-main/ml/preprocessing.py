"""
Preprocessing module for solar panel efficiency dataset.
Extracted from notebooks/eda_cleaning.ipynb for reuse by both training and web app.

Cleaning pipeline:
1. Convert corrupted numeric columns (humidity, wind_speed, pressure) to numeric
2. Clip physically impossible outliers (irradiance, cloud_coverage, temperature, voltage)
3. Treat efficiency == 0 as a data-logging artifact and convert to missing
4. Fill missing categoricals with the literal string 'missing'
5. Fill missing numerics with median imputation
6. One-hot encode categorical columns (error_code, installation_type, string_id)
7. Drop id column (not a real feature)

Output: 20,000 rows × 22 columns, zero missing values, ready for modeling.
"""

import pandas as pd
from pathlib import Path


def clean_and_encode_dataset(input_path, output_path=None):
    """
    Load raw dataset, clean it, encode categoricals, and return cleaned DataFrame.
    Optionally save to CSV.
    
    Args:
        input_path (str or Path): Path to raw CSV file (train.csv)
        output_path (str or Path, optional): If provided, save cleaned CSV here
        
    Returns:
        pd.DataFrame: Cleaned and encoded dataset, ready for modeling
        
    Raises:
        FileNotFoundError: If input_path does not exist
        pd.errors.ParserError: If CSV is malformed
    """
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found at {input_path}")
    
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Step 1: Fix corrupted numeric columns (convert text like 'error', 'unknown' to NaN)
    for col in ['humidity', 'wind_speed', 'pressure']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 2: Clip physically impossible outliers
    # Rationale: irradiance cannot be negative, cloud coverage 0-100%, 
    # temperature <= 90°C reasonable for outdoor panels, voltage <= 100V typical for PV
    df['irradiance'] = df['irradiance'].clip(lower=0)
    df['cloud_coverage'] = df['cloud_coverage'].clip(upper=100)
    df['temperature'] = df['temperature'].clip(upper=90)
    df['voltage'] = df['voltage'].clip(upper=100)
    
    # Step 3: Treat efficiency == 0 as data artifact
    # Per Phase-I notebook: 631 rows had efficiency == 0, statistically identical 
    # to normal rows on all features, concluded to be logging artifact, not real failures
    df.loc[df['efficiency'] == 0, 'efficiency'] = pd.NA
    df['efficiency'] = pd.to_numeric(df['efficiency'], errors='coerce')
    
    # Step 4: Fill missing categoricals with 'missing'
    # Rationale: categorical features have some missing values; imputing with a 
    # distinct category preserves information that data was missing
    for col in ['error_code', 'installation_type']:
        df[col] = df[col].fillna('missing')
    
    # Step 5: Fill missing numeric values with median
    # Rationale: non-parametric, robust to outliers, preserves distribution shape
    numeric_cols = [
        'maintenance_count', 'panel_age', 'soiling_ratio',
        'cloud_coverage', 'temperature', 'voltage', 'irradiance',
        'module_temperature', 'current', 'pressure', 'humidity',
        'wind_speed', 'efficiency'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Step 6: One-hot encode categorical columns
    # drop_first=True to avoid multicollinearity trap (one category is always predictable)
    df_encoded = pd.get_dummies(
        df,
        columns=['error_code', 'installation_type', 'string_id'],
        drop_first=True
    )
    
    # Step 7: Drop id column (row label, not a feature)
    if 'id' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['id'])
    
    # Save if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_encoded.to_csv(output_path, index=False)
        print(f"✓ Cleaned dataset saved to {output_path}")
    
    return df_encoded


if __name__ == '__main__':
    # Quick test: load raw data and clean it
    # Expect output: cleaned dataset with no missing values
    df_clean = clean_and_encode_dataset(
        'data/train.csv',
        'data/train_cleaned.csv'
    )
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Columns: {list(df_clean.columns)}")
