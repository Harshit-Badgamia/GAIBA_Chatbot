# data_cleaning.py

import pandas as pd
import os
import zipfile

# -----------------------------------------
# Function to extract and load marketing data from a local zip
# -----------------------------------------
def load_marketing_data_from_zip(zip_path='marketing_campaign_dataset.zip', csv_filename='marketing_campaign_dataset.csv'):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')
        df = pd.read_csv(os.path.join('data', csv_filename), encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

# -----------------------------------------
# Data Cleaning Functions
# -----------------------------------------
def clean_marketing_data(df):
    # Strip currency symbols and convert to float
    if 'Acquisition_Cost' in df.columns:
        df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(r'[\$,]', '', regex=True).astype(float)

    # Convert Duration (e.g., '2 weeks') to integer number of days
    if 'Duration' in df.columns:
        df['Duration_Days'] = df['Duration'].str.extract(r'(\d+)').astype(float) * 7

    # Standardize campaign types
    if 'Campaign_Type' in df.columns:
        df['Campaign_Type'] = df['Campaign_Type'].str.strip().str.lower()

    # Drop rows with too many missing values
    df.dropna(thresh=len(df.columns) - 2, inplace=True)

    # Fill remaining NaNs with median or mode
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
