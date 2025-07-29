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
            with zip_ref.open(csv_filename) as file:
                df = pd.read_csv(file, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


# -----------------------------------------
# Data Cleaning Functions
# -----------------------------------------
def clean_marketing_data(df):
    df = df.copy()
    # Strip currency symbols and convert to float
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(r'[\$,]', '', regex=True).astype(float)

    # Convert Duration to integer number of days
    df['Duration'] = df['Duration'].replace(r'days',"",regex=True).astype(float)

    # Standardize campaign types
    df['Campaign_Type'] = df['Campaign_Type'].str.strip().str.lower()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows with too many missing values
    df.dropna(thresh=len(df.columns) - 2, inplace=True)

    # Fill remaining NaNs with median or mode
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
