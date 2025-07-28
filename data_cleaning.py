import os
import json
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# If running on Streamlit Cloud, this is needed
import streamlit as st

# -----------------------------------------
# Configure Kaggle credentials from Streamlit Secrets
# -----------------------------------------
def configure_kaggle_credentials():
    kaggle_dict = {
        "username": st.secrets["kaggle"]["username"],
        "key": st.secrets["kaggle"]["key"]
    }
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_path = os.path.join(kaggle_dir, "kaggle.json")

    with open(kaggle_path, "w") as f:
        json.dump(kaggle_dict, f)

    os.chmod(kaggle_path, 0o600)

# -----------------------------------------
# Load dataset from Kaggle
# -----------------------------------------
def load_marketing_data_from_kaggle():
    configure_kaggle_credentials()
    file_path = "marketing_campaign_data.csv"  # Replace with correct filename
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "manishabhatt22/marketing-campaign-performance-dataset",
        file_path
    )
    return df

# -----------------------------------------
# Data Cleaning Function
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
