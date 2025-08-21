# data_loading.py

import pandas as pd
import zipfile

# -----------------------------------------
# Function to extract and load marketing data from a local zip
# -----------------------------------------
def load_cleaned_data(zip_path='marketing_campaign_dataset.zip', csv_filename='marketing_campaign_dataset_cleaned.csv'):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(csv_filename) as file:
                df = pd.read_csv(file, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        return df
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
