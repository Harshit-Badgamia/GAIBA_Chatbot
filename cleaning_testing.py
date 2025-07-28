from data_cleaning import load_marketing_data_from_kaggle, clean_marketing_data

def main():
    # Step 1: Load the marketing data from Kaggle
    df_raw = load_marketing_data_from_kaggle()
    print("✅ Raw marketing data loaded successfully.")
    print(df_raw.head(), "\n")

    # Step 2: Clean the dataset
    df_cleaned = clean_marketing_data(df_raw)
    print("✅ Cleaned marketing data:")
    print(df_cleaned.head())

if __name__ == "__main__":
    main()
