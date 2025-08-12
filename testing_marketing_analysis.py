import pandas as pd
from data_cleaning import load_marketing_data_from_zip, clean_marketing_data
from marketing_analysis import run_full_marketing_analysis

# Load & clean data
df = load_marketing_data_from_zip()
df = clean_marketing_data(df)

# Run analysis
results = run_full_marketing_analysis(df)

# Access outputs directly by key
print("=== Top Campaigns ===")
print(results["Top Campaigns"])

print("\n=== Channel Performance ===")
print(results["Channel Performance"])

print("\n=== Campaign Type Performance ===")
print(results["Campaign Type Performance"])

print("\n=== Location Performance ===")
print(results["Location Performance"])
