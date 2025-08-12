# testing_marketing_analysis.py

from data_cleaning import load_marketing_data_from_zip, clean_marketing_data
from marketing_analysis import calculate_marketing_metrics, summarize_by_group, time_series_analysis, top_performing_campaigns

# Step 1: Load and clean data
df = load_marketing_data_from_zip('marketing_campaign_dataset.zip', 'marketing_campaign_dataset.csv')
df = clean_marketing_data(df)

# Step 2: Calculate marketing metrics
df = calculate_marketing_metrics(df)

# Step 3: Display basic preview
print("\n===== Data with Calculated Metrics =====")
print(df.head())

# Step 4: Group summary by Campaign_Type
print("\n===== Summary by Campaign_Type =====")
print(summarize_by_group(df, 'Campaign_Type'))

# Step 5: Time-series trends
print("\n===== Monthly Trends =====")
print(time_series_analysis(df, date_col='Date', freq='M'))

# Step 6: Top 5 campaigns by ROI
print("\n===== Top 5 Campaigns by ROI =====")
print(top_performing_campaigns(df, metric='ROI', top_n=5))
