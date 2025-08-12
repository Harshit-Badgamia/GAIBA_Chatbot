# marketing_analysis.py

import pandas as pd

# -----------------------------------------
# Function to calculate additional marketing metrics
# -----------------------------------------
def calculate_marketing_metrics(df):
    # Ensure numeric types for calculations
    num_cols = ['Clicks', 'Impressions', 'Acquisition_Cost', 'Engagement_Score', 'ROI', 'Conversion_Rate']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # CTR (Click Through Rate)
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        df['CTR'] = df['Clicks'] / df['Impressions']

    # CPC (Cost Per Click)
    if 'Acquisition_Cost' in df.columns and 'Clicks' in df.columns:
        df['CPC'] = df['Acquisition_Cost'] / df['Clicks']

    # CPM (Cost Per 1000 Impressions)
    if 'Acquisition_Cost' in df.columns and 'Impressions' in df.columns:
        df['CPM'] = (df['Acquisition_Cost'] / df['Impressions']) * 1000

    # Engagement per Click
    if 'Engagement_Score' in df.columns and 'Clicks' in df.columns:
        df['Engagement_per_Click'] = df['Engagement_Score'] / df['Clicks']

    # ROI Category
    if 'ROI' in df.columns:
        df['ROI_Category'] = pd.cut(
            df['ROI'],
            bins=[-float('inf'), 0, 0.5, 1, float('inf')],
            labels=['Loss', 'Low', 'Moderate', 'High']
        )

    return df


# -----------------------------------------
# Group-wise summary
# -----------------------------------------
def summarize_by_group(df, group_col, metric_cols=None):
    if metric_cols is None:
        metric_cols = ['ROI', 'CTR', 'CPC', 'CPM', 'Conversion_Rate']

    return df.groupby(group_col)[metric_cols].mean().reset_index()


# -----------------------------------------
# Time series trends
# -----------------------------------------
def time_series_analysis(df, date_col='Date', freq='M'):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        time_df = df.groupby(pd.Grouper(key=date_col, freq=freq)).agg({
            'ROI': 'mean',
            'CTR': 'mean',
            'Conversion_Rate': 'mean'
        }).reset_index()
        return time_df
    else:
        raise ValueError(f"{date_col} column not found in DataFrame")


# -----------------------------------------
# Top performing campaigns
# -----------------------------------------
def top_performing_campaigns(df, metric='ROI', top_n=5):
    if metric not in df.columns:
        raise ValueError(f"{metric} column not found in DataFrame")
    return df.sort_values(by=metric, ascending=False).head(top_n)
