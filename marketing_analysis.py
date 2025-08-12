import pandas as pd

def calculate_ctr(df):
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
    return df

def calculate_cpc(df):
    if 'Acquisition_Cost' in df.columns and 'Clicks' in df.columns:
        df['CPC'] = df['Acquisition_Cost'] / df['Clicks']
    return df

def calculate_cpm(df):
    if 'Acquisition_Cost' in df.columns and 'Impressions' in df.columns:
        df['CPM'] = (df['Acquisition_Cost'] / df['Impressions']) * 1000
    return df

def top_campaigns_by_roi(df, top_n=5):
    return df.sort_values(by='ROI', ascending=False).head(top_n)

def channel_performance(df):
    return df.groupby('Channel_Used')[['ROI', 'CTR', 'Conversion_Rate']].mean().sort_values(by='ROI', ascending=False)

def campaign_type_performance(df):
    return df.groupby('Campaign_Type')[['ROI', 'CTR', 'Conversion_Rate']].mean().sort_values(by='ROI', ascending=False)

def location_performance(df):
    return df.groupby('Location')[['ROI', 'CTR', 'Conversion_Rate']].mean().sort_values(by='ROI', ascending=False)

def run_full_marketing_analysis(df):
    # Calculate additional metrics
    df = calculate_ctr(df)
    df = calculate_cpc(df)
    df = calculate_cpm(df)

    # Return everything in a single dictionary
    return {
        "dataframe": df,  # Keep full cleaned dataset for reference
        "Top Campaigns": top_campaigns_by_roi(df),
        "Channel Performance": channel_performance(df),
        "Campaign Type Performance": campaign_type_performance(df),
        "Location Performance": location_performance(df)
    }

