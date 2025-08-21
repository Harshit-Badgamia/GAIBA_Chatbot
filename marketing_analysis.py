import pandas as pd
from data_loading import load_cleaned_data


# ----------------------------
# Utility: Apply Filters
# ----------------------------
def apply_filters(df, filters: dict):
    """
    Filter the dataframe based on a dictionary of {column: value}.
    - Supports:
        - Single values (exact match)
        - Lists/tuples/sets (multiple matches)
        - Range filters with operators: 
            {"ROI": (">", 20)}
            {"Acquisition_Cost": ("between", 1000, 5000)}
            {"Date": ("between", "2023-01-01", "2023-06-30")}
    """
    if not filters:
        return df

    for col, val in filters.items():
        if col not in df.columns:
            continue

        # Handle multi-value filters
        if isinstance(val, (list, set)):
            df = df[df[col].isin(val)]

        # Handle range and operator filters
        elif isinstance(val, tuple):
            op = val[0]

            if op == ">":
                df = df[df[col] > val[1]]
            elif op == "<":
                df = df[df[col] < val[1]]
            elif op == ">=":
                df = df[df[col] >= val[1]]
            elif op == "<=":
                df = df[df[col] <= val[1]]
            elif op == "between":
                start, end = val[1], val[2]
                if col.lower() == "date":
                    df[col] = pd.to_datetime(df[col])
                    df = df[(df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))]
                else:
                    df = df[(df[col] >= start) & (df[col] <= end)]

        # Handle single exact match
        else:
            df = df[df[col] == val]

    return df


# ----------------------------
# Metric Calculations
# ----------------------------
def calculate_ctr(df):
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
    return df

def calculate_cpc(df):
    if 'Acquisition_Cost' in df.columns and 'Clicks' in df.columns:
        df['CPC'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, pd.NA)
    return df

def calculate_cpm(df):
    if 'Acquisition_Cost' in df.columns and 'Impressions' in df.columns:
        df['CPM'] = (df['Acquisition_Cost'] / df['Impressions']) * 1000
    return df


# ----------------------------
# Aggregated Analysis
# ----------------------------
def top_campaigns_by_roi(df, n=5):
    if df.empty:
        return "No matching campaigns found."
    return df.sort_values(by='ROI', ascending=False).head(n)

def channel_performance(df):
    if df.empty:
        return "No matching campaigns found."
    return df.groupby('Channel_Used')[['ROI', 'CTR', 'CPC', 'CPM']].mean().sort_values(by='ROI', ascending=False)

def campaign_type_performance(df):
    if df.empty:
        return "No matching campaigns found."
    return df.groupby('Campaign_Type')[['ROI', 'CTR', 'CPC', 'CPM']].mean().sort_values(by='ROI', ascending=False)

def location_performance(df):
    if df.empty:
        return "No matching campaigns found."
    return df.groupby('Location')[['ROI', 'CTR', 'CPC', 'CPM']].mean().sort_values(by='ROI', ascending=False)


# ----------------------------
# Full Analysis Pipeline
# ----------------------------
def run_full_marketing_analysis(filters=None):
    df = load_cleaned_data()
    df = apply_filters(df, filters)

    if df.empty:
        return {"Error": "No matching campaigns found for given filters.", "Filters Applied": filters}

    # Calculate metrics first
    df = calculate_ctr(df)
    df = calculate_cpc(df)
    df = calculate_cpm(df)

    analysis_results = {
        "Filters Applied": filters if filters else "None",
        "Top Campaigns": top_campaigns_by_roi(df),
        "Channel Performance": channel_performance(df),
        "Campaign Type Performance": campaign_type_performance(df),
        "Location Performance": location_performance(df)
    }

    return df, analysis_results
