import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

        if isinstance(val, (list, set)):
            df = df[df[col].isin(val)]

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

        else:
            df = df[df[col] == val]

    return df


# ----------------------------
# Forecasting ROI
# ----------------------------
def forecast_roi(df, periods=6):
    if df.empty or 'ROI' not in df.columns:
        return "No matching campaigns found."

    df = df.copy()

    if "Date" in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    if len(df) < 5:
        return "Not enough data points for forecasting."

    try:
        model = ExponentialSmoothing(df['ROI'], trend='add', seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(periods)
        return forecast
    except Exception as e:
        return f"Forecasting error: {e}"


# ----------------------------
# Budget Scenario Simulation
# ----------------------------
def budget_scenario(df, budget_change_pct=10):
    if df.empty or 'ROI' not in df.columns:
        return "No matching campaigns found."

    avg_roi = df['ROI'].mean()
    adjusted_roi = avg_roi * (1 + (budget_change_pct/100))

    return {
        "Current Avg ROI": avg_roi,
        f"Adjusted ROI ({budget_change_pct}%)": adjusted_roi
    }


# ----------------------------
# Break-Even Analysis
# ----------------------------
def break_even_analysis(df):
    if df.empty or 'Acquisition_Cost' not in df.columns or 'Conversion_Rate' not in df.columns:
        return "No matching campaigns found."

    df = df.copy()
    df['Cost_per_Conversion'] = df['Acquisition_Cost'] / (df['Conversion_Rate'] * 100)

    return {
        "Average Cost per Conversion": df['Cost_per_Conversion'].mean(),
        "Median Cost per Conversion": df['Cost_per_Conversion'].median(),
        "Min Cost per Conversion": df['Cost_per_Conversion'].min(),
        "Max Cost per Conversion": df['Cost_per_Conversion'].max()
    }


# ----------------------------
# Run Full Forecasting Pipeline
# ----------------------------
def run_financial_forecasting(filters=None, forecast_periods=6, budget_change_pct=10):
    df = load_cleaned_data()
    df = apply_filters(df, filters)

    if df.empty:
        return {"Error": "No matching campaigns found for given filters.", "Filters Applied": filters}

    results = {
        "Filters Applied": filters if filters else "None",
        "ROI Forecast": forecast_roi(df, forecast_periods),
        "Budget Scenario": budget_scenario(df, budget_change_pct),
        "Break-Even Analysis": break_even_analysis(df)
    }

    return results
