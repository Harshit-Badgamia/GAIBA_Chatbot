import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from data_loading import load_marketing_data_from_zip   # loads cleaned + clustered data

# ----------------------------
# Forecasting ROI
# ----------------------------
def forecast_roi(df, periods=6, by_cluster=True):
    results = {}

    if by_cluster and "Cluster_Name" in df.columns:
        for cname in df["Cluster_Name"].unique():
            cluster_df = df[df["Cluster_Name"] == cname].copy()

            if "Date" in cluster_df.columns:
                cluster_df['Date'] = pd.to_datetime(cluster_df['Date'])
                cluster_df = cluster_df.set_index('Date')

            if len(cluster_df) > 5:
                model = ExponentialSmoothing(cluster_df['ROI'], trend='add', seasonal=None)
                fit = model.fit()
                forecast = fit.forecast(periods)
                results[cname] = forecast
    else:
        df = df.copy()
        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

        model = ExponentialSmoothing(df['ROI'], trend='add', seasonal=None)
        fit = model.fit()
        results["Overall"] = fit.forecast(periods)

    return results


# ----------------------------
# Budget Scenario Simulation
# ----------------------------
def budget_scenario(df, budget_change_pct=10, by_cluster=True):
    """
    Estimate new ROI if budget is adjusted by a percentage.
    Uses Cluster_Name if available.
    """
    results = {}

    if by_cluster and "Cluster_Name" in df.columns:
        for cname in df["Cluster_Name"].unique():
            avg_roi = df[df["Cluster_Name"] == cname]['ROI'].mean()
            adjusted_roi = avg_roi * (1 + (budget_change_pct/100))
            results[cname] = {
                "Current Avg ROI": avg_roi,
                f"Adjusted ROI ({budget_change_pct}%)": adjusted_roi
            }
    else:
        avg_roi = df['ROI'].mean()
        adjusted_roi = avg_roi * (1 + (budget_change_pct/100))
        results["Overall"] = {
            "Current Avg ROI": avg_roi,
            f"Adjusted ROI ({budget_change_pct}%)": adjusted_roi
        }

    return results


# ----------------------------
# Break-Even Analysis
# ----------------------------
def break_even_analysis(df, by_cluster=True):
    """
    Find average spend required to break even.
    Uses Acquisition_Cost and Conversion_Rate.
    Cluster-aware if Cluster_Name exists.
    """
    results = {}

    if 'Acquisition_Cost' not in df.columns or 'Conversion_Rate' not in df.columns:
        raise ValueError("Required columns missing in dataset.")

    if by_cluster and "Cluster_Name" in df.columns:
        for cname in df["Cluster_Name"].unique():
            cdf = df[df["Cluster_Name"] == cname].copy()
            cdf['Cost_per_Conversion'] = cdf['Acquisition_Cost'] / (cdf['Conversion_Rate'] * 100)
            results[cname] = {
                "Avg": cdf['Cost_per_Conversion'].mean(),
                "Median": cdf['Cost_per_Conversion'].median()
            }
    else:
        df = df.copy()
        df['Cost_per_Conversion'] = df['Acquisition_Cost'] / (df['Conversion_Rate'] * 100)
        results["Overall"] = {
            "Avg": df['Cost_per_Conversion'].mean(),
            "Median": df['Cost_per_Conversion'].median()
        }

    return results


# ----------------------------
# Run Full Forecasting Pipeline
# ----------------------------
def run_financial_forecasting(forecast_periods=6, budget_change_pct=10, by_cluster=True):
    """
    Run a full financial forecasting analysis and return dictionary
    with forecast results, scenario analysis, and break-even metrics.
    """
    df = load_marketing_data_from_zip()

    results = {
        "ROI Forecast": forecast_roi(df, forecast_periods, by_cluster),
        "Budget Scenario": budget_scenario(df, budget_change_pct, by_cluster),
        "Break-Even Analysis": break_even_analysis(df, by_cluster)
    }

    return results
