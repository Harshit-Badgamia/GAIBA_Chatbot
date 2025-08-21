# testing_financial_forecasting.py

from financial_forecasting import run_financial_forecasting

# Example 1: No filters (full dataset)
print("\n=== Financial Forecasting (Full Dataset) ===")
results = run_financial_forecasting(forecast_periods=6, budget_change_pct=15)
print(results)

# Example 2: Filter by cluster and channel
filters = {"Cluster_Name": "High Engagement, Shorter Duration", "Channel_Used": "YouTube"}
print("\n=== Financial Forecasting (Filtered) ===")
results = run_financial_forecasting(filters=filters, forecast_periods=6, budget_change_pct=20)
print("Applied Filters:", results["Filters Applied"])
print("ROI Forecast:\n", results["ROI Forecast"])
print("Budget Scenario:\n", results["Budget Scenario"])
print("Break-Even Analysis:\n", results["Break-Even Analysis"])
