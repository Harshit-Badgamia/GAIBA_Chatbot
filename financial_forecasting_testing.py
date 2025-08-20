from financial_forecasting import run_financial_forecasting

# Run forecasting pipeline
results = run_financial_forecasting(
    forecast_periods=6,   # forecast next 6 periods
    budget_change_pct=15, # simulate +15% budget scenario
    by_cluster=True       # group by Cluster_Name
)

# Print results
print("\n=== ROI Forecast by Cluster ===")
for cluster, forecast in results["ROI Forecast"].items():
    print(f"\n{cluster}:\n{forecast}")

print("\n=== Budget Scenario by Cluster ===")
for cluster, scenario in results["Budget Scenario"].items():
    print(f"\n{cluster}:")
    for k, v in scenario.items():
        print(f"  {k}: {v:.2f}")

print("\n=== Break-Even Analysis by Cluster ===")
for cluster, metrics in results["Break-Even Analysis"].items():
    print(f"\n{cluster}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
