# testing_marketing_analysis.py

from marketing_analysis import run_full_marketing_analysis

# Example 1: No filters (full dataset)
print("\n=== Marketing Analysis (Full Dataset) ===")
df, results = run_full_marketing_analysis()
print("Top Campaigns:\n", results["Top Campaigns"])
print("Channel Performance:\n", results["Channel Performance"])
print("Campaign Type Performance:\n", results["Campaign Type Performance"])
print("Location Performance:\n", results["Location Performance"])

# Example 2: Filter by target audience and channel
filters = {"Target_Audience": "Men 18-24", "Channel_Used": ["YouTube", "Instagram"]}
print("\n=== Marketing Analysis (Filtered) ===")
df, results = run_full_marketing_analysis(filters=filters)
print("Applied Filters:", results["Filters Applied"])
print("Top Campaigns:\n", results["Top Campaigns"])
