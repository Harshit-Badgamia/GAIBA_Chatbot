# marketing_analysis.py
"""
General analysis wrapper for the cleaned marketing dataset.

This file provides JSON-serializable helpers that the backend LLM can call
for general (non user-specific) analysis. For user-specific queries, the LLM
will be allowed to run freeform analysis in-app.
"""
import pandas as pd
import time
import json
import hashlib
from typing import Any, Dict, Optional, List

# Import your loader. The loader signature may vary, so _load_df handles both.
from data_loading import load_cleaned_data

LOG_PATH = "marketing_analysis_calls.log"


def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _log(call_record: Dict[str, Any]):
    call_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(call_record, default=str) + "\n")


# ------------------
# Load once - wrapper uses loader from data_loading.py
# ------------------
_df_cache: Optional[pd.DataFrame] = None


def _load_df() -> pd.DataFrame:
    """
    Try to load cleaned data via the user-provided loader.
    Attempt common signatures so it's robust to how load_cleaned_data was implemented.
    """
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    try:
        # try zero-arg
        _df_cache = load_cleaned_data()
    except TypeError:
        try:
            # try common defaults used earlier (zip + filename)
            _df_cache = load_cleaned_data("marketing_campaign_dataset.zip", "marketing_campaign_dataset_cleaned.csv")
        except Exception as e:
            raise RuntimeError(f"Unable to call load_cleaned_data(): {e}")
    except Exception as e:
        raise RuntimeError(f"Unable to call load_cleaned_data(): {e}")

    if not isinstance(_df_cache, pd.DataFrame):
        raise RuntimeError("load_cleaned_data did not return a pandas DataFrame")
    return _df_cache


# ------------------
# Filter util (keeps simple: equality, list, range by tuple)
# ------------------
def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if not filters:
        return df.copy()
    dff = df.copy()
    for col, val in filters.items():
        if col not in dff.columns:
            # skip silently (LLM tool should check returned 'no_data' later)
            continue
        # between operator: ("between", start, end)
        if isinstance(val, tuple) and len(val) >= 1 and isinstance(val[0], str) and val[0].lower() == "between":
            _, start, end = val[0], val[1], val[2]
            if "date" in col.lower():
                dff[col] = pd.to_datetime(dff[col], errors="coerce")
                dff = dff[(dff[col] >= pd.to_datetime(start)) & (dff[col] <= pd.to_datetime(end))]
            else:
                dff = dff[(dff[col] >= start) & (dff[col] <= end)]
        # other comparison tuple patterns: (">", value), ("<", value), (">=", value), ("<=", value)
        elif isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str):
            op = val[0]
            v = val[1]
            if op == ">":
                dff = dff[dff[col] > v]
            elif op == "<":
                dff = dff[dff[col] < v]
            elif op == ">=":
                dff = dff[dff[col] >= v]
            elif op == "<=":
                dff = dff[dff[col] <= v]
            else:
                # unsupported tuple op; skip
                continue
        elif isinstance(val, (list, set, tuple)):
            dff = dff[dff[col].isin(val)]
        else:
            dff = dff[dff[col] == val]
    return dff


# ------------------
# Metric calculators (safe, non-destructive)
# ------------------
def calculate_ctr(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    # handle case-insensitive column names by checking variants
    impressions_col = next((c for c in dff.columns if c.lower() == "impressions"), None)
    clicks_col = next((c for c in dff.columns if c.lower() == "clicks"), None)
    if impressions_col and clicks_col:
        dff["CTR"] = (dff[clicks_col] / dff[impressions_col]).replace([pd.NA, pd.inf], 0) * 100
    return dff


def calculate_cpc(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    cost_col = next((c for c in dff.columns if c.lower() in ("acquisition_cost", "cost", "spend")), None)
    clicks_col = next((c for c in dff.columns if c.lower() == "clicks"), None)
    if cost_col and clicks_col:
        dff["CPC"] = (dff[cost_col] / dff[clicks_col]).replace([pd.NA, pd.inf], 0)
    return dff


def calculate_cpm(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    cost_col = next((c for c in dff.columns if c.lower() in ("acquisition_cost", "cost", "spend")), None)
    impressions_col = next((c for c in dff.columns if c.lower() == "impressions"), None)
    if cost_col and impressions_col:
        dff["CPM"] = (dff[cost_col] / dff[impressions_col]).replace([pd.NA, pd.inf], 0) * 1000
    return dff


# ------------------
# Aggregations (return serializable dicts)
# ------------------
def _df_head_records(df: pd.DataFrame, max_rows: int = 20):
    return df.head(max_rows).fillna("").to_dict(orient="records")


def top_campaigns_by_roi(df: pd.DataFrame, n: int = 5):
    if df is None or df.empty:
        return []
    # detect ROI column case-insensitively
    roi_col = next((c for c in df.columns if c.lower() == "roi"), None)
    if not roi_col:
        return []
    out = df.sort_values(by=roi_col, ascending=False).head(n)
    cols_to_return = ["campaign_id", roi_col]
    if "Channel_Used" in out.columns:
        cols_to_return.append("Channel_Used")
    elif "channel" in out.columns:
        cols_to_return.append("channel")
    # ensure requested cols exist, otherwise filter
    cols_to_return = [c for c in cols_to_return if c in out.columns]
    return out[cols_to_return].fillna("").to_dict(orient="records")


def channel_performance(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    # Choose grouping column
    group_col = None
    if "Channel_Used" in df.columns:
        group_col = "Channel_Used"
    elif "channel" in df.columns:
        group_col = "channel"
    else:
        return {}

    grp = df.groupby(group_col)
    metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for name, g in grp:
        roi_cols = [c for c in g.columns if c.lower() == "roi"]
        avg_roi = float(g[roi_cols[0]].mean()) if roi_cols else None

        impressions_col = next((c for c in g.columns if c.lower() == "impressions"), None)
        clicks_col = next((c for c in g.columns if c.lower() == "clicks"), None)
        if impressions_col and clicks_col:
            # safe division
            total_impr = g[impressions_col].sum()
            total_clicks = g[clicks_col].sum()
            avg_ctr = float((total_clicks / total_impr) * 100) if total_impr and total_impr != 0 else None
        else:
            avg_ctr = None

        spend_col = next((c for c in g.columns if c.lower() in ("spend", "acquisition_cost", "cost")), None)
        total_spend = float(g[spend_col].sum()) if spend_col else None

        metrics[name] = {"avg_roi": avg_roi, "avg_ctr": avg_ctr, "total_spend": total_spend}
    return metrics


def campaign_type_performance(df: pd.DataFrame):
    if df is None or df.empty or "Campaign_Type" not in df.columns:
        return {}
    grp = df.groupby("Campaign_Type")
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, v in grp:
        roi_cols = [c for c in v.columns if c.lower() == "roi"]
        avg_roi = float(v[roi_cols[0]].mean()) if roi_cols else None

        impressions_col = next((c for c in v.columns if c.lower() == "impressions"), None)
        clicks_col = next((c for c in v.columns if c.lower() == "clicks"), None)
        if impressions_col and clicks_col:
            total_impr = v[impressions_col].sum()
            total_clicks = v[clicks_col].sum()
            avg_ctr = float((total_clicks / total_impr) * 100) if total_impr and total_impr != 0 else None
        else:
            avg_ctr = None

        out[k] = {"avg_roi": avg_roi, "avg_ctr": avg_ctr}
    return out


def location_performance(df: pd.DataFrame):
    if df is None or df.empty or "Location" not in df.columns:
        return {}
    grp = df.groupby("Location")
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k, v in grp:
        roi_cols = [c for c in v.columns if c.lower() == "roi"]
        avg_roi = float(v[roi_cols[0]].mean()) if roi_cols else None
        out[k] = {"avg_roi": avg_roi}
    return out


# ------------------
# Main wrapper: run_full_marketing_analysis
# ------------------
def run_full_marketing_analysis(filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    df = _load_df()
    dff = apply_filters(df, filters)
    if dff is None or dff.empty:
        result = {"error": "no_data_for_filters", "filters": filters}
        _log({"tool": "run_full_marketing_analysis", "params": {"filters": filters}, "result_meta": _checksum(result)})
        return {"metadata": {"checksum": _checksum(result)}, "analysis": result}

    # compute some derived metrics safely
    dff = calculate_ctr(dff)
    dff = calculate_cpc(dff)
    dff = calculate_cpm(dff)

    analysis_results = {
        "filters_applied": filters or {},
        "n_rows": len(dff),
        "top_campaigns": top_campaigns_by_roi(dff, n=10),
        "channel_performance": channel_performance(dff),
        "campaign_type_performance": campaign_type_performance(dff),
        "location_performance": location_performance(dff),
        "sample_rows": _df_head_records(dff, max_rows=20),
    }
    meta = {"checksum": _checksum(analysis_results)}
    _log({"tool": "run_full_marketing_analysis", "params": {"filters": filters}, "result_meta": meta})
    return {"metadata": meta, "analysis": analysis_results}
