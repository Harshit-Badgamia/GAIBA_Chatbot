# email.py
"""
Utilities for reading and validating an uploaded email dataset, sampling, redacting,
and a gated send stub to integrate with actual mail providers later.
"""
import pandas as pd
import time
import json
import hashlib
from typing import List, Dict, Any, Optional

LOG_PATH = "email_tool_calls.log"


def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _log(call_record: Dict[str, Any]):
    call_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(call_record, default=str) + "\n")


# ------------------
# Load / validate uploaded DataFrame (Streamlit will pass the pd.DataFrame)
# ------------------
def validate_email_df(df: pd.DataFrame) -> Dict[str, Any]:
    required = ["email"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        out = {"valid": False, "missing_columns": missing}
        _log({
            "tool": "validate_email_df",
            "params": {"columns": df.columns.tolist()},
            "result_meta": _checksum(out)
        })
        return out

    n_total = len(df)
    n_valid_email = df['email'].dropna().shape[0]
    out = {"valid": True, "n_total": n_total, "n_valid_email": n_valid_email}
    _log({"tool": "validate_email_df", "params": {}, "result_meta": _checksum(out)})
    return out


# ------------------
# Sample & redact
# ------------------
def mask_email(e: str) -> str:
    try:
        local, domain = e.split("@")
        return local[:2] + "***@" + domain
    except Exception:
        return "redacted"


def get_sample_recipients(
    email_df: pd.DataFrame,
    targeting: Optional[Dict[str, Any]] = None,
    n: int = 50,
    redact: bool = True
) -> Dict[str, Any]:
    dff = email_df.copy()

    # enforce opt-in if column exists
    if 'opt_in' in dff.columns:
        dff = dff[dff['opt_in'].isin([True, 'True', 'true', 'yes', 'Yes', 1])]

    # apply simple equality targeting
    if targeting:
        for k, v in targeting.items():
            if k in dff.columns:
                dff = dff[dff[k] == v]

    sampled = dff.sample(n=min(n, max(0, len(dff))), random_state=42) if len(dff) > 0 else dff

    if redact and 'email' in sampled.columns:
        sampled = sampled.copy()
        sampled['email'] = sampled['email'].astype(str).apply(mask_email)

    result = {
        "n_selected": len(sampled),
        "sample": sampled.fillna("").to_dict(orient="records")
    }
    _log({
        "tool": "get_sample_recipients",
        "params": {"n": n, "targeting": targeting},
        "result_meta": _checksum(result)
    })
    return {"metadata": {"checksum": _checksum(result)}, "result": result}


# ------------------
# Send stub (gated)
# ------------------
def send_email_batch_stub(
    campaign_spec: Dict[str, Any],
    template: Dict[str, Any],
    recipients: List[Dict[str, Any]],
    require_confirmation: bool = True
) -> Dict[str, Any]:
    meta = {
        "n_recipients": len(recipients),
        "checksum": _checksum({"campaign_spec": campaign_spec, "template": template})
    }

    if require_confirmation:
        out = {"status": "requires_confirmation", "meta": meta}
        _log({
            "tool": "send_email_batch_stub",
            "params": {"campaign_id": campaign_spec.get("campaign_id")},
            "result_meta": _checksum(out)
        })
        return {"metadata": meta, "result": out}

    # If confirmed: implement provider integration here (SendGrid/SES/Mailgun)
    job_id = f"mock_job_{int(time.time())}"
    out = {"status": "queued", "job_id": job_id, "n_recipients": len(recipients)}
    _log({
        "tool": "send_email_batch_stub",
        "params": {"campaign_id": campaign_spec.get("campaign_id")},
        "result_meta": _checksum(out)
    })
    return {"metadata": meta, "result": out}
