# utils.py
"""
Shared utilities used by the GAIBA marketing agent.

- budget_allocator: simple proportional allocator with optional min/max caps.
- simple logging & checksum helpers (consistent with other modules).
- small helpers for basic formatting / parsing.
"""
from typing import Dict, Any, Optional
import json
import hashlib
import time

LOG_PATH = "utils_calls.log"


def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _log(call_record: Dict[str, Any]):
    call_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_record, default=str) + "\n")
    except Exception:
        # never raise from logging helper
        pass


def budget_allocator(
    total_budget: float,
    expected_rois: Dict[str, float],
    method: str = "proportional",
    min_caps: Optional[Dict[str, float]] = None,
    max_caps: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Allocate `total_budget` across channels given expected ROIs.
    - method: "proportional" (default) or "equal" (fallback); LP not implemented.
    - min_caps / max_caps: dicts mapping channel -> min/max amounts (optional).

    Returns a dict: {"total_budget": float, "by_channel": {...}, "method": str}
    """
    # Defensive copy and normalization
    chans = list(expected_rois.keys())
    rois = [max(0.0, float(expected_rois.get(c, 0.0))) for c in chans]

    allocation: Dict[str, float] = {}

    if method == "proportional":
        s = sum(rois)
        if s == 0:
            # if no signal, fallback to equal split
            allocation = {c: float(total_budget) / max(1, len(chans)) for c in chans}
        else:
            allocation = {c: float(total_budget) * (r / s) for c, r in zip(chans, rois)}
    elif method == "equal":
        allocation = {c: float(total_budget) / max(1, len(chans)) for c in chans}
    else:
        # unknown method -> equal split
        allocation = {c: float(total_budget) / max(1, len(chans)) for c in chans}

    # Apply min caps
    if min_caps:
        for c, v in (min_caps.items()):
            if c in allocation:
                allocation[c] = max(allocation[c], float(v))

    # Apply max caps
    if max_caps:
        for c, v in (max_caps.items()):
            if c in allocation:
                allocation[c] = min(allocation[c], float(v))

    # Re-normalize if total differs slightly from total_budget
    total_alloc = sum(allocation.values())
    if total_alloc == 0:
        # fallback equal split
        allocation = {c: float(total_budget) / max(1, len(chans)) for c in chans}
        total_alloc = sum(allocation.values())

    # scale to match total_budget exactly (preserve ratios after caps)
    if abs(total_alloc - float(total_budget)) > 1e-6:
        factor = float(total_budget) / float(total_alloc)
        allocation = {c: float(v) * factor for c, v in allocation.items()}

    result = {"total_budget": float(total_budget), "by_channel": allocation, "method": method}
    _log({"tool": "budget_allocator", "params": {"total_budget": total_budget, "expected_rois": expected_rois, "min_caps": min_caps, "max_caps": max_caps, "method": method}, "result_meta": _checksum(result)})
    return result


def parse_clusters(cluster_input: str):
    """
    Parse a comma-separated cluster string like '1,2,a' -> ['1','2','a'] (trimmed).
    Returns list of strings.
    """
    if not cluster_input:
        return []
    parts = [p.strip() for p in cluster_input.split(",") if p.strip()]
    return parts


def format_currency(amount: float, symbol: str = "INR"):
    """Simple currency formatting, returns string like 'INR 12,345.67'"""
    try:
        amt = float(amount)
        # minimal formatting (no locale dependency)
        return f"{symbol} {amt:,.2f}"
    except Exception:
        return f"{symbol} {amount}"


# Additional small helpers you may find useful
def safe_divide(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
    try:
        if denominator in (0, 0.0, None):
            return default
        return numerator / denominator
    except Exception:
        return default
