"""Transparency metrics: logging completeness and join integrity.

Transparency in this system is operationalised as auditability: you cannot
explain or review a decision you did not record, and you cannot run fairness
analysis if predictions will not join to demographics. These metrics make those
preconditions measurable.
"""
from __future__ import annotations

import pandas as pd

from .alerts import MetricResult
from .config import REQUIRED_LOG_COLUMNS


def logging_completeness(df: pd.DataFrame) -> MetricResult:
    """Fraction of required fields populated across all rows (lower is worse).

    Computed as the share of (row, required-column) cells that are non-null.
    """
    name = "transparency.logging_completeness"
    dim = "transparency"
    if not len(df):
        return MetricResult(name, None, dim, {"reason": "empty logs"})
    present = [c for c in REQUIRED_LOG_COLUMNS if c in df.columns]
    if not present:
        return MetricResult(name, 0.0, dim, {"reason": "no required columns present"})
    completeness = float(df[present].notna().to_numpy().mean())
    per_col = {c: float(df[c].notna().mean()) for c in present}
    return MetricResult(name, completeness, dim, {"per_column": per_col})


def join_integrity(
    logs: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    key: str = "request_id",
) -> MetricResult:
    """Fraction of inference logs that successfully join to a demographic row.

    A low join rate undermines every fairness metric, so it is treated as a
    transparency precondition (lower is worse).
    """
    name = "transparency.join_integrity"
    dim = "transparency"
    if key not in logs.columns or key not in demographics.columns:
        return MetricResult(name, None, dim, {"reason": f"join key {key!r} missing"})
    if not len(logs):
        return MetricResult(name, None, dim, {"reason": "empty logs"})

    demo_keys = set(demographics[key].dropna().unique())
    matched = logs[key].isin(demo_keys).mean()
    return MetricResult(
        name,
        float(matched),
        dim,
        {
            "matched_rows": int(logs[key].isin(demo_keys).sum()),
            "total_rows": int(len(logs)),
        },
    )
