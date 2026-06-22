"""Robustness metrics: drift, confidence telemetry, and operational health.

These answer "is the model still operating in the world it was trained for, and
is the pipeline healthy?". They compare a current window against a baseline
(typically the training/reference distribution) and are cheap enough to run
daily.
"""
from __future__ import annotations

import pandas as pd

from .alerts import MetricResult
from .config import VALID_PRIORITIES
from .stats import categorical_psi, population_stability_index


def prediction_distribution_drift(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
) -> MetricResult:
    """PSI between the current and baseline priority-class mix."""
    name = "robustness.prediction_drift_psi"
    dim = "robustness"
    cur = current["predicted_priority"].value_counts().to_dict()
    base = baseline["predicted_priority"].value_counts().to_dict()
    psi = categorical_psi(base, cur)
    return MetricResult(name, psi, dim, {"current": cur, "baseline": base})


def feature_drift(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    feature: str,
    *,
    n_bins: int = 10,
) -> MetricResult:
    """PSI for a single metadata feature (numeric binned, categorical aligned)."""
    name = f"robustness.feature_drift_psi.{feature}"
    dim = "robustness"
    if feature not in current.columns or feature not in baseline.columns:
        return MetricResult(name, None, dim, {"reason": f"feature {feature!r} absent"})

    cur_series = current[feature].dropna()
    base_series = baseline[feature].dropna()
    if cur_series.empty or base_series.empty:
        return MetricResult(name, None, dim, {"reason": "empty feature column"})

    if pd.api.types.is_numeric_dtype(base_series):
        # Bin on the baseline's quantile edges so bins are comparable.
        edges = pd.qcut(base_series, q=min(n_bins, base_series.nunique()),
                        retbins=True, duplicates="drop")[1]
        base_counts = (
            pd.cut(base_series, bins=edges, include_lowest=True).value_counts().sort_index()
        )
        cur_counts = (
            pd.cut(cur_series, bins=edges, include_lowest=True).value_counts().sort_index()
        )
        psi = population_stability_index(base_counts.to_numpy(), cur_counts.to_numpy())
    else:
        psi = categorical_psi(
            base_series.value_counts().to_dict(),
            cur_series.value_counts().to_dict(),
        )
    return MetricResult(name, float(psi), dim, {"feature": feature})


def mean_confidence(df: pd.DataFrame) -> MetricResult:
    """Mean prediction confidence over the window (lower is worse)."""
    value = float(df["confidence"].mean()) if len(df) else None
    return MetricResult("robustness.mean_confidence", value, "robustness", {})


def low_confidence_rate(df: pd.DataFrame, *, cutoff: float = 0.5) -> MetricResult:
    """Share of predictions below ``cutoff`` confidence (higher is worse)."""
    if not len(df):
        return MetricResult("robustness.low_confidence_rate", None, "robustness", {})
    rate = float((df["confidence"] < cutoff).mean())
    return MetricResult(
        "robustness.low_confidence_rate", rate, "robustness", {"cutoff": cutoff}
    )


def volume_deviation(df: pd.DataFrame, *, expected_daily_volume: int) -> MetricResult:
    """Relative deviation of observed volume from the expected daily baseline.

    Returns ``|observed - expected| / expected`` so a 30% swing in either
    direction reads as 0.30 regardless of sign.
    """
    name = "robustness.volume_deviation"
    dim = "robustness"
    if expected_daily_volume <= 0:
        return MetricResult(name, None, dim, {"reason": "expected volume not set"})
    observed = len(df)
    dev = abs(observed - expected_daily_volume) / expected_daily_volume
    return MetricResult(
        name, float(dev), dim, {"observed": observed, "expected": expected_daily_volume}
    )


def error_rate(df: pd.DataFrame) -> MetricResult:
    """Fraction of rows with a missing prediction or confidence (higher is worse)."""
    if not len(df):
        return MetricResult("robustness.error_rate", None, "robustness", {})
    bad = df["predicted_priority"].isna() | df["confidence"].isna()
    # Treat priorities outside the vocabulary as errors too.
    bad = bad | ~df["predicted_priority"].isin(VALID_PRIORITIES)
    rate = float(bad.mean())
    return MetricResult("robustness.error_rate", rate, "robustness", {"bad_rows": int(bad.sum())})
