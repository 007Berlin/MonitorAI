"""Fairness metrics over triage predictions joined to demographic data.

All metrics here are *group* metrics: they compare how the classifier behaves
across demographic groups. Two families are provided:

* Selection-rate metrics (demographic parity) -- available immediately from
  predictions alone, so they are the day-one fairness signal.
* Outcome-conditioned metrics (true-positive-rate gap, calibration gap) --
  require confirmed ground truth and so are lagged and partial.

Groups smaller than ``min_group_size`` are skipped: their estimates are too
noisy to act on and small cohorts risk re-identification.
"""
from __future__ import annotations

import pandas as pd

from .alerts import HIGHER_IS_WORSE, MetricResult
from .stats import expected_calibration_error

URGENT = "urgent"


def _eligible_groups(df: pd.DataFrame, group_col: str, min_group_size: int) -> list[str]:
    counts = df[group_col].value_counts()
    return [g for g, n in counts.items() if n >= min_group_size]


def _reference_group(
    df: pd.DataFrame, group_col: str, groups: list[str], configured: str | None
) -> str | None:
    if configured is not None and configured in groups:
        return configured
    if not groups:
        return None
    # Default: the largest eligible group is the comparison baseline.
    return df[df[group_col].isin(groups)][group_col].value_counts().idxmax()


def demographic_parity_difference(
    df: pd.DataFrame,
    group_col: str,
    *,
    positive_label: str = URGENT,
    min_group_size: int = 30,
    reference_group: str | None = None,
) -> MetricResult:
    """Max absolute gap in ``positive_label`` selection rate across groups.

    Reported as the largest deviation of any eligible group from the reference
    group's selection rate. Per-group rates and sample sizes are returned in
    ``detail`` for the audit trail.
    """
    groups = _eligible_groups(df, group_col, min_group_size)
    ref = _reference_group(df, group_col, groups, reference_group)
    name = "fairness.demographic_parity_diff"
    dim = "fairness"

    if ref is None or len(groups) < 2:
        return MetricResult(name, None, dim, {"reason": "insufficient groups", "groups": groups})

    rates: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        sizes[g] = int(len(sub))
        rates[g] = float((sub["predicted_priority"] == positive_label).mean())

    ref_rate = rates[ref]
    diffs = {g: abs(r - ref_rate) for g, r in rates.items()}
    worst_group = max(diffs, key=lambda g: diffs[g])
    return MetricResult(
        name=name,
        value=diffs[worst_group],
        dimension=dim,
        detail={
            "positive_label": positive_label,
            "reference_group": ref,
            "selection_rates": rates,
            "group_sizes": sizes,
            "worst_group": worst_group,
        },
    )


def true_positive_rate_gap(
    df: pd.DataFrame,
    group_col: str,
    *,
    label_col: str = "true_priority",
    positive_label: str = URGENT,
    min_group_size: int = 30,
) -> MetricResult:
    """Max gap in true-positive rate (recall) for ``positive_label`` across groups.

    Equal-opportunity style metric. Requires confirmed outcomes in ``label_col``;
    returns a non-computable result if that column is absent.
    """
    name = "fairness.tpr_gap"
    dim = "fairness"
    if label_col not in df.columns:
        return MetricResult(name, None, dim, {"reason": f"no ground truth ({label_col})"})

    eligible = df[df[label_col] == positive_label]
    groups = _eligible_groups(eligible, group_col, min_group_size)
    if len(groups) < 2:
        return MetricResult(name, None, dim, {"reason": "insufficient positive samples per group"})

    tpr: dict[str, float] = {}
    for g in groups:
        sub = eligible[eligible[group_col] == g]
        tpr[g] = float((sub["predicted_priority"] == positive_label).mean())

    gap = max(tpr.values()) - min(tpr.values())
    return MetricResult(name, float(gap), dim, {"tpr_by_group": tpr, "label": positive_label})


def calibration_gap(
    df: pd.DataFrame,
    group_col: str,
    *,
    label_col: str = "true_priority",
    min_group_size: int = 30,
    n_bins: int = 10,
) -> MetricResult:
    """Max difference in expected calibration error (ECE) across groups.

    A model can be well-calibrated overall yet poorly calibrated for one group;
    this surfaces that. ``correct`` is defined as predicted priority matching the
    confirmed priority. Requires ground truth.
    """
    name = "fairness.calibration_gap"
    dim = "fairness"
    if label_col not in df.columns:
        return MetricResult(name, None, dim, {"reason": f"no ground truth ({label_col})"})

    groups = _eligible_groups(df, group_col, min_group_size)
    if len(groups) < 2:
        return MetricResult(name, None, dim, {"reason": "insufficient groups"})

    ece: dict[str, float] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        correct = (sub["predicted_priority"] == sub[label_col]).to_numpy()
        ece[g] = expected_calibration_error(sub["confidence"].to_numpy(), correct, n_bins=n_bins)

    gap = max(ece.values()) - min(ece.values())
    return MetricResult(name, float(gap), dim, {"ece_by_group": ece})


# Convenience registry so the orchestrator can map config direction uniformly.
DEFAULT_DIRECTION = HIGHER_IS_WORSE
