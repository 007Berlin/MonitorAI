"""Tests for fairness, robustness, and transparency metrics."""
from __future__ import annotations

import numpy as np

from rai_tracker import fairness, robustness, transparency
from tests.conftest import make_dataset

# ---------------- Fairness ----------------

def test_demographic_parity_near_zero_when_balanced(balanced_data) -> None:
    logs, demo = balanced_data
    merged = logs.merge(demo, on="request_id")
    res = fairness.demographic_parity_difference(merged, "group")
    assert res.value is not None
    assert res.value < 0.05


def test_demographic_parity_detects_skew(skewed_data) -> None:
    logs, demo = skewed_data
    merged = logs.merge(demo, on="request_id")
    res = fairness.demographic_parity_difference(merged, "group")
    assert res.value is not None
    assert res.value > 0.10  # the injected gap should surface
    assert res.detail["worst_group"] in {"A", "B"}


def test_demographic_parity_skips_small_groups() -> None:
    logs, demo = make_dataset(n=40)
    merged = logs.merge(demo, on="request_id")
    res = fairness.demographic_parity_difference(merged, "group", min_group_size=1000)
    assert res.value is None


def test_tpr_gap_returns_none_without_ground_truth(balanced_data) -> None:
    logs, demo = balanced_data
    merged = logs.merge(demo, on="request_id")
    res = fairness.true_positive_rate_gap(merged, "group")
    assert res.value is None


def test_tpr_gap_computes_with_ground_truth() -> None:
    logs, demo = make_dataset(n=2000, seed=1)
    merged = logs.merge(demo, on="request_id")
    # Construct ground truth where the model is worse for group B on 'urgent'.
    rng = np.random.default_rng(2)
    merged["true_priority"] = merged["predicted_priority"]
    urgent_b = merged[(merged["group"] == "B") & (merged["predicted_priority"] == "urgent")].index
    flip = rng.choice(urgent_b, size=len(urgent_b) // 2, replace=False)
    merged.loc[flip, "predicted_priority"] = "standard"
    res = fairness.true_positive_rate_gap(merged, "group")
    assert res.value is not None
    assert res.value > 0.0


def test_calibration_gap_with_ground_truth(balanced_data) -> None:
    logs, demo = balanced_data
    merged = logs.merge(demo, on="request_id")
    merged["true_priority"] = merged["predicted_priority"]  # perfectly correct
    res = fairness.calibration_gap(merged, "group")
    assert res.value is not None
    assert res.value >= 0.0


# ---------------- Robustness ----------------

def test_prediction_drift_zero_against_self(balanced_data) -> None:
    logs, _ = balanced_data
    res = robustness.prediction_distribution_drift(logs, logs)
    assert res.value is not None
    assert res.value < 1e-6


def test_prediction_drift_positive_when_mix_changes() -> None:
    base, _ = make_dataset(urgent_rate_a=0.1, urgent_rate_b=0.1, seed=3)
    cur, _ = make_dataset(urgent_rate_a=0.6, urgent_rate_b=0.6, seed=4)
    res = robustness.prediction_distribution_drift(cur, base)
    assert res.value > 0.1


def test_feature_drift_numeric(balanced_data) -> None:
    logs, _ = balanced_data
    shifted = logs.copy()
    shifted["channel_count"] = shifted["channel_count"] + 10
    res = robustness.feature_drift(shifted, logs, "channel_count")
    assert res.value is not None
    assert res.value > 0.1


def test_feature_drift_absent_feature(balanced_data) -> None:
    logs, _ = balanced_data
    res = robustness.feature_drift(logs, logs, "does_not_exist")
    assert res.value is None


def test_low_confidence_rate(balanced_data) -> None:
    logs, _ = balanced_data
    logs = logs.copy()
    logs.loc[logs.index[:200], "confidence"] = 0.1
    res = robustness.low_confidence_rate(logs)
    assert res.value is not None
    assert res.value >= 0.1


def test_volume_deviation() -> None:
    logs, _ = make_dataset(n=1400)
    res = robustness.volume_deviation(logs, expected_daily_volume=2000)
    assert res.value == (2000 - 1400) / 2000


def test_error_rate_flags_bad_rows(balanced_data) -> None:
    logs, _ = balanced_data
    logs = logs.copy()
    logs.loc[logs.index[:10], "confidence"] = np.nan
    res = robustness.error_rate(logs)
    assert res.value is not None
    assert res.detail["bad_rows"] >= 10


# ---------------- Transparency ----------------

def test_logging_completeness_full(balanced_data) -> None:
    logs, _ = balanced_data
    res = transparency.logging_completeness(logs)
    assert res.value == 1.0


def test_logging_completeness_with_gaps(balanced_data) -> None:
    logs, _ = balanced_data
    logs = logs.copy()
    logs.loc[logs.index[:100], "confidence"] = np.nan
    res = transparency.logging_completeness(logs)
    assert res.value < 1.0


def test_join_integrity_full(balanced_data) -> None:
    logs, demo = balanced_data
    res = transparency.join_integrity(logs, demo)
    assert res.value == 1.0


def test_join_integrity_partial(balanced_data) -> None:
    logs, demo = balanced_data
    res = transparency.join_integrity(logs, demo.iloc[: len(demo) // 2])
    assert res.value is not None
    assert 0.4 < res.value < 0.6
