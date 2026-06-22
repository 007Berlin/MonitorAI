"""Reusable statistical primitives shared across metric modules.

These are intentionally dependency-light (numpy only) and pure functions, which
makes them straightforward to unit-test and reason about.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def population_stability_index(
    expected: np.ndarray | list[float],
    actual: np.ndarray | list[float],
) -> float:
    """Population Stability Index between two distributions.

    PSI = sum( (a_i - e_i) * ln(a_i / e_i) ) over bins, where ``e`` and ``a`` are
    the proportion of mass in each bin for the expected (baseline) and actual
    distributions. Common rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift,
    > 0.25 significant shift.

    Both inputs are treated as (unnormalised) bin counts or proportions over the
    *same* bins, in the same order. A small epsilon avoids division-by-zero and
    log-of-zero when a bin is empty in one distribution.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same shape")
    if e.size == 0:
        raise ValueError("distributions must be non-empty")

    e = e / e.sum() if e.sum() > 0 else e
    a = a / a.sum() if a.sum() > 0 else a
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(
    expected_counts: dict[str, float],
    actual_counts: dict[str, float],
) -> float:
    """PSI for categorical distributions given count dicts (aligns categories)."""
    categories = sorted(set(expected_counts) | set(actual_counts))
    e = [expected_counts.get(c, 0.0) for c in categories]
    a = [actual_counts.get(c, 0.0) for c in categories]
    return population_stability_index(e, a)


def expected_calibration_error(
    confidences: np.ndarray | list[float],
    correct: np.ndarray | list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` equal-width confidence bins and
    returns the sample-weighted average gap between mean confidence and observed
    accuracy in each bin. 0 = perfectly calibrated.
    """
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.shape != acc.shape:
        raise ValueError("confidences and correct must have the same shape")
    if conf.size == 0:
        raise ValueError("inputs must be non-empty")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize -> bin index in [1, n_bins]; clip the upper edge (conf == 1.0).
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, n_bins - 1)

    total = conf.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bin_conf = float(conf[mask].mean())
        bin_acc = float(acc[mask].mean())
        ece += (n / total) * abs(bin_conf - bin_acc)
    return ece
