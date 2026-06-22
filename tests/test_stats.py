"""Tests for the statistical primitives."""
from __future__ import annotations

import numpy as np
import pytest

from rai_tracker.stats import (
    categorical_psi,
    expected_calibration_error,
    population_stability_index,
)


def test_psi_identical_distributions_is_zero() -> None:
    dist = [10, 20, 30, 40]
    assert population_stability_index(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_psi_increases_with_divergence() -> None:
    base = [25, 25, 25, 25]
    mild = [20, 30, 25, 25]
    severe = [5, 5, 5, 85]
    assert population_stability_index(base, mild) < population_stability_index(base, severe)


def test_psi_is_symmetric_enough_and_positive() -> None:
    a = [10, 20, 70]
    b = [70, 20, 10]
    assert population_stability_index(a, b) > 0


def test_psi_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [1, 2, 3])


def test_psi_empty_raises() -> None:
    with pytest.raises(ValueError):
        population_stability_index([], [])


def test_categorical_psi_aligns_missing_categories() -> None:
    base = {"a": 50, "b": 50}
    actual = {"a": 50, "c": 50}  # 'b' vanished, 'c' appeared
    assert categorical_psi(base, actual) > 0


def test_ece_perfectly_calibrated_low() -> None:
    # Confidence equals accuracy in each region => near-zero ECE.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, size=5000)
    correct = rng.uniform(0, 1, size=5000) < conf
    assert expected_calibration_error(conf, correct, n_bins=10) < 0.05


def test_ece_overconfident_high() -> None:
    conf = np.full(1000, 0.99)
    correct = np.zeros(1000, dtype=bool)  # always wrong but always confident
    assert expected_calibration_error(conf, correct) > 0.9


def test_ece_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.6], [True])
