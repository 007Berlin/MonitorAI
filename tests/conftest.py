"""Shared fixtures: a synthetic but realistic triage dataset.

The generator can inject a controllable fairness skew so that fairness tests
have a known ground truth to assert against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_dataset(
    n: int = 2000,
    *,
    seed: int = 7,
    urgent_rate_a: float = 0.20,
    urgent_rate_b: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (logs, demographics) joinable on request_id.

    Group "A" and group "B" receive 'urgent' at the configured rates, allowing a
    deterministic demographic-parity gap of ``|urgent_rate_a - urgent_rate_b|``.
    """
    rng = np.random.default_rng(seed)
    request_id = np.arange(1, n + 1)
    group = rng.choice(["A", "B"], size=n)

    priority = np.empty(n, dtype=object)
    for i, g in enumerate(group):
        rate = urgent_rate_a if g == "A" else urgent_rate_b
        r = rng.random()
        if r < rate:
            priority[i] = "urgent"
        elif r < rate + 0.5:
            priority[i] = "standard"
        else:
            priority[i] = "low"

    confidence = np.clip(rng.normal(0.8, 0.1, size=n), 0.0, 1.0)
    region = rng.choice(["north", "south", "east", "west"], size=n)

    logs = pd.DataFrame(
        {
            "request_id": request_id,
            "predicted_priority": priority,
            "confidence": confidence,
            "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 24, n), "h"),
            "region": region,
            "channel_count": rng.poisson(3, size=n),
        }
    )
    demographics = pd.DataFrame(
        {
            "request_id": request_id,
            "age_band": rng.choice(["18-34", "35-54", "55+"], size=n),
            "group": group,
        }
    )
    return logs, demographics


@pytest.fixture
def balanced_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_dataset(urgent_rate_a=0.20, urgent_rate_b=0.20)


@pytest.fixture
def skewed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ~17-point parity gap, comfortably into RED territory.
    return make_dataset(urgent_rate_a=0.30, urgent_rate_b=0.13)
