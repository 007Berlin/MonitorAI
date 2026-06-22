"""Generate realistic synthetic triage data matching the assignment schema.

This is a standalone, importable generator (separate from the test fixtures) so
that the example scripts and a reviewer can produce data without depending on
the test suite. It models the exact schema described in the brief:

    inference logs : request_id, predicted_priority, confidence, timestamp,
                     + input metadata (region, channel, request_type, text_length)
    demographics   : request_id, age_band, gender, ethnicity_band, disability_flag

A controllable ``urgent_rate`` per demographic group lets us inject a known
fairness skew so the monitoring system has something real to detect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIORITIES = ("urgent", "standard", "low")


def generate(
    n: int = 2000,
    *,
    seed: int = 42,
    day: str = "2025-01-01",
    urgent_rate_by_group: dict[str, float] | None = None,
    drift: float = 0.0,
    low_confidence_injection: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (logs, demographics) for a single day of ~``n`` requests.

    Args:
        n: Approximate number of requests for the day.
        seed: RNG seed for reproducibility.
        day: Date stamp for the timestamps.
        urgent_rate_by_group: Per-group probability of an 'urgent' prediction.
            Unequal values create a measurable demographic-parity gap. Defaults
            to an equal 0.20 for every group.
        drift: 0.0 = no drift. Positive values shift the input metadata
            distribution, simulating a changed intake population.
        low_confidence_injection: Fraction of predictions forced to low
            confidence, simulating model degradation.
    """
    rng = np.random.default_rng(seed)
    request_id = np.arange(1, n + 1)

    # --- Demographics (the separate, joinable dataset) ---
    ethnicity = rng.choice(
        ["group_a", "group_b", "group_c", "group_d"], size=n, p=[0.55, 0.25, 0.12, 0.08]
    )
    demographics = pd.DataFrame(
        {
            "request_id": request_id,
            "age_band": rng.choice(["18-34", "35-54", "55+"], size=n),
            "gender": rng.choice(["female", "male", "other"], size=n, p=[0.49, 0.49, 0.02]),
            "ethnicity_band": ethnicity,
            "disability_flag": rng.choice([0, 1], size=n, p=[0.82, 0.18]),
        }
    )

    # --- Predicted priority, with optional per-group skew ---
    rates = urgent_rate_by_group or {g: 0.20 for g in np.unique(ethnicity)}
    priority = np.empty(n, dtype=object)
    for i, g in enumerate(ethnicity):
        urgent_p = rates.get(g, 0.20)
        r = rng.random()
        if r < urgent_p:
            priority[i] = "urgent"
        elif r < urgent_p + 0.50:
            priority[i] = "standard"
        else:
            priority[i] = "low"

    # --- Confidence, with optional degradation ---
    confidence = np.clip(rng.normal(0.82, 0.09, size=n), 0.0, 1.0)
    if low_confidence_injection > 0:
        k = int(n * low_confidence_injection)
        idx = rng.choice(n, size=k, replace=False)
        confidence[idx] = rng.uniform(0.25, 0.45, size=k)

    # --- Input metadata, with optional drift ---
    base_text_len = rng.normal(180, 40, size=n) + drift * 120
    logs = pd.DataFrame(
        {
            "request_id": request_id,
            "predicted_priority": priority,
            "confidence": confidence,
            "timestamp": pd.Timestamp(day) + pd.to_timedelta(rng.integers(0, 24 * 3600, n), "s"),
            "region": rng.choice(["north", "south", "east", "west"], size=n),
            "channel": rng.choice(
                ["web", "phone", "post"], size=n,
                p=[0.6 - drift * 0.2, 0.3, 0.1 + drift * 0.2] if drift else [0.6, 0.3, 0.1],
            ),
            "request_type": rng.choice(["benefit", "appeal", "enquiry", "complaint"], size=n),
            "text_length": np.clip(base_text_len, 10, None).astype(int),
        }
    )
    return logs, demographics


if __name__ == "__main__":  # pragma: no cover
    logs, demo = generate()
    print(logs.head())
    print(demo.head())
