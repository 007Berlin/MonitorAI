"""A lightweight latency check.

This is not a microbenchmark; it is a guardrail. The system runs as a daily
batch over ~2,000 records (and we test at 10x that), so the whole suite must
finish comfortably inside a few seconds. If a future change makes a metric
quadratic, this test catches it.
"""
from __future__ import annotations

import time

from rai_tracker import load_config, run_monitoring
from rai_tracker.synthetic import generate

CONFIG = "config/thresholds.yaml"


def test_full_suite_latency_under_budget() -> None:
    # 20,000 rows = 10 days of volume; generous headroom over the ~2k/day spec.
    logs, demo = generate(n=20_000, seed=11)
    baseline, _ = generate(n=20_000, seed=12)
    config = load_config(CONFIG)

    start = time.perf_counter()
    report = run_monitoring(
        logs,
        config,
        baseline=baseline,
        demographics=demo,
        group_cols=["ethnicity_band", "age_band", "gender"],
        drift_features=["text_length", "channel"],
    )
    elapsed = time.perf_counter() - start

    assert report.alerts  # sanity: it actually produced results
    # 5s is a deliberately loose ceiling so the test is not flaky on CI runners.
    assert elapsed < 5.0, f"monitoring suite too slow: {elapsed:.2f}s for 20k rows"
