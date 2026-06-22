"""End-to-end demo: generate data, run monitoring, print a report.

Run from the repo root:

    python examples/run_demo.py

It produces three scenarios so a reviewer can see the alerting react:
    1. Healthy day      -> overall OK / amber at most
    2. Fairness skew    -> RED on demographic parity
    3. Drift + low conf -> RED on robustness signals
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rai_tracker import load_config, run_monitoring
from rai_tracker.synthetic import generate

CONFIG = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
GROUP_COLS = ["ethnicity_band", "age_band", "gender"]
DRIFT_FEATURES = ["text_length", "channel"]


def show(title: str, report) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"overall severity: {report.overall_severity.label}")
    for dim in ("fairness", "robustness", "transparency"):
        print(f"\n  {dim.upper()}")
        for alert in report.by_dimension(dim):
            val = "n/a" if alert.metric.value is None else f"{alert.metric.value:.4f}"
            print(f"    [{alert.severity.label:5}] {alert.metric.name:45} = {val}")


def main() -> None:
    config = load_config(CONFIG)
    clock = datetime(2025, 1, 2, tzinfo=timezone.utc)

    # Baseline (reference) window the drift metrics compare against.
    baseline, _ = generate(n=2000, seed=1)

    # 1. Healthy day
    logs, demo = generate(n=2000, seed=2)
    show("SCENARIO 1 - Healthy day", run_monitoring(
        logs, config, baseline=baseline, demographics=demo,
        group_cols=GROUP_COLS, drift_features=DRIFT_FEATURES, now=clock))

    # 2. Fairness skew: group_a flagged 'urgent' far more than group_d
    logs, demo = generate(
        n=2000, seed=3,
        urgent_rate_by_group={"group_a": 0.32, "group_b": 0.20, "group_c": 0.15, "group_d": 0.10},
    )
    show("SCENARIO 2 - Injected fairness skew", run_monitoring(
        logs, config, baseline=baseline, demographics=demo,
        group_cols=GROUP_COLS, drift_features=DRIFT_FEATURES, now=clock))

    # 3. Robustness failure: input drift + degraded confidence
    logs, demo = generate(n=2000, seed=4, drift=1.0, low_confidence_injection=0.4)
    show("SCENARIO 3 - Drift + confidence degradation", run_monitoring(
        logs, config, baseline=baseline, demographics=demo,
        group_cols=GROUP_COLS, drift_features=DRIFT_FEATURES, now=clock))


if __name__ == "__main__":
    main()
