"""Integration tests for the orchestrator and CLI."""
from __future__ import annotations

from datetime import datetime, timezone

from rai_tracker import load_config, run_monitoring
from rai_tracker.alerts import Severity
from rai_tracker.cli import main

CONFIG = "config/thresholds.yaml"


def test_run_monitoring_clean_data(balanced_data) -> None:
    logs, demo = balanced_data
    cfg = load_config(CONFIG)
    report = run_monitoring(
        logs,
        cfg,
        baseline=logs,
        demographics=demo,
        group_cols=["group", "age_band"],
        drift_features=["channel_count"],
        now=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    # Balanced, self-baselined data should not produce a RED.
    assert report.overall_severity < Severity.RED
    assert report.generated_at == "2025-01-02T00:00:00+00:00"
    # Every dimension should be represented.
    assert report.by_dimension("fairness")
    assert report.by_dimension("robustness")
    assert report.by_dimension("transparency")


def test_run_monitoring_flags_fairness(skewed_data) -> None:
    logs, demo = skewed_data
    cfg = load_config(CONFIG)
    report = run_monitoring(logs, cfg, demographics=demo, group_cols=["group"])
    parity = next(
        a for a in report.alerts
        if a.metric.name == "fairness.demographic_parity_diff.group"
    )
    assert parity.severity is Severity.RED
    assert report.overall_severity is Severity.RED


def test_report_to_records_serialisable(balanced_data) -> None:
    logs, demo = balanced_data
    cfg = load_config(CONFIG)
    report = run_monitoring(logs, cfg, demographics=demo, group_cols=["group"])
    records = report.to_records()
    assert all({"metric", "value", "severity", "dimension"} <= r.keys() for r in records)


def test_feature_drift_family_threshold_resolves(balanced_data) -> None:
    # The per-feature metric name should fall back to the family threshold.
    logs, demo = balanced_data
    shifted = logs.copy()
    shifted["channel_count"] = shifted["channel_count"] + 20
    cfg = load_config(CONFIG)
    report = run_monitoring(
        shifted, cfg, baseline=logs, demographics=demo,
        group_cols=["group"], drift_features=["channel_count"],
    )
    drift = next(
        a for a in report.alerts
        if a.metric.name == "robustness.feature_drift_psi.channel_count"
    )
    assert drift.threshold is not None
    assert drift.severity is Severity.RED


def test_fairness_family_threshold_resolves(skewed_data) -> None:
    # The group-tagged name (two dots) must still resolve to the family config.
    logs, demo = skewed_data
    cfg = load_config(CONFIG)
    report = run_monitoring(logs, cfg, demographics=demo, group_cols=["group"])
    parity = next(
        a for a in report.alerts
        if a.metric.name == "fairness.demographic_parity_diff.group"
    )
    assert parity.threshold is not None
    assert parity.metric.detail["group_col"] == "group"


def test_cli_end_to_end(tmp_path, skewed_data) -> None:
    logs, demo = skewed_data
    logs_path = tmp_path / "logs.csv"
    demo_path = tmp_path / "demo.csv"
    out_path = tmp_path / "report.json"
    logs.to_csv(logs_path, index=False)
    demo.to_csv(demo_path, index=False)

    code = main(
        [
            "--logs", str(logs_path),
            "--config", CONFIG,
            "--demographics", str(demo_path),
            "--group-cols", "group",
            "--output", str(out_path),
            "--fail-on", "red",
        ]
    )
    assert code == 1  # skewed data trips a RED => non-zero exit
    assert out_path.exists()


def test_cli_clean_data_exit_zero(tmp_path, balanced_data) -> None:
    logs, demo = balanced_data
    logs_path = tmp_path / "logs.csv"
    logs.to_csv(logs_path, index=False)
    code = main(["--logs", str(logs_path), "--config", CONFIG, "--fail-on", "red"])
    assert code == 0
