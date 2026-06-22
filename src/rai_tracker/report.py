"""Orchestration: compute the full metric suite and evaluate alerts.

:func:`run_monitoring` is the single entry point a scheduled job calls. It wires
the metric modules to the threshold config, produces a :class:`MonitoringReport`,
and is deliberately side-effect free (no I/O, no logging of PII) so it is easy
to test and to embed in different runtimes (Airflow, a cron container, a Lambda).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from . import fairness, robustness, transparency
from .alerts import Alert, MetricResult, Severity, evaluate
from .config import MonitoringConfig, validate_logs


@dataclass(frozen=True)
class MonitoringReport:
    """The result of a monitoring run: every metric, evaluated."""

    generated_at: str
    alerts: list[Alert]

    @property
    def overall_severity(self) -> Severity:
        return max((a.severity for a in self.alerts), default=Severity.OK)

    def by_dimension(self, dimension: str) -> list[Alert]:
        return [a for a in self.alerts if a.metric.dimension == dimension]

    def actionable(self) -> list[Alert]:
        """Alerts at amber or above, worst first."""
        flagged = [a for a in self.alerts if a.is_actionable]
        return sorted(flagged, key=lambda a: a.severity, reverse=True)

    def to_records(self) -> list[dict]:
        """Flatten to JSON-serialisable rows for a time-series / metrics store."""
        return [
            {
                "generated_at": self.generated_at,
                "metric": a.metric.name,
                "dimension": a.metric.dimension,
                "value": a.metric.value,
                "severity": a.severity.label,
            }
            for a in self.alerts
        ]


def _collect(
    logs: pd.DataFrame,
    baseline: pd.DataFrame | None,
    demographics: pd.DataFrame | None,
    config: MonitoringConfig,
    group_cols: list[str],
    drift_features: list[str],
) -> list[MetricResult]:
    results: list[MetricResult] = []

    # --- Robustness (predictions only; baseline optional) ---
    results.append(robustness.mean_confidence(logs))
    results.append(robustness.low_confidence_rate(logs))
    results.append(
        robustness.volume_deviation(logs, expected_daily_volume=config.expected_daily_volume)
    )
    results.append(robustness.error_rate(logs))
    if baseline is not None:
        results.append(robustness.prediction_distribution_drift(logs, baseline))
        for feat in drift_features:
            results.append(robustness.feature_drift(logs, baseline, feat))

    # --- Transparency ---
    results.append(transparency.logging_completeness(logs))
    if demographics is not None:
        results.append(transparency.join_integrity(logs, demographics))

    # --- Fairness (requires demographics joined in) ---
    if demographics is not None and group_cols:
        merged = logs.merge(demographics, on="request_id", how="inner", suffixes=("", "_demo"))
        for col in group_cols:
            if col not in merged.columns:
                continue
            # Tag each fairness metric with the group column it was computed over,
            # so "...parity_diff.ethnicity_band" is distinguishable from
            # "...parity_diff.age_band" in the report. Threshold lookup falls
            # back to the family prefix (see run_monitoring), so one config entry
            # still covers every group column.
            for result in (
                fairness.demographic_parity_difference(
                    merged, col,
                    min_group_size=config.min_group_size,
                    reference_group=config.reference_group,
                ),
                fairness.true_positive_rate_gap(merged, col, min_group_size=config.min_group_size),
                fairness.calibration_gap(merged, col, min_group_size=config.min_group_size),
            ):
                results.append(
                    MetricResult(
                        name=f"{result.name}.{col}",
                        value=result.value,
                        dimension=result.dimension,
                        detail={**result.detail, "group_col": col},
                    )
                )
    return results


def run_monitoring(
    logs: pd.DataFrame,
    config: MonitoringConfig,
    *,
    baseline: pd.DataFrame | None = None,
    demographics: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
    drift_features: list[str] | None = None,
    now: datetime | None = None,
) -> MonitoringReport:
    """Run the full monitoring suite over a window of inference logs.

    Args:
        logs: Inference logs for the window under review.
        config: Threshold and settings configuration.
        baseline: Reference window for drift metrics (e.g. training distribution).
        demographics: Demographic dataset joinable on ``request_id``.
        group_cols: Demographic columns to compute fairness across.
        drift_features: Metadata columns to monitor for feature drift.
        now: Injectable clock for deterministic tests.

    Returns:
        A :class:`MonitoringReport` with one evaluated :class:`Alert` per metric.
    """
    validate_logs(logs)
    group_cols = group_cols or []
    drift_features = drift_features or []
    stamp = (now or datetime.now(timezone.utc)).isoformat()

    results = _collect(logs, baseline, demographics, config, group_cols, drift_features)

    alerts: list[Alert] = []
    for result in results:
        # Match the exact metric name first, then fall back to its family prefix
        # (e.g. "robustness.feature_drift_psi.age" -> "robustness.feature_drift_psi").
        cfg = config.threshold_for(result.name)
        if cfg is None and "." in result.name:
            family = result.name.rsplit(".", 1)[0]
            cfg = config.threshold_for(family)
        alerts.append(evaluate(result, cfg))

    return MonitoringReport(generated_at=stamp, alerts=alerts)
