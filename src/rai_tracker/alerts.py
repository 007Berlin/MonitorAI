"""Core types for metric results and two-tier (amber/red) alerting.

The alerting model is deliberately simple and declarative: every metric the
system computes produces a :class:`MetricResult`, and an :class:`AlertConfig`
describes how to turn that scalar into a :class:`Severity`. Keeping thresholds
in configuration (rather than code) is a requirement for a governed system: the
risk owners must be able to tune them without a redeployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    """Ordered severity levels. Higher is worse; ``IntEnum`` so they compare/sort."""

    OK = 0
    AMBER = 1
    RED = 2

    @property
    def label(self) -> str:
        return self.name


class Direction(str):
    """Marker for whether large or small metric values are the problem."""


HIGHER_IS_WORSE = "higher_is_worse"
LOWER_IS_WORSE = "lower_is_worse"


@dataclass(frozen=True)
class AlertConfig:
    """Threshold configuration for a single metric.

    Args:
        amber: Boundary at which the metric becomes a warning.
        red: Boundary at which the metric becomes an actionable alert.
        direction: ``HIGHER_IS_WORSE`` (e.g. drift, disparity) or
            ``LOWER_IS_WORSE`` (e.g. logging completeness).
        description: Human-readable note carried into alerts for auditability.
    """

    amber: float
    red: float
    direction: str = HIGHER_IS_WORSE
    description: str = ""

    def __post_init__(self) -> None:
        if self.direction not in (HIGHER_IS_WORSE, LOWER_IS_WORSE):
            raise ValueError(f"unknown direction: {self.direction!r}")
        if self.direction == HIGHER_IS_WORSE and self.red < self.amber:
            raise ValueError("red threshold must be >= amber when higher is worse")
        if self.direction == LOWER_IS_WORSE and self.red > self.amber:
            raise ValueError("red threshold must be <= amber when lower is worse")

    def classify(self, value: float) -> Severity:
        """Map a metric value to a :class:`Severity` given this config."""
        if self.direction == HIGHER_IS_WORSE:
            if value >= self.red:
                return Severity.RED
            if value >= self.amber:
                return Severity.AMBER
            return Severity.OK
        # LOWER_IS_WORSE
        if value <= self.red:
            return Severity.RED
        if value <= self.amber:
            return Severity.AMBER
        return Severity.OK


@dataclass(frozen=True)
class MetricResult:
    """The outcome of computing one metric.

    Attributes:
        name: Stable identifier, e.g. ``"fairness.demographic_parity_diff"``.
        value: The scalar metric value (``None`` when not computable, e.g. a
            group too small to estimate reliably).
        dimension: One of ``fairness`` / ``robustness`` / ``transparency``.
        detail: Optional structured context (per-group breakdowns, sample sizes).
    """

    name: str
    value: float | None
    dimension: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    """An evaluated metric: a result combined with its severity."""

    metric: MetricResult
    severity: Severity
    threshold: AlertConfig | None

    @property
    def is_actionable(self) -> bool:
        return self.severity >= Severity.AMBER


def evaluate(result: MetricResult, config: AlertConfig | None) -> Alert:
    """Evaluate a metric result against its threshold config.

    A metric with no value (not computable) or no configured threshold is
    returned as :attr:`Severity.OK` rather than silently dropped, so it still
    appears in the audit trail.
    """
    if result.value is None or config is None:
        return Alert(metric=result, severity=Severity.OK, threshold=config)
    severity = config.classify(result.value)
    return Alert(metric=result, severity=severity, threshold=config)
