"""RAI Tracker: responsible-AI monitoring for a request-triage classifier.

The package computes fairness, robustness, and transparency metrics over a
classifier's inference logs (optionally joined to a demographic dataset) and
evaluates them against configurable amber/red thresholds.

Public API:
    Severity, AlertConfig, MetricResult, Alert  -- core types
    load_config                                  -- read thresholds from YAML
    fairness, robustness, transparency           -- metric modules
    MonitoringReport, run_monitoring             -- orchestration
"""
from __future__ import annotations

from .alerts import Alert, AlertConfig, MetricResult, Severity, evaluate
from .config import MonitoringConfig, load_config
from .report import MonitoringReport, run_monitoring

__all__ = [
    "Severity",
    "AlertConfig",
    "MetricResult",
    "Alert",
    "evaluate",
    "MonitoringConfig",
    "load_config",
    "MonitoringReport",
    "run_monitoring",
]

__version__ = "0.1.0"
