"""Configuration loading and inference-log schema validation.

Thresholds live in YAML so that governance can tune them without code changes.
This module turns that YAML into typed :class:`AlertConfig` objects and provides
a single place to validate that an inference-log dataframe has the columns the
rest of the package depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .alerts import AlertConfig

# Columns the inference logs must contain. Demographic columns are validated
# separately because they arrive from a different, access-controlled dataset.
REQUIRED_LOG_COLUMNS: tuple[str, ...] = (
    "request_id",
    "predicted_priority",
    "confidence",
    "timestamp",
)

VALID_PRIORITIES: tuple[str, ...] = ("urgent", "standard", "low")


class ConfigError(ValueError):
    """Raised when configuration is malformed."""


class SchemaError(ValueError):
    """Raised when input data does not match the expected schema."""


@dataclass(frozen=True)
class MonitoringConfig:
    """Top-level monitoring configuration.

    Attributes:
        thresholds: Map of metric name -> :class:`AlertConfig`.
        reference_group: Demographic group treated as the comparison baseline
            for parity metrics. ``None`` uses the largest group at runtime.
        min_group_size: Groups smaller than this are not scored (avoids noisy
            estimates and re-identification of small cohorts).
        expected_daily_volume: Used by the throughput robustness check.
    """

    thresholds: dict[str, AlertConfig]
    reference_group: str | None = None
    min_group_size: int = 30
    expected_daily_volume: int = 2000
    raw: dict[str, Any] = field(default_factory=dict)

    def threshold_for(self, metric_name: str) -> AlertConfig | None:
        return self.thresholds.get(metric_name)


def _parse_thresholds(block: dict[str, Any]) -> dict[str, AlertConfig]:
    thresholds: dict[str, AlertConfig] = {}
    for name, spec in block.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"threshold {name!r} must be a mapping")
        try:
            thresholds[name] = AlertConfig(
                amber=float(spec["amber"]),
                red=float(spec["red"]),
                direction=spec.get("direction", "higher_is_worse"),
                description=spec.get("description", ""),
            )
        except KeyError as exc:
            raise ConfigError(f"threshold {name!r} missing key {exc}") from exc
        except ValueError as exc:
            raise ConfigError(f"threshold {name!r}: {exc}") from exc
    return thresholds


def load_config(path: str | Path) -> MonitoringConfig:
    """Load and validate a monitoring config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if "thresholds" not in data:
        raise ConfigError("config must contain a 'thresholds' section")

    settings = data.get("settings", {})
    return MonitoringConfig(
        thresholds=_parse_thresholds(data["thresholds"]),
        reference_group=settings.get("reference_group"),
        min_group_size=int(settings.get("min_group_size", 30)),
        expected_daily_volume=int(settings.get("expected_daily_volume", 2000)),
        raw=data,
    )


def validate_logs(df: pd.DataFrame) -> None:
    """Validate an inference-log dataframe in place (raises on failure).

    Checks required columns, priority vocabulary, and confidence bounds. Kept
    strict on purpose: a triage monitor that silently accepts malformed logs is
    worse than one that fails loudly.
    """
    missing = [c for c in REQUIRED_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"inference logs missing columns: {missing}")

    bad_priorities = set(df["predicted_priority"].dropna().unique()) - set(VALID_PRIORITIES)
    if bad_priorities:
        raise SchemaError(
            f"unexpected priority labels {sorted(bad_priorities)}; "
            f"expected subset of {VALID_PRIORITIES}"
        )

    conf = pd.to_numeric(df["confidence"], errors="coerce")
    out_of_range = conf.dropna()
    if ((out_of_range < 0) | (out_of_range > 1)).any():
        raise SchemaError("confidence values must lie in [0, 1]")
