"""Tests for alert classification and config loading/validation."""
from __future__ import annotations

import pandas as pd
import pytest

from rai_tracker.alerts import (
    HIGHER_IS_WORSE,
    LOWER_IS_WORSE,
    AlertConfig,
    MetricResult,
    Severity,
    evaluate,
)
from rai_tracker.config import ConfigError, SchemaError, load_config, validate_logs


def test_higher_is_worse_classification() -> None:
    cfg = AlertConfig(amber=0.05, red=0.10, direction=HIGHER_IS_WORSE)
    assert cfg.classify(0.01) is Severity.OK
    assert cfg.classify(0.05) is Severity.AMBER
    assert cfg.classify(0.07) is Severity.AMBER
    assert cfg.classify(0.10) is Severity.RED
    assert cfg.classify(0.5) is Severity.RED


def test_lower_is_worse_classification() -> None:
    cfg = AlertConfig(amber=0.995, red=0.99, direction=LOWER_IS_WORSE)
    assert cfg.classify(1.0) is Severity.OK
    assert cfg.classify(0.995) is Severity.AMBER
    assert cfg.classify(0.99) is Severity.RED
    assert cfg.classify(0.5) is Severity.RED


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        AlertConfig(amber=1, red=2, direction="sideways")


def test_inconsistent_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        AlertConfig(amber=0.10, red=0.05, direction=HIGHER_IS_WORSE)
    with pytest.raises(ValueError):
        AlertConfig(amber=0.90, red=0.95, direction=LOWER_IS_WORSE)


def test_evaluate_handles_missing_value_and_config() -> None:
    result = MetricResult("x", None, "fairness")
    assert evaluate(result, AlertConfig(0.1, 0.2)).severity is Severity.OK
    valued = MetricResult("x", 0.3, "fairness")
    assert evaluate(valued, None).severity is Severity.OK


def test_severity_ordering() -> None:
    assert Severity.OK < Severity.AMBER < Severity.RED
    assert max([Severity.OK, Severity.RED, Severity.AMBER]) is Severity.RED


def test_load_config_roundtrip(tmp_path) -> None:
    yaml_text = """
settings:
  min_group_size: 50
  expected_daily_volume: 2000
thresholds:
  fairness.demographic_parity_diff:
    amber: 0.05
    red: 0.10
    direction: higher_is_worse
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.min_group_size == 50
    assert cfg.threshold_for("fairness.demographic_parity_diff").red == 0.10


def test_load_config_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path.yaml")


def test_load_config_missing_thresholds(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("settings: {}\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_validate_logs_accepts_good_frame() -> None:
    df = pd.DataFrame(
        {
            "request_id": [1, 2],
            "predicted_priority": ["urgent", "low"],
            "confidence": [0.9, 0.3],
            "timestamp": ["2025-01-01", "2025-01-01"],
        }
    )
    validate_logs(df)  # should not raise


def test_validate_logs_rejects_missing_columns() -> None:
    with pytest.raises(SchemaError):
        validate_logs(pd.DataFrame({"request_id": [1]}))


def test_validate_logs_rejects_bad_priority() -> None:
    df = pd.DataFrame(
        {
            "request_id": [1],
            "predicted_priority": ["EXTREME"],
            "confidence": [0.5],
            "timestamp": ["2025-01-01"],
        }
    )
    with pytest.raises(SchemaError):
        validate_logs(df)


def test_validate_logs_rejects_out_of_range_confidence() -> None:
    df = pd.DataFrame(
        {
            "request_id": [1],
            "predicted_priority": ["urgent"],
            "confidence": [1.4],
            "timestamp": ["2025-01-01"],
        }
    )
    with pytest.raises(SchemaError):
        validate_logs(df)
