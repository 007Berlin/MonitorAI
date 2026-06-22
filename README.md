# RAI Tracker

Responsible-AI monitoring for a government request-triage classifier.

A UK government department runs an AI system that triages ~2,000 citizen
requests per day into `urgent` / `standard` / `low`. This package monitors that
system across three dimensions — **fairness**, **robustness**, and
**transparency** — turning raw inference logs into evaluated, two-tier
(amber/red) alerts with a full audit trail.

It is the code companion to the accompanying technical note.

## Architecture

![RAI Tracker monitoring pipeline](docs/architecture.png)

A **validate → compute → evaluate → report** pipeline. There is no model inside
it — the model is the system being observed; this software reads its inference
logs and reports on fairness, robustness, and transparency. See
[`docs/architecture.md`](docs/architecture.md) for the full walkthrough.

## Why it is built this way

* **Outcomes over internals.** The harm a citizen experiences is the priority
  they are assigned, so the primary fairness signal is the selection rate per
  demographic group, not a single aggregate accuracy number.
* **Thresholds are configuration.** All amber/red boundaries live in
  `config/thresholds.yaml`. Governance can tune them without a code change or
  redeployment — a hard requirement for an assured public-sector system.
* **Fails loud, never silent.** Inference logs are schema-validated before any
  metric runs; the CLI returns a non-zero exit code on a breach so a scheduler
  treats it as a job failure and pages the owner.
* **Privacy-aware.** Small demographic groups are suppressed (configurable
  `min_group_size`) to avoid noisy estimates and re-identification, and the
  orchestrator is side-effect free — it logs no PII.

## Install

```bash
pip install -e ".[dev]"
```

## Use as a library

```python
import pandas as pd
from rai_tracker import load_config, run_monitoring

config = load_config("config/thresholds.yaml")
report = run_monitoring(
    logs=pd.read_csv("inference_logs.csv"),
    config=config,
    baseline=pd.read_csv("baseline_logs.csv"),       # for drift
    demographics=pd.read_csv("demographics.csv"),    # joins on request_id
    group_cols=["group", "age_band"],
    drift_features=["channel_count"],
)

print(report.overall_severity.label)        # OK | AMBER | RED
for alert in report.actionable():            # worst first
    print(alert.metric.name, alert.metric.value, alert.severity.label)
report.to_records()                          # JSON-ready rows for a metrics store
```

## Use as a CLI

```bash
rai-tracker \
  --logs inference_logs.csv \
  --config config/thresholds.yaml \
  --demographics demographics.csv \
  --group-cols group age_band \
  --baseline baseline_logs.csv \
  --drift-features channel_count \
  --output report.json \
  --fail-on red          # exit non-zero on RED (use 'amber' to be stricter)
```

## Run the demo (no data needed)

```bash
python examples/run_demo.py
```

This generates synthetic data matching the brief's schema and runs three
scenarios — a healthy day, an injected fairness skew, and a drift/confidence
failure — so you can watch the amber/red alerts react.

## Performance

The suite is a daily batch. On the spec's ~2,000 records/day it completes in
~30 ms; at 20,000 rows (10x volume) it stays under a second. A latency guardrail
test (`tests/test_latency.py`) fails the build if a change makes it slow.

## Metrics

| Dimension | Metric | Module |
|-----------|--------|--------|
| Fairness | Demographic parity difference (selection-rate gap) | `fairness.demographic_parity_difference` |
| Fairness | True-positive-rate gap (equal opportunity)\* | `fairness.true_positive_rate_gap` |
| Fairness | Calibration gap (ECE across groups)\* | `fairness.calibration_gap` |
| Robustness | Prediction-mix drift (PSI) | `robustness.prediction_distribution_drift` |
| Robustness | Input feature drift (PSI) | `robustness.feature_drift` |
| Robustness | Mean confidence / low-confidence rate | `robustness.mean_confidence`, `robustness.low_confidence_rate` |
| Robustness | Volume deviation & error rate | `robustness.volume_deviation`, `robustness.error_rate` |
| Transparency | Logging completeness | `transparency.logging_completeness` |
| Transparency | Join integrity (logs ↔ demographics) | `transparency.join_integrity` |

\* Requires confirmed ground truth (`true_priority`); returns a non-computable
result and is reported as such when outcomes are not yet available.

## Expected schema

**Inference logs** (required): `request_id`, `predicted_priority`
(`urgent`/`standard`/`low`), `confidence` (0–1), `timestamp`. Any extra
metadata columns can be monitored for drift.

**Demographics** (separate dataset): `request_id` plus one column per protected
attribute, joined inside a governed environment.

## Development

```bash
pytest                 # 45 tests, ~95% coverage
ruff check src tests   # lint
mypy src               # type-check
```

## Limitations

The metrics inform human judgement; they do not replace it. Parity is not the
same as fairness, drift is not the same as harm, and accuracy-based fairness
metrics depend on outcome data that is lagged or absent. See the technical note
for the full limitations register.
