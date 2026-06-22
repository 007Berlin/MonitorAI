"""Command-line interface: run monitoring over CSV inputs and print/exit-code.

Designed to slot into a scheduler: a non-zero exit code on a RED alert lets the
orchestrator (cron, Airflow, CI) treat a fairness/robustness breach as a job
failure and page the on-call owner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .alerts import Severity
from .config import load_config
from .report import run_monitoring


def _read_csv(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rai-tracker", description=__doc__)
    p.add_argument("--logs", required=True, help="Inference logs CSV")
    p.add_argument("--config", required=True, help="Thresholds YAML")
    p.add_argument("--baseline", help="Baseline logs CSV for drift")
    p.add_argument("--demographics", help="Demographic dataset CSV (joins on request_id)")
    p.add_argument("--group-cols", nargs="*", default=[], help="Demographic columns for fairness")
    p.add_argument("--drift-features", nargs="*", default=[], help="Metadata columns for drift")
    p.add_argument("--output", help="Write the full report as JSON to this path")
    p.add_argument(
        "--fail-on",
        choices=["never", "amber", "red"],
        default="red",
        help="Severity at which the process exits non-zero",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    logs = pd.read_csv(args.logs)

    report = run_monitoring(
        logs,
        config,
        baseline=_read_csv(args.baseline),
        demographics=_read_csv(args.demographics),
        group_cols=args.group_cols,
        drift_features=args.drift_features,
    )

    records = report.to_records()
    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2))

    print(f"Monitoring run @ {report.generated_at}  overall={report.overall_severity.label}")
    for alert in report.actionable():
        val = alert.metric.value
        print(f"  [{alert.severity.label:5}] {alert.metric.name} = {val:.4f}")
    if not report.actionable():
        print("  no amber/red alerts")

    threshold = {"never": None, "amber": Severity.AMBER, "red": Severity.RED}[args.fail_on]
    if threshold is not None and report.overall_severity >= threshold:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
