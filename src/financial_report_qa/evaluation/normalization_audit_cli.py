"""Command-line interface for normalization issue audit sampling and baseline reporting."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]

from financial_report_qa.evaluation.normalization_audit import (
    AuditSamplingConfig,
    build_issue_sample,
    evaluate_labels,
    load_and_validate_labels,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the normalization-audit CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="normalization-audit",
        description="Sample normalization issues and evaluate baseline metrics.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Generate a deterministic stratified sample of normalization issues.",
    )
    sample_parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="Path to normalized dataset release directory",
    )
    sample_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Target output Parquet file path for the sampled issues",
    )
    sample_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML audit sampling configuration file",
    )

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Evaluate human review labels against sample and produce baseline report.",
    )
    baseline_parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Path to sampled Parquet dataset",
    )
    baseline_parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to human labels CSV file",
    )
    baseline_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory to publish baseline.json and baseline.md reports",
    )
    return parser


def run_sample_subcommand(release_path: Path, output_path: Path, config_path: Path) -> None:
    """Execute the sample generation command."""
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raw_config = {}
    config = AuditSamplingConfig.model_validate(raw_config)

    manifest_path = release_path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest.json in {release_path}")
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_fp = str(manifest_data.get("dataset_fingerprint", ""))
    if not release_fp:
        raise ValueError("manifest.json missing dataset_fingerprint")

    read_table = cast(Any, pq.read_table)
    if output_path.is_file():
        existing_table = read_table(output_path)
        rows = existing_table.to_pylist()
        if rows:
            existing_fp = str(rows[0].get("release_fingerprint", ""))
            if existing_fp != release_fp:
                raise ValueError(
                    f"Refusing to overwrite sample: existing fingerprint {existing_fp} "
                    f"differs from release {release_fp}"
                )

    sample_table = build_issue_sample(release_path, release_fp, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_table = cast(Any, pq.write_table)
    write_table(sample_table, output_path)
    print(f"Sample written to {output_path}")


def run_baseline_subcommand(sample_path: Path, labels_path: Path, output_dir: Path) -> None:
    """Execute the baseline report generation command."""
    if not sample_path.is_file():
        raise ValueError(f"Sample file not found: {sample_path}")

    read_table = cast(Any, pq.read_table)
    sample_table = read_table(sample_path)
    rows = sample_table.to_pylist()
    release_fp = str(rows[0]["release_fingerprint"]) if rows else ""

    labels = load_and_validate_labels(sample_table, labels_path)
    metrics_by_issue = evaluate_labels(sample_table, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline.json"
    md_path = output_dir / "baseline.md"

    report_data = {
        "release_fingerprint": release_fp,
        "metrics_by_issue": {
            code: m.model_dump(mode="json")
            for code, m in sorted(metrics_by_issue.items())
        },
    }
    json_path.write_text(
        json.dumps(report_data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Normalization Issue Audit Baseline Report",
        "",
        f"**Release Fingerprint:** `{release_fp}`",
        "",
        "## Metrics by Issue",
        "",
        (
            "| Issue Code | Sample Count | True Issue | False Positive | "
            "Uncertain | Unlabeled | Conclusive Coverage | False Positive Rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for code, m in sorted(metrics_by_issue.items()):
        fpr = f"{m.false_positive_rate:.4f}" if m.false_positive_rate is not None else "N/A"
        md_lines.append(
            f"| `{code}` | {m.sample_count} | {m.true_issue_count} | "
            f"{m.false_positive_count} | {m.uncertain_count} | {m.unlabeled_count} | "
            f"{m.conclusive_coverage:.4f} | {fpr} |"
        )
    md_lines.extend(["", ""])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Baseline report published to {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for normalization-audit."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    try:
        if args.subcommand == "sample":
            run_sample_subcommand(args.release, args.output, args.config)
        elif args.subcommand == "baseline":
            run_baseline_subcommand(args.sample, args.labels, args.output_dir)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
