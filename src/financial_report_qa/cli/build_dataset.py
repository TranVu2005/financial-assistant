import argparse
import json
import sys
from pathlib import Path

from financial_report_qa.core.errors import DatasetBuildError
from financial_report_qa.data.dataset_builder import DatasetBuildConfig, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic Parquet datasets from ViFinQA inventory manifests."
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        required=True,
        help="Path to snapshot root directory containing text files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to documents.jsonl inventory manifest file.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        required=True,
        help="Path to root output directory for Parquet releases.",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="1",
        help="Release schema version string (default: 1).",
    )

    args = parser.parse_args()

    try:
        config = DatasetBuildConfig(
            snapshot_root=args.snapshot_root,
            manifest_path=args.manifest_path,
            processed_root=args.processed_root,
            schema_version=args.schema_version,
        )
        result = build_dataset(config)
        output = {
            "status": "success",
            "release_path": str(result.release_path),
            "dataset_fingerprint": result.dataset_fingerprint,
            "source_manifest_sha256": result.source_manifest_sha256,
            "document_count": result.document_count,
            "table_count": result.table_count,
            "cell_count": result.cell_count,
            "issue_count": result.issue_count,
            "issue_counts_by_code": result.issue_counts_by_code,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(0)
    except DatasetBuildError as exc:
        output = {"status": "error", "error": str(exc)}
        print(json.dumps(output, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
