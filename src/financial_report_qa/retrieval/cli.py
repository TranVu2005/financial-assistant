"""Command-line interface for reproducible Day 8 BM25 retrieval."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import RetrievalError
from financial_report_qa.retrieval.documents import build_table_documents
from financial_report_qa.retrieval.evaluation import evaluate_retrieval, write_report
from financial_report_qa.retrieval.gold import REQUIRED_GOLD_QUESTION_COUNT, load_reviewed_gold
from financial_report_qa.retrieval.index import build_bm25_index, load_bm25_index, save_bm25_index
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.retrieval.service import RetrievalService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-index", "validate-gold", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--release-lock", type=Path, required=True)
        if name == "build-index":
            command.add_argument("--output-root", type=Path, required=True)
        else:
            command.add_argument("--gold-path", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--index-dir", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = Path.cwd()
        release = resolve_retrieval_release(args.release_lock, repo_root=root)
        if args.command == "build-index":
            documents = build_table_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            index = build_bm25_index(documents, dataset_fingerprint=release.dataset_fingerprint)
            target = args.output_root / release.dataset_fingerprint
            save_bm25_index(index, target)
            print(target)
            return 0
        gold = load_reviewed_gold(
            args.gold_path,
            expected_count=REQUIRED_GOLD_QUESTION_COUNT,
            expected_fingerprint=release.dataset_fingerprint,
        )
        if args.command == "validate-gold":
            print(f"validated {len(gold)} reviewed retrieval questions")
            return 0
        index = load_bm25_index(args.index_dir)
        if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
            raise RetrievalError("BM25 index fingerprint does not match release lock")
        report = evaluate_retrieval(RetrievalService(index), gold)
        json_path, markdown_path = write_report(report, args.output_dir)
        print(json_path)
        print(markdown_path)
        return 0
    except (RetrievalError, OSError, ValueError) as exc:
        print(f"retrieval error: {exc}", file=sys.stderr)
        return 2
