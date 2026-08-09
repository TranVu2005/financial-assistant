"""Command-line interface for reproducible Day 8 BM25 retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from json import JSONDecodeError
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from financial_report_qa.core.errors import (
    DenseArtifactError,
    DenseInputError,
    DenseModelError,
    RetrievalArtifactError,
    RetrievalInputError,
)
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import EncoderName
from financial_report_qa.retrieval.dense_corpus import (
    DenseCorpus,
    build_dense_corpus,
    load_dense_corpus,
    save_dense_corpus,
)
from financial_report_qa.retrieval.dense_encoder import (
    SentenceTransformerDenseEncoder,
    approved_encoder_spec,
    encoder_spec_sha256,
)
from financial_report_qa.retrieval.dense_evaluation import (
    DenseEvaluationRun,
    build_day9_comparison,
    evaluate_cold_and_warm,
    write_day9_comparison,
)
from financial_report_qa.retrieval.dense_index import (
    build_dense_index,
    load_dense_index,
    save_dense_index,
)
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.documents import build_table_documents
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    evaluate_retrieval,
    write_report,
)
from financial_report_qa.retrieval.gold import load_gold_questions
from financial_report_qa.retrieval.index import build_bm25_index, load_bm25_index, save_bm25_index
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.retrieval.service import RetrievalService


class _DenseBuildObservation(BaseModel):
    """Operational evidence emitted by an explicit dense index build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder_name: EncoderName
    encoder_spec_sha256: str
    dataset_fingerprint: str
    build_seconds: float = Field(ge=0)
    index_byte_size: int = Field(ge=0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financial-report-qa retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-index", "validate-gold", "evaluate", "build-dense-corpus"):
        command = commands.add_parser(name)
        command.add_argument("--release-lock", type=Path, required=True)
        if name == "build-index":
            command.add_argument("--output-root", type=Path, required=True)
        elif name == "build-dense-corpus":
            command.add_argument("--output-root", type=Path, required=True)
        else:
            command.add_argument("--gold-path", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--index-dir", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
    dense_index = commands.add_parser("build-dense-index")
    dense_index.add_argument("--release-lock", type=Path, required=True)
    dense_index.add_argument("--corpus-dir", type=Path, required=True)
    dense_index.add_argument(
        "--encoder", choices=("bge-m3", "multilingual-e5-small"), required=True
    )
    dense_index.add_argument("--output-root", type=Path, required=True)
    dense_index.add_argument("--observation-path", type=Path, required=True)
    dense_index.add_argument("--local-files-only", action="store_true")
    dense_evaluation = commands.add_parser("evaluate-dense")
    dense_evaluation.add_argument("--release-lock", type=Path, required=True)
    dense_evaluation.add_argument("--corpus-dir", type=Path, required=True)
    dense_evaluation.add_argument("--index-dir", type=Path, required=True)
    dense_evaluation.add_argument(
        "--encoder", choices=("bge-m3", "multilingual-e5-small"), required=True
    )
    dense_evaluation.add_argument("--gold-path", type=Path, required=True)
    dense_evaluation.add_argument("--cache-dir", type=Path, required=True)
    dense_evaluation.add_argument("--observation-path", type=Path, required=True)
    dense_evaluation.add_argument("--output-path", type=Path, required=True)
    compare = commands.add_parser("compare-day9")
    compare.add_argument("--release-lock", type=Path, required=True)
    compare.add_argument("--bm25-report", type=Path, required=True)
    compare.add_argument("--bge-report", type=Path, required=True)
    compare.add_argument("--e5-report", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_dense_encoder(
    name: EncoderName, *, local_files_only: bool
) -> SentenceTransformerDenseEncoder:
    return SentenceTransformerDenseEncoder(
        approved_encoder_spec(name), local_files_only=local_files_only
    )


def _checked_dense_corpus(
    corpus_dir: Path, *, release_fingerprint: str, lock_sha256: str
) -> DenseCorpus:
    corpus = load_dense_corpus(corpus_dir, release_lock_sha256=lock_sha256)
    if corpus.manifest.dataset_fingerprint != release_fingerprint:
        raise DenseArtifactError("Dense corpus fingerprint does not match release lock")
    return corpus


def _load_build_observation(
    path: Path,
    *,
    encoder_name: EncoderName,
    encoder_hash: str,
    dataset_fingerprint: str,
) -> _DenseBuildObservation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DenseArtifactError("Dense build observation must be a JSON object")
    observation = _DenseBuildObservation.model_validate(payload)
    if (
        observation.encoder_name != encoder_name
        or observation.encoder_spec_sha256 != encoder_hash
        or observation.dataset_fingerprint != dataset_fingerprint
    ):
        raise DenseArtifactError("Dense build observation does not match requested artifacts")
    return observation


def _write_dense_run(run: DenseEvaluationRun, path: Path) -> None:
    write_text_atomic(
        path,
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


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
            index = build_bm25_index(
                documents,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            target = args.output_root / release.dataset_fingerprint
            save_bm25_index(index, target)
            print(target)
            return 0
        if args.command == "build-dense-corpus":
            documents = build_table_documents(
                release.release_dir / "documents.parquet",
                release.release_dir / "tables.parquet",
                release.release_dir / "cells.parquet",
            )
            corpus = build_dense_corpus(
                documents,
                dataset_fingerprint=release.dataset_fingerprint,
                release_lock_sha256=release.lock_sha256,
            )
            target = args.output_root / release.dataset_fingerprint / "corpus"
            save_dense_corpus(corpus, target)
            print(target)
            return 0
        if args.command == "build-dense-index":
            encoder_name = cast(EncoderName, args.encoder)
            encoder = _load_dense_encoder(encoder_name, local_files_only=args.local_files_only)
            corpus = _checked_dense_corpus(
                args.corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            started = time.perf_counter()
            built = build_dense_index(corpus, encoder)
            target = args.output_root / f"{encoder_name}-{encoder_spec_sha256(encoder.spec)[:12]}"
            save_dense_index(built, target)
            observation = _DenseBuildObservation(
                encoder_name=encoder_name,
                encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
                dataset_fingerprint=release.dataset_fingerprint,
                build_seconds=time.perf_counter() - started,
                index_byte_size=(target / "index.faiss").stat().st_size,
            )
            write_text_atomic(
                args.observation_path,
                json.dumps(
                    observation.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            print(target)
            return 0
        if args.command == "validate-gold":
            gold = load_gold_questions(args.gold_path, release)
            print(f"validated {len(gold)} reviewed retrieval questions")
            return 0
        if args.command == "evaluate-dense":
            gold = load_gold_questions(args.gold_path, release)
            encoder_name = cast(EncoderName, args.encoder)
            encoder = _load_dense_encoder(encoder_name, local_files_only=True)
            encoder_hash = encoder_spec_sha256(encoder.spec)
            corpus = _checked_dense_corpus(
                args.corpus_dir,
                release_fingerprint=release.dataset_fingerprint,
                lock_sha256=release.lock_sha256,
            )
            dense_index = load_dense_index(
                args.index_dir,
                corpus,
                expected_encoder_spec_sha256=encoder_hash,
                release_lock_sha256=release.lock_sha256,
            )
            observation = _load_build_observation(
                args.observation_path,
                encoder_name=encoder_name,
                encoder_hash=encoder_hash,
                dataset_fingerprint=release.dataset_fingerprint,
            )
            if observation.index_byte_size != dense_index.manifest.index_byte_size:
                raise DenseArtifactError("Dense build observation index size does not match index")
            service = DenseRetrievalService(
                dense_index,
                encoder,
                QueryEmbeddingCache(args.cache_dir, encoder.spec),
            )
            run = evaluate_cold_and_warm(
                service,
                gold,
                build_seconds=observation.build_seconds,
                index_byte_size=observation.index_byte_size,
            )
            _write_dense_run(run, args.output_path)
            print(args.output_path)
            return 0
        if args.command == "compare-day9":
            bm25_report = RetrievalEvaluationReport.model_validate_json(
                args.bm25_report.read_text(encoding="utf-8")
            )
            bge_run = DenseEvaluationRun.model_validate_json(
                args.bge_report.read_text(encoding="utf-8")
            )
            e5_run = DenseEvaluationRun.model_validate_json(
                args.e5_report.read_text(encoding="utf-8")
            )
            try:
                comparison = build_day9_comparison(bm25_report, bge_run, e5_run)
            except ValueError as exc:
                raise DenseArtifactError("Day 9 comparison inputs are invalid") from exc
            json_path, markdown_path = write_day9_comparison(comparison, args.output_dir)
            print(json_path)
            print(markdown_path)
            return 0
        try:
            gold = load_gold_questions(args.gold_path, release)
            index = load_bm25_index(args.index_dir, release_lock_sha256=release.lock_sha256)
        except ValueError as exc:
            raise RetrievalArtifactError("BM25 index artifact is invalid") from exc
        if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
            raise RetrievalArtifactError("BM25 index fingerprint does not match release lock")
        report = evaluate_retrieval(RetrievalService(index), gold)
        json_path, markdown_path = write_report(report, args.output_dir)
        print(json_path)
        print(markdown_path)
        return 0
    except (
        RetrievalInputError,
        RetrievalArtifactError,
        DenseInputError,
        DenseArtifactError,
        DenseModelError,
        ValidationError,
        JSONDecodeError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"retrieval error: {exc}", file=sys.stderr)
        return 2
