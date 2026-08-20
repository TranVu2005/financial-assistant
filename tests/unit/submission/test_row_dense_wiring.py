"""Unit tests for `submission.cli._load_row_dense_service` (plan.md §7 dense
wiring). Uses a mock encoder -- exactly like `test_row_fusion.py` -- so these
tests never load a real sentence-transformers model, consistent with the
rest of the suite."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease
from financial_report_qa.retrieval.row_dense_corpus import (
    build_row_dense_corpus,
    save_row_dense_corpus,
)
from financial_report_qa.retrieval.row_dense_index import (
    build_row_dense_index,
    save_row_dense_index,
)
from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.submission.cli import _load_row_dense_service

_FINGERPRINT = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
_LOCK_SHA = "2" * 64
_TABLE_ID = "tbl_" + "a" * 64
_DOC_ID = "doc_" + "a" * 64


@dataclass
class _MockEncoder:
    spec: DenseEncoderSpec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _release(dataset_fingerprint: str = _FINGERPRINT) -> ResolvedRetrievalRelease:
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=dataset_fingerprint,
        source_manifest_sha256="0" * 64,
        release_path="fixture/release",
        gate_result_path="fixture/gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    return ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=dataset_fingerprint,
        release_dir=Path("unused"),
        gate_result_path=Path("unused/gate.json"),
        lock_path=Path("unused/lock.json"),
        manifest={},
        lock_sha256=_LOCK_SHA,
    )


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "row_dense_corpus": None,
        "row_dense_index": None,
        "dense_encoder": None,
        "dense_cache_dir": None,
        "dense_local_files_only": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_row_dense_artifacts(
    tmp_path: Path, *, dataset_fingerprint: str, lock_sha256: str, encoder: _MockEncoder
) -> tuple[Path, Path]:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": _DOC_ID,
                    "repo_id": "repo",
                    "revision": "1",
                    "relative_path": "DBC/2023/DBC_report_extracted.txt",
                    "company_code": "DBC",
                    "report_year": 2023,
                    "statement_scope": "consolidated",
                    "sha256": "0" * 64,
                    "file_size_bytes": 10,
                    "encoding": "utf-8",
                    "inventory_status": "ready",
                    "ruleset_version": "1",
                    "normalization_fingerprint": "0" * 64,
                }
            ],
            schema=DOCUMENT_SCHEMA,
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "table_id": _TABLE_ID,
                    "doc_id": _DOC_ID,
                    "source_ordinal": 0,
                    "title_raw": "Bảng cân đối kế toán",
                    "statement_type": "balance_sheet",
                    "unit_raw": "VND",
                    "unit_normalized": "vnd",
                    "line_start": 1,
                    "line_end": 2,
                    "row_count": 1,
                    "column_count": 1,
                    "quality_score": 0.9,
                    "csv_path": None,
                }
            ],
            schema=TABLE_SCHEMA,
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cell_id": "cell_" + "a" * 64,
                    "table_id": _TABLE_ID,
                    "row_idx": 1,
                    "col_idx": 1,
                    "row_label_raw": "Tổng tài sản",
                    "row_label_canonical": "total_assets",
                    "row_group_context_raw": None,
                    "column_label_raw": "Năm 2023",
                    "column_label_canonical": None,
                    "value_raw": "500",
                    "value_numeric": Decimal("500"),
                    "period": "2023",
                    "unit": "VND",
                    "source_line_start": 2,
                    "source_line_end": 2,
                    "extraction_confidence": 0.9,
                }
            ],
            schema=CELL_SCHEMA,
        ),
        release_dir / "cells.parquet",
    )

    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    corpus = build_row_dense_corpus(
        row_docs, dataset_fingerprint=dataset_fingerprint, release_lock_sha256=lock_sha256
    )
    corpus_dir = tmp_path / "row_corpus"
    save_row_dense_corpus(corpus, corpus_dir)

    index = build_row_dense_index(corpus, encoder)
    index_dir = tmp_path / "row_index"
    save_row_dense_index(index, index_dir)

    return corpus_dir, index_dir


@pytest.mark.parametrize(
    "overrides",
    [
        {"row_dense_corpus": Path("x")},
        {"row_dense_index": Path("x")},
        {"dense_encoder": "multilingual-e5-small"},
        {"row_dense_corpus": Path("x"), "row_dense_index": Path("x")},
    ],
)
def test_returns_none_unless_all_three_args_given(
    overrides: dict[str, object],
) -> None:
    assert _load_row_dense_service(_args(**overrides), _release()) is None


def test_returns_none_on_dataset_fingerprint_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _MockEncoder(spec)
    corpus_dir, index_dir = _write_row_dense_artifacts(
        tmp_path, dataset_fingerprint=_FINGERPRINT, lock_sha256=_LOCK_SHA, encoder=encoder
    )

    mismatched_release = _release(dataset_fingerprint="f" * 64)
    args = _args(
        row_dense_corpus=corpus_dir,
        row_dense_index=index_dir,
        dense_encoder="multilingual-e5-small",
    )

    result = _load_row_dense_service(args, mismatched_release)

    assert result is None
    assert "dataset_fingerprint does not match" in capsys.readouterr().err


def test_returns_none_when_index_missing(tmp_path: Path) -> None:
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _MockEncoder(spec)
    corpus_dir, _ = _write_row_dense_artifacts(
        tmp_path, dataset_fingerprint=_FINGERPRINT, lock_sha256=_LOCK_SHA, encoder=encoder
    )

    args = _args(
        row_dense_corpus=corpus_dir,
        row_dense_index=tmp_path / "does_not_exist",
        dense_encoder="multilingual-e5-small",
    )

    assert _load_row_dense_service(args, _release()) is None


def test_happy_path_returns_working_row_dense_service(tmp_path: Path) -> None:
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _MockEncoder(spec)
    corpus_dir, index_dir = _write_row_dense_artifacts(
        tmp_path, dataset_fingerprint=_FINGERPRINT, lock_sha256=_LOCK_SHA, encoder=encoder
    )

    args = _args(
        row_dense_corpus=corpus_dir,
        row_dense_index=index_dir,
        dense_encoder="multilingual-e5-small",
        dense_cache_dir=tmp_path / "cache",
    )

    service = _load_row_dense_service(args, _release(), encoder=encoder)

    assert isinstance(service, RowDenseRetrievalService)
    results = service.retrieve_rows("Tổng tài sản", candidate_table_ids=(_TABLE_ID,))
    assert len(results) == 1
    assert results[0].metadata.row_label_raw == "Tổng tài sản"
