"""Unit tests for stable pilot selection and directory preparation."""

import hashlib
from pathlib import Path

import pytest
from test_week1_dataset import _write_release

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    read_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.evaluation.week1_sampling import (
    prepare_pilot,
    select_pilot_documents,
)
from financial_report_qa.schemas import (
    DocumentRecord,
    stable_document_id,
)


def _make_doc(company: str, year: int, scope: str, idx: int = 1) -> DocumentRecord:
    digest = hashlib.sha256(f"{company}-{year}-{scope}-{idx}".encode()).hexdigest()
    doc_id = stable_document_id(digest)
    return DocumentRecord(
        doc_id=doc_id,
        repo_id="test_repo",
        revision="main",
        relative_path=f"{company}/{year}/{scope}/doc_{idx}.txt",
        company_code=company,
        report_year=year,
        statement_scope=scope,  # type: ignore[arg-type]
        sha256=digest,
        file_size_bytes=1000,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_select_pilot_documents_ranks_deterministically() -> None:
    docs: list[DocumentRecord] = []
    # Create 25 companies, each with 4 documents across scopes/years
    companies = [f"COMP{i:02d}" for i in range(25)]
    scopes = ["consolidated", "separate", "aggregated", "other"]

    for comp in companies:
        for idx in range(4):
            year = 2020 + idx
            scope = scopes[idx % len(scopes)]
            docs.append(_make_doc(comp, year, scope, idx))

    table_doc_ids = {d.doc_id for d in docs}

    pilot_1 = select_pilot_documents(
        tuple(docs), table_doc_ids, company_count=20, documents_per_company=3
    )
    pilot_2 = select_pilot_documents(
        tuple(docs), table_doc_ids, company_count=20, documents_per_company=3
    )

    assert len(pilot_1) == 60
    assert len({d.company_code for d in pilot_1}) == 20
    assert [d.doc_id for d in pilot_1] == [d.doc_id for d in pilot_2]


def test_select_pilot_documents_fails_insufficient_companies() -> None:
    docs: list[DocumentRecord] = []
    for comp in ["COMP01", "COMP02"]:
        for idx in range(3):
            docs.append(_make_doc(comp, 2024, "consolidated", idx))

    table_doc_ids = {d.doc_id for d in docs}

    with pytest.raises(Week1GateInputError, match="Not enough qualifying companies"):
        select_pilot_documents(
            tuple(docs), table_doc_ids, company_count=20, documents_per_company=3
        )


def test_prepare_pilot_creates_valid_template(tmp_path: Path) -> None:
    manifest_path, release_path, _, _, _ = _write_release(tmp_path)

    # We need a release with at least 20 companies x 3 docs for a full pilot,
    # or we can test prepare_pilot with company_count=1, documents_per_company=1
    dataset = load_gate_dataset(manifest_path, release_path)
    annotation_root = tmp_path / "annotations"

    metadata = prepare_pilot(dataset, annotation_root, company_count=1, documents_per_company=1)

    assert metadata.document_count == 1
    assert metadata.company_count == 1
    assert (annotation_root / "pilot-documents.csv").is_file()
    assert (annotation_root / "expected-tables.csv").is_file()
    assert (annotation_root / "pilot-metadata.json").is_file()

    doc_rows = read_csv_rows(annotation_root / "pilot-documents.csv", PILOT_DOCUMENT_COLUMNS)
    assert len(doc_rows) == 1

    exp_rows = read_csv_rows(annotation_root / "expected-tables.csv", EXPECTED_TABLE_COLUMNS)
    assert len(exp_rows) == 0


def test_prepare_pilot_fails_closed_existing_directory(tmp_path: Path) -> None:
    manifest_path, release_path, _, _, _ = _write_release(tmp_path)
    dataset = load_gate_dataset(manifest_path, release_path)

    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    (annotation_root / "some_file.txt").write_text("exists")

    with pytest.raises(Week1GateInputError, match="already exists and is not empty"):
        prepare_pilot(dataset, annotation_root, company_count=1, documents_per_company=1)
