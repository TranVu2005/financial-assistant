"""Stable 20x3 pilot selection, rank calculation, and pilot preparation."""

import hashlib
import tempfile
from pathlib import Path

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    SAMPLING_VERSION,
    PilotDocument,
    PilotMetadata,
    read_csv_rows,
    write_canonical_json,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset
from financial_report_qa.schemas import DocumentRecord


def stable_rank(namespace: str, *parts: object) -> str:
    """Return a deterministic, namespaced rank digest."""
    payload = "\n".join((SAMPLING_VERSION, namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_pilot_documents(
    documents: tuple[DocumentRecord, ...],
    table_doc_ids: set[str],
    *,
    company_count: int = 20,
    documents_per_company: int = 3,
) -> tuple[PilotDocument, ...]:
    """Select a deterministic 20-company x 3-document pilot set."""
    eligible_docs = [
        doc for doc in documents if doc.inventory_status == "ready" and doc.doc_id in table_doc_ids
    ]

    docs_by_company: dict[str, list[DocumentRecord]] = {}
    for doc in eligible_docs:
        docs_by_company.setdefault(doc.company_code, []).append(doc)

    qualifying_companies = [
        comp for comp, docs in docs_by_company.items() if len(docs) >= documents_per_company
    ]
    if len(qualifying_companies) < company_count:
        raise Week1GateInputError(
            f"Not enough qualifying companies with at least {documents_per_company} documents: "
            f"found {len(qualifying_companies)}, required {company_count}"
        )

    qualifying_companies.sort(key=lambda comp: stable_rank("company", comp))
    selected_companies = qualifying_companies[:company_count]

    selected_docs: list[DocumentRecord] = []

    for comp in selected_companies:
        company_docs = docs_by_company[comp]

        # Stratum pass: group by (report_year, statement_scope)
        strata_groups: dict[tuple[int, str], list[DocumentRecord]] = {}
        for doc in company_docs:
            strata_groups.setdefault((doc.report_year, doc.statement_scope), []).append(doc)

        sorted_strata = sorted(
            strata_groups.keys(),
            key=lambda s: stable_rank("stratum", comp, s[0], s[1]),
        )

        chosen_comp_docs: list[DocumentRecord] = []
        for s in sorted_strata:
            if len(chosen_comp_docs) >= documents_per_company:
                break
            group = strata_groups[s]
            group.sort(key=lambda d: stable_rank("document", d.doc_id))
            chosen_comp_docs.append(group[0])

        if len(chosen_comp_docs) < documents_per_company:
            remaining = [d for d in company_docs if d not in chosen_comp_docs]
            remaining.sort(key=lambda d: stable_rank("document", d.doc_id))
            chosen_comp_docs.extend(remaining[: documents_per_company - len(chosen_comp_docs)])

        selected_docs.extend(chosen_comp_docs)

    pilot_docs: list[PilotDocument] = []
    for doc in selected_docs:
        pilot_docs.append(
            PilotDocument(
                annotation_schema_version="1",
                dataset_fingerprint="",  # Set by prepare_pilot
                source_manifest_sha256="",  # Set by prepare_pilot
                doc_id=doc.doc_id,
                relative_path=doc.relative_path,
                company_code=doc.company_code,
                report_year=doc.report_year,
                statement_scope=doc.statement_scope,
            )
        )

    return tuple(pilot_docs)


def prepare_pilot(
    dataset: GateDataset,
    annotation_root: Path,
    *,
    company_count: int = 20,
    documents_per_company: int = 3,
) -> PilotMetadata:
    """Prepare pilot documents and empty template under annotation_root."""
    if annotation_root.exists() and any(annotation_root.iterdir()):
        raise Week1GateInputError(
            f"Annotation root {annotation_root} already exists and is not empty"
        )

    table_doc_ids = {tbl.doc_id for tbl in dataset.tables_by_id.values()}
    all_docs = tuple(dataset.documents_by_id.values())

    raw_pilot_docs = select_pilot_documents(
        all_docs,
        table_doc_ids,
        company_count=company_count,
        documents_per_company=documents_per_company,
    )

    bound_pilot_docs = tuple(
        PilotDocument(
            annotation_schema_version="1",
            dataset_fingerprint=dataset.dataset_fingerprint,
            source_manifest_sha256=dataset.source_manifest_sha256,
            doc_id=doc.doc_id,
            relative_path=doc.relative_path,
            company_code=doc.company_code,
            report_year=doc.report_year,
            statement_scope=doc.statement_scope,
        )
        for doc in raw_pilot_docs
    )

    bound_pilot_docs = tuple(
        sorted(
            bound_pilot_docs,
            key=lambda d: (
                d.company_code,
                d.report_year,
                d.statement_scope,
                d.doc_id,
            ),
        )
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            dir=annotation_root.parent, prefix=f".week1-prepare-{annotation_root.name}-"
        )
    )

    try:
        doc_rows = [doc.model_dump(mode="json") for doc in bound_pilot_docs]
        docs_csv_path = temp_dir / "pilot-documents.csv"
        write_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS, doc_rows)

        docs_sha256 = hashlib.sha256(docs_csv_path.read_bytes()).hexdigest()

        expected_csv_path = temp_dir / "expected-tables.csv"
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, ())

        metadata = PilotMetadata(
            annotation_schema_version="1",
            sampling_version=SAMPLING_VERSION,
            dataset_fingerprint=dataset.dataset_fingerprint,
            source_manifest_sha256=dataset.source_manifest_sha256,
            document_count=len(bound_pilot_docs),
            company_count=len({d.company_code for d in bound_pilot_docs}),
            pilot_documents_sha256=docs_sha256,
        )
        write_canonical_json(temp_dir / "pilot-metadata.json", metadata.model_dump(mode="json"))

        read_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS)
        read_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS)

        temp_dir.replace(annotation_root)
        return metadata
    finally:
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
