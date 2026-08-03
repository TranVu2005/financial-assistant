"""Deterministic atomic JSONL manifests for inventory results."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from financial_report_qa.core.errors import DatasetBuildError
from financial_report_qa.data.inventory import InventoryIssue, InventoryResult
from financial_report_qa.schemas.documents import DocumentRecord


class ManifestSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    inventory: InventoryResult
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _path_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _serialize_entry(record_type: str, model: BaseModel) -> str:
    payload = {"record_type": record_type, **model.model_dump(mode="json")}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_manifest(result: InventoryResult, path: Path) -> None:
    """Atomically write every document and issue in stable path order."""
    entries: list[tuple[str, DocumentRecord | InventoryIssue]] = [
        ("document", document) for document in result.documents
    ]
    entries.extend(("issue", issue) for issue in result.issues)
    entries.sort(key=lambda item: _path_key(item[1].relative_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record_type, model in entries:
                stream.write(_serialize_entry(record_type, model))
                stream.write("\n")
            stream.flush()
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def read_manifest(path: Path) -> ManifestSnapshot:
    """Strictly read and validate a JSONL inventory manifest."""
    if not path.is_file():
        raise DatasetBuildError(f"manifest file does not exist: {path}")

    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise DatasetBuildError("manifest file must end with a newline")

    sha256 = hashlib.sha256(raw).hexdigest()

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetBuildError(f"manifest is not valid UTF-8: {exc}") from exc

    documents: list[DocumentRecord] = []
    issues: list[InventoryIssue] = []
    seen_paths: set[str] = set()
    ready_doc_ids: set[str] = set()

    for idx, line in enumerate(content.splitlines(), start=1):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetBuildError(f"manifest line {idx}: invalid JSON") from exc

        if not isinstance(data, dict):
            raise DatasetBuildError(f"manifest line {idx}: record must be a JSON object")

        record_type = data.pop("record_type", None)
        if record_type == "document":
            try:
                doc = DocumentRecord.model_validate(data)
            except ValidationError as exc:
                raise DatasetBuildError(
                    f"manifest line {idx}: invalid document record"
                ) from exc

            if doc.relative_path in seen_paths:
                raise DatasetBuildError(
                    f"manifest line {idx}: duplicate relative_path {doc.relative_path!r}"
                )
            seen_paths.add(doc.relative_path)

            if doc.inventory_status == "ready":
                if doc.doc_id in ready_doc_ids:
                    raise DatasetBuildError(
                        f"manifest line {idx}: duplicate ready doc_id {doc.doc_id!r}"
                    )
                ready_doc_ids.add(doc.doc_id)

            documents.append(doc)

        elif record_type == "issue":
            try:
                iss = InventoryIssue.model_validate(data)
            except ValidationError as exc:
                raise DatasetBuildError(
                    f"manifest line {idx}: invalid issue record"
                ) from exc

            if iss.relative_path in seen_paths:
                raise DatasetBuildError(
                    f"manifest line {idx}: duplicate relative_path {iss.relative_path!r}"
                )
            seen_paths.add(iss.relative_path)

            issues.append(iss)

        else:
            raise DatasetBuildError(
                f"manifest line {idx}: unknown or missing record_type {record_type!r}"
            )

    inventory = InventoryResult(documents=tuple(documents), issues=tuple(issues))
    return ManifestSnapshot(inventory=inventory, sha256=sha256)
