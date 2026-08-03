"""Deterministic atomic JSONL manifests for inventory results."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from financial_report_qa.data.inventory import InventoryIssue, InventoryResult
from financial_report_qa.schemas.documents import DocumentRecord


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
            temporary_path.unlink(missing_ok=True)
        raise
