# ViFinQA Inventory and Manifest Day 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, read-only ViFinQA TXT inventory and atomic JSONL manifest that preserves every discovered path without weakening `DocumentRecord`.

**Architecture:** `data/inventory.py` discovers the fixed ViFinQA hierarchy, inspects every TXT file once, and returns immutable canonical documents plus explicit issues. `data/manifests.py` merges both result streams by path and atomically writes canonical JSONL. The product CLI delegates `inventory-data` to the inventory command entry point.

**Tech Stack:** Python 3.11, Pydantic 2, hashlib, codecs, pathlib, tempfile, argparse, pytest, Ruff, mypy.

## Global Constraints

- Support only `<ticker>/<year>/<document>/<file>.txt` under the supplied ViFinQA root.
- Never modify source snapshot files.
- Accept only deterministic `utf-8` and `utf-8-sig`; invalid UTF-8 becomes an issue.
- Preserve Unicode paths and serialize JSON with `ensure_ascii=False`.
- Use the immutable Day 1 `DocumentRecord` and `stable_document_id()` unchanged.
- Sort with `(relative_path.casefold(), relative_path)` for deterministic cross-platform output.
- Empty content has status `empty` and never participates in duplicate ownership.
- The first non-empty canonical path for a digest is `ready`; later paths are `duplicate`.
- Manifest output excludes timestamps and absolute paths and is replaced atomically.
- Preserve unrelated working-tree changes in `notebooks/01_dataset_profile.ipynb`, `plan.md`, and `tests/notebooks/test_dataset_profile_notebook.py`.

## File Map

- Create `src/financial_report_qa/data/inventory.py`: inventory models, ViFinQA path parsing, one-pass inspection, duplicate classification, and command entry point.
- Create `src/financial_report_qa/data/manifests.py`: deterministic JSONL encoding and atomic writer.
- Modify `src/financial_report_qa/cli.py`: add `inventory-data` dispatch without changing `download-data` behavior.
- Create `tests/unit/data/test_inventory.py`: path, Unicode, encoding, empty, duplicate, issue, determinism, and source-immutability contracts.
- Create `tests/unit/data/test_manifests.py`: JSONL ordering, round trip, byte stability, and failure atomicity.
- Modify `tests/unit/test_cli.py`: dispatcher contract for `inventory-data`.
- Modify `data/README.md`: replace the stale CSV instruction with the JSONL command and immutability rule.

---

### Task 1: ViFinQA path contract and immutable inventory models

**Files:**
- Create: `src/financial_report_qa/data/inventory.py`
- Create: `tests/unit/data/test_inventory.py`

**Interfaces:**
- Consumes: `DocumentRecord` and `Sha256Digest` from `financial_report_qa.schemas.documents`.
- Produces: `InventoryIssue`, `InventoryResult`, and internal `_parse_vifinqa_path(path, root)`.
- The later tasks rely on `InventoryResult.documents`, `InventoryResult.issues`, and `InventoryIssue.relative_path` exactly as named here.

- [ ] **Step 1: Write failing model and path tests**

Create `tests/unit/data/test_inventory.py` with imports and these initial tests:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.data.inventory import (
    InventoryIssue,
    InventoryResult,
    _parse_vifinqa_path,
)


def test_parse_vifinqa_path_preserves_unicode_and_extracts_metadata(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = root / "vcb" / "2024" / "Báo cáo CONSOLIDATED" / "bảng cân đối.TXT"

    metadata = _parse_vifinqa_path(path, root)

    assert metadata.relative_path == (
        "vcb/2024/Báo cáo CONSOLIDATED/bảng cân đối.TXT"
    )
    assert metadata.company_code == "VCB"
    assert metadata.report_year == 2024
    assert metadata.statement_scope == "consolidated"


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("VCB/2024/file.txt", "exactly ticker/year/document/file"),
        ("v!/2024/report/file.txt", "ticker"),
        ("VCB/year/report/file.txt", "year"),
        ("VCB/2101/report/file.txt", "year"),
    ],
)
def test_parse_vifinqa_path_rejects_invalid_hierarchy(
    tmp_path: Path,
    relative: str,
    message: str,
) -> None:
    root = tmp_path / "financial_statements"

    with pytest.raises(ValueError, match=message):
        _parse_vifinqa_path(root / Path(relative), root)


def test_inventory_models_are_frozen_and_forbid_unknown_fields() -> None:
    issue = InventoryIssue(
        relative_path="bad/year/report/file.txt",
        reason="invalid year directory",
        file_size_bytes=4,
        sha256="a" * 64,
    )
    result = InventoryResult(documents=(), issues=(issue,))

    with pytest.raises(ValidationError):
        InventoryIssue.model_validate({**issue.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError, match="frozen"):
        setattr(result, "issues", ())
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_inventory.py
```

Expected: collection fails with `ModuleNotFoundError` for
`financial_report_qa.data.inventory`.

- [ ] **Step 3: Implement the models and strict ViFinQA path parser**

Create `src/financial_report_qa/data/inventory.py` with:

```python
"""Deterministic inventory for immutable ViFinQA TXT snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.schemas.documents import DocumentRecord, Sha256Digest


class InventoryIssue(BaseModel):
    """A discovered path that cannot become a canonical document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    file_size_bytes: int | None = Field(default=None, ge=0)
    sha256: Sha256Digest | None = None


class InventoryResult(BaseModel):
    """All canonical documents and rejected paths from one snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: tuple[DocumentRecord, ...]
    issues: tuple[InventoryIssue, ...]


class _PathMetadata(NamedTuple):
    relative_path: str
    company_code: str
    report_year: int
    statement_scope: Literal["consolidated", "separate", "aggregated", "other"]


def _parse_vifinqa_path(path: Path, root: Path) -> _PathMetadata:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("path must be inside the inventory root") from error
    parts = relative.parts
    if len(parts) != 4:
        raise ValueError("expected exactly ticker/year/document/file hierarchy")
    ticker_raw, year_raw, document_name, _ = parts
    if not (2 <= len(ticker_raw) <= 10 and ticker_raw.isascii() and ticker_raw.isalnum()):
        raise ValueError("invalid ticker directory")
    if not (len(year_raw) == 4 and year_raw.isascii() and year_raw.isdecimal()):
        raise ValueError("invalid year directory")
    year = int(year_raw)
    if not 1900 <= year <= 2100:
        raise ValueError("invalid year directory")
    normalized_name = document_name.casefold()
    scope: Literal["consolidated", "separate", "aggregated", "other"]
    if "consolidated" in normalized_name:
        scope = "consolidated"
    elif "separate" in normalized_name:
        scope = "separate"
    elif "aggregated" in normalized_name:
        scope = "aggregated"
    else:
        scope = "other"
    return _PathMetadata(relative.as_posix(), ticker_raw.upper(), year, scope)
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_inventory.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
uv run --frozen --no-sync mypy src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
```

Expected: all three commands pass.

- [ ] **Step 5: Commit the path and model contract**

```bash
git add src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
git commit -m "feat: define ViFinQA inventory contract"
```

---

### Task 2: One-pass file inspection and deterministic classification

**Files:**
- Modify: `src/financial_report_qa/data/inventory.py`
- Modify: `tests/unit/data/test_inventory.py`

**Interfaces:**
- Consumes: `_parse_vifinqa_path()`, `DocumentRecord`, `stable_document_id()`.
- Produces: `build_inventory(root: Path, *, repo_id: str, revision: str) -> InventoryResult`.
- `manifests.py` and the CLI consume this exact function and immutable return type.

- [ ] **Step 1: Add failing inventory behavior tests**

Append to `tests/unit/data/test_inventory.py`:

```python
import hashlib

from financial_report_qa.data.inventory import build_inventory


def _write_report(root: Path, relative: str, content: bytes) -> Path:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_build_inventory_hashes_unicode_utf8_and_preserves_source(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    content = "Bảng cân đối kế toán".encode()
    source = _write_report(root, "VCB/2024/Consolidated/Báo cáo.TXT", content)
    before = source.read_bytes()

    result = build_inventory(root, repo_id="org/vifinqa", revision="abc123")

    assert result.issues == ()
    assert len(result.documents) == 1
    record = result.documents[0]
    assert record.relative_path == "VCB/2024/Consolidated/Báo cáo.TXT"
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.file_size_bytes == len(content)
    assert record.encoding == "utf-8"
    assert record.inventory_status == "ready"
    assert source.read_bytes() == before


def test_build_inventory_distinguishes_bom_empty_duplicate_and_issue(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    shared = "doanh thu".encode()
    _write_report(root, "AAA/2023/Separate/a.txt", shared)
    _write_report(root, "AAA/2023/Separate/b.txt", shared)
    _write_report(root, "AAA/2023/Separate/bom.txt", b"\xef\xbb\xbf" + "nợ".encode())
    _write_report(root, "AAA/2023/Separate/empty.txt", b"")
    _write_report(root, "AAA/2023/Separate/bad.txt", b"\xff\xfe")
    _write_report(root, "bad/year/path.txt", b"valid bytes")

    result = build_inventory(root, repo_id="org/vifinqa", revision="abc123")
    by_path = {record.relative_path: record for record in result.documents}

    assert by_path["AAA/2023/Separate/a.txt"].inventory_status == "ready"
    duplicate = by_path["AAA/2023/Separate/b.txt"]
    assert duplicate.inventory_status == "duplicate"
    assert duplicate.notes == ("duplicate_of=AAA/2023/Separate/a.txt",)
    assert by_path["AAA/2023/Separate/bom.txt"].encoding == "utf-8-sig"
    assert by_path["AAA/2023/Separate/empty.txt"].inventory_status == "empty"
    assert {issue.relative_path for issue in result.issues} == {
        "AAA/2023/Separate/bad.txt",
        "bad/year/path.txt",
    }


def test_build_inventory_is_deterministic_and_ignores_non_txt(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    _write_report(root, "ZZZ/2022/Aggregated/z.txt", b"z")
    _write_report(root, "AAA/2022/Other/a.txt", b"a")
    _write_report(root, "AAA/2022/Other/ignored.csv", b"csv")

    first = build_inventory(root, repo_id="org/vifinqa", revision="abc123")
    second = build_inventory(root, repo_id="org/vifinqa", revision="abc123")

    assert first == second
    assert [record.relative_path for record in first.documents] == [
        "AAA/2022/Other/a.txt",
        "ZZZ/2022/Aggregated/z.txt",
    ]


@pytest.mark.parametrize("root_state", ["missing", "file"])
def test_build_inventory_rejects_non_directory_root(
    tmp_path: Path,
    root_state: str,
) -> None:
    root = tmp_path / "financial_statements"
    if root_state == "file":
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="inventory root"):
        build_inventory(root, repo_id="org/vifinqa", revision="abc123")
```

- [ ] **Step 2: Run the focused tests and confirm `build_inventory` is missing**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_inventory.py
```

Expected: collection fails because `build_inventory` is not defined.

- [ ] **Step 3: Add streaming inspection and inventory construction**

Add these imports to `inventory.py`:

```python
import codecs
import hashlib
from dataclasses import dataclass

from financial_report_qa.schemas.documents import (
    DocumentRecord,
    Sha256Digest,
    stable_document_id,
)
```

Then add the implementation:

```python
_CHUNK_SIZE = 1024 * 1024
_UTF8_BOM = codecs.BOM_UTF8


@dataclass(frozen=True)
class _FileInspection:
    file_size_bytes: int
    sha256: str
    encoding: str | None
    decode_error: str | None


def _inspect_file(path: Path) -> _FileInspection:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        prefix = stream.read(len(_UTF8_BOM))
        digest.update(prefix)
        size += len(prefix)
        encoding = "utf-8-sig" if prefix.startswith(_UTF8_BOM) else "utf-8"
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        decode_error = None
        try:
            decoder.decode(prefix)
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            decode_error = str(error)
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
    return _FileInspection(size, digest.hexdigest(), encoding, decode_error)


def _path_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def build_inventory(
    root: Path,
    *,
    repo_id: str,
    revision: str,
) -> InventoryResult:
    if not root.is_dir():
        raise FileNotFoundError(f"inventory root is not a directory: {root}")
    if not repo_id.strip():
        raise ValueError("repo_id must not be empty")
    if not revision.strip():
        raise ValueError("revision must not be empty")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt"),
        key=lambda path: _path_key(path.relative_to(root).as_posix()),
    )
    documents: list[DocumentRecord] = []
    issues: list[InventoryIssue] = []
    primary_by_digest: dict[str, str] = {}

    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        try:
            inspection = _inspect_file(path)
        except OSError as error:
            issues.append(InventoryIssue(relative_path=relative_path, reason=f"read failure: {error}"))
            continue
        try:
            metadata = _parse_vifinqa_path(path, root)
        except ValueError as error:
            issues.append(
                InventoryIssue(
                    relative_path=relative_path,
                    reason=str(error),
                    file_size_bytes=inspection.file_size_bytes,
                    sha256=inspection.sha256,
                )
            )
            continue
        if inspection.decode_error is not None:
            issues.append(
                InventoryIssue(
                    relative_path=relative_path,
                    reason=f"invalid UTF-8: {inspection.decode_error}",
                    file_size_bytes=inspection.file_size_bytes,
                    sha256=inspection.sha256,
                )
            )
            continue

        status: Literal["ready", "empty", "duplicate", "quarantine"]
        if inspection.file_size_bytes == 0:
            status = "empty"
            notes: tuple[str, ...] = ()
        elif inspection.sha256 in primary_by_digest:
            status = "duplicate"
            notes = (f"duplicate_of={primary_by_digest[inspection.sha256]}",)
        else:
            status = "ready"
            notes = ()
            primary_by_digest[inspection.sha256] = metadata.relative_path
        documents.append(
            DocumentRecord(
                doc_id=stable_document_id(inspection.sha256),
                repo_id=repo_id,
                revision=revision,
                relative_path=metadata.relative_path,
                company_code=metadata.company_code,
                report_year=metadata.report_year,
                statement_scope=metadata.statement_scope,
                sha256=inspection.sha256,
                file_size_bytes=inspection.file_size_bytes,
                encoding=inspection.encoding,
                inventory_status=status,
                notes=notes,
            )
        )
    return InventoryResult(documents=tuple(documents), issues=tuple(issues))
```

- [ ] **Step 4: Run focused quality checks**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_inventory.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
uv run --frozen --no-sync mypy src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
```

Expected: all pass. Inspect the temporary source files in the first test to confirm their
bytes remain unchanged.

- [ ] **Step 5: Commit inventory construction**

```bash
git add src/financial_report_qa/data/inventory.py tests/unit/data/test_inventory.py
git commit -m "feat: inventory immutable ViFinQA snapshots"
```

---

### Task 3: Deterministic atomic JSONL manifest

**Files:**
- Create: `src/financial_report_qa/data/manifests.py`
- Create: `tests/unit/data/test_manifests.py`

**Interfaces:**
- Consumes: `InventoryIssue` and `InventoryResult` from Task 1.
- Produces: `write_manifest(result: InventoryResult, path: Path) -> None`.
- Manifest entries contain `record_type` plus the complete Pydantic JSON payload.

- [ ] **Step 1: Write failing manifest tests**

Create `tests/unit/data/test_manifests.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

import financial_report_qa.data.manifests as manifests
from financial_report_qa.data.inventory import InventoryIssue, InventoryResult
from financial_report_qa.data.manifests import write_manifest
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def _document(relative_path: str, digest: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative_path,
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=12,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_write_manifest_sorts_all_paths_preserves_unicode_and_round_trips(
    tmp_path: Path,
) -> None:
    document = _document("VCB/2024/Consolidated/Báo cáo.txt", "b" * 64)
    issue = InventoryIssue(
        relative_path="AAA/2023/Separate/bad.txt",
        reason="invalid UTF-8",
        file_size_bytes=2,
        sha256="a" * 64,
    )
    result = InventoryResult(documents=(document,), issues=(issue,))
    path = tmp_path / "nested" / "documents.jsonl"

    write_manifest(result, path)

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert "Báo cáo".encode() in raw
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert [row["relative_path"] for row in rows] == [
        "AAA/2023/Separate/bad.txt",
        "VCB/2024/Consolidated/Báo cáo.txt",
    ]
    assert rows[0]["record_type"] == "issue"
    assert InventoryIssue.model_validate({k: v for k, v in rows[0].items() if k != "record_type"}) == issue
    assert rows[1]["record_type"] == "document"
    assert DocumentRecord.model_validate(
        {k: v for k, v in rows[1].items() if k != "record_type"}
    ) == document


def test_write_manifest_is_byte_deterministic(tmp_path: Path) -> None:
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )
    path = tmp_path / "documents.jsonl"

    write_manifest(result, path)
    first = path.read_bytes()
    write_manifest(result, path)

    assert path.read_bytes() == first


def test_serialization_failure_preserves_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("previous\n", encoding="utf-8")
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )

    def fail_serialization(record_type: str, model: object) -> str:
        raise TypeError(f"cannot serialize {record_type}: {type(model).__name__}")

    monkeypatch.setattr(manifests, "_serialize_entry", fail_serialization)

    with pytest.raises(TypeError, match="cannot serialize"):
        write_manifest(result, path)

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.iterdir()) == [path]
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_manifests.py
```

Expected: collection fails with `ModuleNotFoundError` for
`financial_report_qa.data.manifests`.

- [ ] **Step 3: Implement canonical serialization and atomic replacement**

Create `src/financial_report_qa/data/manifests.py`:

```python
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
```

`_serialize_entry(..., model: BaseModel)` accepts both concrete immutable models, while
the entry union lets mypy verify that every item has `relative_path`.

- [ ] **Step 4: Run manifest quality checks**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_manifests.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data/manifests.py tests/unit/data/test_manifests.py
uv run --frozen --no-sync mypy src/financial_report_qa/data/manifests.py tests/unit/data/test_manifests.py
```

Expected: all pass; the failure test leaves only the original manifest in the directory.

- [ ] **Step 5: Commit the manifest writer**

```bash
git add src/financial_report_qa/data/manifests.py tests/unit/data/test_manifests.py
git commit -m "feat: write atomic inventory manifests"
```

---

### Task 4: Inventory command and product CLI dispatch

**Files:**
- Modify: `src/financial_report_qa/data/inventory.py`
- Modify: `src/financial_report_qa/cli.py`
- Modify: `tests/unit/data/test_inventory.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `data/README.md`

**Interfaces:**
- Consumes: `build_inventory()` and `write_manifest()`.
- Produces: `financial-report-qa inventory-data` and
  `inventory.main(argv: Sequence[str] | None = None) -> int`.
- Existing `download-data` argument forwarding remains unchanged.

- [ ] **Step 1: Add failing inventory command tests**

Append to `tests/unit/data/test_inventory.py`:

```python
from financial_report_qa.data.inventory import main


def test_inventory_main_writes_manifest_and_prints_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "financial_statements"
    _write_report(root, "AAA/2024/Consolidated/ready.txt", b"ready")
    _write_report(root, "AAA/2024/Consolidated/empty.txt", b"")
    manifest = tmp_path / "manifests" / "documents.jsonl"

    exit_code = main(
        [
            "--root", str(root),
            "--repo-id", "org/vifinqa",
            "--revision", "abc123",
            "--manifest", str(manifest),
        ]
    )

    assert exit_code == 0
    assert manifest.exists()
    output = capsys.readouterr().out
    assert "Documents: 2" in output
    assert "Ready:     1" in output
    assert "Empty:     1" in output
    assert "Duplicate: 0" in output
    assert "Issues:    0" in output


def test_inventory_main_reports_expected_failure_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--root", str(tmp_path / "missing"),
            "--repo-id", "org/vifinqa",
            "--revision", "abc123",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "inventory root" in captured.err
    assert "Traceback" not in captured.err
```

- [ ] **Step 2: Add the failing dispatcher test**

Update `tests/unit/test_cli.py` imports and append:

```python
def test_inventory_data_forwards_arguments() -> None:
    received: list[str] = []

    def fake_inventory_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 0

    exit_code = main(
        ["inventory-data", "--root", "data/raw/vifinqa"],
        inventory_main_fn=fake_inventory_main,
    )

    assert exit_code == 0
    assert received == ["--root", "data/raw/vifinqa"]
```

- [ ] **Step 3: Run tests and confirm missing command behavior**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data/test_inventory.py tests/unit/test_cli.py
```

Expected: failures because `inventory.main`, the CLI dependency injection parameter, and
the `inventory-data` parser branch do not exist.

- [ ] **Step 4: Implement the inventory command entry point**

Add imports to `inventory.py`:

```python
import argparse
import sys
from collections.abc import Sequence

from financial_report_qa.core.errors import FinancialReportQAError
```

Add the parser and entry point at the end of `inventory.py`. The local manifest import
keeps the inventory model module importable by `manifests.py` without a cycle:

```python
DEFAULT_MANIFEST_PATH = Path("data/manifests/documents.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory an immutable ViFinQA snapshot.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and publish a ViFinQA inventory manifest."""
    from financial_report_qa.data.manifests import write_manifest

    args = _parser().parse_args(argv)
    try:
        result = build_inventory(
            args.root,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        write_manifest(result, args.manifest)
    except (FinancialReportQAError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    counts = {status: 0 for status in ("ready", "empty", "duplicate")}
    for document in result.documents:
        if document.inventory_status in counts:
            counts[document.inventory_status] += 1
    print(f"Documents: {len(result.documents)}")
    print(f"Ready:     {counts['ready']}")
    print(f"Empty:     {counts['empty']}")
    print(f"Duplicate: {counts['duplicate']}")
    print(f"Issues:    {len(result.issues)}")
    print(f"Manifest:  {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add product CLI dispatch**

Modify `src/financial_report_qa/cli.py` to import and inject both child commands:

```python
from financial_report_qa.data.inventory import main as inventory_main

InventoryMain = Callable[[Sequence[str] | None], int]
```

Add the parser command:

```python
commands.add_parser(
    "inventory-data",
    add_help=False,
    help="Inventory a ViFinQA TXT snapshot and write its manifest.",
)
```

Change the dispatcher signature and branch:

```python
def main(
    argv: Sequence[str] | None = None,
    *,
    download_main_fn: DownloadMain = download_main,
    inventory_main_fn: InventoryMain = inventory_main,
) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parsed, remaining = _parser().parse_known_args(raw_args)
    if parsed.command == "download-data":
        return download_main_fn(remaining)
    if parsed.command == "inventory-data":
        return inventory_main_fn(remaining)
    raise AssertionError("argparse accepted an unknown command")
```

- [ ] **Step 6: Update the data operator documentation**

Replace the stale CSV sentence in `data/README.md` with:

```markdown
`data/raw/` is append-only. After downloading a revision-pinned ViFinQA snapshot,
create its deterministic manifest without modifying source files:

```bash
financial-report-qa inventory-data \
  --root data/raw/ViFinQA/financial_statements \
  --repo-id tinixai/ViFinQA \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --manifest data/manifests/documents.jsonl
```

Only commit small, redistributable manifests. Investigate every `record_type="issue"`
entry before ingestion.
```

- [ ] **Step 7: Run the complete Day 2 quality gate**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data tests/unit/test_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data src/financial_report_qa/cli.py tests/unit/data tests/unit/test_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/data src/financial_report_qa/cli.py tests/unit/data tests/unit/test_cli.py
```

Expected: all pass. Also run the existing schema regression suite because Day 2 constructs
Day 1 models:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas
```

Expected: all schema tests pass unchanged.

- [ ] **Step 8: Run a synthetic repeatability smoke test**

Run this PowerShell-safe pytest-backed smoke command rather than touching the real raw
snapshot:

```bash
uv run --frozen --no-sync pytest -q \
  tests/unit/data/test_inventory.py::test_build_inventory_is_deterministic_and_ignores_non_txt \
  tests/unit/data/test_manifests.py::test_write_manifest_is_byte_deterministic
```

Expected: `2 passed` and no generated artifact under the repository `data/raw/` tree.

- [ ] **Step 9: Review scope and commit the command integration**

Run:

```bash
git diff --check -- src/financial_report_qa/data/inventory.py src/financial_report_qa/data/manifests.py src/financial_report_qa/cli.py tests/unit/data/test_inventory.py tests/unit/data/test_manifests.py tests/unit/test_cli.py data/README.md
git status --short
git diff -- src/financial_report_qa/data/inventory.py src/financial_report_qa/cli.py tests/unit/data/test_inventory.py tests/unit/test_cli.py data/README.md
```

Expected: only Day 2 files are included in the reviewed diff; the pre-existing notebook,
roadmap, and notebook-test changes remain unstaged.

Commit only the explicit Task 4 paths:

```bash
git add src/financial_report_qa/data/inventory.py src/financial_report_qa/cli.py tests/unit/data/test_inventory.py tests/unit/test_cli.py data/README.md
git commit -m "feat: expose ViFinQA inventory command"
```

---

## Final Verification

- [ ] Run the Day 2 test, lint, and type-check gates from Task 4 again after the final commit.
- [ ] Run `git status --short` and verify only the user's pre-existing unrelated changes remain.
- [ ] Confirm `git log -4 --oneline` shows the three implementation commits after the approved design/plan documentation commits.
- [ ] Do not inventory the full local dataset unless the user explicitly supplies the immutable `repo-id`, resolved `revision`, and intended output path.
