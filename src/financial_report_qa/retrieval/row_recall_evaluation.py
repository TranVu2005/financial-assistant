"""plan.md §20 Row Recall@k / Table Recall@k measurement.

Everything in §7/§9/§15's row-fusion, grounding-provenance, and threshold-
retry work was built with no gold row-level data to measure it against
(flagged repeatedly across those changes). `data/qa/answer-gold-v1.jsonl`
turns out to already carry exactly the missing signal -- each evidence
entry names the real `row_label` an answer came from -- so this module
scores the *live* pipeline (the same `RetrievalService` + `RowFusionService`
`submission/exporter.py` actually uses) against it, end to end: Table
Recall@k first (did the right table even get retrieved), then Row
Recall@1/3/5/10 within whatever tables that step actually returned -- never
against the idealized gold table set, since that would hide a table-
retrieval failure inside an apparently-fine row-recall number.

plan.md §20's own guidance on reading the two numbers together:

    Row Recall@10 = 95%, Cell Accuracy = 60%  -> lỗi nằm ở reranker/grounder
    Row Recall@10 = 55%                       -> lỗi nằm ở row representation/retrieval
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import Field

from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.live_query import TableRetriever, retrieve_candidate_table_ids
from financial_report_qa.retrieval.row_fusion import RowFusionService
from financial_report_qa.retrieval.service import RetrievalService

ROW_RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)


class RowRecallQuestion(_FrozenModel):
    """One gold question reduced to what this measurement needs: the real
    row label(s) its answer's evidence cells came from, and the table(s)
    they live in."""

    question_id: NonEmptyString
    question: NonEmptyString
    gold_row_labels: tuple[NonEmptyString, ...]
    gold_table_ids: tuple[TableId, ...]


def load_row_recall_gold(path: Path) -> tuple[RowRecallQuestion, ...]:
    """Reduce `answer-gold-v1.jsonl` (question_id/question/evidence[]) to
    `RowRecallQuestion`s, deduplicating each question's row labels/table ids
    (a `difference`/`growth_rate` question cites the same row across two
    periods -- one label counted once)."""
    questions: list[RowRecallQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row_labels = tuple(
                dict.fromkeys(
                    evidence["row_label"]
                    for evidence in record["evidence"]
                    if evidence.get("row_label")
                )
            )
            table_ids = tuple(
                dict.fromkeys(evidence["table_id"] for evidence in record["evidence"])
            )
            if not row_labels or not table_ids:
                continue
            questions.append(
                RowRecallQuestion(
                    question_id=record["question_id"],
                    question=record["question"],
                    gold_row_labels=row_labels,
                    gold_table_ids=table_ids,
                )
            )
    return tuple(questions)


class RowRecallOutcome(_FrozenModel):
    """One question's measurement: where (if anywhere) a gold row label
    surfaced in the live pipeline's retrieved tables and fused row ranking."""

    question_id: NonEmptyString
    question: NonEmptyString
    gold_row_labels: tuple[NonEmptyString, ...]
    table_recall_hit: bool
    retrieved_table_count: int = Field(ge=0)
    # Best (lowest) 1-based rank any gold row label reached in fusion, or
    # `None` when no gold label appeared anywhere in the ranked results
    # (including because `table_recall_hit` was already False -- fusion
    # never even searched the right table).
    best_rank: int | None = Field(default=None, ge=1)
    candidate_row_count: int = Field(ge=0)


class RowRecallReport(_FrozenModel):
    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    k_tables: int = Field(gt=0)
    max_row_k: int = Field(gt=0)
    table_recall_at_k: float
    row_recall_at: dict[str, float]
    outcomes: tuple[RowRecallOutcome, ...]


def _label_matches(gold_labels: Sequence[str]) -> frozenset[str]:
    return frozenset(normalized_key(label) for label in gold_labels)


def evaluate_row_recall(
    questions: Sequence[RowRecallQuestion],
    table_service: RetrievalService,
    row_fusion: RowFusionService,
    *,
    dataset_fingerprint: str,
    k_tables: int = 10,
    max_row_k: int = max(ROW_RECALL_KS),
) -> RowRecallReport:
    """Score the live table + row retrieval pipeline against gold row
    labels. Table retrieval and row fusion both run for real, exactly as
    `submission/exporter.py` calls them -- no gold-table shortcut."""
    outcomes: list[RowRecallOutcome] = []
    for question in questions:
        # cast: RetrievalTrace structurally satisfies TableRetriever but mypy
        # cannot prove it against the _RankedResult protocol (same known
        # pattern as retrieval/cli.py's sweep-k wiring).
        retrieved = retrieve_candidate_table_ids(
            question.question, cast(TableRetriever, table_service), k=k_tables
        )
        table_recall_hit = bool(set(question.gold_table_ids) & set(retrieved))

        gold_keys = _label_matches(question.gold_row_labels)
        best_rank: int | None = None
        candidate_row_count = 0
        if retrieved:
            trace = row_fusion.retrieve_rows(
                question.question, candidate_table_ids=retrieved, k=max_row_k
            )
            candidate_row_count = len(trace.results)
            for candidate in trace.results:
                label = candidate.metadata.row_label_raw
                if label and normalized_key(label) in gold_keys:
                    best_rank = candidate.rank
                    break

        outcomes.append(
            RowRecallOutcome(
                question_id=question.question_id,
                question=question.question,
                gold_row_labels=question.gold_row_labels,
                table_recall_hit=table_recall_hit,
                retrieved_table_count=len(retrieved),
                best_rank=best_rank,
                candidate_row_count=candidate_row_count,
            )
        )

    total = len(outcomes)
    table_recall = sum(1 for o in outcomes if o.table_recall_hit) / total if total else 0.0
    row_recall_at = {
        str(k): (
            sum(1 for o in outcomes if o.best_rank is not None and o.best_rank <= k) / total
            if total
            else 0.0
        )
        for k in ROW_RECALL_KS
        if k <= max_row_k
    }

    return RowRecallReport(
        dataset_fingerprint=dataset_fingerprint,
        question_count=total,
        k_tables=k_tables,
        max_row_k=max_row_k,
        table_recall_at_k=table_recall,
        row_recall_at=row_recall_at,
        outcomes=tuple(outcomes),
    )


def _render_row_recall_markdown(report: RowRecallReport) -> str:
    lines = [
        "# plan.md §20 Row Recall@k / Table Recall@k",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Table retrieval depth (k): {report.k_tables}",
        f"- Row fusion depth (k): {report.max_row_k}",
        f"- Table Recall@{report.k_tables}: {report.table_recall_at_k:.3f}",
        "",
        "## Row Recall@k",
        "",
        "| k | Recall |",
        "| ---: | ---: |",
    ]
    for k in ROW_RECALL_KS:
        key = str(k)
        if key in report.row_recall_at:
            lines.append(f"| {k} | {report.row_recall_at[key]:.3f} |")
    lines.extend(("", "## Per-question outcomes", ""))
    for outcome in report.outcomes:
        rank_text = outcome.best_rank if outcome.best_rank is not None else "not found"
        lines.append(
            f"- `{outcome.question_id}` -> table_hit={outcome.table_recall_hit}, "
            f"row_rank={rank_text} (of {outcome.candidate_row_count} candidates)"
        )
    return "\n".join(lines) + "\n"


def write_row_recall_report(report: RowRecallReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"row-recall-{prefix}.json"
    markdown_path = output_dir / f"row-recall-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_row_recall_markdown(report))
    return json_path, markdown_path
