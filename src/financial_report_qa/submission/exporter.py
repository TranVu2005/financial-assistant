"""Day 22 submission exporter: raw question -> live retrieval -> plan ->
execution -> verification -> `SubmissionItem` + replayable CSV rows, plus a
coverage report for every question that did NOT make it into the bundle
(plan.md §2.4: only a verified `answered` result may become a `SubmissionItem`
-- `abstained`/`error` block release, they are never silently dropped, just
reported outside the ZIP).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import SubmissionInputError
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.pipeline.contracts import PipelineStage
from financial_report_qa.planning.cell_grounding import ground_question
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import RowChoiceDecision
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.row_fusion import (
    DEFAULT_ROW_CANDIDATE_COUNT,
    RowFusionService,
)
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.backstop_answer import build_backstop_item
from financial_report_qa.submission.citation_summary import (
    relevant_docs_and_tables as _relevant_docs_and_tables,
)
from financial_report_qa.submission.contracts import (
    QuestionOutcome,
    RawQuestion,
    SubmissionEvidence,
    SubmissionExportReport,
    SubmissionItem,
)
from financial_report_qa.submission.validator import ANSWER_TOLERANCE
from financial_report_qa.verification.builder import build_answer_package
from financial_report_qa.verification.evaluation import build_citation_lookup

CsvRow = Mapping[str, object]


def load_raw_questions(path: Path) -> tuple[RawQuestion, ...]:
    """Parse the official question file (`{"id": int, "question": str}` per
    line, e.g. `data/raw/ViFinQA/questions/questions.jsonl`). Raises on any
    duplicate id -- the submission contract requires an exact, non-duplicate
    id set."""
    records: list[RawQuestion] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            question = RawQuestion.model_validate(json.loads(stripped))
            if question.id in seen_ids:
                raise SubmissionInputError(f"duplicate question id in {path}: {question.id}")
            seen_ids.add(question.id)
            records.append(question)
    return tuple(sorted(records, key=lambda item: item.id))


def _synthetic_question_id(question_id: int) -> str:
    """`AnswerPackage.question_id` requires `^retq_[0-9a-f]{64}$` (Day 8
    contract) -- the official question file's integer `id` never matches
    that pattern. This mapping is used only internally to build the
    `AnswerPackage`/citation chain; it never appears in `submission.json`,
    which uses the real integer `id` (plan.md §2.4)."""
    return "retq_" + hashlib.sha256(f"submission:{question_id}".encode()).hexdigest()


def _real_table_evidence_rows(
    compiled: CompiledQuery, release_dir: Path, *, timeout_seconds: float
) -> tuple[CsvRow, ...] | None:
    """Day 23 evidence-table fix: the packaged CSV must be the real
    extracted table(s) the compiler actually searched (every numeric cell,
    via `build_cell_frame` -- the exact same frame `locate()` searched
    within), not a synthesized single row for just the cell the answer
    used. Scoped to only the tables the evidence cells actually came from
    (usually 1, sometimes up to 4 -- narrower than the full candidate set),
    which cannot introduce new ambiguity: any row `locate()` uniquely found
    in the wider candidate frame stays uniquely findable here.

    Returns `None` when this frame does not independently replay to the
    same answer -- e.g. two evidence cells drawn from tables in different
    units, where `compile_plan` reconverted one but this raw multi-row frame
    cannot. There is no synthetic single-row CSV fallback for that case: the
    caller must treat a `None` return as an execution failure
    (`evidence_frame_replay_mismatch`), not synthesize a replacement row.
    Never trusts the coincidence; always re-verifies via the sandbox.

    Also returns `None` when the evidence table(s) together carry fewer than
    2 numeric cells (Critical 1, 2026-08-21 final review): a 1-row CSV would
    have its single `value` equal to `item.answer`, which is the exact
    hardcode shape (`result = df["answer"].iloc[0]`) compliance checks
    C1+C2 exist to catch. The caller treats this identically to a replay
    mismatch -- fall through to the backstop tier, which independently
    guards against the same singleton-table shape.
    """
    evidence_table_ids: tuple[str, ...] = tuple(
        dict.fromkeys(cell.table_id for cell in compiled.evidence)
    )
    frame = build_cell_frame(release_dir, evidence_table_ids)
    if len(frame) < 2:
        return None
    rows: tuple[CsvRow, ...] = tuple(
        {
            "table_id": record["table_id"],
            "row_idx": record["row_idx"],
            "col_idx": record["col_idx"],
            "company_code": record["company_code"],
            "row_label_canonical": record["row_label_canonical"],
            "row_label_raw": record["row_label_raw"],
            "column_label": record["column_label"],
            "period": record["period"],
            "value": record["value"],
        }
        for record in frame.to_dict(orient="records")
    )
    replay_frame = pd.DataFrame(list(rows))
    replay_frame["period"] = replay_frame["period"].astype("Int64")
    sandbox_result = replay_in_sandbox(
        compiled.pandas_query, replay_frame, timeout_seconds=timeout_seconds
    )
    if sandbox_result.error_code is not None:
        return None
    # Same tolerance `validate_submission_zip` will apply to this very query
    # and CSV. Exact equality here was strictly harsher than the rule the
    # submission is actually judged by: this frame carries float64 straight
    # from `cells.parquet`, while `compiled.answer` is a Decimal, so a
    # converted answer that is right to the last cent still compared unequal.
    assert compiled.answer is not None
    if sandbox_result.value is None:
        return None
    if abs(sandbox_result.value - compiled.answer) > ANSWER_TOLERANCE:
        return None
    return rows


def _run_one_question(
    raw_question: RawQuestion,
    service: RetrievalService,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    k: int,
    row_fusion: RowFusionService | None = None,
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, RowChoiceDecision] | None = None,
) -> tuple[QuestionOutcome, SubmissionItem | None, tuple[CsvRow, ...] | None]:
    question = raw_question.question

    retrieved = retrieve_candidate_table_ids(question, service, k=k)
    if not retrieved:
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "abstained",
                    "stage": "retrieval",
                    "code": "no_candidate_tables",
                }
            ),
            None,
            None,
        )

    # Row fusion feeds the single answering path its ranked row candidates.
    # `k` must match `submission row-batches --rows-per-question`'s default
    # (DEFAULT_ROW_CANDIDATE_COUNT): a decision file produced by `row-batches`
    # may reference `chosen` indices up to that count - 1, and if this
    # candidate list were shorter, those indices would be out of range and
    # silently fall back to rank 1.
    fusion_rows = (
        row_fusion.retrieve_rows(
            question, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT
        ).results
        if row_fusion is not None
        else ()
    )

    entities = parse_query_entities(question)
    # Nhánh answering duy nhất (spec 2026-08-23 §6, nguyên tắc N6): câu hỏi đi
    # qua đúng một đường `ground_question` -- quyết định offline của LLM (hoặc
    # mặc định hạng 1 khi không có) -> plan -> compile. Không tầng thứ hai chạy
    # ra cứu; hỏng ở đâu thì hỏng rõ ở đó với đúng một mã lỗi.
    grounding = ground_question(
        entities=entities,
        decision=(row_decisions or {}).get(raw_question.id),
        fusion_rows=fusion_rows,
        candidate_table_ids=retrieved,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    if grounding.status != "accepted":
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "abstained",
                    "stage": "planning",
                    "code": grounding.error_code,
                    "plan_source": "llm_decision",
                }
            ),
            None,
            None,
        )
    assert grounding.plan is not None and grounding.compiled is not None
    plan = grounding.plan
    compiled = grounding.compiled
    low_confidence = grounding.low_confidence
    cell_ids = tuple(cell_id for cell in compiled.evidence for cell_id in cell.cell_ids)
    citation_lookup = build_citation_lookup(release_dir, cell_ids)
    package = build_answer_package(
        question_id=_synthetic_question_id(raw_question.id),
        question=question,
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset(retrieved),
        citation_lookup=citation_lookup,
        allow_inferred_scope=allow_inferred_scope,
        # plan.md §15: independently re-locate every fact behind the answer.
        release_dir=release_dir,
    )
    if package.verification_status == "rejected":
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "error",
                    "stage": "verification",
                    "code": "+".join(issue.code for issue in package.verification_issues),
                    "plan_source": "llm_decision",
                }
            ),
            None,
            None,
        )

    # BI-1 (design §5.1): evidence CSV luôn là lát cắt bảng nguồn thật. Khi
    # bảng thật không replay ra đúng đáp án, câu này KHÔNG được đóng gói một
    # dòng dựng ngược từ đáp án -- đó chính là mẫu `result = df["answer"]
    # .iloc[0]` mà thể lệ cấm. Trả về thất bại execution để backstop (đã hợp
    # lệ từ Task 4) tiếp quản.
    evidence_rows = _real_table_evidence_rows(
        compiled, release_dir, timeout_seconds=execution_settings.timeout_seconds
    )
    if evidence_rows is None:
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "error",
                    "stage": "execution",
                    "code": "evidence_frame_replay_mismatch",
                    "plan_source": "llm_decision",
                }
            ),
            None,
            None,
        )

    relevant_docs, relevant_tables = _relevant_docs_and_tables(retrieved, release_dir)
    csv_path = f"data/q{raw_question.id:06d}_df1.csv"
    assert compiled.answer is not None
    item = SubmissionItem.model_validate(
        {
            "id": raw_question.id,
            "question": question,
            "answer": float(compiled.answer),
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "evidence": (SubmissionEvidence(variable="df1", csv_path=csv_path),),
            "pandas_query": compiled.pandas_query,
        }
    )
    return (
        QuestionOutcome.model_validate(
            {
                "id": raw_question.id,
                "question": question,
                "status": "answered",
                "stage": None,
                "code": None,
                "plan_source": "llm_decision",
                "grounding_score": grounding.grounding_score,
                "low_confidence": low_confidence,
            }
        ),
        item,
        evidence_rows,
    )


def _apply_backstop(
    raw_question: RawQuestion,
    service: RetrievalService,
    release_dir: Path,
    *,
    k: int,
    outcome: QuestionOutcome,
) -> tuple[QuestionOutcome, SubmissionItem, tuple[CsvRow, ...]]:
    """Day 23 full-coverage strategy tier 4: `outcome` already failed every
    reasoning tier -- fill the gap with `backstop_answer.build_backstop_item`
    so this question still gets a contract-valid `SubmissionItem` (plan.md
    §2.4 rule 1: a single missing id fails the *entire* ZIP). `stage`/`code`
    are carried over from `outcome` so the report still records *why*
    reasoning failed, distinct from a genuinely `"answered"` result.
    """
    retrieved = retrieve_candidate_table_ids(raw_question.question, service, k=k)
    item, rows = build_backstop_item(raw_question, retrieved, release_dir)
    backstopped_outcome = QuestionOutcome.model_validate(
        {
            "id": outcome.id,
            "question": outcome.question,
            "status": "backstopped",
            "stage": outcome.stage,
            "code": outcome.code,
            "plan_source": outcome.plan_source,
        }
    )
    return backstopped_outcome, item, rows


def export_submission(
    raw_questions: Sequence[RawQuestion],
    service: RetrievalService,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    dataset_fingerprint: str,
    k: int = 10,
    row_fusion: RowFusionService | None = None,
    apply_backstop: bool = True,
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, RowChoiceDecision] | None = None,
) -> tuple[SubmissionExportReport, tuple[SubmissionItem, ...], dict[str, tuple[CsvRow, ...]]]:
    """Run every question through the live pipeline once. Returns the
    coverage report (all questions), the ``SubmissionItem``s to package into
    ``submission.json``, and their CSV rows keyed by ``csv_path``.

    The answering path is single (spec 2026-08-23 §6, N6): every question
    goes through exactly one ``cell_grounding.ground_question`` call --
    the offline LLM decision (or the deterministic rank-1 default) is
    assembled into a plan, compiled position-bound, verified. No second tier
    ever runs to the rescue.

    ``row_fusion=None`` keeps grounding from seeing any ranked row candidate:
    every question then abstains with ``no_row_candidates``, so production
    callers pass a ``RowFusionService`` (assembled by ``submission/cli.py``
    from the row BM25 index, optionally plus the dense branch).

    ``row_decisions`` maps ``question_id -> RowChoiceDecision``
    (``planning.question_plan.load_decisions``): the offline LLM's per-question
    choice of row. A missing entry defaults to the rank-1 candidate.

    ``apply_backstop=True`` (the default, Day 23 full-coverage strategy): any
    question the answering path failed gets a contract-valid but
    not-reasoned item (tier 4, ``backstop_answer.build_backstop_item``), so
    ``items`` always covers every question in ``raw_questions`` -- required by
    plan.md §2.4 rule 1 (a single missing id fails the whole ZIP) and safe
    under the official scoring (Answer/Execution Accuracy macro-average over
    the *full* question set, so a wrong answer already scores the same 0
    credit as a missing one). Pass ``False`` to keep the pre-Day-23 partial-
    coverage behavior for pure measurement runs.
    """
    ordered = tuple(sorted(raw_questions, key=lambda item: item.id))
    outcomes: list[QuestionOutcome] = []
    items: list[SubmissionItem] = []
    csv_rows: dict[str, tuple[CsvRow, ...]] = {}
    stage_counts: Counter[PipelineStage] = Counter()

    for raw_question in ordered:
        outcome, item, rows = _run_one_question(
            raw_question,
            service,
            release_dir,
            execution_settings=execution_settings,
            k=k,
            row_fusion=row_fusion,
            allow_inferred_scope=allow_inferred_scope,
            row_decisions=row_decisions,
        )
        if item is None and apply_backstop:
            outcome, item, rows = _apply_backstop(
                raw_question, service, release_dir, k=k, outcome=outcome
            )
        outcomes.append(outcome)
        if outcome.stage is not None:
            stage_counts[outcome.stage] += 1
        if item is not None:
            assert rows is not None
            items.append(item)
            csv_rows[item.evidence[0].csv_path] = rows

    report = SubmissionExportReport.model_validate(
        {
            "dataset_fingerprint": dataset_fingerprint,
            "question_count": len(ordered),
            "answered_count": sum(1 for o in outcomes if o.status == "answered"),
            "backstopped_count": sum(1 for o in outcomes if o.status == "backstopped"),
            "stage_counts": dict(stage_counts),
            "outcomes": tuple(outcomes),
        }
    )
    return report, tuple(items), csv_rows


_CSV_COLUMNS = (
    "table_id",
    "row_idx",
    "col_idx",
    "company_code",
    "row_label_canonical",
    "row_label_raw",
    "column_label",
    "period",
    "value",
)


def _render_csv_bytes(rows: Sequence[CsvRow]) -> bytes:
    """Ghi lát cắt bảng nguồn theo schema cố định.

    `table_id`/`row_idx`/`col_idx` bắt buộc phải có: `pandas_query.py`
    `_position_clauses` sinh predicate tham chiếu trực tiếp `df1.table_id` và
    `df1.row_idx`, và trước Day 27 hai cột đó bị bỏ khỏi CSV -- khiến 84 câu
    ném `KeyError` khi replay trên chính CSV đóng gói kèm.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(_CSV_COLUMNS))
    for row in rows:
        writer.writerow(
            ["" if row.get(column) is None else row.get(column) for column in _CSV_COLUMNS]
        )
    return buffer.getvalue().encode("utf-8")


# Fixed archive timestamp (plan.md §2.4 rule 8): real wall-clock mtimes would
# make two exports of the same input produce different ZIP bytes/SHA-256.
_FIXED_ZIP_DATE_TIME = (2026, 1, 1, 0, 0, 0)


def write_submission_zip(
    items: Sequence[SubmissionItem],
    csv_rows: Mapping[str, Sequence[CsvRow]],
    output_path: Path,
) -> str:
    """Write the deterministic ZIP (one root `submission.json` + `data/`) and
    return its SHA-256. Entries are sorted so identical input always produces
    identical bytes."""
    ordered_items = tuple(sorted(items, key=lambda item: item.id))
    payload = (
        json.dumps(
            [item.model_dump(mode="json") for item in ordered_items],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        info = zipfile.ZipInfo("submission.json", date_time=_FIXED_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, payload)
        for csv_path in sorted(csv_rows):
            info = zipfile.ZipInfo(csv_path, date_time=_FIXED_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, _render_csv_bytes(csv_rows[csv_path]))

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def _render_report_markdown(report: SubmissionExportReport) -> str:
    lines = [
        "# Day 22 Submission Export Coverage",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Answered (in submission.json): {report.answered_count}",
        "",
        "## Not answered, by stage",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    for stage, count in sorted(report.stage_counts.items()):
        lines.append(f"| {stage} | {count} |")
    lines.extend(("", "## Per-question outcomes", ""))
    for outcome in report.outcomes:
        status = "answered" if outcome.status == "answered" else f"{outcome.stage}: {outcome.code}"
        lines.append(f"- `{outcome.id}` -> {status}")
    return "\n".join(lines) + "\n"


def write_export_report(report: SubmissionExportReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"submission-export-{prefix}.json"
    markdown_path = output_dir / f"submission-export-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_report_markdown(report))
    return json_path, markdown_path
