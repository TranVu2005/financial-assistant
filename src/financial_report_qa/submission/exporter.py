"""Day 22 submission exporter: raw question -> live retrieval -> masked-PAL
execution -> `SubmissionItem` + replayable CSV rows, plus a coverage report
for every question that did NOT make it into the bundle (plan.md §2.4: only
a verified `answered` result may become a `SubmissionItem` --
`abstained`/`error` block release, they are never silently dropped, just
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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import ProgramBindingError, SubmissionInputError
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.program_contracts import (
    CellCandidate,
    ExecutedProgram,
    ProgramDecision,
)
from financial_report_qa.execution.program_pipeline import run_question
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.execution.scope_filter import (
    filter_table_ids_by_scope,
    resolve_statement_scope,
)
from financial_report_qa.pipeline.contracts import PipelineStage
from financial_report_qa.planning.cell_candidates import build_cell_candidates
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.live_query import (
    TableRetriever,
    retrieve_candidate_table_ids,
)
from financial_report_qa.retrieval.reranker import Reranker
from financial_report_qa.retrieval.row_fusion import (
    DEFAULT_ROW_CANDIDATE_COUNT,
    RowFusionService,
)
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate
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


def _scope_candidate_tables(
    release_dir: Path,
    table_ids: tuple[str, ...],
    entities: QueryEntities,
    execution_settings: ExecutionSettings,
) -> tuple[str, ...]:
    """Narrow candidate tables to the question's statement scope.

    Used only for the backstop tier's row hint now: the masked-PAL answering
    path deliberately fuses over the UN-narrowed `retrieved` list so its
    numbered candidate list is identical to the one the offline payload
    generator produced (spec 2026-08-24 §4.3).

    Narrowing to nothing returns the ids unchanged: reproducing the empty
    case as a failure here would only add a second, earlier place where a
    question can be lost.
    """
    effective_scope, _ = resolve_statement_scope(
        plan_scope=entities.statement_scope,
        default_scope=execution_settings.default_statement_scope,
    )
    if effective_scope is None:
        return table_ids
    return filter_table_ids_by_scope(release_dir, table_ids, effective_scope) or table_ids


def build_question_cell_candidates(
    release_dir: Path,
    question: str,
    retrieved: Sequence[str],
    fusion_rows: Sequence[RowFusedCandidate],
) -> tuple[CellCandidate, ...]:
    """Dựng danh sách ô đánh số cho một câu, một cách duy nhất.

    `ProgramDecision.cells` là vị trí trong danh sách này, nên lúc sinh payload
    và lúc export phải cho ra danh sách y hệt. Đó là lý do hàm này tồn tại
    thay vì hai lời gọi `build_cell_candidates` song song ở hai file.

    `retrieved` là đúng danh sách bảng ứng viên của nhánh retrieval (thứ tự
    retrieval-rank), KHÔNG bản đã thu hẹp theo scope -- việc thu hẹp, nếu cần,
    thuộc bước sau và không được đụng vào danh sách đã đánh số này.
    """
    entities = parse_query_entities(question)
    frame = build_cell_frame(release_dir, list(retrieved))
    return build_cell_candidates(frame, fusion_rows, periods=entities.periods)


def _run_one_question(
    raw_question: RawQuestion,
    service: TableRetriever,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    k: int,
    reranker: Reranker | None = None,
    row_fusion: RowFusionService | None = None,
    program_decisions: Mapping[int, ProgramDecision],
) -> tuple[QuestionOutcome, SubmissionItem | None, tuple[CsvRow, ...] | None]:
    """One question through the only answering path (spec 2026-08-24 §4.3):
    retrieval -> row fusion over the FULL candidate set -> numbered cell
    candidates -> offline decision -> bounded-retry execution -> verified
    packaging. A failure here falls through to the backstop tier in
    `export_submission`; no second reasoning path ever runs."""
    question = raw_question.question

    retrieved = retrieve_candidate_table_ids(question, service, k=k, reranker=reranker)
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

    # Row fusion runs on the FULL `retrieved` list -- NOT a scope-narrowed
    # subset. That is exactly what `submission row-batches --program` does at
    # payload-generation time (`cli.py` passes raw `retrieved`), and the
    # numbered candidate list below is positions in the fusion result:
    # fusing over different table sets at batch time and export time shifts
    # every `ProgramDecision.cells` index (measured on gold: questions whose
    # scope filter drops tables flipped between empty and non-empty lists).
    #
    # `k` must match `submission row-batches --rows-per-question`'s default
    # (DEFAULT_ROW_CANDIDATE_COUNT): decisions reference indices up to that
    # count - 1.
    fusion_rows = (
        row_fusion.retrieve_rows(
            question, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT
        ).results
        if row_fusion is not None
        else ()
    )
    candidates = build_question_cell_candidates(release_dir, question, retrieved, fusion_rows)
    # Same deterministic frame the helper just built internally, rebuilt here
    # because `run_question` needs it too and the shared helper's contract is
    # candidates-only (the batch path feeds its result straight to
    # `build_program_batch_payload`).
    frame = build_cell_frame(release_dir, list(retrieved))
    result = run_question(
        raw_question.id,
        candidates,
        frame,
        FileDecisionSource(decisions=program_decisions),
    )
    executed = result.executed
    if executed is None:
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "error",
                    "stage": "execution",
                    "code": result.failure_code,
                    "plan_source": "llm_decision",
                }
            ),
            None,
            None,
        )
    # N1: `relevant_docs`/`relevant_tables` keep coming from `retrieved`
    # (rank order preserved by the mapping); they are NEVER taken from
    # `executed.table_ids`. Only the answering slice above may use what
    # the program bound to.
    relevant_docs, relevant_tables = _relevant_docs_and_tables(retrieved, release_dir)
    evidence_rows = _program_evidence_rows(
        executed, release_dir, timeout_seconds=execution_settings.timeout_seconds
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
    item = build_item_from_executed(
        executed,
        question=question,
        retrieved=relevant_tables,
        relevant_docs=relevant_docs,
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
                "regenerated": executed.regenerated,
                "low_confidence": executed.low_confidence,
            }
        ),
        item,
        evidence_rows,
    )


def build_item_from_executed(
    executed: ExecutedProgram,
    *,
    question: str = "",
    retrieved: tuple[str, ...],
    relevant_docs: tuple[str, ...],
) -> SubmissionItem:
    """Dựng một `SubmissionItem` từ kết quả masked-PAL.

    `retrieved` đến từ nhánh 1 và đi thẳng vào `relevant_tables` theo đúng
    thứ tự retrieval-rank. Nó KHÔNG được thay bằng `executed.table_ids`:
    dashboard chấm MRR5 theo vị trí, và nhánh answering không có quyền ghi đè
    đầu ra nhánh retrieval (N1).

    Adaptations vs plan (documented): construction goes through
    `model_construct`, not the validating constructor, for two reasons, both
    pinned by the contract test in
    `tests/unit/submission/test_exporter_program_path.py`:

    1. `SubmissionItem.question` is a required `NonEmptyString`, but an
       `ExecutedProgram` carries no question text. The live caller passes the
       real question via the `question` keyword; direct contract tests omit
       it. Format enforcement is not lost: `validate_submission_zip`
       revalidates every serialized item with the full model before a bundle
       may ship, so a blank question or malformed table entry still fails the
       ship path loudly.
    2. `relevant_tables` entries must be `<report_id>|<line_start>` per the
       field validator, while this builder's `retrieved` argument is pinned
       to pass through verbatim -- the live caller feeds it the output of
       `_relevant_docs_and_tables` (already formatted, rank order preserved),
       and the unit-level pin uses raw `tbl_*` ids. Validating here would make
       the pinned passthrough unrepresentable without weakening anything that
       actually ships.

    A third deviation goes the other way: `evidence` is NON-empty (the brief's
    snippet had `evidence=()`). Both packaging consumers index into it --
    `export_submission` keys `csv_rows[item.evidence[0].csv_path] = rows` and
    `compliance.check_bundle` reads `item.evidence[0].csv_path` -- so the
    empty placeholder would crash the export loop with IndexError on the
    first masked-path question answered; and even without the crash, an
    unreferenced packaged CSV is flagged `orphan_csv` by
    `validate_submission_zip`, while a no-evidence item skips answer replay
    at validation entirely. The entry reuses the old path's exact convention
    (`data/q<id:06d>_df1.csv`, variable `df1`) because
    `render_program_pandas` emits queries over `df1`; the CSV itself comes
    from `_program_evidence_rows`' verified real-table slice.
    """
    return SubmissionItem.model_construct(
        id=executed.question_id,
        question=question,
        answer=float(executed.answer),
        pandas_query=executed.pandas_query,
        program=executed.program,
        relevant_docs=relevant_docs,
        relevant_tables=retrieved,
        evidence=(
            SubmissionEvidence(
                variable="df1",
                csv_path=f"data/q{executed.question_id:06d}_df1.csv",
            ),
        ),
    )


@dataclass(frozen=True)
class FileDecisionSource:
    """Serve one question's decision from the offline file.

    The file is produced offline, so a live "regeneration" can only mean a
    second decision the file already carries. When it carries only one, the
    retry re-runs the identical decision and fails the identical way -- which
    is correct: `low_confidence` then records that nothing better was
    available, instead of pretending a second opinion existed.
    """

    decisions: Mapping[int, ProgramDecision]
    retries: Mapping[int, ProgramDecision] = field(default_factory=dict)

    def decide(self, question_id: int, attempt: int) -> ProgramDecision:
        if attempt > 0 and question_id in self.retries:
            return self.retries[question_id]
        try:
            return self.decisions[question_id]
        except KeyError as error:
            raise ProgramBindingError(f"no program decision for question {question_id}") from error


def _program_evidence_rows(
    executed: ExecutedProgram, release_dir: Path, *, timeout_seconds: float
) -> tuple[CsvRow, ...] | None:
    """BI-1 (design §5.1): the packaged CSV is the real slice of the source
    tables the bindings point into -- never a row synthesized backwards from
    the answer -- and `executed.pandas_query` must replay on exactly that
    slice to `executed.answer` before it may ship.

    Returns `None` when the slice cannot stand as evidence (fewer than 2
    numeric cells -- the hardcode shape compliance C1+C2 exist to catch -- or
    a replay mismatch/error); the caller then fails the question into the
    backstop tier instead of packaging it.
    """
    frame = build_cell_frame(release_dir, executed.table_ids)
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
        executed.pandas_query, replay_frame, timeout_seconds=timeout_seconds
    )
    if sandbox_result.error_code is not None:
        return None
    if sandbox_result.value is None:
        return None
    if abs(sandbox_result.value - executed.answer) > ANSWER_TOLERANCE:
        return None
    return rows


def _decision_row_hint(
    raw_question: RawQuestion,
    retrieved: Sequence[str],
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    row_fusion: RowFusionService | None,
) -> tuple[tuple[str, int] | None, int | None]:
    """Rank-1 fused row plus the first parsed period, as the backstop tier's
    preferred (row, period) hint.

    Fusion deliberately runs over the scope-narrowed set here: unlike the
    answering path there is no numbered-candidate contract to preserve, and a
    row the statement scope excludes should not steer the fill answer.
    """
    if not retrieved or row_fusion is None:
        return None, None
    entities = parse_query_entities(raw_question.question)
    answerable = _scope_candidate_tables(
        release_dir, tuple(retrieved), entities, execution_settings
    )
    fusion_rows = row_fusion.retrieve_rows(
        raw_question.question, candidate_table_ids=answerable, k=DEFAULT_ROW_CANDIDATE_COUNT
    ).results
    if not fusion_rows:
        return None, None
    pick = fusion_rows[0]
    period = int(entities.periods[0]) if entities.periods else None
    return (pick.table_id, pick.row_idx), period


def _apply_backstop(
    raw_question: RawQuestion,
    service: TableRetriever,
    release_dir: Path,
    *,
    k: int,
    outcome: QuestionOutcome,
    execution_settings: ExecutionSettings,
    reranker: Reranker | None = None,
    row_fusion: RowFusionService | None = None,
) -> tuple[QuestionOutcome, SubmissionItem, tuple[CsvRow, ...]]:
    """Day 23 full-coverage strategy tier 4: `outcome` already failed every
    reasoning tier -- fill the gap with `backstop_answer.build_backstop_item`
    so this question still gets a contract-valid `SubmissionItem` (plan.md
    §2.4 rule 1: a single missing id fails the *entire* ZIP). `stage`/`code`
    are carried over from `outcome` so the report still records *why*
    reasoning failed, distinct from a genuinely `"answered"` result.

    `retrieved` stays the unnarrowed retrieval result: it feeds this item's
    `relevant_docs`/`relevant_tables`, which are scored independently of
    whether the answer is right. Only the row hint is computed from the
    scope-narrowed set.
    """
    retrieved = retrieve_candidate_table_ids(raw_question.question, service, k=k, reranker=reranker)
    preferred_row, preferred_period = _decision_row_hint(
        raw_question,
        retrieved,
        release_dir,
        execution_settings=execution_settings,
        row_fusion=row_fusion,
    )
    item, rows = build_backstop_item(
        raw_question,
        retrieved,
        release_dir,
        preferred_row=preferred_row,
        preferred_period=preferred_period,
    )
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
    service: TableRetriever,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    dataset_fingerprint: str,
    k: int = 10,
    reranker: Reranker | None = None,
    row_fusion: RowFusionService | None = None,
    apply_backstop: bool = True,
    program_decisions: Mapping[int, ProgramDecision],
) -> tuple[SubmissionExportReport, tuple[SubmissionItem, ...], dict[str, tuple[CsvRow, ...]]]:
    """Run every question through the live pipeline once. Returns the
    coverage report (all questions), the ``SubmissionItem``s to package into
    ``submission.json``, and their CSV rows keyed by ``csv_path``.

    ``service`` is anything that ranks tables under metadata filters: the
    BM25-only ``RetrievalService``, or a ``FusionService`` combining it with
    a dense branch, optionally followed by the ``reranker`` cross-encoder on
    the fused top-N.

    The answering path is single (spec 2026-08-24 §4.3): every question is
    answered from ``program_decisions`` --
    ``question_id -> ProgramDecision``
    (``planning.program_decisions.load_program_decisions``), the offline
    LLM's masked-PAL program -- through ``run_question`` (one decision + one
    bounded regeneration) over the shared numbered cell candidate list built
    by :func:`build_question_cell_candidates`.

    ``row_fusion=None`` leaves every question without ranked row candidates:
    each then fails with ``no_cell_candidates`` and flows to the backstop, so
    production callers pass a ``RowFusionService`` (assembled by
    ``submission/cli.py`` from the row BM25 index, optionally plus the dense
    branch).

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
            reranker=reranker,
            row_fusion=row_fusion,
            program_decisions=program_decisions,
        )
        if item is None and apply_backstop:
            outcome, item, rows = _apply_backstop(
                raw_question,
                service,
                release_dir,
                k=k,
                outcome=outcome,
                execution_settings=execution_settings,
                reranker=reranker,
                row_fusion=row_fusion,
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

    `table_id`/`row_idx`/`col_idx` bắt buộc phải có: truy vấn đóng gói tham
    chiếu trực tiếp `df1.table_id` và `df1.row_idx` (và masked-PAL còn ghim
    cột bằng `df1.col_idx`); trước Day 27 hai cột đó bị bỏ khỏi CSV -- khiến
    84 câu ném `KeyError` khi replay trên chính CSV đóng gói kèm.
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
