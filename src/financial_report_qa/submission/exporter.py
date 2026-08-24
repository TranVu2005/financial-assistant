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
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import ProgramBindingError, SubmissionInputError
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.contracts import CompiledQuery
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
from financial_report_qa.planning.cell_grounding import ground_question
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import RowChoiceDecision
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


def _scope_candidate_tables(
    release_dir: Path,
    table_ids: tuple[str, ...],
    entities: QueryEntities,
    execution_settings: ExecutionSettings,
) -> tuple[str, ...]:
    """Narrow candidate tables to the question's statement scope, before row
    fusion ranks rows out of them.

    `compile_plan` applies exactly this filter, but it runs *after* row fusion
    has ranked rows and after the offline decision has indexed into that
    ranking. So a "cong ty me" question could have its row picked out of a
    consolidated table, and the compiler would then drop that table and report
    `metric_not_found` -- the row existed, held the right value for the right
    company, and was discarded for its document's scope. Measured on the real
    1012-question export: 373 of 554 `metric_not_found` outcomes were exactly
    this, the single largest remaining failure mode.

    Filtering here changes no rule, only when the existing rule applies. The
    filter in `compile_plan` stays where it is and becomes a no-op backstop.

    Narrowing to nothing returns the ids unchanged: `compile_plan` reports that
    case as `candidate_table_ids_scope_empty`, and reproducing the decision
    here would only add a second, earlier place where a question can be lost.
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
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, RowChoiceDecision] | None = None,
    program_decisions: Mapping[int, ProgramDecision] | None = None,
) -> tuple[QuestionOutcome, SubmissionItem | None, tuple[CsvRow, ...] | None]:
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

    # Row fusion feeds the answering path its ranked row candidates. `k` must
    # match `submission row-batches --rows-per-question`'s default
    # (DEFAULT_ROW_CANDIDATE_COUNT): a decision file produced by `row-batches`
    # may reference indices up to that count - 1, and if this candidate list
    # were shorter, those indices would be out of range and silently fall back
    # to rank 1.
    #
    # It is deliberately a SEPARATE name from `retrieved`. Retrieval is scored
    # on its own at 50% weight with a recall-favouring F2, and `relevant_docs`/
    # `relevant_tables` below are built from `retrieved` -- narrowing that list
    # would hand back retrieval score to buy answer score. Only the answering
    # path sees the narrowed set.
    entities = parse_query_entities(question)
    answerable = _scope_candidate_tables(release_dir, retrieved, entities, execution_settings)

    # Masked-PAL answering branch (spec 2026-08-24 §4.3), behind
    # --program-decisions. The numbered candidate list comes from exactly ONE
    # place -- `build_question_cell_candidates` -- which is also what
    # `submission row-batches --program` calls at payload-generation time:
    # `ProgramDecision.cells` are positions in this list, and a list that
    # differs between payload generation and export shifts every index.
    #
    # Row fusion therefore runs on the FULL `retrieved` list here, NOT on the
    # scope-narrowed `answerable` -- that is what the batch generator does
    # (`cli.py` row-batches passes raw `retrieved`). Fusing over the narrowed
    # set at export time while payloads were generated over the full set is
    # exactly the index-shifting divergence this helper exists to prevent:
    # measured on gold, questions whose scope filter drops tables flipped
    # between empty and non-empty candidate lists across the two paths.
    if program_decisions is not None:
        fusion_rows = (
            row_fusion.retrieve_rows(
                question, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT
            ).results
            if row_fusion is not None
            else ()
        )
        candidates = build_question_cell_candidates(release_dir, question, retrieved, fusion_rows)
        # Same deterministic frame the helper just built internally, rebuilt
        # here because `run_question` needs it as well and the shared
        # helper's contract is candidates-only (the batch path feeds its
        # result straight to `build_program_batch_payload`).
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

    # Nhánh answering duy nhất (spec 2026-08-23 §6, nguyên tắc N6): câu hỏi đi
    # qua đúng một đường `ground_question` -- quyết định offline của LLM (hoặc
    # mặc định hạng 1 khi không có) -> plan -> compile. Không tầng thứ hai chạy
    # ra cứu; hỏng ở đâu thì hỏng rõ ở đó với đúng một mã lỗi.
    #
    # Scope narrowing runs before fusion on purpose (see
    # `_scope_candidate_tables`): fusion must not spend the decision's single
    # pick on a row the compiler is going to drop for its document's scope.
    fusion_rows = (
        row_fusion.retrieve_rows(
            question, candidate_table_ids=answerable, k=DEFAULT_ROW_CANDIDATE_COUNT
        ).results
        if row_fusion is not None
        else ()
    )
    grounding = ground_question(
        entities=entities,
        decision=(row_decisions or {}).get(raw_question.id),
        fusion_rows=fusion_rows,
        candidate_table_ids=answerable,
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
    """BI-1 (design §5.1) for the masked-PAL branch: the packaged CSV is the
    real slice of the source tables the bindings point into -- never a row
    synthesized backwards from the answer -- and `executed.pandas_query` must
    replay on exactly that slice to `executed.answer` before it may ship.

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
    row_decisions: Mapping[int, RowChoiceDecision] | None,
) -> tuple[tuple[str, int] | None, int | None]:
    """`((table_id, row_idx), period)` that the offline decision points at.

    The backstop tier answers 823/1012 questions, and until now it ignored the
    decision entirely -- it took whichever row `_uniquely_addressable_row`
    found first in the first candidate table. Measured over those 823: 404 have
    a cell at the decision's row *at the exact period asked*, 304 more have one
    at another period. Recovering that hint is worth more than every gate fix
    in this pipeline combined.
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
    decision = (row_decisions or {}).get(raw_question.id)
    index = decision.chosen[0] if decision is not None and decision.chosen else 0
    if not 0 <= index < len(fusion_rows):
        index = 0
    pick = fusion_rows[index]
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
    row_decisions: Mapping[int, RowChoiceDecision] | None = None,
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
        row_decisions=row_decisions,
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
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, RowChoiceDecision] | None = None,
    program_decisions: Mapping[int, ProgramDecision] | None = None,
) -> tuple[SubmissionExportReport, tuple[SubmissionItem, ...], dict[str, tuple[CsvRow, ...]]]:
    """Run every question through the live pipeline once. Returns the
    coverage report (all questions), the ``SubmissionItem``s to package into
    ``submission.json``, and their CSV rows keyed by ``csv_path``.

    ``service`` is anything that ranks tables under metadata filters: the
    BM25-only ``RetrievalService`` (the historical default), or a
    ``FusionService`` combining it with a dense branch, optionally followed
    by the ``reranker`` cross-encoder on the fused top-N.

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

    ``program_decisions`` maps ``question_id -> ProgramDecision``
    (``planning.program_decisions.load_program_decisions``, spec 2026-08-24
    §4.3): the offline LLM's masked-PAL program per question. When not
    ``None`` it REPLACES the compiler answering path above -- each question
    runs through ``run_question`` (one decision + one bounded regeneration)
    instead of ``ground_question`` -- while retrieval, scoring outputs and
    backstop coverage stay exactly as they are. ``None`` (the default, and
    what the CLI passes without ``--program-decisions``) keeps the historical
    path byte-for-byte.

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
            allow_inferred_scope=allow_inferred_scope,
            row_decisions=row_decisions,
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
                row_decisions=row_decisions,
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
