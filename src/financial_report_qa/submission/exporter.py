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
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import SubmissionInputError
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.pipeline.contracts import PipelineStage
from financial_report_qa.planning import llm_planner
from financial_report_qa.planning.cell_grounding import compile_grounded, ground_with_recovery
from financial_report_qa.planning.column_refinement import plan_with_column
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.evidence_rendering import (
    evidence_row_labels,
    evidence_table_context,
    plan_grounding_score,
    row_label_confidence,
)
from financial_report_qa.planning.llm_cell_grounding import choose_column_label
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.llm_evidence_planner import plan_with_evidence
from financial_report_qa.planning.plan_contracts import _PERIOD_PATTERN, map_requested_unit
from financial_report_qa.planning.plan_router import route_plan
from financial_report_qa.planning.raw_metric_grounding import (
    candidate_column_labels,
    candidate_row_labels,
    plan_with_raw_grounding_fallback,
)
from financial_report_qa.planning.table_context_rendering import render_table_context
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


def _bare_year_periods(periods: Sequence[str]) -> tuple[int, ...]:
    """plan.md §12: the periods `enumerate_candidate_facts` filters on.

    `entities.periods` (entity_parser.py) is not always a bare 4-digit year
    -- a question naming an explicit date ("ngày 31/12/2015") resolves to an
    ISO date string ("2015-12-31"), and a quarter can resolve to its own
    non-year token. `rule_planner.py` already guards this: it abstains with
    `period_grammar_unsupported` unless every period matches
    `_PERIOD_PATTERN`. This tier has to apply the same filter -- silently
    dropping a non-bare-year entry, rather than crashing the whole export on
    one question's phrasing, since a live full-export run did exactly that
    (`int('2015-12-31')` raising `ValueError` out of `_run_one_question`).
    """
    return tuple(int(period) for period in periods if _PERIOD_PATTERN.match(period))


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
    if sandbox_result.error_code is not None or sandbox_result.value != compiled.answer:
        return None
    return rows


def _run_one_question(
    raw_question: RawQuestion,
    service: RetrievalService,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    k: int,
    llm_client: ChatCompletionClient | None,
    row_fusion: RowFusionService | None = None,
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, int] | None = None,
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

    # Evidence-aware planner: run row fusion to get ranked rows for downstream
    # tiers. When row_fusion is None (backward compatible), fusion_rows stays
    # empty and all tiers fall back to their existing unranked behavior.
    # `k` must match `submission row-batches --rows-per-question`'s default
    # (DEFAULT_ROW_CANDIDATE_COUNT): an LLM decision file produced by
    # `row-batches` may reference `chosen_index` values up to that count - 1,
    # and if this candidate list were shorter, those indices would look
    # "out of range" here and silently fall back to a worse rank-1 choice.
    fusion_rows = (
        row_fusion.retrieve_rows(
            question, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT
        ).results
        if row_fusion is not None
        else ()
    )

    entities = parse_query_entities(question)
    row_labels = (
        evidence_row_labels(fusion_rows)
        if fusion_rows
        else candidate_row_labels(release_dir, retrieved)
    )
    # Day 23 plan Step 1: try the rule planner, with one grounded retry when
    # it abstains purely on `metric_unknown` (raw_metric_grounding.py). Only
    # once *that* combined attempt still has no plan does the Day 22
    # coverage-improvement follow-up (`route_plan`, ADR 0006 A1) get to try
    # the LLM planner -- never the reverse, so the Day 16
    # false-plan-rate-0.0 guarantee is unaffected. `llm_client=None`
    # reproduces the exact pre-fallback behavior (no network call attempted).
    plan_result, grounded = plan_with_raw_grounding_fallback(
        entities,
        candidate_table_ids=retrieved,
        known_table_ids=frozenset(retrieved),
        release_dir=release_dir,
    )
    plan_source: Literal[
        "rule",
        "rule_raw_grounded",
        "llm",
        "llm_grounded",
        "llm_row_choice",
        "row_choice_fallback_rank1",
        "llm_cell_grounded",
        "llm_cell_grounded_recovered",
        "llm_cell_grounded_context_expanded",
        "llm_column_refined",
        "llm_evidence_planner",
    ]
    compiled: CompiledQuery | None = None
    low_confidence = False

    if plan_result.plan is not None:
        plan_source = "rule_raw_grounded" if grounded else "rule"
        # plan.md §9/§14: bind the plan's row selector to the position row
        # retrieval ranked first, and extract with `df.loc[row_index, column]`.
        # This has to happen on the primary path, not only inside the recovery
        # ladder below: a plan that picked the wrong row still *compiles*, so
        # recovery never sees it -- which is exactly how 71 of 88 wrong dev
        # benchmark answers were produced.
        primary_plan, compiled = compile_grounded(
            plan_result.plan, fusion_rows, release_dir, execution_settings
        )
        plan_result = plan_result.model_copy(update={"plan": primary_plan})
        # Column refinement on deterministic rule plan if ambiguous
        if compiled.error_code == "cell_ambiguous" and llm_client is not None:
            refined = plan_with_column(
                primary_plan,
                lambda row_label, columns: choose_column_label(
                    question, row_label, columns, client=llm_client
                ),
                columns_for=lambda row_label: candidate_column_labels(
                    release_dir, primary_plan.candidate_table_ids, row_label
                ),
            )
            if refined is not None:
                retried_plan, retried = compile_grounded(
                    refined, fusion_rows, release_dir, execution_settings
                )
                if retried.status == "answered":
                    plan_result = plan_result.model_copy(update={"plan": retried_plan})
                    compiled = retried
                    plan_source = "llm_column_refined"
    elif llm_client is not None:
        # plan.md §12 Evidence-Aware Planner, tried before the typed-plan
        # router. Given the facts row retrieval already grounded, the model
        # only has to name `{operation, operands}` -- it is never asked to
        # invent a table, a row locator, a column locator or a metric name,
        # which is the reasoning chain §12 blames for the 231
        # `llm_plan_invalid` questions. Purely additive: every decline path
        # (no rows, no facts, a model that will not choose, an operand set
        # that fails `build_plan_from_facts`) returns None and leaves the
        # existing router chain below exactly as it was.
        evidence_plan = (
            plan_with_evidence(
                question,
                fusion_rows,
                release_dir,
                client=llm_client,
                company_code=entities.company_codes[0] if entities.company_codes else None,
                periods=_bare_year_periods(entities.periods),
                expected_unit=map_requested_unit(entities.requested_unit),
            )
            if fusion_rows
            else None
        )
        plan_source = "llm_evidence_planner"
        if evidence_plan is not None:
            evidence_compiled = compile_plan(
                evidence_plan, release_dir, execution_settings=execution_settings
            )
            if evidence_compiled.status == "answered":
                plan_result = plan_result.model_copy(update={"plan": evidence_plan})
                compiled = evidence_compiled

        if compiled is None:
            # LLM Planner (Attempt 0 LLM)
            routed = route_plan(
                entities,
                client=llm_client,
                candidate_table_ids=retrieved,
                known_table_ids=frozenset(retrieved),
            )
            plan_result, plan_source = routed.result, routed.source

            if plan_result.plan is None:
                table_context = (
                    evidence_table_context(fusion_rows)
                    if fusion_rows
                    else render_table_context(release_dir, retrieved)
                )
                grounded_result = llm_planner.build_plan_grounded(
                    question,
                    client=llm_client,
                    candidate_table_ids=retrieved,
                    known_table_ids=frozenset(retrieved),
                    table_context=table_context,
                )
                if grounded_result.plan is not None:
                    plan_result, plan_source = grounded_result, "llm_grounded"

            if plan_result.plan is not None:
                routed_plan, compiled = compile_grounded(
                    plan_result.plan, fusion_rows, release_dir, execution_settings
                )
                plan_result = plan_result.model_copy(update={"plan": routed_plan})
                if compiled.error_code == "cell_ambiguous":
                    refined = plan_with_column(
                        routed_plan,
                        lambda row_label, columns: choose_column_label(
                            question, row_label, columns, client=llm_client
                        ),
                        columns_for=lambda row_label: candidate_column_labels(
                            release_dir, routed_plan.candidate_table_ids, row_label
                        ),
                    )
                    if refined is not None:
                        retried_plan, retried = compile_grounded(
                            refined, fusion_rows, release_dir, execution_settings
                        )
                        if retried.status == "answered":
                            plan_result = plan_result.model_copy(update={"plan": retried_plan})
                            compiled = retried
                            plan_source = "llm_column_refined"
    else:
        plan_source = "rule"
        compiled = None

    # Decoupled Cell Grounding & Grounding Recovery (Attempt 1-2). Runs
    # whenever the plan produced above -- by the rule planner or the LLM
    # planner, or no plan at all -- did not compile to an answer, so a
    # rule-planner failure (e.g. `cell_ambiguous`) gets the same recovery
    # ladder as an LLM-planner failure instead of skipping it.
    needs_grounding_recovery = (
        plan_result.plan is None or compiled is None or compiled.status != "answered"
    )
    if needs_grounding_recovery and (llm_client is not None or row_decisions is not None):
        grounding_res = ground_with_recovery(
            question=question,
            entities=entities,
            retrieved=retrieved,
            row_labels=row_labels,
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=execution_settings,
            llm_client=llm_client,
            row_decisions=row_decisions,
            question_id=raw_question.id,
        )
        if grounding_res.status == "accepted":
            assert grounding_res.plan is not None
            assert grounding_res.compiled is not None

            # Log recovery success if it was recovered
            if grounding_res.plan_source == "llm_cell_grounded_recovered":
                first_tried = (
                    plan_result.plan.metric.raw_text
                    if plan_result.plan and plan_result.plan.metric
                    else None
                )
                previous_confidence = row_label_confidence(first_tried, fusion_rows) or 0.0
                new_confidence = (
                    row_label_confidence(grounding_res.plan.metric.raw_text, fusion_rows) or 0.0
                )
                log_entry = {
                    "question_id": raw_question.id,
                    "attempt": grounding_res.recovery_attempts,
                    "strategy": "candidate_switching",
                    "previous_confidence": round(previous_confidence, 4),
                    "new_confidence": round(new_confidence, 4),
                    "selected_row": grounding_res.plan.metric.raw_text,
                    "selected_column": (
                        grounding_res.plan.metric.column_text if grounding_res.plan.metric else None
                    ),
                    "status": "accepted",
                }
                print(
                    f"Grounding Recovery Success: {json.dumps(log_entry, ensure_ascii=False)}",
                    file=sys.stderr,
                )

            plan_result = plan_result.model_copy(update={"plan": grounding_res.plan})
            compiled = grounding_res.compiled
            plan_source = grounding_res.plan_source
            low_confidence = grounding_res.low_confidence
        elif plan_result.plan is not None:
            plan_result = plan_result.model_copy(
                update={"plan": None, "abstain_codes": (grounding_res.error_code,)}
            )

    if compiled is None or compiled.status != "answered":
        if compiled is None:
            abstain_code = (
                "+".join(plan_result.abstain_codes)
                if plan_result.abstain_codes
                else "planning_failed"
            )
            return (
                QuestionOutcome.model_validate(
                    {
                        "id": raw_question.id,
                        "question": question,
                        "status": "abstained",
                        "stage": "planning",
                        "code": abstain_code,
                        "plan_source": plan_source,
                    }
                ),
                None,
                None,
            )
        assert compiled.error_code is not None
        # Day 22 plan §1 §3: no gold exists for this question set, so the
        # retrieval/planning/normalization/execution stage split
        # `pipeline/evaluation.py` computes from `gold_in_retrieved` cannot be
        # replicated here -- every `compile_plan` error is reported under
        # "execution", a documented simplification, not a hidden one.
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "error",
                    "stage": "execution",
                    "code": compiled.error_code,
                    "plan_source": plan_source,
                }
            ),
            None,
            None,
        )

    plan = plan_result.plan
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
                    "plan_source": plan_source,
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
                    "plan_source": plan_source,
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
                "plan_source": plan_source,
                "grounding_score": plan_grounding_score(plan, fusion_rows),
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
    llm_client: ChatCompletionClient | None = None,
    row_fusion: RowFusionService | None = None,
    apply_backstop: bool = True,
    allow_inferred_scope: bool = False,
    row_decisions: Mapping[int, int] | None = None,
) -> tuple[SubmissionExportReport, tuple[SubmissionItem, ...], dict[str, tuple[CsvRow, ...]]]:
    """Run every question through the live pipeline once. Returns the
    coverage report (all questions), the ``SubmissionItem``s to package into
    ``submission.json``, and their CSV rows keyed by ``csv_path``.

    ``llm_client=None`` (the default) keeps the pre-Day-22-coverage-follow-up
    behavior: only the rule planner ever runs. Passing a client routes every
    rule-planner abstain through ``plan_router.route_plan``'s LLM fallback
    (ADR 0006 decision A1), then a grounded LLM retry (Day 23 tier 3) -- the
    rule planner still always runs first and is never overridden once it
    succeeds.

    ``row_fusion=None`` (the default) keeps the pre-evidence-aware behavior:
    LLM grounding tiers use unranked row labels and full table context.
    Passing a ``RowFusionService`` enables ranked, evidence-aware label lists
    and focused context for the LLM.

    ``row_decisions=None`` (the default) keeps grounding recovery's Attempt 0
    LLM row decision unavailable unless a live ``llm_client`` is also given.
    Passing a ``{question_id: chosen_index}`` mapping (`row_choice_decision.
    load_decisions`) makes that decision come from an *offline* file instead
    -- grounding recovery then runs even when ``llm_client`` is ``None``,
    which is why the gate below checks either, not just the client.

    ``apply_backstop=True`` (the default, Day 23 full-coverage strategy): any
    question every reasoning tier still failed gets a contract-valid but
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
            llm_client=llm_client,
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
