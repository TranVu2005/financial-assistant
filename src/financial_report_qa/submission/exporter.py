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
from pathlib import Path, PurePosixPath

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import SubmissionInputError
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.contracts import CompiledQuery, ReplayRow
from financial_report_qa.pipeline.contracts import PipelineStage
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.plan_router import route_plan
from financial_report_qa.planning.rule_planner import build_plan
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.service import RetrievalService
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


def _synthetic_question_id(question_id: int) -> str:
    """`AnswerPackage.question_id` requires `^retq_[0-9a-f]{64}$` (Day 8
    contract) -- the official question file's integer `id` never matches
    that pattern. This mapping is used only internally to build the
    `AnswerPackage`/citation chain; it never appears in `submission.json`,
    which uses the real integer `id` (plan.md §2.4)."""
    return "retq_" + hashlib.sha256(f"submission:{question_id}".encode()).hexdigest()


def _replay_rows_to_csv_rows(replay_rows: tuple[ReplayRow, ...]) -> tuple[CsvRow, ...]:
    return tuple(
        {
            "company_code": row.company_code,
            "row_label_canonical": row.row_label_canonical,
            "row_label_raw": row.row_label_raw,
            "period": row.period,
            "value": row.value,
        }
        for row in replay_rows
    )


def _relevant_docs_and_tables(
    compiled: CompiledQuery, citation_lookup: Mapping[str, Mapping[str, object]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Day 22 plan §2 decision C: uses each evidence CELL's own resolved
    source line (not a table-level line_start) -- see the plan doc for why."""
    docs: dict[str, None] = {}
    tables: dict[str, None] = {}
    for cell in compiled.evidence:
        for cell_id in cell.cell_ids:
            provenance = citation_lookup[cell_id]
            relative_path = str(provenance["doc_relative_path"])
            report_id = PurePosixPath(relative_path).name
            if report_id.endswith(".txt"):
                report_id = report_id[: -len(".txt")]
            docs.setdefault(report_id, None)
            tables.setdefault(f"{report_id}|{provenance['source_line_start']}", None)
    return tuple(docs), tuple(tables)


def _run_one_question(
    raw_question: RawQuestion,
    service: RetrievalService,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    k: int,
    llm_client: ChatCompletionClient | None,
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

    entities = parse_query_entities(question)
    # Day 22 coverage-improvement follow-up: `route_plan` (ADR 0006 A1) tries
    # the rule planner first and only calls the LLM planner when it abstains
    # -- never the reverse, so the Day 16 false-plan-rate-0.0 guarantee is
    # unaffected. `llm_client=None` reproduces the exact pre-fallback
    # behavior (no network call attempted).
    if llm_client is not None:
        routed = route_plan(
            entities,
            client=llm_client,
            candidate_table_ids=retrieved,
            known_table_ids=frozenset(retrieved),
        )
        plan_result, plan_source = routed.result, routed.source
    else:
        plan_result = build_plan(
            entities, candidate_table_ids=retrieved, known_table_ids=frozenset(retrieved)
        )
        plan_source = "rule"
    if plan_result.plan is None:
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "abstained",
                    "stage": "planning",
                    "code": "+".join(plan_result.abstain_codes),
                    "plan_source": plan_source,
                }
            ),
            None,
            None,
        )
    plan = plan_result.plan

    compiled = compile_plan(plan, release_dir, execution_settings=execution_settings)
    if compiled.status != "answered":
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

    cell_ids = tuple(cell_id for cell in compiled.evidence for cell_id in cell.cell_ids)
    citation_lookup = build_citation_lookup(release_dir, cell_ids)
    package = build_answer_package(
        question_id=_synthetic_question_id(raw_question.id),
        question=question,
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset(retrieved),
        citation_lookup=citation_lookup,
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

    relevant_docs, relevant_tables = _relevant_docs_and_tables(compiled, citation_lookup)
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
            }
        ),
        item,
        _replay_rows_to_csv_rows(compiled.replay_rows),
    )


def export_submission(
    raw_questions: Sequence[RawQuestion],
    service: RetrievalService,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    dataset_fingerprint: str,
    k: int = 10,
    llm_client: ChatCompletionClient | None = None,
) -> tuple[SubmissionExportReport, tuple[SubmissionItem, ...], dict[str, tuple[CsvRow, ...]]]:
    """Run every question through the live pipeline once. Returns the
    coverage report (all questions), the `SubmissionItem`s that qualify for
    `submission.json` (answered + verified only), and their CSV rows keyed by
    `csv_path`.

    `llm_client=None` (the default) keeps the pre-Day-22-coverage-follow-up
    behavior: only the rule planner ever runs. Passing a client routes every
    rule-planner abstain through `plan_router.route_plan`'s LLM fallback
    (ADR 0006 decision A1) -- the rule planner still always runs first and is
    never overridden once it succeeds.
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
            "answered_count": len(items),
            "stage_counts": dict(stage_counts),
            "outcomes": tuple(outcomes),
        }
    )
    return report, tuple(items), csv_rows


def _render_csv_bytes(rows: Sequence[CsvRow]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["company_code", "row_label_canonical", "row_label_raw", "period", "value"])
    for row in rows:
        canonical = row["row_label_canonical"]
        raw = row["row_label_raw"]
        writer.writerow(
            [
                row["company_code"],
                "" if canonical is None else canonical,
                "" if raw is None else raw,
                row["period"],
                row["value"],
            ]
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
