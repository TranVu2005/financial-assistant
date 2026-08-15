"""Day 21 E2E evaluation: retrieval -> plan -> execution -> verification, for
real (ADR 0010 decision E1).

Unlike `verification/evaluation.py::evaluate_answer_packages_on_gold` (Day
20), which feeds `gold_table_ids` into both plan construction and
`retrieved_table_ids` -- bypassing retrieval entirely (Day 21 plan §1.1
measured this: answered fell 30 -> 9 when a real BM25 ranking replaced gold
tables) -- `run_e2e_pipeline` takes a saved retrieval ranking as its only
source of candidate tables per question, and records every abstain as a
first-class `PipelineQuestionResult`, never a swallowed `continue` (§1.9:
19/70 abstains were invisible in the Day 20 report).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.pipeline.contracts import (
    NORMALIZATION_ISSUE_CODES,
    PipelineQuestionResult,
    PipelineReport,
    PipelineStage,
    ScopePolicy,
    ScopePolicyComparisonReport,
    ScopePolicyResult,
)
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.rule_planner import build_plan
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.verification.builder import build_answer_package
from financial_report_qa.verification.evaluation import build_citation_lookup


def load_rankings(path: Path) -> tuple[dict[str, tuple[str, ...]], str]:
    """Load a saved `retrieval-v2-*.json` report's per-question rankings.

    Returns `(question_id -> predicted_table_ids, sha256_of_file)` so a
    pipeline report can record exactly which ranking snapshot it replayed --
    no index is rebuilt here.
    """
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    rankings = {
        entry["question_id"]: tuple(entry["predicted_table_ids"])
        for entry in document["per_question"]
    }
    return rankings, sha256


def _classify_execution_stage(code: str, *, gold_in_retrieved: bool) -> PipelineStage:
    """ADR 0010 decision E1's per-question rule: `cell_ambiguous` and the
    normalization-family codes are stage-ambiguous by nature (Day 21 plan
    §1.2 measured 22/24 real `cell_ambiguous` failures had gold IN the
    retrieved set) -- classify from `gold_in_retrieved`, not a static table.
    Genuine execution-layer codes (row_limit_exceeded, division_by_zero,
    ...) are unrelated to which tables were retrieved.
    """
    if code == "cell_ambiguous":
        return "retrieval" if not gold_in_retrieved else "planning"
    if code in NORMALIZATION_ISSUE_CODES:
        return "retrieval" if not gold_in_retrieved else "normalization"
    return "execution"


def _run_one_question(
    question: GoldRetrievalQuestion,
    retrieved: tuple[str, ...],
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    abstain_when_scope_unstated: bool = False,
) -> PipelineQuestionResult:
    gold_in_retrieved = set(question.gold_table_ids) <= set(retrieved)
    base: dict[str, object] = {
        "question_id": question.question_id,
        "question": question.question,
        "gold_in_retrieved": gold_in_retrieved,
        "retrieved_table_count": len(retrieved),
    }

    if not retrieved:
        return PipelineQuestionResult.model_validate(
            {
                **base,
                "stage": "retrieval",
                "code": "no_candidate_tables",
                "message": "retrieval returned no candidate tables for this question",
            }
        )

    entities = parse_query_entities(question.question)
    plan_result = build_plan(
        entities, candidate_table_ids=retrieved, known_table_ids=frozenset(retrieved)
    )
    if plan_result.plan is None:
        return PipelineQuestionResult.model_validate(
            {
                **base,
                "stage": "planning",
                "code": "+".join(plan_result.abstain_codes),
                "message": "rule planner abstained: " + ", ".join(plan_result.abstain_codes),
            }
        )
    plan = plan_result.plan

    # Day 21 plan §1.5 policy sweep: the `abstain_when_unstated` scope policy
    # is deliberately NOT the production behavior (ADR 0010 decision B1 chose
    # infer-and-block instead) -- this flag exists only so the tradeoff
    # report in `evaluate_scope_policies` can measure it against the other
    # two policies without a separate code path duplicating everything above.
    if abstain_when_scope_unstated and plan.statement_scope is None:
        return PipelineQuestionResult.model_validate(
            {
                **base,
                "stage": "planning",
                "code": "scope_unstated",
                "message": "question does not state separate/consolidated and "
                "abstain_when_scope_unstated policy is active",
            }
        )

    compiled = compile_plan(plan, release_dir, execution_settings=execution_settings)
    if compiled.status != "answered":
        assert compiled.error_code is not None and compiled.error_message is not None
        stage = _classify_execution_stage(compiled.error_code, gold_in_retrieved=gold_in_retrieved)
        return PipelineQuestionResult.model_validate(
            {**base, "stage": stage, "code": compiled.error_code, "message": compiled.error_message}
        )

    cell_ids = tuple(cell_id for cell in compiled.evidence for cell_id in cell.cell_ids)
    citation_lookup = build_citation_lookup(release_dir, cell_ids)
    package = build_answer_package(
        question_id=question.question_id,
        question=question.question,
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset(retrieved),
        citation_lookup=citation_lookup,
    )
    if package.verification_status == "rejected":
        return PipelineQuestionResult.model_validate(
            {
                **base,
                "stage": "verification",
                "code": "+".join(issue.code for issue in package.verification_issues),
                "message": "; ".join(issue.message for issue in package.verification_issues),
            }
        )

    return PipelineQuestionResult.model_validate(
        {
            **base,
            "stage": None,
            "code": None,
            "message": None,
            "answer": package.answer,
            "scope_inferred": compiled.scope_inferred,
        }
    )


def run_e2e_pipeline(
    questions: Sequence[GoldRetrievalQuestion],
    rankings: Mapping[str, tuple[str, ...]],
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    rankings_source: str,
    rankings_sha256: str,
    answer_gold: Mapping[str, Decimal] | None = None,
    abstain_when_scope_unstated: bool = False,
) -> PipelineReport:
    """Run every question through retrieval(saved) -> plan -> execution ->
    verification once, classify every failure by stage, and score verified
    answers against `answer_gold` when a label exists."""
    answer_gold = answer_gold or {}
    ordered = sorted(questions, key=lambda item: item.question_id)
    results: list[PipelineQuestionResult] = []
    stage_counts: Counter[PipelineStage] = Counter()
    scored = correct = overconfident_wrong = 0

    for question in ordered:
        retrieved = rankings.get(question.question_id, ())
        result = _run_one_question(
            question,
            retrieved,
            release_dir,
            execution_settings=execution_settings,
            abstain_when_scope_unstated=abstain_when_scope_unstated,
        )
        results.append(result)
        if result.stage is not None:
            stage_counts[result.stage] += 1

        gold_answer = answer_gold.get(question.question_id)
        if result.stage is None and gold_answer is not None:
            scored += 1
            if result.answer == gold_answer:
                correct += 1
            else:
                overconfident_wrong += 1

    return PipelineReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "0" * 64,
        rankings_source=rankings_source,
        rankings_sha256=rankings_sha256,
        default_statement_scope=execution_settings.default_statement_scope,
        question_count=len(ordered),
        verified_count=sum(1 for result in results if result.stage is None),
        stage_counts=dict(stage_counts),
        scored_against_gold_count=scored,
        correct_count=correct,
        overconfident_wrong_count=overconfident_wrong,
        results=tuple(results),
    )


def _render_pipeline_markdown(report: PipelineReport) -> str:
    accuracy = report.accuracy_against_gold
    lines = [
        "# Day 21 E2E Pipeline (retrieval in the loop)",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Rankings source: `{report.rankings_source}` (sha256 `{report.rankings_sha256}`)",
        f"- default_statement_scope: `{report.default_statement_scope}`",
        f"- Questions: {report.question_count}",
        f"- Verified: {report.verified_count}",
        f"- Scored against answer-gold: {report.scored_against_gold_count}",
        f"- Correct: {report.correct_count}",
        f"- Overconfident wrong (verified but != gold): {report.overconfident_wrong_count}",
        f"- Accuracy against gold: {accuracy}",
        "",
        "## Failures by stage",
        "",
        "| Stage | Count |",
        "| --- | ---: |",
    ]
    for stage, count in sorted(report.stage_counts.items()):
        lines.append(f"| {stage} | {count} |")
    lines.extend(("", "## Per-question results", ""))
    for result in report.results:
        status = "verified" if result.stage is None else f"{result.stage}: {result.code}"
        lines.append(
            f"- `{result.question_id}` gold_in_retrieved={result.gold_in_retrieved} "
            f"retrieved={result.retrieved_table_count} -> {status}"
        )
    return "\n".join(lines) + "\n"


_SCOPE_POLICY_SETTINGS: dict[ScopePolicy, tuple[str | None, bool]] = {
    # (default_statement_scope, abstain_when_scope_unstated)
    "none": (None, False),
    "default_consolidated": ("consolidated", False),
    "abstain_when_unstated": (None, True),
}


def _evaluate_one_scope_policy(
    ordered: Sequence[GoldRetrievalQuestion],
    rankings: Mapping[str, tuple[str, ...]],
    release_dir: Path,
    *,
    policy: ScopePolicy,
    base_execution_settings: ExecutionSettings,
    answer_gold: Mapping[str, Decimal],
) -> ScopePolicyResult:
    default_scope, abstain_when_unstated = _SCOPE_POLICY_SETTINGS[policy]
    settings = base_execution_settings.model_copy(update={"default_statement_scope": default_scope})
    answered = scored = correct = overconfident_wrong = 0

    for question in ordered:
        retrieved = rankings.get(question.question_id, ())
        if not retrieved:
            continue
        entities = parse_query_entities(question.question)
        plan_result = build_plan(
            entities, candidate_table_ids=retrieved, known_table_ids=frozenset(retrieved)
        )
        if plan_result.plan is None:
            continue
        plan = plan_result.plan
        if abstain_when_unstated and plan.statement_scope is None:
            continue

        compiled = compile_plan(plan, release_dir, execution_settings=settings)
        if compiled.status != "answered":
            continue
        answered += 1

        gold_answer = answer_gold.get(question.question_id)
        if gold_answer is not None:
            scored += 1
            if compiled.answer == gold_answer:
                correct += 1
            else:
                overconfident_wrong += 1

    return ScopePolicyResult(
        policy=policy,
        answered_count=answered,
        scored_against_gold_count=scored,
        correct_count=correct,
        overconfident_wrong_count=overconfident_wrong,
    )


def evaluate_scope_policies(
    questions: Sequence[GoldRetrievalQuestion],
    rankings: Mapping[str, tuple[str, ...]],
    release_dir: Path,
    *,
    base_execution_settings: ExecutionSettings,
    rankings_source: str,
    rankings_sha256: str,
    answer_gold: Mapping[str, Decimal] | None = None,
) -> ScopePolicyComparisonReport:
    """Measure the same question set under all three scope policies (Day 21
    plan §1.5/ADR 0010 decision G1) so the gate report shows the tradeoff
    (coverage vs. accuracy vs. overconfident-wrong count) instead of a single
    accuracy number that hides which policy produced it.

    Deliberately measures raw `compile_plan` coverage, not
    `run_e2e_pipeline`'s post-verification `verified_count` -- see
    `ScopePolicyResult`'s docstring for why.
    """
    ordered = sorted(questions, key=lambda item: item.question_id)
    answer_gold = answer_gold or {}
    all_policies: tuple[ScopePolicy, ...] = (
        "abstain_when_unstated",
        "default_consolidated",
        "none",
    )
    policy_results = tuple(
        _evaluate_one_scope_policy(
            ordered,
            rankings,
            release_dir,
            policy=policy,
            base_execution_settings=base_execution_settings,
            answer_gold=answer_gold,
        )
        for policy in all_policies
    )

    return ScopePolicyComparisonReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "0" * 64,
        rankings_source=rankings_source,
        rankings_sha256=rankings_sha256,
        question_count=len(ordered),
        policies=policy_results,
    )


def _render_scope_policy_markdown(report: ScopePolicyComparisonReport) -> str:
    lines = [
        "# Day 21 Statement-Scope Policy Tradeoff",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Rankings source: `{report.rankings_source}` (sha256 `{report.rankings_sha256}`)",
        f"- Questions: {report.question_count}",
        "",
        "| Policy | Answered | Scored | Correct | Overconfident wrong | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report.policies:
        lines.append(
            f"| {result.policy} | {result.answered_count} | "
            f"{result.scored_against_gold_count} | {result.correct_count} | "
            f"{result.overconfident_wrong_count} | {result.accuracy_against_gold} |"
        )
    return "\n".join(lines) + "\n"


def write_scope_policy_report(
    report: ScopePolicyComparisonReport, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"scope-policy-tradeoff-{prefix}.json"
    markdown_path = output_dir / f"scope-policy-tradeoff-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_scope_policy_markdown(report))
    return json_path, markdown_path


def write_pipeline_report(report: PipelineReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"e2e-pipeline-{prefix}.json"
    markdown_path = output_dir / f"e2e-pipeline-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_pipeline_markdown(report))
    return json_path, markdown_path
