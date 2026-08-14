"""Day 17 LLM-planner and router evaluation harness (§17.8, ADR 0006 decision C1).

Two independent measurements, never mixed:

- `evaluate_llm_plan_cases`: the LLM planner alone (`llm_planner.build_plan`,
  bypassing the rule planner) on the plan cases with an `expected_operation`
  -> `operation_accuracy`, `invalid_json_rate`, `repair_success_rate`.
- `evaluate_router_abstain_cases`: the full A1 router (`plan_router.route_plan`)
  on the plan cases with an `expected_abstain_code` -> `false_plan_rate` must
  stay 0.0, the same hard KPI Day 16 already enforces for the rule planner
  alone (`plan_evaluation.py`).

`ReplayCacheClient` makes both runnable with no LLM server (Day 17 plan
§1.8): it looks up `sha256(model identity + system prompt + user prompt)` in
an in-memory cache (loaded from a JSONL file via `load_replay_cache`) instead
of calling out. A cache miss with no `underlying` client configured raises
`LLMUnavailableError` — honest under-coverage from a partial cache, never a
crash. Passing an `underlying` client (a real `LLMClient`) falls through on a
miss and appends the new response to `record_path`, growing the cache for
next time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from financial_report_qa.core.errors import LLMUnavailableError
from financial_report_qa.planning import llm_planner
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.plan_cases import PlanCase
from financial_report_qa.planning.plan_router import route_plan
from financial_report_qa.retrieval.contracts import _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import sha256_bytes, write_text_atomic

_DUMMY_TABLE_ID = "tbl_" + "0" * 64
_DUMMY_KNOWN_TABLES = frozenset({_DUMMY_TABLE_ID})
_CACHE_KEY_SEPARATOR = "\x1f"  # ASCII unit separator: never appears in prompt text


def cache_key(model_identity: str, system_prompt: str, user_prompt: str) -> str:
    """One fingerprint identifying a (model, prompt) pair for replay caching."""
    payload = _CACHE_KEY_SEPARATOR.join((model_identity, system_prompt, user_prompt))
    return sha256_bytes(payload.encode("utf-8"))


def load_replay_cache(path: Path) -> dict[str, str]:
    """Load a previously recorded replay-cache JSONL file; missing is empty."""
    if not path.is_file():
        return {}
    cache: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["key"]] = row["content"]
    return cache


def _append_cache_entry(path: Path, key: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"key": key, "content": content}, ensure_ascii=False) + "\n")


class ReplayCacheClient:
    """A `ChatCompletionClient` backed by a replay cache, with optional live fallback."""

    def __init__(
        self,
        *,
        cache: dict[str, str],
        model_identity: str,
        underlying: ChatCompletionClient | None = None,
        record_path: Path | None = None,
    ) -> None:
        self._cache = cache
        self._model_identity = model_identity
        self._underlying = underlying
        self._record_path = record_path

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, object]
    ) -> str:
        key = cache_key(self._model_identity, system_prompt, user_prompt)
        if key in self._cache:
            return self._cache[key]
        if self._underlying is None:
            raise LLMUnavailableError(f"no cached response for key {key} and no live client set")
        content = self._underlying.complete_json(
            system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema
        )
        self._cache[key] = content
        if self._record_path is not None:
            _append_cache_entry(self._record_path, key, content)
        return content


# --- LLM planner alone ----------------------------------------------------

LLMPlanCaseOutcome = Literal["wrong_operation", "invalid_json", "plan_invalid", "llm_unavailable"]


class LLMPlanCaseFailure(_FrozenModel):
    case_id: str
    question: str
    expected_operation: str
    actual_operation: str | None
    outcome: LLMPlanCaseOutcome


class LLMEvaluationReport(_FrozenModel):
    """Scored against the same `plan_cases.py` answer key Day 16 uses."""

    case_set_sha256: str
    case_count: int = Field(ge=0)
    operation_accuracy: float = Field(ge=0, le=1)
    invalid_json_rate: float = Field(ge=0, le=1)
    repair_success_rate: float = Field(ge=0, le=1)
    failures: tuple[LLMPlanCaseFailure, ...]


def evaluate_llm_plan_cases(
    cases: tuple[PlanCase, ...], *, client: ChatCompletionClient, case_set_sha256: str
) -> LLMEvaluationReport:
    """Score the LLM planner alone (no rule planner) on `expected_operation` cases."""
    operable = sorted(
        (case for case in cases if case.expected_operation is not None),
        key=lambda case: case.case_id,
    )
    correct = invalid_json = repaired_total = repaired_correct = 0
    failures: list[LLMPlanCaseFailure] = []

    for case in operable:
        result = llm_planner.build_plan(
            case.question,
            client=client,
            candidate_table_ids=(_DUMMY_TABLE_ID,),
            known_table_ids=_DUMMY_KNOWN_TABLES,
        )
        actual_operation = result.plan.operation if result.plan is not None else None
        is_correct = actual_operation == case.expected_operation

        if is_correct:
            correct += 1
        if result.repaired:
            repaired_total += 1
            if is_correct:
                repaired_correct += 1

        if not is_correct:
            outcome: LLMPlanCaseOutcome
            if "llm_invalid_json" in result.abstain_codes:
                outcome = "invalid_json"
                invalid_json += 1
            elif "llm_unavailable" in result.abstain_codes:
                outcome = "llm_unavailable"
            elif result.plan is None:
                outcome = "plan_invalid"
            else:
                outcome = "wrong_operation"
            failures.append(
                LLMPlanCaseFailure(
                    case_id=case.case_id,
                    question=case.question,
                    expected_operation=case.expected_operation,  # type: ignore[arg-type]
                    actual_operation=actual_operation,
                    outcome=outcome,
                )
            )

    total = len(operable)
    return LLMEvaluationReport(
        case_set_sha256=case_set_sha256,
        case_count=total,
        operation_accuracy=(correct / total) if total else 1.0,
        invalid_json_rate=(invalid_json / total) if total else 0.0,
        repair_success_rate=(repaired_correct / repaired_total) if repaired_total else 0.0,
        failures=tuple(failures),
    )


def _render_llm_plan_case_markdown(report: LLMEvaluationReport) -> str:
    lines = [
        "# Day 17 LLM Planner Evaluation (LLM alone, no rule planner)",
        "",
        f"- Case set SHA-256: `{report.case_set_sha256}`",
        f"- Cases: {report.case_count}",
        f"- Operation accuracy: {report.operation_accuracy:.6f}",
        f"- Invalid-JSON rate: {report.invalid_json_rate:.6f}",
        f"- Repair success rate: {report.repair_success_rate:.6f}",
        "",
        f"## Failures ({len(report.failures)})",
        "",
    ]
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.case_id}",
                "",
                f"- Question: {failure.question}",
                f"- Operation expected/actual: {failure.expected_operation} / "
                f"{failure.actual_operation}",
                f"- Outcome: {failure.outcome}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_llm_plan_case_report(report: LLMEvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.case_set_sha256[:12]
    json_path = output_dir / f"llm-plan-cases-{prefix}.json"
    markdown_path = output_dir / f"llm-plan-cases-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_llm_plan_case_markdown(report))
    return json_path, markdown_path


# --- Router on abstain-expected cases --------------------------------------


class RouterAbstainFailure(_FrozenModel):
    case_id: str
    question: str
    expected_abstain_code: str
    source: Literal["rule", "llm"]
    actual_operation: str


class RouterAbstainReport(_FrozenModel):
    """Scored on `expected_abstain_code` cases: did the A1 router still abstain?"""

    case_set_sha256: str
    case_count: int = Field(ge=0)
    abstain_recall: float = Field(ge=0, le=1)
    false_plan_rate: float = Field(ge=0, le=1)
    failures: tuple[RouterAbstainFailure, ...]


def evaluate_router_abstain_cases(
    cases: tuple[PlanCase, ...], *, client: ChatCompletionClient, case_set_sha256: str
) -> RouterAbstainReport:
    """`false_plan_rate` (DoD hard KPI, must be 0.0): does the LLM fallback ever
    fabricate a plan for a question the rule planner correctly abstains on?"""
    abstain_cases = sorted(
        (case for case in cases if case.expected_abstain_code is not None),
        key=lambda case: case.case_id,
    )
    abstain_correct = false_plan = 0
    failures: list[RouterAbstainFailure] = []

    for case in abstain_cases:
        entities = parse_query_entities(case.question)
        routed = route_plan(
            entities,
            client=client,
            candidate_table_ids=(_DUMMY_TABLE_ID,),
            known_table_ids=_DUMMY_KNOWN_TABLES,
        )
        if routed.result.plan is None:
            abstain_correct += 1
        else:
            false_plan += 1
            failures.append(
                RouterAbstainFailure(
                    case_id=case.case_id,
                    question=case.question,
                    expected_abstain_code=case.expected_abstain_code,  # type: ignore[arg-type]
                    source=routed.source,
                    actual_operation=routed.result.plan.operation,
                )
            )

    total = len(abstain_cases)
    return RouterAbstainReport(
        case_set_sha256=case_set_sha256,
        case_count=total,
        abstain_recall=(abstain_correct / total) if total else 1.0,
        false_plan_rate=(false_plan / total) if total else 0.0,
        failures=tuple(failures),
    )


def _render_router_abstain_markdown(report: RouterAbstainReport) -> str:
    lines = [
        "# Day 17 Router Abstain Evaluation (rule planner + LLM fallback)",
        "",
        f"- Case set SHA-256: `{report.case_set_sha256}`",
        f"- Cases: {report.case_count}",
        f"- Abstain recall: {report.abstain_recall:.6f}",
        f"- **False-plan rate (DoD hard KPI, must be 0.0): {report.false_plan_rate:.6f}**",
        "",
        f"## Failures ({len(report.failures)})",
        "",
    ]
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.case_id}",
                "",
                f"- Question: {failure.question}",
                f"- Expected abstain code: {failure.expected_abstain_code}",
                f"- Source that produced a plan instead: {failure.source}",
                f"- Fabricated operation: {failure.actual_operation}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_router_abstain_report(report: RouterAbstainReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.case_set_sha256[:12]
    json_path = output_dir / f"router-abstain-{prefix}.json"
    markdown_path = output_dir / f"router-abstain-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_router_abstain_markdown(report))
    return json_path, markdown_path
