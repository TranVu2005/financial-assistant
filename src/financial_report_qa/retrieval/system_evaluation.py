"""Derive complete V2 metric artifacts from authoritative persisted rankings."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.evaluation import (
    RetrievalMetricsExtended,
    score_extended_at_10,
)

SystemSourceKind = Literal["legacy", "dense", "fusion", "expansion"]


class SystemQuestionEvaluationV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    intent: str
    gold_cardinality: str
    period_cardinality: str
    statement_filter: str
    report_era: str | None
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    metrics: RetrievalMetricsExtended


class RetrievalSystemReportV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["retrieval-system-evaluation-v2"] = "retrieval-system-evaluation-v2"
    system_name: str
    source_kind: SystemSourceKind
    source_path: str
    source_sha256: str
    selection: str
    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    macro: RetrievalMetricsExtended
    by_intent: dict[str, RetrievalMetricsExtended]
    by_gold_cardinality: dict[str, RetrievalMetricsExtended]
    by_period_cardinality: dict[str, RetrievalMetricsExtended]
    by_statement_filter: dict[str, RetrievalMetricsExtended]
    by_report_era: dict[str, RetrievalMetricsExtended]
    per_question: tuple[SystemQuestionEvaluationV2, ...]

    @model_validator(mode="after")
    def validate_complete_evidence(self) -> RetrievalSystemReportV2:
        if self.question_count != len(self.per_question):
            raise ValueError("question_count must equal per_question length")
        question_ids = tuple(item.question_id for item in self.per_question)
        if question_ids != tuple(sorted(set(question_ids))):
            raise ValueError("per_question must be sorted and unique by question_id")
        return self


def _average(values: Iterable[RetrievalMetricsExtended]) -> RetrievalMetricsExtended:
    metrics = tuple(values)
    count = len(metrics)
    if not metrics:
        return RetrievalMetricsExtended(
            true_positive=0,
            precision_at_10=0,
            recall_at_3=0,
            recall_at_5=0,
            recall_at_10=0,
            f2_at_10=0,
            mrr=0,
            precision_at_r=0,
            f2_at_r=0,
        )
    return RetrievalMetricsExtended(
        true_positive=sum(item.true_positive for item in metrics),
        precision_at_10=sum(item.precision_at_10 for item in metrics) / count,
        recall_at_3=sum(item.recall_at_3 for item in metrics) / count,
        recall_at_5=sum(item.recall_at_5 for item in metrics) / count,
        recall_at_10=sum(item.recall_at_10 for item in metrics) / count,
        f2_at_10=sum(item.f2_at_10 for item in metrics) / count,
        mrr=sum(item.mrr for item in metrics) / count,
        precision_at_r=sum(item.precision_at_r for item in metrics) / count,
        f2_at_r=sum(item.f2_at_r for item in metrics) / count,
    )


def _selected_questions(
    payload: Mapping[str, object], source_kind: SystemSourceKind
) -> tuple[Sequence[Mapping[str, object]], str]:
    if source_kind == "legacy":
        return _mapping_sequence(payload.get("per_question")), "legacy-report"
    if source_kind == "dense":
        cold = _mapping(payload.get("cold_report"), "dense cold_report")
        return _mapping_sequence(cold.get("per_question")), "cold-report"
    grid = _mapping_sequence(payload.get("grid"))
    selector_name = "best_weights" if source_kind == "fusion" else "best_params"
    point_name = "weights" if source_kind == "fusion" else "params"
    selector = _mapping(payload.get(selector_name), selector_name)
    matches = [item for item in grid if _mapping(item.get(point_name), point_name) == selector]
    if len(matches) != 1:
        raise ValueError(f"{source_kind} source must have exactly one selected grid point")
    return _mapping_sequence(matches[0].get("per_question")), (
        f"{selector_name} "
        + json.dumps(selector, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("source per_question/grid must be an array of objects")
    return tuple(value)


def _gold_cardinality(question: GoldRetrievalQuestion) -> str:
    count = len(question.gold_table_ids)
    return "one_table" if count == 1 else "two_tables" if count == 2 else "three_or_more"


def _period_cardinality(question: GoldRetrievalQuestion) -> str:
    return "one_period" if len(question.filters.periods) == 1 else "multiple_periods"


def _report_era(question: GoldRetrievalQuestion) -> str | None:
    if not question.filters.periods:
        return None
    latest = max(int(period) for period in question.filters.periods)
    if not 2015 <= latest <= 2025:
        raise ValueError("report-era breakdown requires periods from 2015 through 2025")
    return "2015_2019" if latest <= 2019 else "2020_2023" if latest <= 2023 else "2024_2025"


def derive_system_report_v2(
    *,
    system_name: str,
    source_path: Path,
    source_kind: SystemSourceKind,
    questions: Sequence[GoldRetrievalQuestion],
) -> RetrievalSystemReportV2:
    """Re-score a persisted system ranking without rerunning its model/index."""
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "source report")
    source_questions, selection = _selected_questions(payload, source_kind)
    question_by_id = {item.question_id: item for item in questions}
    source_by_id = {
        str(item.get("question_id")): item
        for item in source_questions
        if isinstance(item.get("question_id"), str)
    }
    if set(source_by_id) != set(question_by_id) or len(source_by_id) != len(source_questions):
        raise ValueError("source report question IDs do not exactly match reviewed gold")

    results: list[SystemQuestionEvaluationV2] = []
    groups: dict[str, dict[str, list[RetrievalMetricsExtended]]] = {
        "intent": defaultdict(list),
        "gold": defaultdict(list),
        "period": defaultdict(list),
        "statement": defaultdict(list),
        "era": defaultdict(list),
    }
    for question_id in sorted(question_by_id):
        question = question_by_id[question_id]
        source_question = source_by_id[question_id]
        predicted_raw = source_question.get("predicted_table_ids")
        if not isinstance(predicted_raw, list) or not all(
            isinstance(item, str) for item in predicted_raw
        ):
            raise ValueError("source predicted_table_ids must be an array of strings")
        predicted = tuple(predicted_raw)
        metrics = score_extended_at_10(predicted, question.gold_table_ids)
        gold_cardinality = _gold_cardinality(question)
        period_cardinality = _period_cardinality(question)
        statement_filter = "filtered" if question.filters.statement_types else "unfiltered"
        era = _report_era(question)
        results.append(
            SystemQuestionEvaluationV2(
                question_id=question_id,
                intent=question.intent,
                gold_cardinality=gold_cardinality,
                period_cardinality=period_cardinality,
                statement_filter=statement_filter,
                report_era=era,
                predicted_table_ids=predicted[:10],
                gold_table_ids=question.gold_table_ids,
                metrics=metrics,
            )
        )
        groups["intent"][question.intent].append(metrics)
        groups["gold"][gold_cardinality].append(metrics)
        groups["period"][period_cardinality].append(metrics)
        groups["statement"][statement_filter].append(metrics)
        if era is not None:
            groups["era"][era].append(metrics)

    def averaged(
        values: dict[str, list[RetrievalMetricsExtended]],
    ) -> dict[str, RetrievalMetricsExtended]:
        return {key: _average(values[key]) for key in sorted(values)}

    fingerprint = questions[0].dataset_fingerprint if questions else ""
    payload_fingerprint = payload.get("dataset_fingerprint")
    if source_kind == "dense":
        payload_fingerprint = _mapping(payload.get("cold_report"), "dense cold_report").get(
            "dataset_fingerprint"
        )
    if payload_fingerprint != fingerprint:
        raise ValueError("source report fingerprint does not match reviewed gold")
    return RetrievalSystemReportV2(
        system_name=system_name,
        source_kind=source_kind,
        source_path=source_path.as_posix(),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        selection=selection,
        dataset_fingerprint=fingerprint,
        question_count=len(results),
        macro=_average(item.metrics for item in results),
        by_intent=averaged(groups["intent"]),
        by_gold_cardinality=averaged(groups["gold"]),
        by_period_cardinality=averaged(groups["period"]),
        by_statement_filter=averaged(groups["statement"]),
        by_report_era=averaged(groups["era"]),
        per_question=tuple(results),
    )


def _render_markdown(report: RetrievalSystemReportV2) -> str:
    lines = [
        f"# Retrieval V2 â€” {report.system_name}",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Source: `{report.source_path}` (`{report.source_sha256}`)",
        f"- Selection: {report.selection}",
        f"- Questions: {report.question_count}",
    ]
    for title, values in (
        ("By intent", report.by_intent),
        ("By gold cardinality", report.by_gold_cardinality),
        ("By period cardinality", report.by_period_cardinality),
        ("By statement filter", report.by_statement_filter),
        ("By report era", report.by_report_era),
    ):
        lines.extend(
            (
                "",
                f"## {title}",
                "",
                "| Group | R@3 | R@5 | R@10 | MRR | P@R | F2@R |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        lines.extend(
            f"| {label} | {metrics.recall_at_3:.6f} | {metrics.recall_at_5:.6f} | "
            f"{metrics.recall_at_10:.6f} | {metrics.mrr:.6f} | "
            f"{metrics.precision_at_r:.6f} | {metrics.f2_at_r:.6f} |"
            for label, metrics in values.items()
        )
    return "\n".join(lines) + "\n"


def write_system_report_v2(report: RetrievalSystemReportV2, output_dir: Path) -> tuple[Path, Path]:
    """Write deterministic per-system V2 JSON and Markdown."""
    safe_name = report.system_name.replace("/", "-").replace(" ", "-")
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-v2-{safe_name}-{prefix}.json"
    markdown_path = output_dir / f"retrieval-v2-{safe_name}-{prefix}.md"
    content = json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
    write_text_atomic(json_path, content + "\n")
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
