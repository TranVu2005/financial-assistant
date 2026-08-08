"""Fixed-denominator Retrieval@10 evaluation and deterministic report rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.service import RetrievalService


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    true_positive: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f2: float = Field(ge=0, le=1)


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_fingerprint: str
    question_count: int
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    per_question: dict[str, RetrievalMetrics]


def score_at_10(predicted: tuple[str, ...], gold: tuple[str, ...]) -> RetrievalMetrics:
    """Score fixed Retrieval@10 precision, recall, and F2."""
    if len(set(predicted)) != len(predicted):
        raise ValueError("predicted table IDs must be unique")
    if len(set(gold)) != len(gold):
        raise ValueError("gold table IDs must be unique")
    if not gold:
        raise ValueError("gold table IDs must not be empty")
    true_positive = len(set(predicted[:10]).intersection(gold))
    precision = true_positive / 10
    recall = true_positive / len(gold)
    denominator = 4 * precision + recall
    return RetrievalMetrics(
        true_positive=true_positive,
        precision=precision,
        recall=recall,
        f2=0.0 if denominator == 0 else 5 * precision * recall / denominator,
    )


def _average(metrics: list[RetrievalMetrics]) -> RetrievalMetrics:
    if not metrics:
        return RetrievalMetrics(true_positive=0, precision=0, recall=0, f2=0)
    count = len(metrics)
    return RetrievalMetrics(
        true_positive=sum(item.true_positive for item in metrics),
        precision=sum(item.precision for item in metrics) / count,
        recall=sum(item.recall for item in metrics) / count,
        f2=sum(item.f2 for item in metrics) / count,
    )


def evaluate_retrieval(
    retriever: RetrievalService, questions: tuple[GoldRetrievalQuestion, ...], *, k: int = 10
) -> RetrievalEvaluationReport:
    """Evaluate reviewed questions in stable ID order."""
    per_question: dict[str, RetrievalMetrics] = {}
    per_intent: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    for question in sorted(questions, key=lambda item: item.question_id):
        trace = retriever.retrieve(
            question.question, filters=question.filters, k=k, question_id=question.question_id
        )
        metrics = score_at_10(
            tuple(item.table_id for item in trace.results), question.gold_table_ids
        )
        per_question[question.question_id] = metrics
        per_intent[question.intent].append(metrics)
    return RetrievalEvaluationReport(
        dataset_fingerprint=questions[0].dataset_fingerprint if questions else "",
        question_count=len(questions),
        macro=_average(list(per_question.values())),
        by_intent={intent: _average(per_intent[intent]) for intent in sorted(per_intent)},
        per_question={
            question_id: per_question[question_id] for question_id in sorted(per_question)
        },
    )


def write_report(report: RetrievalEvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    """Write byte-stable JSON and Markdown report artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-evaluation-{prefix}.json"
    markdown_path = output_dir / f"retrieval-evaluation-{prefix}.md"
    json_path.write_text(
        json.dumps(report.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    macro = report.macro
    markdown_path.write_text(
        "\n".join(
            (
                "# Day 8 BM25 Retrieval Evaluation",
                "",
                f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
                f"- Questions: {report.question_count}",
                f"- Precision@10: {macro.precision:.6f}",
                f"- Recall@10: {macro.recall:.6f}",
                f"- F2@10: {macro.f2:.6f}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path
