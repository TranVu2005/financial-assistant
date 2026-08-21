"""Tests for the plan.md §20 metrics missing before this pass: Cell Accuracy,
Period Accuracy, Unit Accuracy, and Valid Plan Rate. Before this module,
`grep -rln "cell_accuracy|period_accuracy|unit_accuracy|valid_plan_rate"`
found nothing in `src/` at all (v2-remaining-gaps.md).

Cell/Period/Unit Accuracy are scored against `dev-benchmark-v1.gold.jsonl`
(ADR 0009 decision A2: annotated independently of the pipeline, straight from
the source report text) rather than against the pipeline's own idea of what
it retrieved -- otherwise the metric would just measure self-consistency.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.evaluation.quality_metrics import (
    GoldFactAnnotation,
    evaluate_grounding_quality,
    load_gold_fact_annotations,
    valid_plan_rate,
)
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

TABLE_ID = "tbl_" + "1" * 64


def _gold(**overrides: object) -> GoldFactAnnotation:
    payload: dict[str, object] = {
        "question_id": 4,
        "row_labels": ("Lợi nhuận sau thuế",),
        "column_labels": ("Năm 2023",),
        "period": 2023,
        "unit": "VND_million",
    }
    payload.update(overrides)
    return GoldFactAnnotation.model_validate(payload)


def _fact(**overrides: object) -> GroundedFact:
    payload: dict[str, object] = {
        "fact_id": "F1",
        "table_id": TABLE_ID,
        "row_index": 14,
        "row_label": "Lợi nhuận sau thuế",
        "column": "Năm 2023",
        "period": 2023,
        "raw_value": Decimal("444918"),
        "unit": "VND_million",
        "grounding_score": 0.94,
    }
    payload.update(overrides)
    return GroundedFact.model_validate(payload)


# ---------------------------------------------------------------------------
# GoldFactAnnotation / loader
# ---------------------------------------------------------------------------


def test_gold_fact_annotation_requires_at_least_one_row_label() -> None:
    with pytest.raises(ValidationError):
        GoldFactAnnotation.model_validate(
            {
                "question_id": 4,
                "row_labels": (),
                "column_labels": (),
                "period": 2023,
                "unit": "VND_million",
            }
        )


def test_load_gold_fact_annotations_reads_only_annotated_records(tmp_path: Path) -> None:
    """`needs_review` records (v2-remaining-gaps.md / the gold set's own
    provenance doc) explicitly do not settle a row/column/value -- scoring
    against one would grade the pipeline on a question with no right answer."""
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"vifinqa_id": 4, "status": "annotated", "period": 2023, '
                    '"gold_rows": ["Lợi nhuận sau thuế"], "gold_columns": ["Năm 2023"], '
                    '"gold_values": [{"raw": "444.918", "numeric": 444918.0, '
                    '"source_unit": "VND_million"}]}'
                ),
                '{"vifinqa_id": 907, "status": "needs_review"}',
            ]
        ),
        encoding="utf-8",
    )
    golds = load_gold_fact_annotations(path)
    assert [gold.question_id for gold in golds] == [4]
    assert golds[0].row_labels == ("Lợi nhuận sau thuế",)
    assert golds[0].column_labels == ("Năm 2023",)
    assert golds[0].period == 2023
    assert golds[0].unit == "VND_million"


def test_load_gold_fact_annotations_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '\n{"vifinqa_id": 4, "status": "annotated", "period": 2023, '
        '"gold_rows": ["X"], "gold_columns": ["Y"], '
        '"gold_values": [{"raw": "1", "numeric": 1.0, "source_unit": "VND"}]}\n\n',
        encoding="utf-8",
    )
    golds = load_gold_fact_annotations(path)
    assert len(golds) == 1


# ---------------------------------------------------------------------------
# Cell / Period / Unit Accuracy
# ---------------------------------------------------------------------------


def test_a_fact_agreeing_on_row_period_and_unit_scores_all_three() -> None:
    report = evaluate_grounding_quality([_gold()], {4: _fact()})
    assert report.question_count == 1
    assert report.cell_correct == 1
    assert report.period_correct == 1
    assert report.unit_correct == 1
    assert report.cell_accuracy == 1.0
    assert report.period_accuracy == 1.0
    assert report.unit_accuracy == 1.0


def test_row_matching_is_normalization_tolerant() -> None:
    """The same tolerance `locator.py`/`fact_grounding.py` already use for
    matching a selector's raw text against a corpus label -- a gold label
    copied from the source text can differ from the pipeline's in casing,
    NFKC form, or whitespace while naming the same row."""
    gold = _gold(row_labels=("  Lợi Nhuận Sau Thuế  ",))
    report = evaluate_grounding_quality([gold], {4: _fact()})
    assert report.cell_correct == 1


def test_a_wrong_row_fails_cell_accuracy_only() -> None:
    """v2-remaining-gaps.md's dominant failure mode: a plausible number read
    off the wrong line item. Period and unit can both still be right even
    when the row is wrong -- they are independent checks (plan.md §20's
    table lists them as separate rows/metrics)."""
    fact = _fact(row_label="Doanh thu thuần")
    report = evaluate_grounding_quality([_gold()], {4: fact})
    assert report.cell_correct == 0
    assert report.period_correct == 1
    assert report.unit_correct == 1


def test_a_wrong_period_fails_period_accuracy_only() -> None:
    fact = _fact(period=2022)
    report = evaluate_grounding_quality([_gold()], {4: fact})
    assert report.cell_correct == 1
    assert report.period_correct == 0
    assert report.unit_correct == 1


def test_a_wrong_unit_fails_unit_accuracy_only() -> None:
    fact = _fact(unit="VND_billion")
    report = evaluate_grounding_quality([_gold()], {4: fact})
    assert report.cell_correct == 1
    assert report.period_correct == 1
    assert report.unit_correct == 0


def test_a_question_with_no_produced_fact_counts_as_wrong_on_all_three() -> None:
    """Matches how `score_value.py` computes 'accuracy on all gold-annotated':
    a question the pipeline never grounded at all is a grounding failure, not
    something to quietly exclude from the denominator."""
    report = evaluate_grounding_quality([_gold()], {})
    assert report.question_count == 1
    assert report.graded_count == 0
    assert report.cell_accuracy == 0.0
    assert report.period_accuracy == 0.0
    assert report.unit_accuracy == 0.0


def test_accuracy_averages_across_multiple_questions() -> None:
    golds = [_gold(question_id=4), _gold(question_id=6, row_labels=("Chi phí",))]
    facts = {
        4: _fact(),  # fully correct
        6: _fact(row_label="Doanh thu"),  # wrong row
    }
    report = evaluate_grounding_quality(golds, facts)
    assert report.question_count == 2
    assert report.cell_accuracy == 0.5
    assert report.period_accuracy == 1.0
    assert report.unit_accuracy == 1.0


def test_column_labels_do_not_affect_cell_accuracy() -> None:
    """§20's table lists Cell Accuracy, Period Accuracy and Unit Accuracy as
    three separate rows for the row/period/unit dimensions grounding must get
    right -- column narrowing is a finer-grained concern `fact_checks.py`
    (plan.md §15) already re-verifies independently, not this aggregate."""
    fact = _fact(column="Số cuối năm")
    report = evaluate_grounding_quality([_gold()], {4: fact})
    assert report.cell_correct == 1


# ---------------------------------------------------------------------------
# Valid Plan Rate
# ---------------------------------------------------------------------------


def _lookup_plan(**overrides: object) -> FinancialQueryPlan:
    payload: dict[str, object] = {
        "operation": "lookup",
        "companies": ("ACB",),
        "periods": ("2023",),
        "candidate_table_ids": (TABLE_ID,),
        "metric": MetricSelector(canonical="cash_and_cash_equivalents"),
    }
    payload.update(overrides)
    return FinancialQueryPlan.model_validate(payload)


def test_valid_plan_rate_is_one_when_every_plan_passes_semantic_validation() -> None:
    assert valid_plan_rate([_lookup_plan(), _lookup_plan()]) == 1.0


def test_valid_plan_rate_counts_a_semantically_invalid_plan_as_invalid() -> None:
    """plan.md §12: the failure mode being measured is a plan the (LLM)
    planner *built* but that `plan_validator.validate_plan_semantics` rejects
    -- e.g. a `lookup` naming two periods, which is structurally a valid
    `FinancialQueryPlan` (arity is not enforced at that layer, ADR 0004) but
    semantically wrong."""
    bad = _lookup_plan(periods=("2022", "2023"))
    assert valid_plan_rate([_lookup_plan(), bad]) == 0.5


def test_valid_plan_rate_is_zero_for_no_attempted_plans() -> None:
    assert valid_plan_rate([]) == 0.0
