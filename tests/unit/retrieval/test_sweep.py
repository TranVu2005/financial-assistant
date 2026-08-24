import pytest
from tests.unit.retrieval.test_live_query import _FakeBm25Service, _table_id

from financial_report_qa.retrieval.sweep import (
    SweepResult,
    recommend_k,
    render_sweep_markdown,
    run_sweep,
)


class _GoldStub:
    def __init__(self, question: str, gold_table_ids: tuple[str, ...]) -> None:
        self.question = question
        self.gold_table_ids = gold_table_ids


def test_sweep_reports_f2_and_mrr5_at_every_k() -> None:
    service = _FakeBm25Service(("a", "b", "c"))
    questions = (_GoldStub("Doanh thu VCB 2023?", (_table_id("b"),)),)

    results = run_sweep(questions, service, ks=(1, 2, 3))

    assert [item.k for item in results] == [1, 2, 3]
    # gold nằm ở hạng 2 -> k=1 trượt hoàn toàn, k>=2 bắt được.
    assert results[0].f2 == pytest.approx(0.0)
    assert results[0].mrr5 == pytest.approx(0.0)
    assert results[1].mrr5 == pytest.approx(0.5)
    assert results[2].mrr5 == pytest.approx(0.5)


def test_precision_penalty_makes_f2_fall_as_k_grows_past_the_gold() -> None:
    # hậu tố phải là hex: TableMetadata bắt table_id khớp ^tbl_[0-9a-f]{64}$
    service = _FakeBm25Service(tuple("0123456789"))
    questions = (_GoldStub("Doanh thu VCB 2023?", (_table_id("0"),)),)

    results = {item.k: item for item in run_sweep(questions, service, ks=(1, 10))}

    assert results[1].f2 > results[10].f2


def test_recommend_k_prefers_the_best_f2_and_breaks_ties_on_mrr5() -> None:
    results = (
        SweepResult(k=1, f2=0.40, mrr5=0.40),
        SweepResult(k=5, f2=0.60, mrr5=0.55),
        SweepResult(k=8, f2=0.60, mrr5=0.58),
        SweepResult(k=10, f2=0.55, mrr5=0.58),
    )

    assert recommend_k(results) == 8


def test_recommend_k_rejects_an_empty_sweep() -> None:
    with pytest.raises(ValueError):
        recommend_k(())


def test_markdown_marks_the_recommended_row() -> None:
    results = (SweepResult(k=5, f2=0.6, mrr5=0.5), SweepResult(k=10, f2=0.5, mrr5=0.5))

    rendered = render_sweep_markdown(results, recommended_k=5)

    assert "| 5 |" in rendered
    assert "**5**" in rendered or "<-- k*" in rendered
