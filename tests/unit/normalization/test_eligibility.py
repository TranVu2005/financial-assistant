# mypy: ignore-errors
from decimal import Decimal

import pytest

from financial_report_qa.normalization.eligibility import classify_cell_eligibility
from financial_report_qa.schemas.tables import CellRecord


@pytest.fixture
def cell_factory():
    def _factory(**kwargs):
        tbl_id = "tbl_" + "a" * 64
        defaults = {
            "cell_id": "cell_" + "b" * 64,
            "table_id": tbl_id,
            "row_idx": 0,
            "col_idx": 0,
            "row_label_raw": "Metric",
            "row_label_canonical": "net_revenue",
            "column_label_raw": "2024",
            "column_label_canonical": "2024",
            "value_raw": "1000",
            "value_numeric": Decimal("1000"),
            "period": "2024",
            "unit": "VND_million",
            "source_line_start": 1,
            "source_line_end": 1,
            "extraction_confidence": 1.0,
        }
        defaults.update(kwargs)
        return CellRecord(**defaults)

    return _factory


def test_cell_eligibility_levels(cell_factory):
    raw = classify_cell_eligibility(cell_factory(value_numeric=None), set())
    assert raw.searchable and not raw.comparable and not raw.calculable

    comparable = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit=None), set()
    )
    assert comparable.comparable and not comparable.calculable

    calculable = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit="VND"), set()
    )
    assert calculable.calculable

    # Verify unit_conflict blocks calculable
    blocked_unit = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit="VND"), {"unit_conflict"}
    )
    assert not blocked_unit.calculable

    # Verify number_ambiguous blocks calculable
    blocked_num = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit="VND"), {"number_ambiguous"}
    )
    assert not blocked_num.calculable

    # Verify period_ambiguous blocks calculable
    blocked_per = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit="VND"), {"period_ambiguous"}
    )
    assert not blocked_per.calculable
