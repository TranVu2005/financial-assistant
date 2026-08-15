"""Day 20 Vietnamese template renderer (ADR 0009 decision F1).

The default, LLM-free path from a locked `CompiledQuery` to display text and
a Vietnamese sentence. Every number in the output is either the locked
answer itself (possibly rescaled per the presentation table below) or a
`plan.periods`/`plan.top_k` value already present in the plan -- the
template never invents a new number, so it needs no `numeric_guard` pass of
its own (that module exists for the *optional* LLM paraphrase path, Day 20
plan Sec 1.8/2.F).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

_CURRENCY_LABELS: dict[str, str] = {
    "VND": "VND",
    "VND_thousand": "nghìn VND",
    "VND_million": "triệu VND",
    "VND_billion": "tỷ VND",
}

_DEFAULT_PRECISION_BY_UNIT: dict[str, int] = {
    "VND": 0,
    "VND_thousand": 0,
    "VND_million": 0,
    "VND_billion": 0,
    "percent": 2,
    "ratio": 4,
}


def _metric_label(selector: MetricSelector | None) -> str:
    if selector is None:
        return "chỉ số"
    return selector.canonical if selector.canonical is not None else selector.raw_text  # type: ignore[return-value]


def _quantize(value: Decimal, precision: int) -> Decimal:
    step = Decimal(1).scaleb(-precision)
    return value.quantize(step, rounding=ROUND_HALF_UP)


def _format_number(value: Decimal, precision: int) -> str:
    quantized = _quantize(value, precision)
    sign = "-" if quantized < 0 else ""
    text = f"{abs(quantized):,.{precision}f}"
    return f"{sign}{text}"


def render_answer(plan: FinancialQueryPlan, compiled: CompiledQuery) -> tuple[str, int]:
    """Return `(display, display_precision)` for one answered `CompiledQuery`.

    Raises `ValueError` on a non-`answered` result -- there is no display for
    an error, by design (ADR 0007 decision D1's "never guess" extends here).
    """
    if compiled.status != "answered":
        raise ValueError("cannot render a display for a non-answered CompiledQuery")
    assert compiled.answer is not None and compiled.unit is not None
    unit: CanonicalUnit = compiled.unit
    value = compiled.answer

    if unit == "ratio" and plan.expected_unit == "percent":
        precision = _DEFAULT_PRECISION_BY_UNIT["percent"]
        return f"{_format_number(value * 100, precision)}%", precision

    if unit == "percent":
        precision = _DEFAULT_PRECISION_BY_UNIT["percent"]
        return f"{_format_number(value, precision)}%", precision

    if unit == "ratio":
        precision = _DEFAULT_PRECISION_BY_UNIT["ratio"]
        return _format_number(value, precision), precision

    precision = _DEFAULT_PRECISION_BY_UNIT[unit]
    label = _CURRENCY_LABELS[unit]
    return f"{_format_number(value, precision)} {label}", precision


def render_sentence(plan: FinancialQueryPlan, compiled: CompiledQuery, display: str) -> str:
    """Render the Vietnamese answer sentence. `display` must come from
    `render_answer` (or something with the same numeric content) -- this
    function introduces no numbers of its own beyond `plan.periods`/`top_k`.
    """
    if compiled.status != "answered":
        raise ValueError("cannot render a sentence for a non-answered CompiledQuery")
    operation = compiled.operation
    company = plan.companies[0]
    period = plan.periods[0] if plan.periods else ""

    if operation == "lookup":
        return f"{_metric_label(plan.metric)} của {company} năm {period} là {display}."

    if operation in ("difference", "growth_rate"):
        start, end = plan.periods[0], plan.periods[-1]
        metric = _metric_label(plan.metric)
        if operation == "difference":
            return f"{metric} của {company} thay đổi {display} từ năm {start} đến năm {end}."
        assert compiled.answer is not None
        direction = "tăng" if compiled.answer >= 0 else "giảm"
        magnitude = display.lstrip("-")
        return f"{metric} của {company} {direction} {magnitude} từ năm {start} đến năm {end}."

    if operation == "compare":
        metric_a = _metric_label(plan.metric_a)
        metric_b = _metric_label(plan.metric_b)
        return f"So sánh {metric_a} và {metric_b} của {company} năm {period}: chênh lệch {display}."

    if operation == "compare_companies":
        company_a, company_b = plan.companies[0], plan.companies[1]
        metric = _metric_label(plan.metric)
        return (
            f"So sánh {metric} của {company_a} và {company_b} năm {period}: chênh lệch {display}."
        )

    if operation == "ratio":
        numerator = _metric_label(plan.numerator_metric)
        denominator = _metric_label(plan.denominator_metric)
        return f"Tỷ lệ {numerator} trên {denominator} của {company} năm {period} là {display}."

    if operation in ("average", "sum"):
        metric = _metric_label(plan.metric)
        verb = "Trung bình" if operation == "average" else "Tổng"
        periods = ", ".join(plan.periods)
        return f"{verb} {metric} của {company} các năm {periods} là {display}."

    if operation == "rank":
        assert plan.top_k is not None
        metric = _metric_label(plan.metric)
        return (
            f"{metric} đứng thứ {plan.top_k} trong số {len(plan.companies)} công ty "
            f"năm {period} là {display}."
        )

    raise ValueError(f"no sentence template for operation '{operation}'")
