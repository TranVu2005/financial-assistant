"""Deterministic entity-parser evaluation cases generated from the locked release.

Every case's expected entities are computed from a frozen template plus real
values drawn from the locked Week 1 release (company codes, periods,
statement types) and the curated metric dictionary — never by running the
parser and copying its output. That is what makes this an evaluation set
instead of a mirror: a bug in the parser cannot also corrupt its own answer
key.

**Freeze discipline:** `TEMPLATE_INVENTORY` and this generator were committed
before `financial_report_qa.planning.entity_parser` grew any rule beyond its
module skeleton. If a template must change later, the reason and the new
`entity_case_set_sha256` value belong in README, not a silent edit.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import pyarrow.parquet as pq
from pydantic import field_validator

from financial_report_qa.normalization.companies import COMPANY_REGISTRY
from financial_report_qa.normalization.metrics import (
    BALANCE_SHEET_ALIASES,
    CASH_FLOW_ALIASES,
    INCOME_STATEMENT_ALIASES,
)
from financial_report_qa.normalization.statements import _STATEMENT_FAMILIES
from financial_report_qa.planning.entity_contracts import AmbiguityCode
from financial_report_qa.retrieval.contracts import NonEmptyString, _canonical_tuple, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import (
    canonical_json_bytes,
    sha256_bytes,
    write_text_atomic,
)
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease

TemplateCategory = Literal["lookup", "compare", "growth", "adversarial"]

_QUARTER_ROMAN = ("I", "II", "III", "IV")
_STATEMENT_TEMPLATE_FAMILIES = ("balance_sheet", "income_statement", "cash_flow_statement")


class EntityCase(_FrozenModel):
    """One generated question with its structurally derived expected entities."""

    case_id: str
    template_id: NonEmptyString
    category: TemplateCategory
    question: NonEmptyString
    expected_company_codes: tuple[str, ...] = ()
    expected_periods: tuple[str, ...] = ()
    expected_metrics: tuple[str, ...] = ()
    expected_statement_types: tuple[str, ...] = ()
    expected_ambiguity: tuple[AmbiguityCode, ...] = ()

    @field_validator(
        "expected_company_codes",
        "expected_periods",
        "expected_metrics",
        "expected_statement_types",
    )
    @classmethod
    def validate_canonical_values(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _canonical_tuple(values, label=str(getattr(info, "field_name", "expected")))

    @field_validator("expected_ambiguity")
    @classmethod
    def validate_ambiguity(cls, values: tuple[AmbiguityCode, ...]) -> tuple[AmbiguityCode, ...]:
        if tuple(sorted(set(values))) != values:
            raise ValueError("expected_ambiguity must be sorted and unique")
        return values


@dataclass(frozen=True)
class ValuePools:
    """Deterministic, sorted value pools drawn from the locked release."""

    companies: tuple[tuple[str, str], ...]  # (ticker, canonical_name)
    years: tuple[str, ...]
    year_pairs: tuple[tuple[str, str], ...]
    quarters: tuple[str, ...]
    dates: tuple[str, ...]  # ISO YYYY-MM-DD
    metrics: tuple[tuple[str, str], ...]  # (raw_alias, canonical)
    statement_families: tuple[str, ...] = field(
        default=tuple(f for f in _STATEMENT_TEMPLATE_FAMILIES)
    )


def _load_value_pools(release: ResolvedRetrievalRelease) -> ValuePools:
    documents = pq.read_table(  # type: ignore[no-untyped-call]
        release.release_dir / "documents.parquet", columns=["company_code"]
    ).to_pylist()
    document_codes = {str(row["company_code"]) for row in documents}
    company_codes = sorted(document_codes & COMPANY_REGISTRY.keys())
    companies = tuple((code, COMPANY_REGISTRY[code].canonical_name) for code in company_codes)

    cells = pq.read_table(  # type: ignore[no-untyped-call]
        release.release_dir / "cells.parquet", columns=["period"]
    ).to_pylist()
    raw_periods = {str(row["period"]) for row in cells if row["period"] is not None}
    years = tuple(sorted(value for value in raw_periods if len(value) == 4 and value.isdigit()))
    year_set = set(years)
    year_pairs = tuple(
        (year, str(int(year) + 1)) for year in years if str(int(year) + 1) in year_set
    )
    quarters = tuple(sorted(value for value in raw_periods if len(value) == 7 and value[5] == "Q"))
    dates = tuple(
        sorted(value for value in raw_periods if len(value) == 10 and value.endswith("-12-31"))
    )

    tables = pq.read_table(  # type: ignore[no-untyped-call]
        release.release_dir / "tables.parquet", columns=["statement_type"]
    ).to_pylist()
    observed_families = {str(row["statement_type"]) for row in tables if row["statement_type"]}
    statement_families = tuple(
        family for family in _STATEMENT_TEMPLATE_FAMILIES if family in observed_families
    )

    merged_metrics = {**INCOME_STATEMENT_ALIASES, **BALANCE_SHEET_ALIASES, **CASH_FLOW_ALIASES}
    metrics = tuple(sorted(merged_metrics.items()))

    if not (
        companies and years and year_pairs and quarters and dates and metrics and statement_families
    ):
        raise ValueError("Locked release does not have enough value diversity for entity cases")

    return ValuePools(
        companies=companies,
        years=years,
        year_pairs=year_pairs,
        quarters=quarters,
        dates=dates,
        metrics=metrics,
        statement_families=statement_families,
    )


_T = TypeVar("_T")


def _pick(pool: tuple[_T, ...], index: int) -> _T:
    return pool[index % len(pool)]


@dataclass(frozen=True)
class RenderedCase:
    question: str
    expected_company_codes: tuple[str, ...] = ()
    expected_periods: tuple[str, ...] = ()
    expected_metrics: tuple[str, ...] = ()
    expected_statement_types: tuple[str, ...] = ()
    expected_ambiguity: tuple[AmbiguityCode, ...] = ()


@dataclass(frozen=True)
class EntityTemplate:
    """One question pattern plus the pure function that renders it deterministically.

    `render(pools, index)` must be a pure function of `pools` and `index`: no
    randomness, no clock, no filesystem access beyond the already-loaded pools.
    """

    template_id: str
    category: TemplateCategory
    render: Callable[[ValuePools, int], RenderedCase]


def _lookup_ticker(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"Tra cứu {raw_metric} của {ticker} năm {year}.",
        expected_company_codes=(ticker,),
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
    )


def _lookup_name(pools: ValuePools, index: int) -> RenderedCase:
    ticker, name = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"Tra cứu {raw_metric} của {name} năm {year}.",
        expected_company_codes=(ticker,),
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
    )


def _compare_years(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year_a, year_b = _pick(pools.year_pairs, index)
    return RenderedCase(
        question=f"So sánh {raw_metric} của {ticker} giữa năm {year_a} và năm {year_b}.",
        expected_company_codes=(ticker,),
        expected_periods=tuple(sorted({year_a, year_b})),
        expected_metrics=(canonical_metric,),
    )


def _growth_years(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year_a, year_b = _pick(pools.year_pairs, index)
    return RenderedCase(
        question=(
            f"Tính tốc độ tăng trưởng {raw_metric} của {ticker} từ năm {year_a} đến năm {year_b}."
        ),
        expected_company_codes=(ticker,),
        expected_periods=tuple(sorted({year_a, year_b})),
        expected_metrics=(canonical_metric,),
    )


def _quarter_lookup(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    quarter_period = _pick(pools.quarters, index)
    year_str, quarter_str = quarter_period.split("-Q")
    roman = _QUARTER_ROMAN[int(quarter_str) - 1]
    return RenderedCase(
        question=f"{raw_metric} của {ticker} quý {roman} năm {year_str} là bao nhiêu?",
        expected_company_codes=(ticker,),
        expected_periods=(quarter_period,),
        expected_metrics=(canonical_metric,),
    )


def _date_lookup(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    iso_date = _pick(pools.dates, index)
    day, month, year = datetime.date.fromisoformat(iso_date).strftime("%d/%m/%Y").split("/")
    return RenderedCase(
        question=f"Tra cứu {raw_metric} của {ticker} tại ngày {day}/{month}/{year}.",
        expected_company_codes=(ticker,),
        expected_periods=(iso_date,),
        expected_metrics=(canonical_metric,),
    )


def _statement_lookup(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    family = _pick(pools.statement_families, index)
    phrase = _STATEMENT_FAMILIES[family][0]
    return RenderedCase(
        question=f"Tra cứu {raw_metric} trên {phrase} của {ticker} năm {year}.",
        expected_company_codes=(ticker,),
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
        expected_statement_types=(family,),
    )


def _missing_company(pools: ValuePools, index: int) -> RenderedCase:
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"Tra cứu {raw_metric} năm {year} là bao nhiêu?",
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
        expected_ambiguity=("company_missing",),
    )


def _two_companies(pools: ValuePools, index: int) -> RenderedCase:
    ticker_a, _ = pools.companies[index % len(pools.companies)]
    ticker_b, _ = pools.companies[(index + 1) % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"So sánh {raw_metric} giữa {ticker_a} và {ticker_b} năm {year}.",
        expected_company_codes=tuple(sorted({ticker_a, ticker_b})),
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
    )


def _company_conflict(pools: ValuePools, index: int) -> RenderedCase:
    ticker_named, name = pools.companies[index % len(pools.companies)]
    ticker_explicit, _ = pools.companies[(index + 1) % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"Tra cứu {raw_metric} của {name} (mã CK: {ticker_explicit}) năm {year}.",
        expected_periods=(str(year),),
        expected_metrics=(canonical_metric,),
        expected_ambiguity=("company_conflict",),
    )


def _relative_period(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    return RenderedCase(
        question=f"{raw_metric} năm nay của {ticker} là bao nhiêu?",
        expected_company_codes=(ticker,),
        expected_metrics=(canonical_metric,),
        expected_ambiguity=("period_relative_unresolved",),
    )


def _incomplete_quarter(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    quarter_period = _pick(pools.quarters, index)
    roman = _QUARTER_ROMAN[int(quarter_period.split("-Q")[1]) - 1]
    return RenderedCase(
        question=f"{raw_metric} của {ticker} quý {roman} là bao nhiêu?",
        expected_company_codes=(ticker,),
        expected_metrics=(canonical_metric,),
        expected_ambiguity=("period_incomplete",),
    )


def _unknown_metric(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    year = _pick(pools.years, index)
    return RenderedCase(
        question=f"Tra cứu tổng lợi thế cạnh tranh của {ticker} năm {year}.",
        expected_company_codes=(ticker,),
        expected_periods=(str(year),),
        expected_ambiguity=("metric_unknown",),
    )


def _missing_period(pools: ValuePools, index: int) -> RenderedCase:
    ticker, _ = pools.companies[index % len(pools.companies)]
    raw_metric, canonical_metric = _pick(pools.metrics, index)
    return RenderedCase(
        question=f"{raw_metric} của {ticker} là bao nhiêu?",
        expected_company_codes=(ticker,),
        expected_metrics=(canonical_metric,),
        expected_ambiguity=("period_missing",),
    )


TEMPLATE_INVENTORY: tuple[EntityTemplate, ...] = (
    EntityTemplate("lookup_ticker", "lookup", _lookup_ticker),
    EntityTemplate("lookup_name", "lookup", _lookup_name),
    EntityTemplate("compare_years", "compare", _compare_years),
    EntityTemplate("growth_years", "growth", _growth_years),
    EntityTemplate("quarter_lookup", "lookup", _quarter_lookup),
    EntityTemplate("date_lookup", "lookup", _date_lookup),
    EntityTemplate("statement_lookup", "lookup", _statement_lookup),
    EntityTemplate("missing_company", "adversarial", _missing_company),
    EntityTemplate("two_companies", "adversarial", _two_companies),
    EntityTemplate("company_conflict", "adversarial", _company_conflict),
    EntityTemplate("relative_period", "adversarial", _relative_period),
    EntityTemplate("incomplete_quarter", "adversarial", _incomplete_quarter),
    EntityTemplate("unknown_metric", "adversarial", _unknown_metric),
    EntityTemplate("missing_period", "adversarial", _missing_period),
)


def _case_id(
    template_id: str,
    question: str,
    expected: Mapping[str, tuple[str, ...]],
) -> str:
    payload = {
        "contract_version": "entity-case-v1",
        "template_id": template_id,
        "question": question,
        "expected": {key: list(value) for key, value in sorted(expected.items())},
    }
    return "ecase_" + sha256_bytes(canonical_json_bytes(payload))


def generate_entity_cases(release: ResolvedRetrievalRelease) -> tuple[EntityCase, ...]:
    """Generate the full deterministic entity-case set from the locked release.

    One case per (template, company) pair — every template cycles through
    the full company pool, drawing period/metric/statement values by
    deterministic modulo indexing over their own pools. No randomness.
    """
    pools = _load_value_pools(release)
    cases: list[EntityCase] = []
    for template in TEMPLATE_INVENTORY:
        for index in range(len(pools.companies)):
            rendered = template.render(pools, index)
            expected = {
                "company_codes": rendered.expected_company_codes,
                "periods": rendered.expected_periods,
                "metrics": rendered.expected_metrics,
                "statement_types": rendered.expected_statement_types,
                "ambiguity": rendered.expected_ambiguity,
            }
            case_id = _case_id(template.template_id, rendered.question, expected)
            cases.append(
                EntityCase(
                    case_id=case_id,
                    template_id=template.template_id,
                    category=template.category,
                    question=rendered.question,
                    expected_company_codes=rendered.expected_company_codes,
                    expected_periods=rendered.expected_periods,
                    expected_metrics=rendered.expected_metrics,
                    expected_statement_types=rendered.expected_statement_types,
                    expected_ambiguity=rendered.expected_ambiguity,
                )
            )
    ordered = tuple(sorted(cases, key=lambda case: case.case_id))
    if len({case.case_id for case in ordered}) != len(ordered):
        raise ValueError("Generated entity cases contain duplicate case_id values")
    return ordered


def entity_case_set_sha256(cases: tuple[EntityCase, ...]) -> str:
    """Return one fingerprint identifying the entire generated case set."""
    payload = [case.model_dump(mode="json") for case in cases]
    return sha256_bytes(canonical_json_bytes(payload))


def write_entity_cases(cases: tuple[EntityCase, ...], path: Path) -> None:
    """Publish the case set as sorted JSONL, one case per line."""
    lines = (case.model_dump_json() for case in cases)
    write_text_atomic(path, "\n".join(lines) + "\n")


def load_entity_cases(path: Path) -> tuple[EntityCase, ...]:
    """Load a previously generated, immutable entity-case JSONL file."""
    cases = tuple(
        EntityCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Loaded entity cases contain duplicate case_id values")
    return cases
