"""Contract tests for Day 12 graph expansion and reranking."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.expansion_contracts import (
    PRE_REGISTERED_EXPANSION_GRID,
    ExpansionParams,
    RerankedCandidate,
    RerankFeatures,
    SeedCandidate,
)

TABLE_ID = "tbl_" + "a" * 64


def _metadata() -> TableMetadata:
    return TableMetadata(
        table_id=TABLE_ID,
        doc_id="doc-1",
        source_path="report.txt",
        line_start=1,
        line_end=2,
    )


def test_seed_candidate_keeps_the_common_seed_shape() -> None:
    candidate = SeedCandidate(
        table_id=TABLE_ID,
        rank=1,
        metadata=_metadata(),
        snippet="balance sheet",
    )

    assert candidate.rank == 1
    assert candidate.snippet == "balance sheet"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": -0.01}, "greater than or equal to 0"),
        ({"fan_out": 0}, "greater than 0"),
        ({"relations": ("same_document", "adjacent_period")}, "sorted and unique"),
        ({"relations": ("same_document", "same_document")}, "sorted and unique"),
    ],
)
def test_expansion_params_reject_invalid_or_noncanonical_values(
    kwargs: dict[str, float | int | tuple[str, ...]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ExpansionParams(**kwargs)  # type: ignore[arg-type]


def test_expansion_params_accepts_sorted_unique_relations() -> None:
    params = ExpansionParams(relations=("adjacent_period", "same_document"))

    assert params.relations == ("adjacent_period", "same_document")


def test_preregistered_grid_has_the_fixed_thirteen_experiments() -> None:
    assert len(PRE_REGISTERED_EXPANSION_GRID) == 13
    assert all(
        point.seed_depth == 50 and point.rrf_k == 60 and point.fan_out == 25
        for point in PRE_REGISTERED_EXPANSION_GRID
    )
    assert sum(point.alpha == 0 for point in PRE_REGISTERED_EXPANSION_GRID) == 1
    assert {
        (point.relations, point.alpha, point.expand_non_seeds)
        for point in PRE_REGISTERED_EXPANSION_GRID
        if point.alpha > 0
    } == {
        (relations, alpha, expand_non_seeds)
        for relations in (
            ("same_document",),
            (
                "adjacent_period",
                "explained_by_note",
                "same_document",
                "same_statement_type",
                "shared_metric",
            ),
        )
        for alpha in (0.25, 0.5, 1.0)
        for expand_non_seeds in (False, True)
    }


def test_reranked_candidate_preserves_explainable_features() -> None:
    features = RerankFeatures(
        seed_rrf=1 / 61,
        neighbor_rrf=1 / 62,
        supporting_seed_count=1,
        relations=("same_document",),
    )
    candidate = RerankedCandidate(
        table_id=TABLE_ID,
        rank=1,
        score=0.1,
        source="seed",
        features=features,
        contradiction_count=0,
        metadata=_metadata(),
        snippet="balance sheet",
    )

    assert candidate.features.supporting_seed_count == 1
    assert candidate.source == "seed"
