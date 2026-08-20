"""Fuzzy character-similarity and metric-alias row retrieval branches.

Design V2 §7 asks for two more signals besides BM25/dense in row fusion:
character-level similarity (catches OCR noise, whitespace, and label
variants BM25's exact-token matching misses) and a metric-alias dictionary
bonus (catches a row whose canonical metric the question names by a known
synonym, even when the raw label shares no tokens with the question).

Both branches scan only the row documents belonging to this question's own
candidate tables -- never the whole corpus. ADR 0004 rejected exactly this
kind of matching when left unscoped ("Option B"): with no scoping, several
near-matching rows makes picking one a guess. Scoped to a handful of already
table-retrieval-validated tables, each branch instead just contributes one
more ranked candidate list into fusion; the eventual cell is still decided
by rule/LLM selection plus compile-time verification, never by this score
alone (plan.md §7: "Alias/dictionary chỉ đóng góp vào score, không quyết
định query có được xử lý hay không" -- the same principle applies to fuzzy).
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import Field, field_validator

from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.normalization.metrics import METRIC_ALIASES
from financial_report_qa.retrieval.contracts import TableId, _FrozenModel
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_index import RowBM25Index

# Longest-alias-first so "Chi phí bán hàng và QLDN" matches whole before the
# shorter "Chi phí bán hàng" it contains.
_ALIASES_BY_LENGTH: tuple[tuple[str, str], ...] = tuple(
    sorted(METRIC_ALIASES.items(), key=lambda item: -len(item[0]))
)


@dataclass(frozen=True)
class _Hit:
    score: float
    metadata: RowMetadata
    snippet: str
    row_idx: int
    table_id: str
    row_id: str


class RowLexicalCandidate(_FrozenModel):
    """One row candidate scored by a non-learned lexical signal."""

    row_id: str
    table_id: TableId
    row_idx: int
    score: float
    rank: int = Field(ge=1)
    metadata: RowMetadata
    snippet: str

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("score must be within [0, 1]")
        return value


def _query_alias_canonicals(query: str) -> frozenset[str]:
    """Canonical metric ids named verbatim, as a whole known alias phrase,
    somewhere in the question text. Whole-phrase substring lookup only --
    never per-token or fuzzy -- mirrors `normalize_metric`'s own
    no-partial-match contract, so this cannot turn an unrelated question
    into a guessed alias hit."""
    key = normalized_key(query)
    if not key:
        return frozenset()
    return frozenset(
        canonical for alias_key, canonical in _ALIASES_BY_LENGTH if alias_key and alias_key in key
    )


class RowFuzzyRetrievalService:
    """Character-similarity branch: `SequenceMatcher` ratio between the
    question text and each candidate row's raw label."""

    def __init__(self, index: RowBM25Index) -> None:
        self._index = index

    def retrieve_rows(
        self, query: str, *, candidate_table_ids: Sequence[str], k: int = 10
    ) -> tuple[RowLexicalCandidate, ...]:
        if k < 1:
            raise ValueError("k must be positive")
        if not candidate_table_ids:
            return ()

        query_key = normalized_key(query)
        if not query_key:
            return ()

        candidate_table_set = frozenset(candidate_table_ids)
        scored: list[_Hit] = []
        for document in self._index.documents:
            if document.table_id not in candidate_table_set:
                continue
            label = document.metadata.row_label_raw
            if not label:
                continue
            ratio = difflib.SequenceMatcher(None, query_key, normalized_key(label)).ratio()
            if ratio <= 0.0:
                continue
            scored.append(
                _Hit(
                    score=ratio,
                    metadata=document.metadata,
                    snippet=document.text[:500],
                    row_idx=document.row_idx,
                    table_id=document.table_id,
                    row_id=document.row_id,
                )
            )

        ranked = sorted(scored, key=lambda hit: (-hit.score, hit.row_id))[:k]
        return tuple(
            RowLexicalCandidate(
                row_id=hit.row_id,
                table_id=hit.table_id,
                row_idx=hit.row_idx,
                score=hit.score,
                rank=rank,
                metadata=hit.metadata,
                snippet=hit.snippet,
            )
            for rank, hit in enumerate(ranked, start=1)
        )


class RowAliasRetrievalService:
    """Alias-dictionary branch: every row whose stored canonical metric was
    named verbatim as a known alias phrase in the question text."""

    def __init__(self, index: RowBM25Index) -> None:
        self._index = index

    def retrieve_rows(
        self, query: str, *, candidate_table_ids: Sequence[str], k: int = 10
    ) -> tuple[RowLexicalCandidate, ...]:
        if k < 1:
            raise ValueError("k must be positive")
        if not candidate_table_ids:
            return ()

        matched_canonicals = _query_alias_canonicals(query)
        if not matched_canonicals:
            return ()

        candidate_table_set = frozenset(candidate_table_ids)
        hits: list[_Hit] = []
        for document in self._index.documents:
            if document.table_id not in candidate_table_set:
                continue
            if document.metadata.row_label_canonical in matched_canonicals:
                hits.append(
                    _Hit(
                        score=1.0,
                        metadata=document.metadata,
                        snippet=document.text[:500],
                        row_idx=document.row_idx,
                        table_id=document.table_id,
                        row_id=document.row_id,
                    )
                )

        # Every hit is an exact dictionary match -- there is no ranking
        # signal between them, so order is fixed by row_id for determinism.
        ordered = sorted(hits, key=lambda hit: hit.row_id)[:k]
        return tuple(
            RowLexicalCandidate(
                row_id=hit.row_id,
                table_id=hit.table_id,
                row_idx=hit.row_idx,
                score=hit.score,
                rank=rank,
                metadata=hit.metadata,
                snippet=hit.snippet,
            )
            for rank, hit in enumerate(ordered, start=1)
        )
