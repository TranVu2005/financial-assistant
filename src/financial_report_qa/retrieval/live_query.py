"""Day 22 plan §1/§2 decision B: retrieval for a raw, never-before-seen
question.

Every prior evaluation harness (Day 8-21) scores BM25 against a question that
already has a hand-labeled `RetrievalFilters` (`GoldRetrievalQuestion.filters`)
-- none of them derive filters from question text. `to_retrieval_filters`
(Day 10, `planning/entity_contracts.py`) already does exactly that from
`parse_query_entities`'s output, dropping any field the parser itself flagged
as ambiguous. This module is the thin, previously-missing wire between the
two: raw text -> entities -> filters -> `RetrievalService.retrieve`.
"""

from __future__ import annotations

from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import TableId
from financial_report_qa.retrieval.service import RetrievalService


def retrieve_candidate_table_ids(
    question: str, service: RetrievalService, *, k: int = 10
) -> tuple[TableId, ...]:
    """Rank candidate tables for one raw question, in retrieval-rank order."""
    entities = parse_query_entities(question)
    filters = to_retrieval_filters(entities)
    trace = service.retrieve(question, filters=filters, k=k)
    return tuple(candidate.table_id for candidate in trace.results)
