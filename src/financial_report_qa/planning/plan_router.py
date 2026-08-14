"""Day 17 plan router (§17.7): ADR 0006 decision A1.

The rule planner runs first; the LLM planner is only invoked when the rule
planner abstains. `route_plan` is the single point in the codebase allowed to
call `llm_planner.build_plan` — no other module should bypass the rule
planner and go straight to the LLM. This ordering is load-bearing: Day 16
already measured the rule planner at operation accuracy 1.0 and false-plan
rate 0.0 on 1,400 plan cases, so letting the LLM run first (or override a
successful rule-planner result) could only make those numbers worse, never
better.
"""

from __future__ import annotations

from typing import Literal

from financial_report_qa.planning import llm_planner, rule_planner
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.rule_planner import RulePlanResult
from financial_report_qa.retrieval.contracts import TableId, _FrozenModel

PlanSource = Literal["rule", "llm"]


class RoutedPlanResult(_FrozenModel):
    """A plan result tagged with which planner produced it."""

    result: RulePlanResult
    source: PlanSource


def route_plan(
    entities: QueryEntities,
    *,
    client: ChatCompletionClient,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
) -> RoutedPlanResult:
    """Try the rule planner; fall back to the LLM planner only on abstain."""
    rule_result = rule_planner.build_plan(
        entities, candidate_table_ids=candidate_table_ids, known_table_ids=known_table_ids
    )
    if rule_result.plan is not None:
        return RoutedPlanResult(result=rule_result, source="rule")

    llm_result = llm_planner.build_plan(
        entities.question,
        client=client,
        candidate_table_ids=candidate_table_ids,
        known_table_ids=known_table_ids,
    )
    return RoutedPlanResult(result=llm_result, source="llm")
