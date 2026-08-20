"""plan.md §19: curate a ~100-150 question dev benchmark from a real full
export run, deliberately covering the pipeline's named failure categories
(metric_not_found, cell_ambiguous, period_unresolved, unit_missing,
llm_plan_invalid, no_candidate_tables, llm_invalid_json) plus a slice of
already-answered questions as a regression control group.

This is a *selection* script, not an annotation one: it produces
`data/qa/dev-benchmark-v1.jsonl` with `gold_*` fields left null and
`needs_annotation: true`. Real gold_table/gold_rows/gold_columns/gold_values
require reading the source financial statements by hand -- see the
provenance doc for why that is a deliberate follow-up, not something this
script fabricates.

Usage:
    python scripts/build_dev_benchmark_v1.py \
        --report <path to a submission-export-*.json full run> \
        --out data/qa/dev-benchmark-v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# plan.md §19's own named categories, in the order they should be read.
# "answered" is not a failure category -- it is included as a regression
# control group, since a dev benchmark that only contains failures cannot
# detect a change that breaks a previously-working question.
CATEGORY_QUOTAS: dict[str, int] = {
    "answered": 25,
    "metric_not_found": 20,
    "llm_plan_invalid": 20,
    "cell_ambiguous": 20,
    "period_unresolved": 20,
    "no_candidate_tables": 15,
    "unit_missing": 12,
    "llm_invalid_json": 12,
}

SAMPLE_SEED = 19  # plan.md section number, fixed for reproducibility.


def _category_of(outcome: dict) -> str | None:
    if outcome.get("status") == "answered":
        return "answered"
    return outcome.get("code")


def build(report_path: Path, out_path: Path) -> dict[str, int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fingerprint = report["dataset_fingerprint"]

    buckets: dict[str, list[dict]] = {category: [] for category in CATEGORY_QUOTAS}
    for outcome in report["outcomes"]:
        category = _category_of(outcome)
        if category in buckets:
            buckets[category].append(outcome)

    rng = random.Random(SAMPLE_SEED)
    selected: list[dict] = []
    achieved: dict[str, int] = {}
    for category, quota in CATEGORY_QUOTAS.items():
        pool = sorted(buckets[category], key=lambda o: o["id"])
        rng.shuffle(pool)
        picked = sorted(pool[:quota], key=lambda o: o["id"])
        achieved[category] = len(picked)
        for outcome in picked:
            selected.append(
                {
                    "question_id": f"devq_{fingerprint[:12]}_{outcome['id']:04d}",
                    "vifinqa_id": outcome["id"],
                    "question": outcome["question"],
                    "error_category": category,
                    "source_status": outcome.get("status"),
                    "source_stage": outcome.get("stage"),
                    "source_code": outcome.get("code"),
                    "source_plan_source": outcome.get("plan_source"),
                    "gold_table": None,
                    "gold_rows": None,
                    "gold_columns": None,
                    "gold_values": None,
                    "operation": None,
                    "needs_annotation": True,
                }
            )

    selected.sort(key=lambda record: record["vifinqa_id"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return achieved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    achieved = build(args.report, args.out)
    total = sum(achieved.values())
    print(f"Wrote {total} questions to {args.out}")
    for category, quota in CATEGORY_QUOTAS.items():
        got = achieved[category]
        flag = "" if got == quota else f"  (wanted {quota})"
        print(f"  {category}: {got}{flag}")


if __name__ == "__main__":
    main()
