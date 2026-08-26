"""Strict local validation of a generated `ProgramDecision` file.

Kaggle side (notebooks/kaggle_program_decisions_qwen3_4b.ipynb) only does a
light check because the repo package cannot be installed there
(requires-python <3.12). This script is the opposite: it runs inside the repo,
loads the decisions through the real `load_program_decisions` loader, replays
the full AST guard from `execution.masked_program`, cross-checks every
question_id against the payload batches, and reports distributions.

Exit code 0 only when every answerable payload question (>=1 candidate) has a
valid decision. Questions with 0 candidates can never have one -- the contract
requires `cells` to be non-empty and the pipeline records them as
`no_cell_candidates` failures -- so they are reported separately instead of
failing the gate.

Usage::

    uv run python scripts/validate_program_decisions.py \
        --batches-dir artifacts/batches/program-full \
        --decisions data/decisions/program-full.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import (
    PlanningArtifactError,
    ProgramGuardError,
)
from financial_report_qa.execution.masked_program import (
    parse_program,
    substitute_placeholders,
)
from financial_report_qa.execution.program_contracts import (
    ProgramDecision,
    UseClaim,
)
from financial_report_qa.planning.program_decisions import load_program_decisions

_BINOP_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_DECISION_FIELDS: tuple[str, ...] = ("question_id", "cells", "program", "uses", "scale")


def _operator_signature(program: str) -> str | None:
    """Classify one guarded program by the operators it uses, e.g. `-,/`.

    Returns `None` when the program does not parse (already reported as an
    error; never raise from a reporting helper).
    """
    try:
        tree = ast.parse(substitute_placeholders(program), mode="eval")
    except SyntaxError:
        return None
    parts: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            symbol = _BINOP_SYMBOLS.get(type(node.op))
            if symbol is not None:
                parts.add(symbol)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            parts.add("neg")
        elif isinstance(node, ast.Call):
            parts.add("abs")
    if not parts:
        return "direct"
    return ",".join(sorted(parts))


def _check_use_claims(uses: tuple[UseClaim, ...], cells_count: int) -> list[str]:
    """Every cell position must carry exactly one use claim (verification B).

    `execution.bind_values` binds one `[NUM_i]` per `cells` entry, and
    `verification.use_checks.check_use_bindings` fails on both a missing claim
    and a leftover claim -- so a decision that skips either is unusable
    downstream even though the pydantic contract alone would accept it.
    """
    claimed = [claim.num for claim in uses]
    problems: list[str] = []
    duplicated = sorted({num for num in claimed if claimed.count(num) > 1})
    if duplicated:
        problems.append(f"duplicate use claims for nums {duplicated}")
    expected = set(range(cells_count))
    missing = sorted(expected - set(claimed))
    if missing:
        problems.append(f"use claims missing for cell positions {missing}")
    extra = sorted(set(claimed) - expected)
    if extra:
        problems.append(f"use claims reference nonexistent cell positions {extra}")
    return problems


def _validate_one(decision: ProgramDecision, candidate_counts: dict[int, int]) -> list[str]:
    """Run every non-pydantic check on one decision; empty list means valid."""
    errors: list[str] = []
    count = candidate_counts.get(decision.question_id)
    if count is None:
        errors.append("question_id not present in payload batches")
        return errors
    out_of_range = sorted({index for index in decision.cells if index >= count})
    if out_of_range:
        errors.append(
            f"candidate_index_out_of_range: {out_of_range} "
            f"is not below the {count} candidates of this question"
        )
    try:
        ast.parse(substitute_placeholders(decision.program), mode="eval")
    except SyntaxError as error:
        errors.append(f"program does not parse: {error}")
        return errors
    try:
        # Full guard replay: length/node budgets, no literal, allowed nodes
        # and operators only, `[NUM_i]` indices below len(cells).
        parse_program(decision.program, value_count=len(decision.cells))
    except ProgramGuardError as error:
        errors.append(f"program guard: {error}")
    errors.extend(_check_use_claims(decision.uses, len(decision.cells)))
    return errors


def _load_payload_candidate_counts(batches_dir: Path) -> dict[int, int]:
    """Map every payload `question_id` to its number of cell candidates."""
    batch_paths = sorted(batches_dir.glob("batch_*.jsonl"))
    if not batch_paths:
        raise SystemExit(f"no batch_*.jsonl under {batches_dir}")
    counts: dict[int, int] = {}
    for path in batch_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}: invalid JSON on line {line_number}") from error
                question_id = payload["question_id"]
                if question_id in counts:
                    raise SystemExit(f"{path}: duplicate question_id {question_id}")
                counts[question_id] = len(payload["candidates"])
    return counts


def _scan_raw_lines(
    decisions_path: Path, candidate_counts: dict[int, int]
) -> tuple[dict[int, ProgramDecision], list[tuple[int, str, list[str]]]]:
    """Permissive first pass: collect every bad line instead of stopping.

    Returns the decisions that parsed cleanly, plus `(line_number, raw,
    errors)` triples for the rest. `load_program_decisions` is the
    authoritative gate and stops at the first bad line; this scan exists so
    the operator sees ALL problems in one run.
    """
    parsed: dict[int, ProgramDecision] = {}
    invalid: list[tuple[int, str, list[str]]] = []
    text = decisions_path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            invalid.append((line_number, line, [f"invalid JSON: {error}"]))
            continue
        if not isinstance(payload, dict):
            invalid.append((line_number, line, ["line is not a JSON object"]))
            continue
        field_errors = [
            f"unexpected key {key!r}" for key in sorted(set(payload) - set(_DECISION_FIELDS))
        ]
        field_errors += [f"missing key {key!r}" for key in _DECISION_FIELDS if key not in payload]
        if field_errors:
            invalid.append((line_number, line, field_errors))
            continue
        try:
            decision = ProgramDecision.model_validate(payload)
        except ValidationError as error:
            invalid.append((line_number, line, [str(error)]))
            continue
        errors = _validate_one(decision, candidate_counts)
        if errors:
            invalid.append((line_number, line, errors))
        else:
            parsed[decision.question_id] = decision
    return parsed, invalid


def _print_distribution(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  (empty)")
        return
    for value, count in counter.most_common():
        print(f"  {value}: {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly validate a generated ProgramDecision JSONL file against "
            "the masked-PAL contracts and the payload batches."
        )
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path("artifacts/batches/program-full"),
        help="Directory holding the batch_*.jsonl payload files.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/decisions/program-full.jsonl"),
        help="Generated decision file to validate.",
    )
    args = parser.parse_args(argv)

    candidate_counts = _load_payload_candidate_counts(args.batches_dir)
    answerable = {qid for qid, count in candidate_counts.items() if count > 0}
    unanswerable = set(candidate_counts) - answerable
    print(
        f"payloads: {len(candidate_counts)} questions from {args.batches_dir} "
        f"({len(answerable)} answerable, {len(unanswerable)} with 0 candidates)"
    )

    if not args.decisions.is_file():
        print(f"FAIL: decision file not found: {args.decisions}")
        return 1

    try:
        load_program_decisions(args.decisions)
        loader_ok = True
    except PlanningArtifactError as error:
        loader_ok = False
        print(f"\nauthoritative loader rejected the file:\n  {error}")

    parsed, invalid = _scan_raw_lines(args.decisions, candidate_counts)

    print(f"\ntotal decisions: {len(parsed)} (of {len(candidate_counts)} payload questions)")

    if invalid:
        print(f"\ninvalid lines ({len(invalid)}):")
        for line_number, _raw, errors in invalid:
            for problem in errors:
                print(f"  line {line_number}: {problem}")
    else:
        print("\ninvalid lines: none")

    missing = sorted(answerable - set(parsed))
    extra = sorted(set(parsed) - set(candidate_counts))
    if missing:
        print(f"\nmissing decisions for question_ids ({len(missing)}): {missing[:20]}...")
    if unanswerable:
        print(
            f"\nunanswerable payloads, 0 candidates, no decision possible "
            f"({len(unanswerable)}): handled as no_cell_candidates failures"
        )
    if extra:
        print(f"\nextra decisions for unknown question_ids ({len(extra)}): {extra[:20]}...")
    if not missing and not extra:
        print("\nquestion_id coverage: every answerable payload has a decision")

    _print_distribution("scale distribution", Counter(d.scale for d in parsed.values()))
    _print_distribution(
        "programs by operator shape",
        Counter(
            sig
            for sig in (_operator_signature(d.program) for d in parsed.values())
            if sig is not None
        ),
    )

    ok = loader_ok and not invalid and not missing and not extra
    print(f"\n{'PASS' if ok else 'FAIL'}: {args.decisions}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
