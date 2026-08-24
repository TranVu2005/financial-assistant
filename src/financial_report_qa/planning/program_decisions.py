"""Read the offline masked-PAL decision file.

One JSONL line per question. `ProgramDecision` forbids extra fields, so a
line that smuggles in a numeric value is rejected rather than ignored -- that
rejection is what makes N7 enforceable on a file someone edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import PlanningArtifactError
from financial_report_qa.execution.program_contracts import ProgramDecision


def load_program_decisions(path: Path) -> dict[int, ProgramDecision]:
    """Load every decision, keyed by `question_id`, in file order."""
    if not path.is_file():
        raise PlanningArtifactError(f"program decision file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        # A hand-edited decision file saved in a legacy Windows encoding must
        # fail like every other corrupt artifact, not as a bare codec error.
        raise PlanningArtifactError(f"{path}: not valid UTF-8") from error
    decisions: dict[int, ProgramDecision] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlanningArtifactError(f"{path}: invalid JSON on line {number}") from error
        try:
            decision = ProgramDecision.model_validate(payload)
        except ValidationError as error:
            raise PlanningArtifactError(f"{path}: invalid decision on line {number}") from error
        if decision.question_id in decisions:
            raise PlanningArtifactError(
                f"{path}: duplicate question_id {decision.question_id} on line {number}"
            )
        decisions[decision.question_id] = decision
    return decisions
