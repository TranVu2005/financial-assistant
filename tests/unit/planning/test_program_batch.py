import json
from pathlib import Path

import pytest

from financial_report_qa.core.errors import PlanningArtifactError
from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.program_decisions import load_program_decisions
from financial_report_qa.planning.row_choice_batch import build_program_batch_payload

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(index: int) -> CellCandidate:
    return CellCandidate(
        index=index,
        table_id=_TABLE_ID,
        company_code="VCB",
        row_idx=3,
        col_idx=index + 1,
        row_path="Doanh thu > Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path=f"Năm_{2022 + index}",
        period=2022 + index,
        unit="triệu VND",
    )


def _entities() -> QueryEntities:
    return QueryEntities(company_codes=("VCB",), periods=("2022", "2023"), question="Tăng trưởng?")


def test_payload_never_carries_a_value_or_a_score() -> None:
    payload = build_program_batch_payload(7, "Tăng trưởng?", _entities(), (_candidate(0),))

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "value" not in serialized
    assert "score" not in serialized


def test_payload_numbers_candidates_in_the_given_order() -> None:
    payload = build_program_batch_payload(
        7, "Tăng trưởng?", _entities(), (_candidate(0), _candidate(1))
    )

    assert [item["index"] for item in payload["candidates"]] == [0, 1]


def test_payload_carries_the_parsed_companies_and_periods() -> None:
    payload = build_program_batch_payload(7, "Tăng trưởng?", _entities(), (_candidate(0),))

    assert payload["question_id"] == 7
    assert payload["companies"] == ["VCB"]
    assert payload["periods"] == ["2022", "2023"]


def test_decisions_load_keyed_by_question_id(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": 7,
                "cells": [1, 0],
                "program": "[NUM_0] - [NUM_1]",
                "uses": [
                    {"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},
                    {"num": 1, "row": "Doanh thu thuần", "col": "Năm 2022"},
                ],
                "scale": "none",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = load_program_decisions(path)

    assert decisions[7].cells == (1, 0)
    assert decisions[7].uses[0].col == "Năm 2023"


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        "\n" + json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]"}) + "\n\n",
        encoding="utf-8",
    )

    assert list(load_program_decisions(path)) == [1]


def test_a_duplicate_question_id_is_rejected(tmp_path: Path) -> None:
    line = json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]"})
    path = tmp_path / "decisions.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")

    with pytest.raises(PlanningArtifactError, match="duplicate"):
        load_program_decisions(path)


def test_malformed_json_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(PlanningArtifactError, match="line 1"):
        load_program_decisions(path)


def test_a_decision_carrying_a_numeric_value_field_is_rejected(tmp_path: Path) -> None:
    # N7: file quyết định không được mang giá trị số. `extra="forbid"` là chốt.
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]", "value": 4500}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PlanningArtifactError):
        load_program_decisions(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PlanningArtifactError):
        load_program_decisions(tmp_path / "absent.jsonl")
