import json
from pathlib import Path

from financial_report_qa.retrieval.cli import write_sweep_report
from financial_report_qa.retrieval.sweep import SweepResult


def test_report_writes_both_json_and_markdown(tmp_path: Path) -> None:
    results = (SweepResult(k=5, f2=0.61, mrr5=0.52), SweepResult(k=10, f2=0.55, mrr5=0.52))

    json_path, markdown_path = write_sweep_report(results, 5, tmp_path / "day8-sweep")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["recommended_k"] == 5
    assert payload["results"] == [
        {"k": 5, "f2": 0.61, "mrr5": 0.52},
        {"k": 10, "f2": 0.55, "mrr5": 0.52},
    ]
    assert "| 5 |" in markdown_path.read_text(encoding="utf-8")
