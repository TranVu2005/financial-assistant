"""Tests for the Day 25 LLM cell-grounding tier.

Drives a real `LLMClient` over `httpx.MockTransport` (the Day 17 pattern) so
the client/parser wiring is exercised end to end without a socket.
"""

from __future__ import annotations

import json

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.planning.llm_cell_grounding import choose_column_label, choose_row_label
from financial_report_qa.planning.llm_client import LLMClient

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen2.5:7b",
    timeout_seconds=5.0,
    max_output_tokens=32,
    temperature=0.0,
    context_length=32768,
    json_schema_constrained=True,
)
_LABELS = ("Doanh thu thuần", "Giá vốn hàng bán", "Lợi nhuận sau thuế")
_QUESTION = "Lợi nhuận sau thuế của ACB năm 2023 là bao nhiêu?"


def _client(handler: object) -> LLMClient:
    return LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)  # type: ignore[arg-type]


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def test_choose_row_label_returns_the_selected_label() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 3})))

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) == "Lợi nhuận sau thuế"


def test_choose_row_label_accepts_a_bare_integer_reply() -> None:
    """The schema should prevent this, but a small model occasionally ignores
    it -- a bare integer must still be usable rather than wasting the call."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("2"))

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) == "Giá vốn hàng bán"


def test_choose_row_label_returns_none_on_explicit_decline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 0})))

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) is None


def test_choose_row_label_returns_none_on_out_of_range_index() -> None:
    """Never clamp to the nearest valid row -- an out-of-range answer means
    the model did not actually choose, so this must abstain."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 99})))

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) is None


def test_choose_row_label_returns_none_on_unparseable_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("tôi không chắc"))

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) is None


def test_choose_row_label_returns_none_when_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    assert choose_row_label(_QUESTION, _LABELS, client=_client(handler)) is None


def test_choose_row_label_returns_none_without_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the model when there is nothing to choose from")

    assert choose_row_label(_QUESTION, (), client=_client(handler)) is None


def test_choose_row_label_deduplicates_labels_before_numbering() -> None:
    """The same label repeats across periods/columns in a candidate frame;
    numbering must be over distinct labels so the index the model returns
    maps back to what it was shown."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 2})))

    labels = ("Doanh thu thuần", "Doanh thu thuần", "Giá vốn hàng bán")
    assert choose_row_label(_QUESTION, labels, client=_client(handler)) == "Giá vốn hàng bán"
    user_message = captured["body"]["messages"][1]["content"]  # type: ignore[index]
    assert user_message.count("Doanh thu thuần") == 1


def test_choose_row_label_only_ever_returns_a_real_candidate() -> None:
    """Structural guarantee: whatever the model says, the result is either
    None or a member of the supplied list -- it can never fabricate a label."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 1})))

    result = choose_row_label(_QUESTION, _LABELS, client=_client(handler))
    assert result in _LABELS


_COLUMNS = ("Số phải nộpđầu năm", "Số phải nộptrong năm", "Số cuối nămVND", "Số phải nộpcuối năm")
_COLUMN_QUESTION = (
    "Số dư thuế giá trị gia tăng phải nộp cuối năm 2025 của PC1 là bao nhiêu triệu đồng?"
)


def test_choose_column_label_returns_the_selected_column() -> None:
    """A row alone does not identify a cell. Measured on the 117 questions the
    locator still abstains on: only 5 name a column deterministically, so the
    column has to be chosen by reading the real headers against the question --
    the same "index into real strings" shape the row tier already proves out."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 4})))

    chosen = choose_column_label(
        _COLUMN_QUESTION, "Thuế giá trị gia tăng", _COLUMNS, client=_client(handler)
    )
    assert chosen == "Số phải nộpcuối năm"


def test_choose_column_label_returns_none_when_the_model_declines() -> None:
    """Choice 0 means "none of these"; it must abstain rather than clamp to a
    column, or the answer silently comes from an unrelated amount."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 0})))

    assert (
        choose_column_label(
            _COLUMN_QUESTION, "Thuế giá trị gia tăng", _COLUMNS, client=_client(handler)
        )
        is None
    )


def test_choose_column_label_returns_none_without_candidate_columns() -> None:
    """4.4% of tables carry no column label at all; that is an abstain, not a
    reason to call the model."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("model must not be called without candidates")

    assert choose_column_label(_COLUMN_QUESTION, "Thuế", (), client=_client(handler)) is None


def test_choose_column_label_prompt_shows_the_row_being_disambiguated() -> None:
    """The row is the context that makes the column meaningful: "Số cuối năm"
    versus "Số phải nộp cuối năm" is only decidable knowing which line is read."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][-1]["content"])
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 4})))

    choose_column_label(
        _COLUMN_QUESTION, "Thuế giá trị gia tăng", _COLUMNS, client=_client(handler)
    )
    assert "Thuế giá trị gia tăng" in seen[0]
    assert "Số phải nộpcuối năm" in seen[0]
