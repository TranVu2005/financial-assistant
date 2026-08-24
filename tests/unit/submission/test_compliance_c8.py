import pytest

from financial_report_qa.submission.compliance import (
    _strip_structural_tokens,
    check_program_literals,
)


def test_a_clean_program_passes() -> None:
    assert check_program_literals("([NUM_0] - [NUM_1]) / [NUM_1]") is None


def test_a_bare_lookup_passes() -> None:
    assert check_program_literals("[NUM_0]") is None


@pytest.mark.parametrize(
    "program", ["[NUM_0] * 100", "[NUM_0] + 0", "abs([NUM_0] - 1)", "round([NUM_0])"]
)
def test_a_literal_or_forbidden_call_is_reported(program: str) -> None:
    detail = check_program_literals(program)

    assert detail is not None


def test_the_scale_suffix_is_stripped_before_the_c4_literal_scan() -> None:
    query = 'df1[(df1.row_idx == 3)]["value"].iloc[0] * 100'

    assert "100" not in _strip_structural_tokens(query)


@pytest.mark.parametrize("suffix", [" * 100", " / 1000", " / 1000000", " / 1000000000"])
def test_every_scale_suffix_is_stripped(suffix: str) -> None:
    query = f'df1[(df1.row_idx == 3)]["value"].iloc[0]{suffix}'

    stripped = _strip_structural_tokens(query)

    assert not any(character.isdigit() for character in stripped.split("]")[-1])


def test_a_real_literal_elsewhere_is_still_visible_to_c4() -> None:
    query = 'df1[(df1.row_idx == 3)]["value"].iloc[0] - 4500'

    assert "4500" in _strip_structural_tokens(query)
