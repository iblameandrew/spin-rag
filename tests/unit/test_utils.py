import pytest

from spin_rag import SpinType, clean_llm_output, cosine_similarity, parse_spin


def test_cosine_identical_and_orthogonal():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_norm():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 1], [0, 0]) == 0.0


def test_clean_llm_output_strips_fences_and_quotes():
    assert clean_llm_output(None) == ""
    assert clean_llm_output(12) == "12"
    assert clean_llm_output("  hello  ") == "hello"
    assert clean_llm_output("```TOP```") == "TOP"
    assert clean_llm_output('"wrapped"') == "wrapped"
    assert clean_llm_output("'wrapped'") == "wrapped"


def test_parse_spin_regex_not_substring():
    # First whole-word match wins (same as the production classifier).
    assert parse_spin("Not a TOP, it is BOTTOM") == SpinType.TOP
    assert parse_spin("NOTATOP it is BOTTOM") == SpinType.BOTTOM
    assert parse_spin("yes TOP") == SpinType.TOP
    assert parse_spin("left") == SpinType.LEFT
    assert parse_spin("RIGHT.") == SpinType.RIGHT
    assert parse_spin("no label here") == SpinType.TOP
    assert parse_spin("") == SpinType.TOP
