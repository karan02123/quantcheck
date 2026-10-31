"""Tests for scorers — edge cases: missing answer, malformed output, formatting."""

from __future__ import annotations

import pytest

from quantcheck.scoring import (
    get_scorer,
    multiple_choice_scorer,
    numeric_answer_scorer,
    instruction_keyword_scorer,
    exact_match_scorer,
)


# --------------------------- multiple choice --------------------------------- #
@pytest.mark.parametrize(
    "resp,gold,expected",
    [
        ("The answer is B.", "B", 1),
        ("answer: c", "C", 1),
        ("I think the correct option is (D)", "D", 1),
        ("A) because ...", "A", 1),
        ("The answer is B.", "A", 0),
        ("", "A", 0),                       # no answer at all
        ("I cannot decide.", "A", 0),       # no letter
        ("blah blah", "B", 0),
    ],
)
def test_multiple_choice(resp, gold, expected):
    assert multiple_choice_scorer(resp, gold) == expected


def test_multiple_choice_prefers_answer_cue_over_stray_letter():
    # A stray "A" early, real answer cue later -> cue wins.
    resp = "Option A looks tempting, but the answer is C."
    assert multiple_choice_scorer(resp, "C") == 1


# ------------------------------- numeric ------------------------------------- #
@pytest.mark.parametrize(
    "resp,gold,expected",
    [
        ("The answer is 42", "42", 1),
        ("= 42.0", "42", 1),
        ("Total: $1,234", "1234", 1),
        ("So we get 7 apples. #### 7", "7", 1),
        ("First 3, then 4, the result is 12", "12", 1),
        ("the answer is 41", "42", 0),
        ("no numbers here", "42", 0),
        ("", "42", 0),
        ("-5 degrees", "-5", 1),
    ],
)
def test_numeric(resp, gold, expected):
    assert numeric_answer_scorer(resp, gold) == expected


def test_numeric_takes_last_number_without_cue():
    assert numeric_answer_scorer("steps: 2, 3, 5", "5") == 1


def test_numeric_rejects_nonnumeric_gold():
    with pytest.raises(ValueError):
        numeric_answer_scorer("42", "forty-two")


# --- regression: chain-of-thought answers must not be stolen by intermediate --- #
# arithmetic. The first version of this scorer used a single regex with a bare
# "=" cue and took the FIRST match, so "2*5 = 10 ... 10/2 = 5" scored the
# working value 10 instead of the answer 5. Caught by the GSM8K template gate.
@pytest.mark.parametrize(
    "resp,gold",
    [
        ("2 children * 5 changes = 10 changes. Wife does 10 / 2 = 5 diapers.", "5"),
        ("4 rows * 18 = 72 seats. Admins took 72/4 = 18. Remaining = 36", "36"),
        ("5 * 4 = 20 legs. 2 * 4 = 8 legs. 10 * 2 = 20. Total = 48", "48"),
        ("Slow lane is 60 / 2 = 30 mph. Time = 480 / 30 = 16 hours", "16"),
        ("Available = 5000 - 3755 = 1245. Boxes = 1245 / 15 = 83", "83"),
    ],
)
def test_numeric_uses_last_cue_not_first(resp, gold):
    """The final result comes last in a chain of thought; earlier '=' are steps."""
    assert numeric_answer_scorer(resp, gold) == 1


def test_numeric_hash_delimiter_beats_later_equals_noise():
    """'#### n' is the instructed format and outranks a trailing '=' step."""
    assert numeric_answer_scorer("#### 52\nCheck: 47 + 52 = 99", "52") == 1
    assert numeric_answer_scorer("47+52+57 = 156. Avg = 52.\n#### 52", "52") == 1


@pytest.mark.parametrize(
    "resp,gold",
    [
        ("The average is 50.33.\n#### 50.33", "52"),   # model simply got it wrong
        ("Jordan changes 10 per day. #### 10", "5"),
        ("2 * 5 = 10 changes total.", "5"),             # answer never stated
    ],
)
def test_numeric_still_scores_wrong_answers_zero(resp, gold):
    """The last-match fix must not inflate scores for genuinely wrong answers."""
    assert numeric_answer_scorer(resp, gold) == 0


# ----------------------------- instruction ----------------------------------- #
@pytest.mark.parametrize(
    "resp,gold,expected",
    [
        ("I love bananas", "banana", 1),
        ("Paris is the capital of France", "paris|france", 1),
        ("Paris only", "paris|france", 0),
        ("nothing relevant", "banana", 0),
    ],
)
def test_instruction(resp, gold, expected):
    assert instruction_keyword_scorer(resp, gold) == expected


# ----------------------------- exact match ----------------------------------- #
def test_exact_match_normalizes_whitespace_and_case():
    assert exact_match_scorer("The   Eiffel   Tower", "eiffel tower") == 1
    assert exact_match_scorer("something else", "eiffel tower") == 0


# ------------------------------- lookup -------------------------------------- #
def test_get_scorer_unknown_raises():
    with pytest.raises(ValueError):
        get_scorer("does_not_exist")


def test_scorers_always_return_int_0_or_1():
    for name in ("multiple_choice", "numeric", "instruction", "exact_match"):
        s = get_scorer(name)
        gold = "1" if name == "numeric" else "A" if name == "multiple_choice" else "x"
        v = s("garbage response with no signal", gold)
        assert v in (0, 1) and isinstance(v, int)
