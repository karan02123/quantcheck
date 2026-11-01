"""Tests for the response-health validity diagnostic.

Why these exist
---------------
`assess_health` is the part of the report that tells a reader whether to trust
the verdict above it. A diagnostic that cries wolf is worse than none at all: it
teaches the reader to discount valid results.

That is not hypothetical. An earlier version warned that degradation events were
"candidate truncations rather than demonstrated wrong answers" whenever the
candidate response *looked* cut off. On the bundled knowledge suite the prompts
end with "Answer:", so models reply with the letter and explain afterwards --
the token budget truncates the explanation long after the answer is recorded.
The warning fired on 75 of 80 events and was wrong about every one of them. The
figure that actually mattered was 21: the events where the scorer could read no
answer at all.

These tests pin that distinction.
"""

from __future__ import annotations

from quantcheck.report import assess_health


def test_truncated_after_answering_is_not_a_validity_warning():
    """The real shape from the knowledge runs: answer first, then cut off.

    Every candidate response here states a letter and is then truncated. None
    is unparseable, so no effect-size warning should be raised, however high
    the raw truncation count climbs.
    """
    base = [" C\nExplanation: because the predicate"] * 20
    cand = [" A\nExplanation: because the predicate"] * 20
    health = assess_health(
        base, cand, [1] * 20, [0] * 20, scorers=["multiple_choice"] * 20
    )
    assert health.baseline.truncated == 20       # the raw count is still shown
    assert health.b_candidate_truncated == 20    # cut off inside b, reported
    assert health.b_candidate_unparseable == 0   # but all answers were readable
    assert not any("upper bound" in w for w in health.warnings)


def test_unparseable_candidate_answers_do_warn():
    """Cut off *before* answering is a genuine threat to the effect size."""
    base = [" C\nExplanation: because the predicate"] * 20
    cand = ["Let me work through each option in turn"] * 20
    health = assess_health(
        base, cand, [1] * 20, [0] * 20, scorers=["multiple_choice"] * 20
    )
    assert health.b_candidate_unparseable == 20
    assert any("upper bound" in w for w in health.warnings)


def test_warning_is_suppressed_rather_than_guessed_without_scorers():
    """Parseability cannot be determined without knowing the scorer, so the
    gated warning is withheld instead of approximated by truncation."""
    base = [" C\nExplanation: because the predicate"] * 20
    cand = ["Let me work through each option in turn"] * 20
    health = assess_health(base, cand, [1] * 20, [0] * 20)
    assert health.b_candidate_unparseable == 0
    assert not any("upper bound" in w for w in health.warnings)


def test_empty_responses_are_always_reported():
    health = assess_health(
        ["", ""], ["answer: A", "answer: A"], [0, 0], [1, 1],
        scorers=["multiple_choice"] * 2,
    )
    assert health.baseline.empty == 2
    assert any("empty" in w for w in health.warnings)


def test_health_survives_a_round_trip_through_as_dict():
    """The diagnostic is persisted into every result JSON, so the new fields
    have to serialize or they are lost to offline auditing."""
    health = assess_health(
        [" C\nbecause"], ["no answer here"], [1], [0],
        scorers=["multiple_choice"],
    )
    d = health.as_dict()
    assert d["b_candidate_unparseable"] == 1
    assert d["c_baseline_unparseable"] == 0
    assert "warnings" in d and isinstance(d["warnings"], list)
