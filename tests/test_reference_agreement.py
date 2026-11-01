"""Differential validation against the reference implementation for the paper.

Why this file exists
--------------------
The rest of `tests/test_stats.py` checks QuantCheck's exact test against values
derived independently *by us* -- closed-form binomial probabilities, Monte-Carlo
simulation. That is necessary but self-contained: a shared misreading of the
method would pass every one of those tests.

This file closes that gap by comparing against a third party: the reference
implementation released with

    Kuebler, Budhathoki, Kleindessner, Zhou, Yin, Khetan & Karypis,
    "When LLMs get significantly worse: A statistical approach to detect model
    degradations", ICLR 2026. arXiv:2602.10144.
    github.com/amazon-science/LLM-Accuracy-Stats

Provenance and licensing
------------------------
That repository is licensed **CC BY-NC 4.0**, which is incompatible with this
project's MIT terms. It is therefore:

  * never vendored into this repository,
  * never read for implementation detail, and
  * used here strictly as a **black box** -- we call its public
    ``pooled_test(b_list, c_list)`` and compare the number it returns.

QuantCheck's statistics were implemented from the paper's description, and these
tests do not change that. See `NOTICE`.

Running it
----------
Skipped by default, since it needs an external checkout. To run:

    git clone https://github.com/amazon-science/LLM-Accuracy-Stats.git /tmp/ref
    QUANTCHECK_REFERENCE_REPO=/tmp/ref pytest tests/test_reference_agreement.py -v

Recorded outcome (514 tables, incl. the seven runs published at the time): the
two implementations agree **exactly** -- maximum absolute difference 0.0, not
merely within floating-point tolerance. Run 8 was pinned after that figure was
recorded, so the next run of this file covers eight published runs and 515
tables; do not quote 515 as agreeing until it has been observed. The single
divergence found is documented in
`test_reference_crashes_on_zero_flips_quantcheck_does_not` below.
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import pytest

from quantcheck.stats import ContingencyTable, paired_degradation_test

REF_ENV = "QUANTCHECK_REFERENCE_REPO"


def _p(b: int, c: int, n: int) -> float:
    """QuantCheck's exact one-sided p-value. `with_ci=False` skips the bootstrap,
    which the p-value does not depend on."""
    table = ContingencyTable(a=0, b=b, c=c, d=max(n - b - c, 0))
    return paired_degradation_test(table, with_ci=False).p_value


@pytest.fixture(scope="module")
def reference():
    """Load `pooled_test` from an external checkout, or skip."""
    root = os.environ.get(REF_ENV)
    if not root:
        pytest.skip(f"set {REF_ENV} to a checkout of amazon-science/LLM-Accuracy-Stats")
    target = Path(root) / "aggregation_script.py"
    if not target.exists():
        pytest.skip(f"{target} not found -- is {REF_ENV} pointing at the repo root?")
    spec = importlib.util.spec_from_file_location("_quantcheck_reference", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pooled_test


def test_reference_uses_the_same_tail(reference):
    """Both must put the small p-value on b > c, or the comparison is meaningless.

    Establishes the convention by observed behaviour rather than by reading the
    reference: a lopsided-toward-degradation table must score lower than its
    mirror image.
    """
    assert reference([8], [2]) < reference([2], [8])
    assert _p(8, 2, 10) < _p(2, 8, 10)


@pytest.mark.parametrize(
    "label,b,c,n",
    [
        ("run 1  Llama-3.2-3B Q4_K_M knowledge", 5, 13, 200),
        ("run 2  Gemma-3-4B   Q4_K_M knowledge", 11, 15, 200),
        ("run 3  Gemma-3-4B   Q2_K   knowledge", 32, 20, 200),
        ("run 4  Gemma-3-4B   Q2_K   knowledge", 89, 47, 500),
        ("run 5  Gemma-3-4B   Q2_K   reasoning", 33, 14, 200),
        ("run 6  Gemma-3-4B   Q4_K_M reasoning", 7, 10, 200),
        ("run 7  Qwen2.5-7B   Q4_K_M reasoning", 16, 11, 500),
        # Run 8 supersedes run 4 and carries the project's strongest claim, so
        # it has to be pinned here too. Run 4 stays as the historical record.
        ("run 8  Gemma-3-4B   Q2_K   knowledge", 80, 41, 500),
    ],
)
def test_published_runs_match_the_reference(reference, label, b, c, n):
    """Every verdict published in examples/REAL_RUN.md, re-derived by the
    reference implementation. These are the numbers that carry the project's
    claims, so they are pinned individually rather than in a loop."""
    assert _p(b, c, n) == pytest.approx(float(reference([b], [c])), abs=1e-15), label


def test_agreement_on_randomized_tables(reference):
    """500 random tables plus boundary shapes. Exact equality is asserted, which
    is what was observed -- loosen this only with a documented reason."""
    rng = random.Random(0)
    cases = [(1, 0, 10), (0, 1, 10), (5, 5, 10), (1, 1, 2), (0, 5, 10), (5, 0, 10)]
    cases += [
        (rng.randint(0, 60), rng.randint(0, 60), 0) for _ in range(500)
    ]
    cases = [(b, c, max(n, b + c + rng.randint(0, 500))) for b, c, n in cases]

    worst = 0.0
    for b, c, n in cases:
        if b + c == 0:
            continue  # the reference raises here; covered by the test below
        worst = max(worst, abs(_p(b, c, n) - float(reference([b], [c]))))
    assert worst == 0.0, f"max divergence {worst!r} across {len(cases)} tables"


def test_reference_crashes_on_zero_flips_quantcheck_does_not(reference):
    """A documented behavioural difference, not a QuantCheck bug.

    When two models agree on every prompt (b = c = 0) there are no discordant
    pairs, the test statistic is undefined, and the correct answer is p = 1.0:
    zero evidence of degradation. QuantCheck returns that. The reference
    implementation raises ZeroDivisionError.

    This is not a corner case invented to score a point. b = c = 0 is exactly
    what a *lossless* optimization produces, and detecting that case without a
    false positive is a stated goal of the paper. The scenario is reachable with
    a small suite or a genuinely lossless quantization.
    """
    assert _p(0, 0, 10) == 1.0
    with pytest.raises(ZeroDivisionError):
        reference([0], [0])
