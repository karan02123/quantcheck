"""Correctness tests for the statistical core (quantcheck/stats.py).

This is the most important test file in the project: the credibility of the
whole tool rests on the paired test being correct. Every test here checks the
implementation against a value that is **known analytically** or verified by
**direct simulation**, never against the implementation's own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from quantcheck.stats import (
    build_contingency,
    paired_degradation_test,
    bootstrap_delta_ci,
    power_analysis,
)


# --------------------------------------------------------------------------- #
# Contingency table
# --------------------------------------------------------------------------- #
def test_contingency_counts_known():
    # baseline: 1 1 1 0 0   candidate: 1 0 0 1 0
    base = [1, 1, 1, 0, 0]
    cand = [1, 0, 0, 1, 0]
    t = build_contingency(base, cand)
    # a: both right  -> index 0                 -> 1
    # b: base right, cand wrong -> indices 1,2  -> 2  (degradations)
    # c: base wrong, cand right -> index 3      -> 1  (improvements)
    # d: both wrong  -> index 4                 -> 1
    assert (t.a, t.b, t.c, t.d) == (1, 2, 1, 1)
    assert t.n == 5
    assert t.n_flips == 3
    assert t.baseline_accuracy == pytest.approx(3 / 5)
    assert t.candidate_accuracy == pytest.approx(2 / 5)
    assert t.flip_rate == pytest.approx(3 / 5)


def test_contingency_rejects_nonbinary_and_mismatch():
    with pytest.raises(ValueError):
        build_contingency([1, 2], [0, 1])  # non-binary
    with pytest.raises(ValueError):
        build_contingency([1, 0, 1], [0, 1])  # length mismatch
    with pytest.raises(ValueError):
        build_contingency([], [])  # empty


# --------------------------------------------------------------------------- #
# The exact p-value — checked against an analytically known value
# --------------------------------------------------------------------------- #
def test_pvalue_matches_hand_computed_binomial():
    # Construct b=8 degradations, c=2 improvements -> n_flips=10.
    # Under H0, b ~ Binomial(10, 0.5). One-sided P(B >= 8) is exactly:
    #   sum_{k=8}^{10} C(10,k) (0.5)^10 = (45 + 10 + 1)/1024 = 56/1024.
    expected = 56 / 1024
    base = [1] * 8 + [0] * 2 + [1] * 5
    cand = [0] * 8 + [1] * 2 + [1] * 5  # 8 degr, 2 impr, 5 agree-correct
    t = build_contingency(base, cand)
    assert (t.b, t.c) == (8, 2)
    res = paired_degradation_test(t, baseline_scores=base, candidate_scores=cand)
    assert res.p_value == pytest.approx(expected, rel=1e-12)
    # And it must equal scipy's own one-sided binomtest on (b, b+c).
    assert res.p_value == pytest.approx(
        binomtest(8, 10, 0.5, alternative="greater").pvalue, rel=1e-12
    )


def test_pvalue_symmetric_case_is_half_ish():
    # b == c: no net direction. P(B >= b) for symmetric split is > 0.5.
    base = [1] * 5 + [0] * 5
    cand = [0] * 5 + [1] * 5  # b=5, c=5
    t = build_contingency(base, cand)
    assert (t.b, t.c) == (5, 5)
    res = paired_degradation_test(t, baseline_scores=base, candidate_scores=cand)
    # P(Binom(10,0.5) >= 5) = 0.623046875 (hand-computed).
    assert res.p_value == pytest.approx(0.623046875, rel=1e-9)


def test_zero_flips_pvalue_is_one():
    # Models agree on everything -> no signal -> p-value defined as 1.0.
    base = [1, 1, 0, 0]
    cand = [1, 1, 0, 0]
    t = build_contingency(base, cand)
    assert t.n_flips == 0
    res = paired_degradation_test(t, baseline_scores=base, candidate_scores=cand)
    assert res.p_value == 1.0
    assert res.delta_degradation == 0.0


def test_strong_degradation_is_significant():
    # 18 degradations (base=1,cand=0), 2 improvements (base=0,cand=1).
    base = [1] * 18 + [0] * 2
    cand = [0] * 18 + [1] * 2
    t = build_contingency(base, cand)
    res = paired_degradation_test(t, baseline_scores=base, candidate_scores=cand)
    assert res.b == 18 and res.c == 2
    assert res.p_value < 0.001
    assert res.delta_degradation == pytest.approx((18 - 2) / 20)


def test_only_disagreements_drive_pvalue():
    # Adding agreements (both correct / both wrong) must not change the p-value,
    # because agreements are conditioned out of the exact test.
    base_small = [1] * 6 + [0] * 1  # b=6, c=1
    cand_small = [0] * 6 + [1] * 1
    t1 = build_contingency(base_small, cand_small)
    p1 = paired_degradation_test(
        t1, baseline_scores=base_small, candidate_scores=cand_small, with_ci=False
    ).p_value
    # Same b, c but pad with 50 both-correct and 50 both-wrong agreements.
    base_big = base_small + [1] * 50 + [0] * 50
    cand_big = cand_small + [1] * 50 + [0] * 50
    t2 = build_contingency(base_big, cand_big)
    assert (t2.b, t2.c) == (t1.b, t1.c)
    p2 = paired_degradation_test(
        t2, baseline_scores=base_big, candidate_scores=cand_big, with_ci=False
    ).p_value
    assert p1 == pytest.approx(p2, rel=1e-12)


def test_requires_scores_for_ci():
    t = build_contingency([1, 0], [0, 1])
    with pytest.raises(ValueError):
        paired_degradation_test(t, with_ci=True)


# --------------------------------------------------------------------------- #
# Bootstrap CI — checked for correct centering and coverage by simulation
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    n = 400
    base = rng.integers(0, 2, n)
    cand = base.copy()
    # Force b=40 degradations, c=10 improvements among the first 50 prompts.
    cand[:40] = 0
    base[:40] = 1
    base[40:50] = 0
    cand[40:50] = 1
    t = build_contingency(base, cand)
    point = (t.b - t.c) / t.n
    lo, hi = bootstrap_delta_ci(base, cand, n_boot=2000, seed=1)
    assert lo <= point <= hi
    assert hi > lo  # non-degenerate


def test_bootstrap_ci_coverage_by_simulation():
    """The 95% paired bootstrap CI should cover the true (b-c)/N about 95% of
    the time. We simulate paired data with a known per-prompt degradation
    structure and check empirical coverage is in a sane band."""
    rng = np.random.default_rng(42)
    n = 300
    # True model: P(degrade)=0.10, P(improve)=0.04, else agree.
    p_degrade, p_improve = 0.10, 0.04
    true_delta = p_degrade - p_improve
    covered = 0
    trials = 200
    for _ in range(trials):
        u = rng.random(n)
        base = np.ones(n, dtype=int)
        cand = np.ones(n, dtype=int)
        # degrade: base=1, cand=0
        deg = u < p_degrade
        # improve: base=0, cand=1
        imp = (u >= p_degrade) & (u < p_degrade + p_improve)
        base[imp] = 0
        cand[deg] = 0
        lo, hi = bootstrap_delta_ci(base, cand, n_boot=400, seed=int(rng.integers(1e9)))
        if lo <= true_delta <= hi:
            covered += 1
    coverage = covered / trials
    # Percentile bootstrap on a difference of proportions: expect ~0.95, allow band.
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage} out of expected band"


# --------------------------------------------------------------------------- #
# Power analysis — checked against direct Monte-Carlo of the exact test
# --------------------------------------------------------------------------- #
def _simulate_exact_test_power(n, flip_rate, drop, alpha, trials, rng):
    """Monte-Carlo the *exact* test's power for a true accuracy drop.

    Per prompt: flips with prob flip_rate; given a flip it is a degradation with
    prob q = 0.5 + drop/(2*flip_rate). Run the real exact test each trial.
    """
    q = 0.5 + drop / (2 * flip_rate)
    rejections = 0
    for _ in range(trials):
        u = rng.random(n)
        flipped = u < flip_rate
        n_flips = int(flipped.sum())
        if n_flips == 0:
            continue
        b = int(rng.binomial(n_flips, q))
        p = binomtest(b, n_flips, 0.5, alternative="greater").pvalue
        if p < alpha:
            rejections += 1
    return rejections / trials


def test_power_formula_tracks_simulation():
    rng = np.random.default_rng(7)
    n, flip_rate, drop, alpha = 500, 0.20, 0.05, 0.05
    pr = power_analysis(
        n, flip_rate, min_detectable_drop=drop, alpha=alpha, target_power=0.95
    )
    sim = _simulate_exact_test_power(n, flip_rate, drop, alpha, trials=2000, rng=rng)
    # Normal approximation vs exact test: should agree to within ~0.07.
    assert abs(pr.achieved_power - sim) < 0.07, (pr.achieved_power, sim)


def test_required_n_actually_reaches_target_power():
    rng = np.random.default_rng(11)
    flip_rate, drop, alpha = 0.15, 0.03, 0.05
    pr = power_analysis(
        100, flip_rate, min_detectable_drop=drop, alpha=alpha, target_power=0.90
    )
    assert pr.required_n is not None
    # Simulate the exact test at the recommended N; power should be near target.
    sim = _simulate_exact_test_power(
        pr.required_n, flip_rate, drop, alpha, trials=3000, rng=rng
    )
    assert sim >= 0.85, f"required_n={pr.required_n} gave power {sim}, target 0.90"


def test_low_flip_rate_is_underpowered():
    # Tiny flip rate => underpowered even with a fair number of prompts.
    pr = power_analysis(200, flip_rate=0.01, min_detectable_drop=0.02)
    assert not pr.adequate
    assert pr.achieved_power < 0.95


def test_zero_flip_rate_zero_power():
    pr = power_analysis(200, flip_rate=0.0, min_detectable_drop=0.02)
    assert pr.achieved_power == 0.0
    assert pr.required_n is None
    assert not pr.adequate


def test_power_increases_with_n():
    p_small = power_analysis(100, 0.2, min_detectable_drop=0.05).achieved_power
    p_big = power_analysis(2000, 0.2, min_detectable_drop=0.05).achieved_power
    assert p_big > p_small


def test_power_input_validation():
    with pytest.raises(ValueError):
        power_analysis(100, flip_rate=1.5)
    with pytest.raises(ValueError):
        power_analysis(100, flip_rate=0.2, min_detectable_drop=0.0)


# ---------------------------------------------------------------------------
# Multiplicity correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------
def test_bh_matches_hand_computed_values():
    """BH on a known vector, checked against the definition p_i * m / i."""
    from quantcheck.stats import benjamini_hochberg

    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    q = benjamini_hochberg(p)
    # raw scaled values: 0.05, 0.05, 0.05, 0.05, 0.05 -> all equal
    assert all(abs(v - 0.05) < 1e-12 for v in q)


def test_bh_preserves_input_order():
    from quantcheck.stats import benjamini_hochberg

    q = benjamini_hochberg([0.9, 0.001, 0.5])
    assert q[1] == min(q)  # the smallest p keeps the smallest q, at index 1


def test_bh_is_monotone_and_bounded():
    from quantcheck.stats import benjamini_hochberg

    p = [0.001, 0.008, 0.02, 0.2, 0.7, 0.99]
    q = benjamini_hochberg(p)
    assert all(0.0 <= v <= 1.0 for v in q)
    assert q == sorted(q)  # input sorted ascending => q must be non-decreasing
    assert all(a <= b + 1e-12 for a, b in zip(p, q))  # BH never shrinks a p-value


def test_bh_edge_cases():
    from quantcheck.stats import benjamini_hochberg

    assert benjamini_hochberg([]) == []
    assert benjamini_hochberg([0.03]) == [0.03]  # m=1 is the identity


def test_bh_reproduces_the_real_run_result():
    """Regression pin: the 14 per-task p-values from the N=500 Q2_K knowledge
    run must yield ZERO significant tasks after correction. This is the finding
    that corrected an earlier over-claim in examples/REAL_RUN.md."""
    from quantcheck.stats import benjamini_hochberg

    p = [0.00647, 0.01123, 0.01562, 0.05469, 0.1133, 0.2539, 0.2539,
         0.2744, 0.2744, 0.5, 0.6562, 0.6562, 0.9102, 0.9844]
    assert len(p) == 14
    q = benjamini_hochberg(p)
    assert sum(1 for v in q if v < 0.05) == 0
    assert abs(min(q) - 0.0729) < 1e-3  # the three smallest collapse to ~0.073


# ---------------------------------------------------------------------------
# Minimum detectable effect (the inverse of the power question)
# ---------------------------------------------------------------------------
def test_mde_is_self_consistent_with_power():
    """Power evaluated AT the reported MDE must equal the target power."""
    from quantcheck.stats import power_analysis

    r = power_analysis(500, 0.272, min_detectable_drop=0.02, target_power=0.95)
    assert r.min_detectable_effect is not None
    check = power_analysis(
        500, 0.272, min_detectable_drop=r.min_detectable_effect, target_power=0.95
    )
    assert abs(check.achieved_power - 0.95) < 1e-3


def test_mde_shrinks_as_n_grows():
    """More prompts => able to detect smaller drops. The core design fact."""
    from quantcheck.stats import power_analysis

    mdes = [
        power_analysis(n, 0.272, min_detectable_drop=0.02).min_detectable_effect
        for n in (100, 200, 500, 1000)
    ]
    assert all(a > b for a, b in zip(mdes, mdes[1:])), mdes


def test_mde_explains_the_n200_vs_n500_verdict_flip():
    """Regression pin on the project's clearest result: at N=200 the true ~8-9%
    drop sits BELOW the detectable minimum, at N=500 it sits above it."""
    from quantcheck.stats import power_analysis

    mde_200 = power_analysis(200, 0.260, min_detectable_drop=0.02).min_detectable_effect
    mde_500 = power_analysis(500, 0.272, min_detectable_drop=0.02).min_detectable_effect
    assert mde_200 > 0.084 > mde_500


def test_mde_is_none_when_nothing_is_detectable():
    from quantcheck.stats import power_analysis

    assert power_analysis(500, 0.0, min_detectable_drop=0.02).min_detectable_effect is None
    # A handful of prompts cannot reliably detect even a total collapse.
    assert power_analysis(3, 0.3, min_detectable_drop=0.02).min_detectable_effect is None


def test_mde_never_exceeds_the_flip_rate():
    """A drop larger than the disagreement rate is arithmetically impossible."""
    from quantcheck.stats import power_analysis

    for flip in (0.05, 0.2, 0.5, 0.9):
        mde = power_analysis(2000, flip, min_detectable_drop=0.01).min_detectable_effect
        if mde is not None:
            assert mde <= flip + 1e-9
