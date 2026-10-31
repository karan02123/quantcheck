"""Statistical core of QuantCheck.

This module implements the **paired per-sample degradation test** used to decide
whether a quantized ("candidate") model is significantly worse than a
higher-precision ("baseline") model when both are scored on the *same* prompts.

IMPORTANT — HONEST FRAMING
--------------------------
The statistics here are **not novel** and are **not "our method."** The paired
disagreement test (an exact one-sided McNemar / sign test on the off-diagonal
counts of a 2x2 agreement table) is a textbook procedure. Its application to
LLM quantization degradation follows:

    Kübler, Budhathoki, Kleindessner, Zhou, Yin, Khetan & Karypis (Amazon),
    "When LLMs get significantly worse: A statistical approach to detect model
    degradations", ICLR 2026. arXiv:2602.10144.
    Code: github.com/amazon-science/LLM-Accuracy-Stats (CC BY-NC 4.0; this
    module implements the method from the paper, not their code).

Note that their paper addresses detecting model degradations *in general*
(inference optimizations, model updates, serving changes) — quantization is one
application, not the paper's scope. QuantCheck's contribution is *engineering
and accessibility*: wrapping this established test in a one-command CLI for
local GGUF models. This module simply implements the established procedure
carefully and documents the reasoning so the result can be trusted.

Scope boundary — what this module deliberately does NOT do
----------------------------------------------------------
There is no equivalence / non-inferiority test here, and that omission is
deliberate. The exactness of the test below rests on a conditional argument
that holds *only* at the null delta = 0: given the number of flips n, the count
b is distributed Binomial(n, 0.5) with no nuisance parameters left over. At a
non-zero margin that is no longer true, so declaring "equivalent within m
percentage points" requires a different machinery — score-based intervals
(Tango 1998, Stat Med 17(8):891-908; Nam 1997, Biometrics 53(4):1422) or
exact unconditional methods (reviewed in Fagerland, Lydersen & Laake 2014,
Stat Med 33(16):2850-75). Those address a margin on the *difference*; a margin
on the *ratio* is a separate literature (Nam & Blackwelder 2002, Stat Med
21(5):689-699). Rather than ship
one CLI whose superiority test is exact and whose equivalence test is
asymptotic, QuantCheck implements the delta = 0 test exactly and reports the
*minimum detectable effect* (see :func:`power_analysis`) so that a null result
is quantified rather than merely reassuring.

The statistical reasoning, step by step
---------------------------------------
We score every prompt with a deterministic scorer ``L(response) -> {0, 1}``
(1 = correct). Because *both* models see the *same* prompts, their per-prompt
outcomes are **paired and correlated**. Treating the two overall accuracies as
two independent samples and running a two-sample test would be **wrong** — it
ignores the pairing and generally inflates the variance, costing power, and can
also mislead when the two models largely agree.

Instead we build the 2x2 agreement table over the N paired outcomes:

                         candidate correct   candidate wrong
    baseline correct            a                  b
    baseline wrong              c                  d

Only the **disagreements** carry information about a *change* in skill:

    b = #(baseline correct AND candidate wrong)   -> degradations
    c = #(baseline wrong   AND candidate correct) -> improvements

The agreements (a, d) tell us nothing about which model is better — they cancel.

Null hypothesis H0: quantization causes no degradation, i.e. a prompt that
flips is equally likely to flip in either direction. Then, conditional on the
number of flips n = b + c, the count b follows::

    b ~ Binomial(n, 0.5)

We want evidence that the candidate is *worse*, i.e. that b is *larger* than
chance. So we run a **one-sided exact binomial test** with
``alternative='greater'`` on b out of n. This is exactly McNemar's test in its
exact (binomial) form, one-sided.

Effect size: the empirical accuracy difference
``delta = acc_candidate - acc_baseline = (c - b) / N``. We report the
*degradation* magnitude as ``(b - c) / N`` for convenience, always with a
bootstrap confidence interval that resamples *prompts* (preserving pairing).

The p-value and the CI are NOT the same quantity
------------------------------------------------
Read them as answering different questions, because they do:

* The **p-value** is *conditional* and exact — it conditions on the number of
  flips n = b + c and asks whether b is larger than a fair coin would give.
  This is the number the verdict is based on.
* The **CI** is *unconditional* and approximate — a percentile bootstrap over
  the marginal quantity (b - c) / N, describing how big the drop is.

They can therefore disagree at the boundary: a CI that just touches zero
alongside a p-value just under alpha is not a contradiction, it is two
different questions. When they conflict, the conditional p-value governs the
decision and the CI describes the magnitude.

Power: a non-significant p-value is **not** proof of equivalence. With few flips
the test is underpowered. We therefore compute, given the observed flip rate,
whether N was large enough to detect a user-specified minimum drop, and if not,
the approximate N needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy.stats import binomtest, norm

__all__ = [
    "ContingencyTable",
    "DegradationResult",
    "PowerResult",
    "build_contingency",
    "paired_degradation_test",
    "bootstrap_delta_ci",
    "power_analysis",
    "benjamini_hochberg",
]


# --------------------------------------------------------------------------- #
# Multiplicity correction
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (FDR q-values), original order.

    Why this is needed
    ------------------
    Running the paired test once per task and reading off the raw p-values
    inflates the false-positive rate: with 14 tasks at alpha=0.05 you expect
    roughly 0.7 significant results **by chance alone**. A per-task p-value is
    therefore not interpretable on its own.

    We use Benjamini-Hochberg (control of the false discovery rate) rather than
    Bonferroni (control of the family-wise error rate) because the per-task
    question is exploratory — "which tasks look worth investigating?" — and FDR
    is the standard, less conservative answer to that question. Bonferroni
    remains available to the reader as ``alpha / n_tests``.

    Method:
        Sort the m p-values ascending. The adjusted value for rank i is
        ``p_i * m / i``, then enforced monotonically non-decreasing from the
        largest rank downwards, and clipped at 1.0.

    Args:
        p_values: raw p-values, one per test.

    Returns:
        Adjusted q-values in the **same order as the input**. An empty input
        returns an empty list.

    Note:
        BH assumes independence or positive regression dependence across tests.
        Per-task subsets of one eval set are disjoint prompt sets, so treating
        them as independent is reasonable; both models are shared across tasks,
        which is a mild positive dependence that BH tolerates.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running_min = 1.0
    # Walk from the largest p-value down, enforcing monotonicity as we go.
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        scaled = p_values[idx] * m / rank
        running_min = min(running_min, scaled)
        q[idx] = min(1.0, running_min)
    return q


# --------------------------------------------------------------------------- #
# Contingency table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContingencyTable:
    """2x2 paired agreement table over N prompts.

    Attributes:
        a: #(baseline correct AND candidate correct) — agreement, both right.
        b: #(baseline correct AND candidate wrong)   — DEGRADATION (signal).
        c: #(baseline wrong   AND candidate correct) — IMPROVEMENT (signal).
        d: #(baseline wrong   AND candidate wrong)   — agreement, both wrong.
    """

    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        """Total paired prompts, N = a + b + c + d."""
        return self.a + self.b + self.c + self.d

    @property
    def n_flips(self) -> int:
        """Number of disagreements, n = b + c (the only informative outcomes)."""
        return self.b + self.c

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy of the baseline model = (a + b) / N."""
        return (self.a + self.b) / self.n if self.n else float("nan")

    @property
    def candidate_accuracy(self) -> float:
        """Accuracy of the candidate model = (a + c) / N."""
        return (self.a + self.c) / self.n if self.n else float("nan")

    @property
    def flip_rate(self) -> float:
        """Disagreement rate p_flip = (b + c) / N.

        This drives statistical power: more flips => more signal per prompt.
        """
        return self.n_flips / self.n if self.n else float("nan")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            n=self.n,
            n_flips=self.n_flips,
            baseline_accuracy=self.baseline_accuracy,
            candidate_accuracy=self.candidate_accuracy,
            flip_rate=self.flip_rate,
        )
        return d


def build_contingency(
    baseline_scores: Sequence[int],
    candidate_scores: Sequence[int],
) -> ContingencyTable:
    """Build the 2x2 paired agreement table from per-prompt 0/1 scores.

    The two score arrays must be **aligned**: index i is the same prompt scored
    by each model. This pairing is the whole point — do not sort or shuffle one
    without the other.

    Args:
        baseline_scores: per-prompt scores for the baseline model, each in {0,1}.
        candidate_scores: per-prompt scores for the candidate model, each in {0,1}.

    Returns:
        ContingencyTable with counts a, b, c, d.

    Raises:
        ValueError: if the arrays differ in length, are empty, or contain values
            other than 0/1.

    Failure modes:
        - Misaligned arrays silently produce a meaningless table; the caller is
          responsible for keeping prompt order identical across both models.
    """
    base = np.asarray(baseline_scores)
    cand = np.asarray(candidate_scores)
    if base.shape != cand.shape:
        raise ValueError(
            f"score arrays must be the same length; got {base.shape} and {cand.shape}"
        )
    if base.size == 0:
        raise ValueError("score arrays are empty; nothing to compare")
    if not (np.isin(base, (0, 1)).all() and np.isin(cand, (0, 1)).all()):
        raise ValueError("scores must be binary 0/1 (output of L(response) -> {0,1})")

    base = base.astype(bool)
    cand = cand.astype(bool)
    a = int(np.sum(base & cand))
    b = int(np.sum(base & ~cand))
    c = int(np.sum(~base & cand))
    d = int(np.sum(~base & ~cand))
    return ContingencyTable(a=a, b=b, c=c, d=d)


# --------------------------------------------------------------------------- #
# The exact one-sided paired test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DegradationResult:
    """Result of the paired one-sided degradation test.

    Attributes:
        n: total paired prompts.
        b: degradations (baseline right, candidate wrong).
        c: improvements (baseline wrong, candidate right).
        n_flips: disagreements b + c.
        flip_rate: (b + c) / n.
        delta_degradation: (b - c) / n. Positive => candidate worse overall.
        p_value: one-sided exact-binomial (exact McNemar) p-value for H1 "candidate
            degraded". Small p => evidence of degradation.
        test_name: human-readable name of the test used.
        ci_low / ci_high: bootstrap 95% CI for delta_degradation (filled in by
            ``paired_degradation_test`` when ``with_ci=True``).
    """

    n: int
    b: int
    c: int
    n_flips: int
    flip_rate: float
    delta_degradation: float
    p_value: float
    test_name: str
    ci_low: float | None = None
    ci_high: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def paired_degradation_test(
    table: ContingencyTable,
    *,
    baseline_scores: Sequence[int] | None = None,
    candidate_scores: Sequence[int] | None = None,
    with_ci: bool = True,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int | None = 0,
) -> DegradationResult:
    """Run the exact one-sided paired degradation test (exact McNemar).

    Statistical reasoning:
        Under H0 (no degradation) a flipped prompt is equally likely to flip in
        either direction, so b ~ Binomial(b + c, 0.5). We test H1: candidate is
        worse, i.e. b is *large*. Hence ``binomtest(b, b + c, 0.5,
        alternative='greater')``. Agreements (a, d) are conditioned out and never
        enter the p-value — only the disagreements carry signal.

    Edge case — zero flips:
        If b + c == 0 the models agreed on every prompt; there is no evidence of
        any difference and the p-value is defined as 1.0. This is *not* proof of
        equivalence (see ``power_analysis``).

    Args:
        table: the 2x2 contingency table.
        baseline_scores / candidate_scores: per-prompt 0/1 scores, required only
            when ``with_ci=True`` so the bootstrap can resample paired prompts.
        with_ci: if True, attach a bootstrap CI for the degradation effect size.
        n_boot: bootstrap resamples (default 2000).
        ci_level: confidence level for the CI (default 0.95).
        seed: RNG seed for the bootstrap, logged for reproducibility.

    Returns:
        DegradationResult with the test name, counts, p-value, effect size, and
        (optionally) the bootstrap CI.

    Raises:
        ValueError: if ``with_ci=True`` but per-prompt scores were not supplied.
    """
    b, c, n = table.b, table.c, table.n
    n_flips = b + c

    if n_flips == 0:
        p_value = 1.0
    else:
        # Exact one-sided binomial test == exact McNemar, alternative 'greater'.
        p_value = binomtest(b, n_flips, 0.5, alternative="greater").pvalue

    delta = (b - c) / n  # degradation magnitude; positive => candidate worse

    ci_low = ci_high = None
    if with_ci:
        if baseline_scores is None or candidate_scores is None:
            raise ValueError(
                "with_ci=True requires baseline_scores and candidate_scores so the "
                "bootstrap can resample paired prompts"
            )
        ci_low, ci_high = bootstrap_delta_ci(
            baseline_scores,
            candidate_scores,
            n_boot=n_boot,
            ci_level=ci_level,
            seed=seed,
        )

    return DegradationResult(
        n=n,
        b=b,
        c=c,
        n_flips=n_flips,
        flip_rate=table.flip_rate,
        delta_degradation=delta,
        p_value=p_value,
        test_name="exact one-sided McNemar (binomial sign test on disagreements)",
        ci_low=ci_low,
        ci_high=ci_high,
    )


# --------------------------------------------------------------------------- #
# Bootstrap CI for the (paired) effect size
# --------------------------------------------------------------------------- #
def bootstrap_delta_ci(
    baseline_scores: Sequence[int],
    candidate_scores: Sequence[int],
    *,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the degradation effect size (b - c) / N.

    Why bootstrap *prompts* (not the b/c counts): the data are paired, so we must
    resample the **prompt index** with replacement and recompute (b - c) / N on
    each resample. This preserves the paired structure — each resampled prompt
    carries *both* model outcomes together — which a parametric CI on two
    independent proportions would not.

    The reported quantity matches ``DegradationResult.delta_degradation``:
    positive => candidate worse.

    Args:
        baseline_scores: per-prompt 0/1 scores, aligned with candidate_scores.
        candidate_scores: per-prompt 0/1 scores, aligned with baseline_scores.
        n_boot: number of bootstrap resamples (default 2000).
        ci_level: central interval mass (default 0.95 -> 2.5%/97.5% percentiles).
        seed: RNG seed, logged for reproducibility.

    Returns:
        (ci_low, ci_high) for (b - c) / N.

    Failure modes:
        - With very few prompts the percentile interval is coarse and may be wide
          or degenerate; this honestly reflects the lack of data.
    """
    base = np.asarray(baseline_scores).astype(int)
    cand = np.asarray(candidate_scores).astype(int)
    n = base.size
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")

    rng = np.random.default_rng(seed)
    # diff[i] = +1 if degradation on prompt i (baseline right, candidate wrong),
    #           -1 if improvement, 0 if the two models agreed.
    diff = base - cand  # in {-1, 0, +1}; mean(diff) == (b - c) / N
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    alpha = 1.0 - ci_level
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------- #
# Power analysis
# --------------------------------------------------------------------------- #
def _conditional_power(
    delta: float, n: int, flip_rate: float, *, z_alpha: float
) -> float:
    """Power of the one-sided conditional test against a true drop ``delta``.

    Shared by :func:`power_analysis` and :func:`_solve_mde` so that the reported
    power and the reported minimum detectable effect can never disagree.

    See :func:`power_analysis` for the derivation. Returns a value in [0, 1].
    """
    if flip_rate <= 0.0 or delta <= 0.0:
        return 0.0
    q = min(0.5 + delta / (2.0 * flip_rate), 1.0 - 1e-9)
    sd_alt = np.sqrt(q * (1 - q))
    n_flips = flip_rate * n
    power = float(
        norm.cdf(((q - 0.5) * np.sqrt(n_flips) - z_alpha * 0.5) / sd_alt)
    )
    return max(0.0, min(1.0, power))


def _solve_mde(
    n: int, flip_rate: float, *, target_power: float, z_alpha: float
) -> float | None:
    """Smallest drop this design could detect at ``target_power`` (the MDE).

    This is the inverse of the power question, and the more useful direction in
    practice. "You would need 7340 prompts to detect a 2% drop" tells a user
    their run was inadequate; "at your N and flip rate this run could only have
    detected drops of 7% or more" tells them *what their run actually means*,
    which is what makes a null result interpretable instead of merely
    disappointing.

    Method:
        :func:`_conditional_power` is monotonically increasing in ``delta`` (a
        larger true drop pushes more of the flip mass into b), so we bisect on
        ``delta`` in ``(0, flip_rate]``. The upper bound is the flip rate
        because ``|b - c| <= b + c``: a drop larger than the disagreement rate
        is arithmetically impossible.

    Returns:
        The MDE as a fraction (e.g. 0.07 = 7 percentage points), or None when
        even the largest arithmetically possible drop would not reach
        ``target_power`` at this N and flip rate.
    """
    if flip_rate <= 0.0 or n <= 0:
        return None
    hi = flip_rate
    if _conditional_power(hi, n, flip_rate, z_alpha=z_alpha) < target_power:
        return None  # not even a total wipeout would be reliably detected

    lo = 0.0
    for _ in range(60):  # ~1e-18 resolution; far finer than we report
        mid = (lo + hi) / 2.0
        if _conditional_power(mid, n, flip_rate, z_alpha=z_alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass(frozen=True)
class PowerResult:
    """Result of the power check for the paired degradation test.

    Attributes:
        min_detectable_drop: the accuracy drop (as a fraction, e.g. 0.02 = 2%)
            the user cares about detecting.
        flip_rate: observed disagreement rate p_flip = (b + c) / N; power depends
            on this, not on N alone.
        n: the eval set size actually used.
        achieved_power: estimated probability this design would detect a true drop
            of ``min_detectable_drop`` at the given significance level.
        required_n: approximate N needed to reach ``target_power`` for that drop
            at the observed flip rate (None if the flip rate is ~0, where the
            required N is effectively unbounded).
        min_detectable_effect: the smallest drop this design *could* have
            detected at ``target_power``, given the N and flip rate actually
            observed. This is the honest summary of a null result: any true
            degradation smaller than this would likely have been missed. None
            when no drop is reliably detectable at this N and flip rate.
        target_power: power target (default 0.95).
        alpha: one-sided significance level (default 0.05).
        adequate: True iff achieved_power >= target_power.
    """

    min_detectable_drop: float
    flip_rate: float
    n: int
    achieved_power: float
    required_n: int | None
    min_detectable_effect: float | None
    target_power: float
    alpha: float
    adequate: bool

    def as_dict(self) -> dict:
        return asdict(self)


def power_analysis(
    n: int,
    flip_rate: float,
    *,
    min_detectable_drop: float = 0.02,
    target_power: float = 0.95,
    alpha: float = 0.05,
) -> PowerResult:
    """Estimate power and required N for the paired degradation test.

    Statistical reasoning (normal approximation to the exact test):
        The test looks at b ~ Binomial(n_flips, q) where n_flips = p_flip * N is
        the expected number of disagreements and q is the conditional probability
        that a flip is a degradation. Under H0, q = 0.5.

        A true overall accuracy drop ``delta = (b - c)/N`` together with the flip
        rate pins down the conditional split:
            b - c = delta * N  and  b + c = p_flip * N
            => q = P(degradation | flip) = 1/2 + delta / (2 * p_flip).

        Treating the conditional count b as approximately normal, a one-sided
        test at level ``alpha`` has power

            power ≈ Φ( ( (q - 0.5) * sqrt(n_flips)
                         - z_alpha * sqrt(0.25) )
                       / sqrt(q * (1 - q)) )

        where n_flips = p_flip * N and z_alpha = Φ⁻¹(1 - alpha). This is the
        standard one-sample-proportion power formula applied to the *conditional*
        binomial, which is what makes power depend on the flip rate: a low flip
        rate means few informative trials, so even a real drop is hard to detect.

    The required N is obtained by solving the same formula for n_flips and
    dividing by p_flip.

    Note: this uses a normal approximation for speed and interpretability; it is
    a guide to adequacy, not an exact operating characteristic of the exact test.
    The exact test is always what produces the reported p-value.

    Args:
        n: eval set size actually used.
        flip_rate: observed disagreement rate p_flip in [0, 1].
        min_detectable_drop: target accuracy drop to detect (fraction, default 0.02).
        target_power: desired power (default 0.95).
        alpha: one-sided significance level (default 0.05).

    Returns:
        PowerResult with achieved power, required N, and an adequacy flag.

    Failure modes:
        - If flip_rate == 0 the test has no signal; achieved_power is 0 and
          required_n is None (effectively unbounded). This is the honest answer:
          you cannot detect a drop you never observed flipping.
        - If the implied conditional q exceeds 1 (drop larger than the flip rate
          can support), q is clamped to just below 1.
    """
    if not (0.0 <= flip_rate <= 1.0):
        raise ValueError(f"flip_rate must be in [0, 1]; got {flip_rate}")
    if min_detectable_drop <= 0:
        raise ValueError("min_detectable_drop must be positive")

    z_alpha = norm.ppf(1 - alpha)
    z_target = norm.ppf(target_power)

    if flip_rate <= 0.0:
        return PowerResult(
            min_detectable_drop=min_detectable_drop,
            flip_rate=flip_rate,
            n=n,
            achieved_power=0.0,
            required_n=None,
            min_detectable_effect=None,
            target_power=target_power,
            alpha=alpha,
            adequate=False,
        )

    achieved_power = _conditional_power(
        min_detectable_drop, n, flip_rate, z_alpha=z_alpha
    )

    # Conditional probability that a flip is a degradation under the alternative.
    q = 0.5 + min_detectable_drop / (2.0 * flip_rate)
    q = min(q, 1.0 - 1e-9)  # clamp: drop cannot exceed what the flip rate supports
    sd_alt = np.sqrt(q * (1 - q))
    sd_null = 0.5  # sqrt(0.5 * 0.5)

    # Solve for the number of flips needed to reach target_power, then convert to N.
    # (q - 0.5)*sqrt(m) - z_alpha*sd_null = z_target * sd_alt
    # => sqrt(m) = (z_alpha*sd_null + z_target*sd_alt) / (q - 0.5)
    numerator = z_alpha * sd_null + z_target * sd_alt
    sqrt_m = numerator / (q - 0.5)
    m_required = sqrt_m ** 2
    required_n = int(np.ceil(m_required / flip_rate))

    return PowerResult(
        min_detectable_drop=min_detectable_drop,
        flip_rate=flip_rate,
        n=n,
        achieved_power=achieved_power,
        required_n=required_n,
        min_detectable_effect=_solve_mde(
            n, flip_rate, target_power=target_power, z_alpha=z_alpha
        ),
        target_power=target_power,
        alpha=alpha,
        adequate=achieved_power >= target_power,
    )
