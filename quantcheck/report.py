"""Scoring orchestration, verdict construction, and output formatting.

This module ties the pieces together: it scores both models' raw responses with
each suite item's scorer, builds the pooled and per-task contingency tables,
runs the paired degradation test and power check from :mod:`quantcheck.stats`,
and renders an honest verdict.

HONESTY GUARDRAILS (enforced here, by design)
---------------------------------------------
- We never print "no degradation" as if proving equivalence. A non-significant
  result is reported as "no statistically significant degradation detected (not
  proof of equivalence; see power check)."
- If the power check says N was too small to detect the drop the user cares
  about, the verdict is **INCONCLUSIVE**, not "passed".
- We always show together: the test used, N, b, c, flip rate, p-value, and CI.
- Per-task results are shown alongside the pooled result so a degradation
  concentrated in one task type is not hidden by the aggregate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Sequence

from .scoring import answer_present, get_scorer
from .stats import (
    ContingencyTable,
    DegradationResult,
    PowerResult,
    benjamini_hochberg,
    build_contingency,
    paired_degradation_test,
    power_analysis,
)
from .suites import Suite

__all__ = [
    "score_responses",
    "TaskResult",
    "ResponseHealth",
    "HealthCheck",
    "assess_health",
    "Report",
    "evaluate",
    "format_report",
]


def score_responses(suite: Suite, responses: Sequence[str]) -> list[int]:
    """Score each raw response with its suite item's scorer.

    Args:
        suite: the suite (provides per-item scorer name and gold answer).
        responses: raw model outputs, aligned with ``suite.items``.

    Returns:
        A list of 0/1 scores, aligned with the suite items.

    Raises:
        ValueError: if the number of responses does not match the suite length.
    """
    if len(responses) != len(suite.items):
        raise ValueError(
            f"got {len(responses)} responses for {len(suite.items)} suite items"
        )
    scores: list[int] = []
    for item, resp in zip(suite.items, responses):
        scorer = get_scorer(item.scorer)
        scores.append(int(scorer(resp, item.answer)))
    return scores


@dataclass
class TaskResult:
    """Degradation result for a single task type (or the pooled set).

    Attributes:
        task: the task tag, or "ALL" for the pooled result.
        table: the 2x2 paired table for this subset.
        test: the paired degradation test on this subset.
        q_value: Benjamini-Hochberg adjusted p-value across the per-task family
            of tests. ``None`` for the pooled result (which is a single
            pre-specified test and needs no adjustment) and whenever there is
            only one task, where there is no multiplicity to correct.
        power: per-task power check, so a task line carries its own adequacy
            rather than borrowing the pooled run's. ``None`` for the pooled row,
            whose power is reported separately as the headline power check.
    """

    task: str
    table: ContingencyTable
    test: DegradationResult
    q_value: float | None = None
    power: PowerResult | None = None

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "table": self.table.as_dict(),
            "test": self.test.as_dict(),
            "q_value": self.q_value,
            "power": self.power.as_dict() if self.power is not None else None,
        }


# --------------------------------------------------------------------------- #
# Response health -- a validity diagnostic on the raw outputs
# --------------------------------------------------------------------------- #
# A paired run can look like a clean degradation when it is really a token-budget
# artifact: if the candidate runs out of tokens mid-derivation more often than
# the baseline, it loses points for being cut off rather than for being wrong.
# That is a different claim, so the report surfaces it instead of leaving it to
# manual forensics.
_ENDERS = (".", "!", "?", '"', "'", ")", "}", "$", "%")
_TAIL_ANSWER_RE = re.compile(
    r"(?:####|answer|result|total)\D{0,12}-?\$?\d[\d,]*(?:\.\d+)?\s*$",
    re.IGNORECASE,
)


def _looks_truncated(text: str) -> bool:
    """Heuristic: did generation stop mid-thought (i.e. hit the token budget)?

    A response counts as truncated when it is non-empty, does not end on
    terminal punctuation, and does not end with an answer statement (so a
    response ending "#### 83" is not flagged).

    This is a **heuristic**, not ground truth: llama.cpp reports a
    ``finish_reason`` but that is not currently threaded through the backend
    protocol. Replacing this rule with that flag would make the diagnostic
    exact, and is the right fix.

    Caveat on symmetry: the rule is applied identically to both models, but
    that does not make its *error rate* equal across them. An identical rule
    meets two different response-length distributions, and more heavily
    quantized models tend to generate longer, more rambling answers — so they
    are differentially exposed to a length-correlated heuristic. Read the
    truncation counts as an indicator that needs the asymmetry warning below,
    not as a calibrated measurement.
    """
    s = text.rstrip()
    if not s:
        return False
    if s.endswith(_ENDERS):
        return False
    if _TAIL_ANSWER_RE.search(s):
        return False
    return True


@dataclass
class ResponseHealth:
    """Raw-output diagnostics for one model.

    Attributes:
        n: number of responses.
        empty: responses that were blank.
        truncated: responses that look cut off mid-thought (heuristic).
        avg_chars / max_chars: response length, for spotting runaway generation.
    """

    n: int
    empty: int
    truncated: int
    avg_chars: int
    max_chars: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthCheck:
    """Validity diagnostic for a paired run.

    Attributes:
        baseline / candidate: per-model response health.
        b_candidate_truncated: degradation events (b) where the candidate was
            cut off rather than demonstrably wrong.
        c_baseline_truncated: the mirror -- improvement events (c) where the
            baseline was cut off.
        warnings: human-readable validity warnings; empty means nothing found.
    """

    baseline: ResponseHealth
    candidate: ResponseHealth
    b_candidate_truncated: int
    c_baseline_truncated: int
    warnings: list[str]
    b_candidate_unparseable: int = 0
    c_baseline_unparseable: int = 0

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "b_candidate_truncated": self.b_candidate_truncated,
            "c_baseline_truncated": self.c_baseline_truncated,
            "b_candidate_unparseable": self.b_candidate_unparseable,
            "c_baseline_unparseable": self.c_baseline_unparseable,
            "warnings": list(self.warnings),
        }


def _health_of(responses: Sequence[str]) -> ResponseHealth:
    n = len(responses)
    chars = [len(r) for r in responses] or [0]
    return ResponseHealth(
        n=n,
        empty=sum(1 for r in responses if not r.strip()),
        truncated=sum(1 for r in responses if _looks_truncated(r)),
        avg_chars=sum(chars) // max(1, n),
        max_chars=max(chars),
    )


def assess_health(
    baseline_responses: Sequence[str],
    candidate_responses: Sequence[str],
    baseline_scores: Sequence[int],
    candidate_scores: Sequence[int],
    scorers: Sequence[str] | None = None,
) -> HealthCheck:
    """Build the validity diagnostic from raw responses and their scores.

    Args:
        baseline_responses / candidate_responses: raw model outputs, aligned.
        baseline_scores / candidate_scores: the 0/1 scores, aligned.

    Returns:
        A HealthCheck. ``warnings`` is empty when no asymmetry was detected.

    Thresholds (deliberately conservative, documented so they can be argued
    with): >10% truncated for either model; a truncation gap between models
    wider than max(5, 5% of N); or >15% of degradation events where the
    candidate produced no parseable answer.

    On that last threshold: an earlier version counted degradation events where
    the candidate response merely *looked truncated*, and warned that the
    effect size was therefore an upper bound. That is wrong whenever the prompt
    format puts the answer first -- the bundled knowledge prompts end with
    "Answer:", the model replies with the letter and explains afterwards, so the
    token budget cuts the explanation and leaves the answer intact. The warning
    then discredits a perfectly valid verdict. Truncation is only a validity
    threat where the scorer *also* extracted nothing, so the warning is gated
    on that intersection. Raw truncation counts are still reported, because a
    90% truncation rate is worth seeing even when it is harmless.

    Args (continued):
        scorers: per-item scorer names, aligned with the responses. Without
            them, parse failure cannot be determined and the gated warning is
            suppressed rather than guessed at.
    """
    base = _health_of(baseline_responses)
    cand = _health_of(candidate_responses)
    n = len(baseline_scores)
    b_idx = [i for i in range(n)
             if baseline_scores[i] == 1 and candidate_scores[i] == 0]
    c_idx = [i for i in range(n)
             if baseline_scores[i] == 0 and candidate_scores[i] == 1]
    b_ct = sum(1 for i in b_idx if _looks_truncated(candidate_responses[i]))
    c_bt = sum(1 for i in c_idx if _looks_truncated(baseline_responses[i]))

    # Parse failure: the scorer found no answer at all. This, not truncation,
    # is what makes a disagreement uninterpretable.
    if scorers is not None and len(scorers) == n:
        b_cu = sum(1 for i in b_idx
                   if not answer_present(candidate_responses[i], scorers[i]))
        c_bu = sum(1 for i in c_idx
                   if not answer_present(baseline_responses[i], scorers[i]))
    else:
        b_cu = c_bu = 0

    warnings: list[str] = []
    for label, h in (("baseline", base), ("candidate", cand)):
        if h.empty:
            warnings.append(f"{label}: {h.empty}/{h.n} responses were empty")
        if h.n and h.truncated / h.n > 0.10:
            warnings.append(
                f"{label}: {h.truncated}/{h.n} responses look truncated "
                f"({100 * h.truncated / h.n:.0f}%) -- raise --max-tokens"
            )
    if base.n and abs(base.truncated - cand.truncated) > max(5, 0.05 * base.n):
        warnings.append(
            f"ASYMMETRIC truncation (baseline {base.truncated} vs candidate "
            f"{cand.truncated}): part of the measured effect may be a "
            f"token-budget artifact rather than a quality difference. Re-check "
            f"with a larger --max-tokens before trusting the effect size."
        )
    if scorers is not None and b_idx and b_cu / len(b_idx) > 0.15:
        warnings.append(
            f"{b_cu}/{len(b_idx)} degradation events are candidate responses "
            f"the scorer could not read an answer from, rather than "
            f"demonstrated wrong answers -- treat the effect size as an upper "
            f"bound, and re-check with a larger --max-tokens."
        )
    return HealthCheck(base, cand, b_ct, c_bt, warnings,
                       b_candidate_unparseable=b_cu, c_baseline_unparseable=c_bu)


@dataclass
class Report:
    """Full QuantCheck report for one paired run.

    Attributes:
        suite_name / suite_source: identification + provenance of the eval set.
        pooled: TaskResult over all items.
        per_task: one TaskResult per task tag (sorted by task name).
        power: power check for the pooled result.
        significant: True iff pooled p-value < alpha.
        verdict: one of "DEGRADED", "INCONCLUSIVE", "NO_SIGNIFICANT_DEGRADATION".
        plain_summary: a single action-oriented sentence for non-statisticians.
        alpha: significance level used.
        baseline_spec / candidate_spec / generation / seeds: reproducibility log.
        timings: wall-clock seconds per model.
        created_utc: ISO timestamp.
        health: optional validity diagnostic on the raw responses (truncation
            and emptiness, per model and inside the disagreements). None when
            responses were not supplied.
        per_item: optional per-prompt record (prompt, gold, task, both raw
            responses and both scores). Persisted so scoring can be audited or
            re-run offline without re-invoking the models -- a long paired run
            is expensive, and a scorer bug found later should not cost another
            full run. Empty when responses were not supplied.
    """

    suite_name: str
    suite_source: str
    pooled: TaskResult
    per_task: list[TaskResult]
    power: PowerResult
    significant: bool
    verdict: str
    plain_summary: str
    alpha: float
    baseline_spec: dict
    candidate_spec: dict
    generation: dict
    seeds: dict
    timings: dict
    created_utc: str
    per_item: list[dict] = field(default_factory=list)
    health: HealthCheck | None = None

    def as_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "suite_source": self.suite_source,
            "verdict": self.verdict,
            "significant": self.significant,
            "alpha": self.alpha,
            "plain_summary": self.plain_summary,
            "pooled": self.pooled.as_dict(),
            "per_task": [t.as_dict() for t in self.per_task],
            "power": self.power.as_dict(),
            "baseline_spec": self.baseline_spec,
            "candidate_spec": self.candidate_spec,
            "generation": self.generation,
            "seeds": self.seeds,
            "timings": self.timings,
            "created_utc": self.created_utc,
            "per_item": self.per_item,
            "health": self.health.as_dict() if self.health is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full report to JSON (numbers are plain floats/ints)."""

        def _default(o):
            try:
                return float(o)
            except (TypeError, ValueError):
                return str(o)

        return json.dumps(self.as_dict(), indent=indent, default=_default)


def _verdict(
    significant: bool, power: PowerResult
) -> tuple[str, str]:
    """Decide the verdict and a plain-English summary, with honesty guardrails.

    Logic:
        - significant degradation -> "DEGRADED".
        - not significant AND power adequate -> "NO_SIGNIFICANT_DEGRADATION"
          (explicitly *not* a proof of equivalence).
        - not significant AND power inadequate -> "INCONCLUSIVE" (the eval set
          was too small to detect the drop the user cares about).
    """
    if significant:
        return (
            "DEGRADED",
            "The candidate is statistically significantly worse than the baseline "
            "on this suite. Treat this quantization as a real quality regression.",
        )
    mde = (
        f"{power.min_detectable_effect*100:.1f}%"
        if power.min_detectable_effect is not None
        else "any"
    )
    if power.adequate:
        return (
            "NO_SIGNIFICANT_DEGRADATION",
            "No statistically significant degradation was detected, and the eval "
            "set was large enough to catch a "
            f"{power.min_detectable_drop*100:.1f}% drop if one existed. This rules "
            f"out drops larger than about {mde}; smaller ones could still be "
            "present. It is NOT proof the models are equivalent.",
        )
    need = (
        f"~{power.required_n} prompts" if power.required_n is not None
        else "many more prompts"
    )
    return (
        "INCONCLUSIVE",
        "No significant degradation was detected, BUT the eval set was too small "
        f"to reliably catch a {power.min_detectable_drop*100:.1f}% drop. What this "
        f"run could detect is a drop of about {mde} or larger, so anything "
        f"smaller would likely have been missed. Reaching your "
        f"{power.min_detectable_drop*100:.1f}% target needs {need}. Do not read "
        "this as 'passed'.",
    )


def evaluate(
    suite: Suite,
    baseline_scores: Sequence[int],
    candidate_scores: Sequence[int],
    *,
    baseline_responses: Sequence[str] | None = None,
    candidate_responses: Sequence[str] | None = None,
    baseline_spec: dict,
    candidate_spec: dict,
    generation: dict,
    timings: dict,
    alpha: float = 0.05,
    min_detectable_drop: float = 0.02,
    target_power: float = 0.95,
    n_boot: int = 2000,
    bootstrap_seed: int = 0,
) -> Report:
    """Build the full report from per-prompt 0/1 scores.

    Args:
        suite: the evaluation suite (for task tags and provenance).
        baseline_scores / candidate_scores: aligned per-prompt 0/1 scores.
        baseline_responses / candidate_responses: optional raw model outputs,
            aligned with the suite. When given they are recorded per item in
            the report so scoring can be re-audited offline, and the
            response-health diagnostic is computed from them.
        baseline_spec / candidate_spec / generation: reproducibility metadata
            (dicts) logged verbatim into the report.
        timings: wall-clock seconds per model, logged.
        alpha: one-sided significance level.
        min_detectable_drop: the drop the user cares about, for the power check.
        target_power: target power for the power check.
        n_boot: bootstrap resamples for the CI.
        bootstrap_seed: seed for the bootstrap, logged.

    Returns:
        A fully populated Report.

    Raises:
        ValueError: on length mismatch between scores and the suite.
    """
    n = len(suite.items)
    if not (len(baseline_scores) == len(candidate_scores) == n):
        raise ValueError("score arrays must match the suite length")

    # Per-item record + validity diagnostic (only when responses were supplied).
    per_item: list[dict] = []
    health: HealthCheck | None = None
    if baseline_responses is not None and candidate_responses is not None:
        if not (len(baseline_responses) == len(candidate_responses) == n):
            raise ValueError("response arrays must match the suite length")
        health = assess_health(
            baseline_responses, candidate_responses,
            baseline_scores, candidate_scores,
            scorers=[item.scorer for item in suite.items],
        )
        for i, item in enumerate(suite.items):
            per_item.append({
                "i": i,
                "task": item.task,
                "prompt": item.prompt,
                "gold": item.answer,
                "scorer": item.scorer,
                "baseline_response": baseline_responses[i],
                "candidate_response": candidate_responses[i],
                "baseline_score": int(baseline_scores[i]),
                "candidate_score": int(candidate_scores[i]),
            })

    # Pooled result over all items.
    pooled_table = build_contingency(baseline_scores, candidate_scores)
    pooled_test = paired_degradation_test(
        pooled_table,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        n_boot=n_boot,
        ci_level=1 - alpha,
        seed=bootstrap_seed,
    )

    # Per-task results.
    per_task: list[TaskResult] = []
    for task in suite.tasks:
        idx = [i for i, it in enumerate(suite.items) if it.task == task]
        b_t = [baseline_scores[i] for i in idx]
        c_t = [candidate_scores[i] for i in idx]
        table_t = build_contingency(b_t, c_t)
        test_t = paired_degradation_test(
            table_t,
            baseline_scores=b_t,
            candidate_scores=c_t,
            n_boot=n_boot,
            ci_level=1 - alpha,
            seed=bootstrap_seed,
        )
        per_task.append(
            TaskResult(
                task=task,
                table=table_t,
                test=test_t,
                power=power_analysis(
                    n=table_t.n,
                    flip_rate=table_t.flip_rate,
                    min_detectable_drop=min_detectable_drop,
                    target_power=target_power,
                    alpha=alpha,
                ),
            )
        )

    # Multiplicity: one test per task inflates the false-positive rate, so the
    # per-task lines carry BH-adjusted q-values alongside the raw p-values. With
    # a single task there is no family to correct, so q is left as None.
    if len(per_task) > 1:
        q_values = benjamini_hochberg([tr.test.p_value for tr in per_task])
        for tr, q in zip(per_task, q_values):
            tr.q_value = q

    power = power_analysis(
        n=n,
        flip_rate=pooled_table.flip_rate,
        min_detectable_drop=min_detectable_drop,
        target_power=target_power,
        alpha=alpha,
    )

    significant = pooled_test.p_value < alpha
    verdict, plain = _verdict(significant, power)

    return Report(
        suite_name=suite.name,
        suite_source=suite.source,
        pooled=TaskResult(task="ALL", table=pooled_table, test=pooled_test),
        per_task=per_task,
        power=power,
        significant=significant,
        verdict=verdict,
        plain_summary=plain,
        alpha=alpha,
        baseline_spec=baseline_spec,
        candidate_spec=candidate_spec,
        generation=generation,
        seeds={"bootstrap_seed": bootstrap_seed, "generation_seed": generation.get("seed")},
        timings=timings,
        created_utc=datetime.now(timezone.utc).isoformat(),
        per_item=per_item,
        health=health,
    )


# --------------------------------------------------------------------------- #
# Human-readable formatting
# --------------------------------------------------------------------------- #
_VERDICT_BANNER = {
    "DEGRADED": "[X] DEGRADED -- significant quality regression",
    "INCONCLUSIVE": "[!] INCONCLUSIVE -- eval set too small to decide",
    "NO_SIGNIFICANT_DEGRADATION": "[OK] NO SIGNIFICANT DEGRADATION (not proof of equivalence)",
}


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def _fmt_task_line(tr: TaskResult, *, alpha: float = 0.05) -> str:
    t = tr.test
    ci = (
        f"[{_fmt_pct(t.ci_low)}, {_fmt_pct(t.ci_high)}]"
        if t.ci_low is not None
        else "[CI n/a]"
    )
    if tr.q_value is None:
        q = ""
    else:
        flag = " *" if tr.q_value < alpha else "  "
        q = f" q={tr.q_value:.3g}{flag}"
    return (
        f"  {tr.task:<28} N={t.n:<4} b={t.b:<3} c={t.c:<3} "
        f"flip={t.flip_rate*100:5.1f}%  "
        f"drop={_fmt_pct(t.delta_degradation)} CI95={ci}  "
        f"p={t.p_value:.4g}{q}"
    )


def format_report(report: Report, *, color: bool = True) -> str:
    """Render the report as human-readable text for the terminal.

    Always shows, together: the test used, N, b, c, flip rate, p-value, and CI,
    plus the power check and the plain-English verdict. Per-task lines appear
    beneath the pooled line so concentrated degradation is visible.

    Args:
        report: the Report to render.
        color: reserved for future ANSI coloring; currently unused (kept for a
            stable signature).

    Returns:
        A multi-line string.
    """
    p = report.pooled.test
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"QuantCheck verdict -- suite: {report.suite_name}")
    lines.append(f"  source: {report.suite_source}")
    lines.append("=" * 72)
    lines.append(_VERDICT_BANNER.get(report.verdict, report.verdict))
    lines.append("")
    lines.append(f"Test used : {p.test_name}")
    lines.append(
        f"Pooled    : N={p.n}  b(degraded)={p.b}  c(improved)={p.c}  "
        f"flips={p.n_flips} ({p.flip_rate*100:.1f}%)"
    )
    lines.append(
        f"Accuracy  : baseline {report.pooled.table.baseline_accuracy*100:.2f}%  "
        f"-> candidate {report.pooled.table.candidate_accuracy*100:.2f}%"
    )
    ci = (
        f"[{_fmt_pct(p.ci_low)}, {_fmt_pct(p.ci_high)}]"
        if p.ci_low is not None
        else "[CI n/a]"
    )
    lines.append(
        f"Effect    : degradation {_fmt_pct(p.delta_degradation)}  "
        f"(95% bootstrap CI {ci})"
    )
    lines.append(f"p-value   : {p.p_value:.4g}  (one-sided, alpha={report.alpha})")
    lines.append("")

    # Power check — always shown.
    pw = report.power
    lines.append("Design sensitivity (what this run could have detected):")
    if pw.min_detectable_effect is not None:
        lines.append(
            f"  smallest drop detectable: {pw.min_detectable_effect*100:.1f}% "
            f"at {pw.target_power*100:.0f}% power "
            f"(N={pw.n}, flip rate {pw.flip_rate*100:.1f}%)"
        )
    else:
        lines.append(
            f"  smallest drop detectable: none -- at a flip rate of "
            f"{pw.flip_rate*100:.1f}% and N={pw.n} no drop is reliably detectable."
        )
    lines.append(
        f"  your target was {pw.min_detectable_drop*100:.1f}%  ->  "
        f"sensitivity for that target: {pw.achieved_power*100:.0f}%"
    )
    if pw.adequate:
        lines.append("  -> eval set was large enough for your target drop.")
    elif pw.required_n is not None:
        lines.append(
            f"  -> too small for your {pw.min_detectable_drop*100:.1f}% target: "
            f"that would need ~{pw.required_n} prompts at this flip rate."
        )
    else:
        lines.append("  -> no flips observed; cannot detect a drop at all.")
    # Under a DEGRADED verdict this block is about what a *different*, smaller
    # effect would have required. Saying so prevents reading a low sensitivity
    # number as undermining a significant result (the post-hoc power fallacy).
    if report.significant:
        lines.append(
            "  (This run DID detect its effect, so the numbers above do not weaken"
        )
        lines.append(
            "   the verdict -- they describe smaller hypothetical drops. A drop can"
        )
        lines.append(
            "   also be detected while sitting below the minimum above: that bound"
        )
        lines.append(
            f"   is set at {pw.target_power*100:.0f}% power, not at certainty.)"
        )
    else:
        lines.append(
            "  (A null result only rules out drops LARGER than the detectable"
        )
        lines.append(
            "   minimum above. Sensitivity depends on the FLIP RATE, not just N.)"
        )
    lines.append("")

    # Per-task breakdown.
    if len(report.per_task) > 1:
        m = len(report.per_task)
        lines.append(
            f"Per-task breakdown ({m} tasks -- exploratory, read q not p):"
        )
        for tr in report.per_task:
            lines.append(_fmt_task_line(tr, alpha=report.alpha))
        n_sig = sum(
            1 for tr in report.per_task
            if tr.q_value is not None and tr.q_value < report.alpha
        )
        lines.append(
            f"  q = Benjamini-Hochberg adjusted p across these {m} tests; "
            f"'*' marks q < {report.alpha}."
        )
        lines.append(
            f"  {n_sig} of {m} tasks remain significant after correction. "
            f"Raw p is NOT interpretable here: at {m} tests you expect "
            f"~{m*report.alpha:.1f} to clear p<{report.alpha} by chance."
        )
        underpowered = sum(
            1 for tr in report.per_task
            if tr.power is not None and not tr.power.adequate
        )
        if underpowered:
            lines.append(
                f"  {underpowered} of {m} per-task subsets are individually "
                f"underpowered for a {report.power.min_detectable_drop*100:.1f}% "
                f"drop; treat task lines as direction, not evidence."
            )
        lines.append("")

    # Response health -- only when raw responses were recorded.
    h = report.health
    if h is not None:
        lines.append("Response health (validity check on raw outputs):")
        lines.append(
            f"  baseline  : truncated {h.baseline.truncated}/{h.baseline.n}  "
            f"empty {h.baseline.empty}  avg {h.baseline.avg_chars} chars  "
            f"max {h.baseline.max_chars}"
        )
        lines.append(
            f"  candidate : truncated {h.candidate.truncated}/{h.candidate.n}  "
            f"empty {h.candidate.empty}  avg {h.candidate.avg_chars} chars  "
            f"max {h.candidate.max_chars}"
        )
        lines.append(
            f"  cut off inside disagreements  : {h.b_candidate_truncated} of "
            f"b={report.pooled.test.b} (candidate), "
            f"{h.c_baseline_truncated} of c={report.pooled.test.c} (baseline)"
        )
        lines.append(
            f"  NO ANSWER found               : "
            f"{h.b_candidate_unparseable} of b={report.pooled.test.b} (candidate), "
            f"{h.c_baseline_unparseable} of c={report.pooled.test.c} (baseline)"
        )
        lines.append(
            "  (The second line is the one that bears on validity. A response that states"
        )
        lines.append(
            "   its answer and is then cut off mid-explanation is still fully scorable,"
        )
        lines.append(
            "   so a high truncation count on an answer-first prompt format is harmless.)"
        )
        if h.warnings:
            for w in h.warnings:
                lines.append(f"  [!] {w}")
        else:
            lines.append("  -> no truncation or emptiness asymmetry detected.")
        lines.append("  (Truncation is a documented heuristic applied identically to")
        lines.append("   both models -- but its error rate is not equal across them,")
        lines.append("   since response lengths differ; see report._looks_truncated.)")
        lines.append("")

    lines.append("Plain English:")
    lines.append(f"  {report.plain_summary}")
    lines.append("")
    lines.append(
        "Caveats: this measures one scorer on one benchmark; it is not proof of "
        "equivalence, does not capture style/tone/safety, and temp-0 GGUF "
        "inference can be slightly nondeterministic. The p-value is exact and "
        "conditional (it decides the verdict); the CI is an unconditional "
        "bootstrap (it describes magnitude), so they answer different questions "
        "and may disagree at the boundary. See README."
    )
    lines.append("=" * 72)
    return "\n".join(lines)
