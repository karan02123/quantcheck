"""Deterministic scorers: ``L(response) -> {0, 1}`` for each suite type.

The contingency table in :mod:`quantcheck.stats` is built from binary per-prompt
outcomes. A *scorer* turns a model's raw text response (plus the gold answer)
into that 0/1 outcome. Scoring must be **deterministic** and **well-defined**:
the same response + gold must always yield the same score, on both models, so
the comparison isolates the quantization effect rather than scorer noise.

Each scorer documents its parsing rule and its **known failure modes**, because
a scorer that mis-parses one model's formatting more than the other's would
manufacture a fake degradation. When in doubt, prefer a lenient-but-symmetric
rule (applied identically to baseline and candidate).
"""

from __future__ import annotations

import re
from typing import Callable, Protocol

__all__ = [
    "Scorer",
    "multiple_choice_scorer",
    "numeric_answer_scorer",
    "instruction_keyword_scorer",
    "exact_match_scorer",
    "get_scorer",
    "SCORERS",
]


class Scorer(Protocol):
    """A scorer maps (response, gold_answer) -> 1 (correct) or 0 (incorrect)."""

    def __call__(self, response: str, gold: str) -> int: ...


# --------------------------------------------------------------------------- #
# Multiple choice (MMLU-style: answer is a single letter A/B/C/D)
# --------------------------------------------------------------------------- #
_MC_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-D])\)?[\.\):]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b([A-D])\b"),  # last-resort: first standalone A-D token
]


def multiple_choice_scorer(response: str, gold: str) -> int:
    """Score a multiple-choice answer by extracting a single letter A-D.

    Parsing rule (tried in order, first match wins):
        1. "answer is X" / "answer: X"
        2. "option X" / "choice X"
        3. a line beginning with "X." / "X)" / "X:"
        4. the first standalone A-D token anywhere in the text

    The same rule is applied to both models, so any parser bias is symmetric.

    Args:
        response: raw model output.
        gold: the correct option letter (e.g. "B"); case-insensitive.

    Returns:
        1 if the extracted letter equals gold, else 0.

    Known failure modes:
        - If the model restates the question (which contains the option letters)
          before answering, rule 4 may grab the wrong letter. Rules 1–3 guard
          against this for well-formatted answers.
        - A model that never emits a letter scores 0 (treated as incorrect),
          which is the honest outcome for an unparseable answer.
    """
    gold_letter = gold.strip().upper()[:1]
    for pat in _MC_PATTERNS:
        m = pat.search(response)
        if m:
            return int(m.group(1).upper() == gold_letter)
    return 0


# --------------------------------------------------------------------------- #
# Numeric answer (GSM8K-style arithmetic)
# --------------------------------------------------------------------------- #
_NUM = r"(-?\$?\d[\d,]*(?:\.\d+)?)"
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

# Answer cues, strongest first. Within a tier we take the LAST match, because in
# a chain-of-thought answer the final result comes last and intermediate steps
# come earlier. A bare "=" is deliberately the weakest tier: every intermediate
# arithmetic step contains one, so treating it as a strong cue makes the scorer
# grab a working number instead of the answer.
_CUE_TIERS = (
    # Tier 1: the delimiter the bundled suites explicitly instruct models to use.
    re.compile(r"####\s*" + _NUM),
    # Tier 2: an explicit answer statement.
    re.compile(r"(?:final\s+)?(?:answer|result)\s*(?:is|:|=)?\s*" + _NUM,
               re.IGNORECASE),
    # Tier 3: a stated total.
    re.compile(r"total\s*(?:is|:|=)?\s*" + _NUM, re.IGNORECASE),
    # Tier 4: bare equals — last resort before "last number in the text".
    re.compile(r"=\s*" + _NUM),
)


def _to_number(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def numeric_answer_scorer(response: str, gold: str) -> int:
    """Score an arithmetic answer by extracting the final number.

    Parsing rule (tiered; the first tier that matches wins, and within that
    tier the **last** match is used):
        1. "#### <number>" — the delimiter the bundled suites instruct models
           to emit.
        2. "answer is/: <number>" or "result ...".
        3. "total ... <number>".
        4. "= <number>" — weakest, because every intermediate arithmetic step
           has one.
        5. Otherwise the **last** number anywhere in the response.
        Taking the last match matters: chain-of-thought answers state the
        result last, so a first-match rule returns an intermediate working
        value instead of the answer.
        Comparison is numeric with a small tolerance (1e-6), so "42", "42.0",
        "$42", and "42" all match a gold of 42.

    Args:
        response: raw model output.
        gold: the correct numeric answer as a string (e.g. "42" or "3.5").

    Returns:
        1 if the extracted number equals gold within tolerance, else 0.

    Known failure modes:
        - If the model prints intermediate arithmetic *after* the final answer
          with no answer cue, the last-number fallback may grab the wrong one.
          The tiered cue rules mitigate this for models that mark their answer.
        - The same rule is applied to both models, so any residual parser bias
          is symmetric and cannot manufacture a one-sided degradation.
        - Non-numeric gold raises ValueError (a suite authoring bug, surfaced
          early rather than silently scored 0).
    """
    gold_num = _to_number(gold.strip())
    if gold_num is None:
        raise ValueError(f"numeric scorer received non-numeric gold answer: {gold!r}")

    for cue in _CUE_TIERS:
        matches = cue.findall(response)
        if matches:
            # Last match in the strongest tier that fired.
            val = _to_number(matches[-1])
            if val is not None:
                return int(abs(val - gold_num) < 1e-6)

    numbers = _NUMBER_RE.findall(response)
    for tok in reversed(numbers):
        val = _to_number(tok)
        if val is not None:
            return int(abs(val - gold_num) < 1e-6)
    return 0


# --------------------------------------------------------------------------- #
# Instruction following (keyword / constraint check)
# --------------------------------------------------------------------------- #
def instruction_keyword_scorer(response: str, gold: str) -> int:
    """Score instruction-following by requiring all gold keywords be present.

    The gold field is a ``|``-separated list of required substrings, matched
    case-insensitively. The response passes (1) only if **every** required
    substring appears. This is a deliberately simple, deterministic proxy for
    instruction adherence (e.g. "include the word 'banana'" -> gold "banana").

    Args:
        response: raw model output.
        gold: ``|``-separated required substrings (e.g. "paris|france").

    Returns:
        1 if all required substrings are present, else 0.

    Known failure modes:
        - Substring matching cannot verify structural constraints (e.g. "exactly
          3 bullet points"); it only checks presence. Suites should phrase
          constraints as keyword presence to stay within this scorer's ability.
        - Empty gold ("") trivially passes; suite authors should avoid empty gold.
    """
    required = [s.strip().lower() for s in gold.split("|") if s.strip()]
    low = response.lower()
    return int(all(sub in low for sub in required))


# --------------------------------------------------------------------------- #
# Exact match (generic fallback)
# --------------------------------------------------------------------------- #
def exact_match_scorer(response: str, gold: str) -> int:
    """Score by case-insensitive, whitespace-normalized exact match.

    The gold string must appear (normalized) as a contiguous span in the
    normalized response. Useful for short factual answers.

    Args:
        response: raw model output.
        gold: the expected answer string.

    Returns:
        1 if gold (normalized) is contained in the response (normalized), else 0.

    Known failure modes:
        - Verbose responses that paraphrase the answer without the exact span
          score 0. Use a more specific scorer when the answer format is known.
    """
    norm_resp = " ".join(response.lower().split())
    norm_gold = " ".join(gold.lower().split())
    return int(norm_gold in norm_resp)


SCORERS: dict[str, Scorer] = {
    "multiple_choice": multiple_choice_scorer,
    "numeric": numeric_answer_scorer,
    "instruction": instruction_keyword_scorer,
    "exact_match": exact_match_scorer,
}


def get_scorer(name: str) -> Scorer:
    """Look up a scorer by name.

    Args:
        name: one of ``SCORERS`` ("multiple_choice", "numeric", "instruction",
            "exact_match").

    Returns:
        The scorer callable.

    Raises:
        ValueError: if the name is unknown, listing the valid options.
    """
    try:
        return SCORERS[name]
    except KeyError:
        raise ValueError(
            f"unknown scorer {name!r}; valid scorers: {sorted(SCORERS)}"
        ) from None
