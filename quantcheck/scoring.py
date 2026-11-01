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
    "extract_choice",
    "numeric_answer_scorer",
    "extract_number",
    "instruction_keyword_scorer",
    "exact_match_scorer",
    "answer_present",
    "get_scorer",
    "SCORERS",
]


class Scorer(Protocol):
    """A scorer maps (response, gold_answer) -> 1 (correct) or 0 (incorrect)."""

    def __call__(self, response: str, gold: str) -> int: ...


# --------------------------------------------------------------------------- #
# Multiple choice (MMLU-style: answer is a single letter A/B/C/D)
# --------------------------------------------------------------------------- #
# A bare option letter at the very start of the response. The bundled knowledge
# prompts end with "Answer:", so a model answering directly -- the raw-completion
# regime -- emits the letter immediately and nothing else matters. Anchored, so
# there is no ambiguity about which letter it is.
_MC_ANCHORED = re.compile(r"^\s*\**\(?([A-D])\)?\**\s*(?:[.):,\]]|$|\s)")

# Answer cues, strongest first. Within a tier the LAST match wins: in a
# chain-of-thought answer the conclusion comes last and the options being
# weighed up come earlier. This mirrors _CUE_TIERS in numeric_answer_scorer;
# the two scorers deliberately share the convention.
_MC_TIERS = (
    # Tier 1: an explicit answer statement. "the answer is C", "Answer: C".
    re.compile(r"(?:final\s+)?answer\s*(?:is|:|=)?\s*\**\(?([A-D])\)?\**",
               re.IGNORECASE),
    # Tier 2: the letter asserted as correct. "C is the correct option."
    re.compile(r"\**\(?([A-D])\)?\**\s+is\s+(?:the\s+)?"
               r"(?:correct|right|best|answer)", re.IGNORECASE),
    # Tier 3: "option C" / "choice C". Weak on its own -- a response that
    # dismisses options one by one is full of these -- but the last one is
    # usually the conclusion.
    re.compile(r"\b(?:option|choice)\s*\**\(?([A-D])\)?\**", re.IGNORECASE),
    # Tier 4: a line that begins with the letter, e.g. "C) Tokyo".
    re.compile(r"^\s*\**\(?([A-D])\)?\**[.):]", re.MULTILINE),
    # Tier 5: last resort -- the last standalone A-D token in the text.
    re.compile(r"\b([A-D])\b"),
)


def extract_choice(response: str) -> str | None:
    """Extract the answer letter from a multiple-choice response, or None.

    Exposed separately from the scorer so the parsing rule can be tested and
    audited on its own, independently of whether it matches a gold answer.
    """
    # An explicit answer statement always outranks position. A response may
    # open by quoting an option it goes on to reject -- "A) Seoul is not the
    # capital. The correct answer is D." -- so the anchor must yield to tiers
    # 1 and 2 rather than short-circuit them.
    for tier in _MC_TIERS[:2]:
        matches = tier.findall(response)
        if matches:
            return matches[-1].upper()

    anchored = _MC_ANCHORED.match(response)
    if anchored:
        return anchored.group(1).upper()

    for tier in _MC_TIERS[2:]:
        matches = tier.findall(response)
        if matches:
            return matches[-1].upper()
    return None


def multiple_choice_scorer(response: str, gold: str) -> int:
    """Score a multiple-choice answer by extracting a single letter A-D.

    Parsing rule:
        0. A bare letter at the very start of the response wins outright. The
           prompts end with "Answer:", so this is a model answering directly.
        Otherwise, tiers are tried strongest-first and the **last** match within
        the winning tier is taken:
        1. "answer is X" / "answer: X" / "final answer X"
        2. "X is the correct answer" / "X is right"
        3. "option X" / "choice X"
        4. a line beginning with "X." / "X)" / "X:"
        5. the last standalone A-D token anywhere in the text

    Why last-match: in a chain-of-thought answer the conclusion comes last,
    while the options under consideration come earlier. An earlier version took
    the *first* match and therefore scored "Option A is incorrect, ... the
    answer is C" as A. See tests/test_scoring.py for the regression tests.

    Args:
        response: raw model output.
        gold: the correct option letter (e.g. "B"); case-insensitive.

    Returns:
        1 if the extracted letter equals gold, else 0.

    Known failure modes:
        - A response that names the right letter and then keeps talking about
          another option can still be mis-parsed by tier 3 or 5.
        - A model that never emits a letter scores 0, which is the honest
          outcome for an unparseable answer.
        - The same rule is applied to both models, but that makes the parser
          bias *symmetrically applied*, not symmetric in effect. If one model
          is markedly more verbose than the other -- and quantization does
          change response length, measurably so in examples/REAL_RUN.md -- then
          it offers more text for a weak tier to trip over, and the error rates
          are not equal. Treat a verdict built on heavy chain-of-thought
          multiple-choice output with corresponding caution.
    """
    gold_letter = gold.strip().upper()[:1]
    found = extract_choice(response)
    return int(found is not None and found == gold_letter)


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

    val = extract_number(response)
    if val is None:
        return 0
    return int(abs(val - gold_num) < 1e-6)


def extract_number(response: str) -> float | None:
    """Extract the answer number from a response, or None if there is none.

    Holds the tiered rule documented on :func:`numeric_answer_scorer`, split
    out so that *whether an answer exists* can be asked without also asking
    whether it is correct. :func:`answer_present` needs exactly that: a
    response cut off before stating any number is unscorable, while one that
    states its answer and is then truncated mid-explanation is fine.
    """
    for cue in _CUE_TIERS:
        matches = cue.findall(response)
        if matches:
            # Last match in the strongest tier that fired.
            val = _to_number(matches[-1])
            if val is not None:
                return val

    for tok in reversed(_NUMBER_RE.findall(response)):
        val = _to_number(tok)
        if val is not None:
            return val
    return None


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


def answer_present(response: str, scorer: str) -> bool:
    """Did this response contain anything the scorer could read as an answer?

    This is deliberately *not* "was the answer correct". It separates two very
    different failures that both score 0:

      * the model answered and was wrong -- a real quality signal;
      * the model never produced a parseable answer -- which may just mean it
        ran out of tokens, and says little about what it knows.

    The distinction is what makes the truncation diagnostic meaningful. A
    response that states its answer and is then cut off mid-explanation is
    fully scorable, so flagging it as a validity threat is a false alarm. See
    :func:`quantcheck.report.assess_health`.

    Scorers whose answer is the whole response (``instruction``,
    ``exact_match``) are treated as always parseable: there is no extraction
    step that can fail, so truncation cannot create this particular ambiguity.

    Args:
        response: raw model output.
        scorer: the scorer name from the suite item.

    Returns:
        True if an answer could be extracted (or if extraction does not apply).
    """
    if scorer == "multiple_choice":
        return extract_choice(response) is not None
    if scorer == "numeric":
        return extract_number(response) is not None
    return True


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
