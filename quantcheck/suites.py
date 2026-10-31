"""Evaluation suites: bundled small benchmarks + custom JSONL loader.

A *suite* is an ordered list of :class:`SuiteItem`. Each item carries a prompt,
a gold answer, the name of the scorer to use, and a ``task`` tag for per-task
breakdown (see the "aggregate hides the damage" insight in the README).

Bundled suites are stored as JSONL under the package's ``suites/`` directory so
the tool runs out of the box on a laptop. Each bundled suite ships with a
``SOURCE.md`` documenting provenance and licensing. They are intentionally small
(a few hundred items at most) so a full paired run finishes in reasonable time
on CPU.

JSONL item format (one JSON object per line)::

    {"prompt": "...", "answer": "...", "scorer": "numeric", "task": "arithmetic"}

- ``prompt``  (str, required): the text sent to the model.
- ``answer``  (str, required): gold answer, interpreted by the named scorer.
- ``scorer``  (str, optional): scorer name; defaults to the suite's default.
- ``task``    (str, optional): task tag for per-task aggregation; defaults to
  the suite name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .scoring import SCORERS

__all__ = [
    "SuiteItem",
    "Suite",
    "BUNDLED_SUITES",
    "load_suite",
    "load_jsonl_suite",
    "list_bundled_suites",
]

# Bundled suites live next to the installed package: <repo>/suites/<name>.jsonl
_SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"

# Default scorer per bundled suite name.
BUNDLED_SUITES: dict[str, str] = {
    "reasoning": "numeric",          # GSM8K-style arithmetic word problems
    "knowledge": "multiple_choice",  # MMLU-style factual multiple choice
    "instruction": "instruction",    # simple instruction-following prompts
}


@dataclass(frozen=True)
class SuiteItem:
    """One evaluation item.

    Attributes:
        prompt: text sent to the model.
        answer: gold answer string, interpreted by ``scorer``.
        scorer: scorer name (key into :data:`quantcheck.scoring.SCORERS`).
        task: task tag used for per-task aggregation.
    """

    prompt: str
    answer: str
    scorer: str
    task: str


@dataclass(frozen=True)
class Suite:
    """A named collection of evaluation items.

    Attributes:
        name: suite identifier.
        items: ordered list of SuiteItem. Order is preserved and is the pairing
            key — both models see items in this exact order.
        source: human-readable provenance string.
    """

    name: str
    items: list[SuiteItem]
    source: str

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[SuiteItem]:
        return iter(self.items)

    @property
    def tasks(self) -> list[str]:
        """Sorted unique task tags present in the suite."""
        return sorted({it.task for it in self.items})


def _validate_item(obj: dict, default_scorer: str, default_task: str, lineno: int,
                   src: str) -> SuiteItem:
    if "prompt" not in obj or "answer" not in obj:
        raise ValueError(
            f"{src}:{lineno}: each item needs 'prompt' and 'answer' keys; got "
            f"{sorted(obj)}"
        )
    scorer = obj.get("scorer", default_scorer)
    if scorer not in SCORERS:
        raise ValueError(
            f"{src}:{lineno}: unknown scorer {scorer!r}; valid: {sorted(SCORERS)}"
        )
    return SuiteItem(
        prompt=str(obj["prompt"]),
        answer=str(obj["answer"]),
        scorer=scorer,
        task=str(obj.get("task", default_task)),
    )


def load_jsonl_suite(
    path: str | Path,
    *,
    name: str | None = None,
    default_scorer: str = "exact_match",
    limit: int | None = None,
) -> Suite:
    """Load a custom suite from a JSONL file.

    Args:
        path: path to the .jsonl file.
        name: suite name; defaults to the file stem.
        default_scorer: scorer used for items that omit the ``scorer`` field.
        limit: if set, load at most this many items. NOTE: this takes a
            **prefix** of the file, not a fresh random sample. Bundled suites are
            already shuffled at fetch time, so a prefix is an unbiased subset of
            the benchmark -- but on a multi-task suite the per-task counts will
            drift away from equal, so check the realized per-task N in the
            report rather than assuming balance.

    Returns:
        A Suite.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if a line is not valid JSON, an item lacks required keys, or
            names an unknown scorer. The error includes the line number.

    Failure modes:
        - Blank lines are skipped. A trailing newline is fine.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"suite file not found: {p}")
    if default_scorer not in SCORERS:
        raise ValueError(
            f"unknown default_scorer {default_scorer!r}; valid: {sorted(SCORERS)}"
        )

    suite_name = name or p.stem
    items: list[SuiteItem] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from None
            items.append(
                _validate_item(obj, default_scorer, suite_name, lineno, str(p))
            )
            if limit is not None and len(items) >= limit:
                break

    if not items:
        raise ValueError(f"suite {p} contained no items")
    return Suite(name=suite_name, items=items, source=f"custom JSONL: {p}")


def load_suite(name_or_path: str, *, limit: int | None = None) -> Suite:
    """Load a bundled suite by name, or a custom suite by file path.

    If ``name_or_path`` matches a bundled suite name (see :data:`BUNDLED_SUITES`),
    the bundled JSONL is loaded with that suite's default scorer. Otherwise the
    argument is treated as a path to a custom JSONL file.

    Args:
        name_or_path: bundled suite name ("reasoning"/"knowledge"/"instruction")
            or a path to a .jsonl file.
        limit: optional cap on number of items, taken as a prefix of the
            file (see :func:`load_jsonl_suite` for what that means for per-task
            balance).

    Returns:
        A Suite.

    Raises:
        FileNotFoundError: bundled suite name given but its data file is missing,
            or a custom path does not exist.
        ValueError: malformed suite file.
    """
    if name_or_path in BUNDLED_SUITES:
        path = _SUITES_DIR / f"{name_or_path}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"bundled suite {name_or_path!r} expected at {path} but the file "
                f"is missing from the installation"
            )
        suite = load_jsonl_suite(
            path,
            name=name_or_path,
            default_scorer=BUNDLED_SUITES[name_or_path],
            limit=limit,
        )
        return Suite(
            name=suite.name,
            items=suite.items,
            source=f"bundled suite {name_or_path!r} ({path.name})",
        )
    return load_jsonl_suite(name_or_path, limit=limit)


def list_bundled_suites() -> list[str]:
    """Return the names of bundled suites that are present in the installation."""
    return [n for n in BUNDLED_SUITES if (_SUITES_DIR / f"{n}.jsonl").exists()]
