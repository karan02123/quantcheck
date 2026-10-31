"""Re-render a saved result JSON through the current reporting code.

Why this exists
---------------
A paired run costs hours of inference, but the *analysis* of it is cheap. Every
result file stores the full 2x2 tables (pooled and per-task), so improvements to
the reporting layer -- multiplicity correction, the minimum-detectable-effect
inversion, reworded verdicts -- can be applied retroactively to runs that were
executed under older code. Without this, fixing a reporting bug would mean
re-running the models.

This script reads **only** the stored contingency counts and re-derives
everything statistical from them. It does not load a model, does not re-score
responses, and does not touch the input file.

Scope: counts (a, b, c, d) are sufficient statistics for the test, the bootstrap
CI, and the sensitivity analysis, so the re-derived numbers are exactly what a
fresh run under current code would produce -- except the bootstrap CI, which is
re-drawn from the reconstructed per-prompt scores and so is identical only up to
the resampling seed.

Usage:

    python scripts/rerender_report.py results/quantcheck_knowledge_*.json
    python scripts/rerender_report.py results/*.json --write
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantcheck.report import Report, TaskResult, _verdict  # noqa: E402
from quantcheck.stats import (  # noqa: E402
    ContingencyTable,
    benjamini_hochberg,
    build_contingency,
    paired_degradation_test,
    power_analysis,
)
from quantcheck.report import format_report  # noqa: E402


def _scores_from_counts(t: ContingencyTable) -> tuple[list[int], list[int]]:
    """Rebuild per-prompt 0/1 score vectors consistent with a 2x2 table.

    Prompt order is lost in the counts, but order does not affect the test, the
    effect size, or the bootstrap distribution -- only the pairing pattern does,
    and that is exactly what (a, b, c, d) preserves.
    """
    base = [1] * (t.a + t.b) + [0] * (t.c + t.d)
    cand = [1] * t.a + [0] * t.b + [1] * t.c + [0] * t.d
    return base, cand


def _task_from_dict(d: dict, *, alpha: float, n_boot: int, seed: int,
                    min_drop: float, target_power: float) -> TaskResult:
    tbl = d["table"]
    table = ContingencyTable(a=tbl["a"], b=tbl["b"], c=tbl["c"], d=tbl["d"])
    base, cand = _scores_from_counts(table)
    test = paired_degradation_test(
        table, baseline_scores=base, candidate_scores=cand,
        n_boot=n_boot, ci_level=1 - alpha, seed=seed,
    )
    return TaskResult(
        task=d["task"],
        table=table,
        test=test,
        power=power_analysis(
            n=table.n, flip_rate=table.flip_rate,
            min_detectable_drop=min_drop, target_power=target_power, alpha=alpha,
        ),
    )


def rerender(path: Path) -> tuple[str, Report]:
    """Re-derive a Report from a saved result JSON and render it."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    alpha = raw.get("alpha", 0.05)
    pw_old = raw.get("power", {}) or {}
    min_drop = pw_old.get("min_detectable_drop", 0.02)
    target_power = pw_old.get("target_power", 0.95)
    seed = (raw.get("seeds", {}) or {}).get("bootstrap_seed", 0) or 0
    n_boot = 2000

    kw = dict(alpha=alpha, n_boot=n_boot, seed=seed,
              min_drop=min_drop, target_power=target_power)
    pooled = _task_from_dict(raw["pooled"], **kw)
    pooled.power = None  # pooled power is reported as the headline check
    per_task = [_task_from_dict(t, **kw) for t in raw.get("per_task", [])]

    if len(per_task) > 1:
        for tr, q in zip(per_task, benjamini_hochberg([t.test.p_value for t in per_task])):
            tr.q_value = q

    power = power_analysis(
        n=pooled.table.n, flip_rate=pooled.table.flip_rate,
        min_detectable_drop=min_drop, target_power=target_power, alpha=alpha,
    )
    significant = pooled.test.p_value < alpha
    verdict, plain = _verdict(significant, power)

    report = Report(
        suite_name=raw.get("suite_name", "?"),
        suite_source=raw.get("suite_source", "?") + f"  [re-rendered from {path.name}]",
        pooled=pooled,
        per_task=per_task,
        power=power,
        significant=significant,
        verdict=verdict,
        plain_summary=plain,
        alpha=alpha,
        baseline_spec=raw.get("baseline_spec", {}),
        candidate_spec=raw.get("candidate_spec", {}),
        generation=raw.get("generation", {}),
        seeds=raw.get("seeds", {}),
        timings=raw.get("timings", {}),
        created_utc=raw.get("created_utc", ""),
    )
    return format_report(report), report


def main(argv: list[str] | None = None) -> int:
    pa = argparse.ArgumentParser(prog="rerender_report")
    pa.add_argument("paths", nargs="+", help="result JSON files (globs allowed)")
    pa.add_argument("--write", action="store_true",
                    help="also save alongside as <name>.rerendered.txt")
    args = pa.parse_args(argv)

    files: list[Path] = []
    for pat in args.paths:
        files.extend(Path(p) for p in sorted(glob.glob(pat)))
    if not files:
        print("no files matched", file=sys.stderr)
        return 1

    for f in files:
        try:
            text, report = rerender(f)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"-- {f.name}: SKIPPED ({type(e).__name__}: {e})")
            continue
        base = report.baseline_spec.get("path_or_id", "?")
        cand = report.candidate_spec.get("path_or_id", "?")
        print(f"\n########## {f.name}")
        print(f"########## {Path(str(base)).name}  vs  {Path(str(cand)).name}\n")
        print(text)
        if args.write:
            out = f.with_suffix(".rerendered.txt")
            out.write_text(text, encoding="utf-8")
            print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
