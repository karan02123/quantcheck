"""Worked example: a simulated Q4_K_M candidate that degrades on reasoning.

This script does NOT need any model files. It fabricates a realistic paired
outcome — the candidate is fine on knowledge/instruction but quietly worse on
multi-step reasoning — and runs the *real* QuantCheck statistics + report on it.
Run it to reproduce the output embedded in the README:

    python examples/worked_example.py

The point: the pooled number can look mild while a single task is badly hit.
QuantCheck's per-task breakdown is what surfaces that.
"""

from __future__ import annotations

import numpy as np

from quantcheck.report import evaluate, format_report
from quantcheck.suites import Suite, SuiteItem


def build_simulated_suite_and_scores(seed: int = 0):
    """Fabricate aligned baseline/candidate 0/1 scores with a known structure.

    Structure (per task):
        reasoning   : 120 items, baseline strong; candidate degrades a real ~12%.
        knowledge   : 120 items, candidate essentially unchanged (tiny noise).
        instruction :  60 items, candidate unchanged.

    Returns:
        (suite, baseline_scores, candidate_scores)
    """
    rng = np.random.default_rng(seed)
    items: list[SuiteItem] = []
    base: list[int] = []
    cand: list[int] = []

    def add(task, n, base_acc, degrade, improve):
        for _ in range(n):
            items.append(SuiteItem(prompt="(simulated)", answer="x",
                                   scorer="exact_match", task=task))
            b = int(rng.random() < base_acc)
            base.append(b)
            u = rng.random()
            if b == 1 and u < degrade:        # baseline right -> candidate wrong
                cand.append(0)
            elif b == 0 and u < improve:      # baseline wrong -> candidate right
                cand.append(1)
            else:
                cand.append(b)

    # reasoning: real degradation (P(degrade | base correct) ~ 0.18 of an 0.8 base)
    add("reasoning", 120, base_acc=0.80, degrade=0.18, improve=0.03)
    # knowledge: essentially unchanged
    add("knowledge", 120, base_acc=0.75, degrade=0.03, improve=0.03)
    # instruction: unchanged
    add("instruction", 60, base_acc=0.90, degrade=0.02, improve=0.02)

    suite = Suite(name="demo-mixed", items=items, source="examples/worked_example.py (simulated)")
    return suite, base, cand


def main() -> None:
    suite, base, cand = build_simulated_suite_and_scores(seed=0)
    report = evaluate(
        suite,
        base,
        cand,
        baseline_spec={"path_or_id": "tinyllama-f16.gguf", "backend": "gguf"},
        candidate_spec={"path_or_id": "tinyllama-q4_k_m.gguf", "backend": "gguf"},
        generation={"temperature": 0.0, "seed": 0, "max_tokens": 256},
        timings={"baseline_s": 0.0, "candidate_s": 0.0},
        alpha=0.05,
        min_detectable_drop=0.02,
        target_power=0.95,
        n_boot=2000,
        bootstrap_seed=0,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
