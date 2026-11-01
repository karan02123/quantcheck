# Measured runs — the evidence behind `examples/REAL_RUN.md`

Every paired run reported in [`../REAL_RUN.md`](../REAL_RUN.md), as the tool
emitted it. These files are **tracked in git on purpose.** A project whose claim
is honest measurement has to let a reader check the measurements without
spending 31 hours of inference to reproduce them.

`results/` is git-ignored scratch space; this directory is the curated record.
Files here are byte-identical copies of what `quantcheck` wrote — unedited, and
verified by SHA-256 against their originals.

## Index

| Run | File stem | Suite | Pair | N | b | c | p | Template | Compute | Auditable | Verdict |
|-----|-----------|-------|------|---|---|---|---|----------|---------|-----------|---------|
| 1 | `quantcheck_knowledge_20260609T170942` | knowledge | Llama-3.2-3B-Instruct-Q8_0 → Llama-3.2-3B-Instruct-Q4_K_M | 200 | 5 | 13 | 0.9846 | (raw) | 1.0 h | counts only | INCONCLUSIVE |
| 2 | `quantcheck_knowledge_20260609T203502` | knowledge | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q4_K_M | 200 | 11 | 15 | 0.8365 | (raw) | 1.0 h | counts only | INCONCLUSIVE |
| 3 | `quantcheck_knowledge_20260610T071231` | knowledge | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q2_K | 200 | 32 | 20 | 0.06317 | (raw) | 1.7 h | counts only | INCONCLUSIVE |
| 4 | `quantcheck_knowledge_20260610T122700` | knowledge | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q2_K | 500 | 89 | 47 | 0.0001993 | (raw) | 4.6 h | counts only | DEGRADED |
| 5 | `quantcheck_reasoning_20261030T160054` | reasoning | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q2_K | 200 | 33 | 14 | 0.003971 | gemma | 3.0 h | yes | DEGRADED |
| 6 | `quantcheck_reasoning_20261030T222037` | reasoning | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q4_K_M | 200 | 7 | 10 | 0.8338 | gemma | 3.1 h | yes | INCONCLUSIVE |
| 7 | `quantcheck_reasoning_20261031T175024` | reasoning | Qwen2.5-7B-Instruct-Q8_0 → Qwen2.5-7B-Instruct-Q4_K_M | 500 | 16 | 11 | 0.221 | qwen | 14.2 h | yes | INCONCLUSIVE |
| 8 | `quantcheck_knowledge_20261101T131526` | knowledge | gemma-3-4b-it-Q8_0 → gemma-3-4b-it-Q2_K | 500 | 80 | 41 | 0.0002493 | (raw) | 2.9 h | yes | DEGRADED |

`b` = baseline correct, candidate wrong (degradation). `c` = the reverse.
Degradation is reported as a **positive** number of percentage points worse,
δ=(b−c)/N. Every number in this table was read out of the JSON files beside it,
not transcribed by hand. Total measured inference: **31.4 hours**.

**The "Auditable" column is the one to notice.** Runs 1–4 predate `per_item`
persistence and stored counts only. That is enough to re-derive the test, the
effect size, the CI and the sensitivity analysis — counts are sufficient
statistics — but *not* enough to re-check the scoring. When a bug was later
found in `multiple_choice_scorer`, run 4 could not be re-analysed and had to be
re-run from scratch as **run 8**. Runs 5–8 would not need that.

Run 8 supersedes run 4 on scoring; run 4's counts are kept as the historical
record. The verdict was unchanged (DEGRADED both times) but the effect moved
from +8.40% to +7.80%. See *Run 8* in `REAL_RUN.md`.

## What each file holds

- **`.json`** — the full result: both 2×2 contingency tables (pooled and
  per-task), model specs, every generation setting and seed, timings, response
  health counts, and (runs 5–8) `per_item` with the complete prompt, gold
  answer, and both models' raw responses.
- **`.txt`** — the human-readable report printed at the end of the run.

Any of these runs can be re-analysed under current code with **no model and no
re-scoring**:

```
python scripts/rerender_report.py examples/runs/quantcheck_reasoning_20261031T175024.json
```

That is how runs executed under older reporting code acquired the
minimum-detectable-effect and Benjamini–Hochberg output they now carry. Run 7
was re-rendered as a check and reproduced its live report exactly.

Runs 5–8 also carry `per_item`, so their responses can be **re-scored** offline
if a scorer bug is found later — which is not hypothetical: one was, and
`REAL_RUN.md` documents what it cost.

## Spot-checks

`spotcheck_*.jsonl` are the cheap pre-run gates (`scripts/template_spotcheck.py`,
10 items, one model) that decide whether a prompt format is wired correctly
before committing hours to a full run. Each line holds one item's prompt, gold
answer, and the raw and chat-templated responses with token counts and
truncation flags.

These are evidence for a finding, not just a diagnostic: the Gemma gate showed
raw and templated prompting scoring identically, the Qwen gate showed templated
prompting winning decisively on format compliance (p=0.031). The template's
effect is model-dependent, so the gate is not skippable. See
*The template's effect is model-dependent* in `REAL_RUN.md`.

## Not included

Two `backend=mock` files (N=10 and N=2) from early development, and a smoke-test
directory. They measure nothing and were deleted rather than shipped as if they
were results.
