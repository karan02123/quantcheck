# Running QuantCheck on real GGUF files

This page documents real measured runs on real GGUF files — not the mock/synthetic
example in the main README. Numbers here are actual outputs.

---

## Run 1 — Llama-3.2-3B-Instruct Q8_0 vs Q4_K_M, MMLU knowledge suite

**Date:** 2026-06-09  
**Models:** `bartowski/Llama-3.2-3B-Instruct-GGUF`  
**Backend:** CPU only (`--n-gpu-layers 0`)  
**Suite:** `knowledge` (200 real MMLU test-split items, 14 subjects, seed=42)  
**Command:**
```powershell
quantcheck `
  --baseline  models\Llama-3.2-3B-Instruct-Q8_0.gguf `
  --candidate models\Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  --suite knowledge --n-gpu-layers 0 --max-tokens 64
```

**Full output:**
```
========================================================================
QuantCheck verdict -- suite: knowledge
  source: bundled suite 'knowledge' (knowledge.jsonl)
========================================================================
[!] INCONCLUSIVE -- eval set too small to decide

Test used : exact one-sided McNemar (binomial sign test on disagreements)
Pooled    : N=200  b(degraded)=5  c(improved)=13  flips=18 (9.0%)
Accuracy  : baseline 42.50%  -> candidate 46.50%
Effect    : degradation -4.00%  (95% bootstrap CI [-8.01%, +0.00%])
p-value   : 0.9846  (one-sided, alpha=0.05)

Power check:
  target drop to detect : 2.0%   target power : 95%
  achieved power        : 24% (at flip rate 9.0%, N=200)
  -> UNDERPOWERED: need ~2375 prompts to detect a 2.0% drop at this flip rate.

Per-task breakdown:
  college-mathematics   N=15  b=2  c=1  drop=+6.67%  p=0.5
  logical-fallacies     N=13  b=1  c=0  drop=+7.69%  p=0.5
  [12 other tasks: no degradation signal, all p>0.5]

Plain English:
  No significant degradation was detected, BUT the eval set was too small to
  reliably catch a 2.0% drop (estimated power 24%). Do not read this as 'passed'.
========================================================================
```

### What this result means

**The tool worked correctly.** Load/unload lifecycle on real GGUFs functioned as
designed — one model in memory at a time confirmed by RAM monitoring.

**INCONCLUSIVE is the honest verdict here.** With 18 flips across 200 items
(9% flip rate), the test has only ~24% power to detect a 2% drop. That is not
"the models are equivalent" — it is "the data cannot distinguish." The tool
correctly refuses to call this "passed."

**The Q4_K_M appears *slightly better* than Q8_0 on this run** (b=5, c=13).
This is within noise — the 95% CI spans [−8%, 0%] and p=0.98. Don't read it as
"quantization improves quality"; read it as "18 flips is not enough signal to
conclude anything about direction."

**The 42.5% baseline accuracy is lower than expected** (~58–63% is typical for
Llama-3.2-3B on MMLU). The reason: this is an instruct-tuned model being run
with raw completion prompts — no chat template (`<|user|>...<|assistant|>`
wrapping) is applied. Instruct models underperform on raw prompts. This is a
known gap in QuantCheck's current implementation (see below).

---

## Run 2 — Gemma-3-4b-it Q8_0 vs Q4_K_M, MMLU knowledge suite

**Date:** 2026-06-09  
**Models:** `bartowski/google_gemma-3-4b-it-GGUF`  
**Backend:** CPU only (`--n-gpu-layers 0`), llama-cpp-python 0.3.28 built from source  
**Suite:** `knowledge` (200 real MMLU test-split items, 14 subjects, seed=42)  
**Command:**
```powershell
quantcheck `
  --baseline  models\google_gemma-3-4b-it-Q8_0.gguf `
  --candidate models\google_gemma-3-4b-it-Q4_K_M.gguf `
  --suite knowledge --n-gpu-layers 0 --max-tokens 64
```

**Result:**
```
[!] INCONCLUSIVE -- eval set too small to decide

Pooled    : N=200  b(degraded)=11  c(improved)=15  flips=26 (13.0%)
Accuracy  : baseline 45.00%  -> candidate 47.00%
Effect    : degradation -2.00%  (95% bootstrap CI [-7.00%, +3.00%])
p-value   : 0.8365  (one-sided, alpha=0.05)

Power check:
  achieved power : 19% (at flip rate 13.0%, N=200)
  -> UNDERPOWERED: need ~3476 prompts to detect a 2.0% drop at this flip rate.
```

### What this result means

Same honest verdict as Run 1, on a different model family: **INCONCLUSIVE**.
Gemma-3-4B's Q4_K_M is statistically indistinguishable from its Q8_0 on MMLU at
this N — candidate marginally *higher* (within noise; CI [-7%, +3%], p=0.84).

### Cross-run finding (the actual takeaway)

Across **two independent model families** (Llama-3.2-3B Run 1, Gemma-3-4B Run 2),
Q8 -> Q4_K_M quantization shows **no detectable accuracy degradation on MMLU**,
and the flip rate is low enough (9-13%) that **200 items cannot conclude either
way** — the tool correctly reports this rather than rubber-stamping "passed."

This is the tool doing its job: a naive eval would have printed "Q4 scored 47%,
Q8 scored 45%, ship it" and been quietly wrong about its own certainty.
QuantCheck instead says: *the data cannot support a 2% claim; you would need
~3500 prompts.* That power-check honesty is the entire point.

(Note Gemma's flip rate 13% > Llama's 9%, and Gemma-it responds usefully to raw
MMLU prompts despite the chat-template gap below — its baseline 45% is closer to
expected than Llama-instruct's 42.5%.)

---

## Run 4 — Gemma-3-4b-it Q8_0 vs Q2_K, MMLU knowledge suite, N=500 (DEGRADED)

**Date:** 2026-06-10  
**Models:** `bartowski/google_gemma-3-4b-it-GGUF`  
**Backend:** CPU only (`--n-gpu-layers 0`)  
**Suite:** `knowledge` (500 real MMLU test-split items, 14 subjects)  
**Command:**
```powershell
quantcheck `
  --baseline  models\google_gemma-3-4b-it-Q8_0.gguf `
  --candidate models\google_gemma-3-4b-it-Q2_K.gguf `
  --suite knowledge --n-gpu-layers 0 --max-tokens 64
```

**Full output:**
```
========================================================================
QuantCheck verdict -- suite: knowledge
  source: bundled suite 'knowledge' (knowledge.jsonl)
========================================================================
[X] DEGRADED -- significant quality regression

Test used : exact one-sided McNemar (binomial sign test on disagreements)
Pooled    : N=500  b(degraded)=89  c(improved)=47  flips=136 (27.2%)
Accuracy  : baseline 47.20%  -> candidate 38.80%
Effect    : degradation +8.40%  (95% bootstrap CI [+4.00%, +12.80%])
p-value   : 0.0001993  (one-sided, alpha=0.05)

Power check:
  target drop to detect : 2.0%   target power : 95%
  achieved power        : 21% (at flip rate 27.2%, N=500)
  -> UNDERPOWERED: need ~7340 prompts to detect a 2.0% drop at this flip rate.
  (Power depends on the FLIP RATE, not just N: few disagreements
   means little signal, so more prompts are needed.)

Per-task breakdown (watch for damage hidden by the pooled number):
  abstract-algebra          N=36  b=8   c=2  flip=27.8%  drop=+16.67%  CI95=[+0.00%, +33.33%]   p=0.05469
  college-mathematics       N=36  b=6   c=0  flip=16.7%  drop=+16.67%  CI95=[+5.56%, +30.56%]   p=0.01562
  computer-security         N=35  b=8   c=7  flip=42.9%  drop= +2.86%  CI95=[-17.14%, +22.86%]  p=0.5
  formal-logic              N=36  b=1   c=5  flip=16.7%  drop=-11.11%  CI95=[-25.00%, +2.78%]   p=0.9844
  high-school-biology       N=35  b=6   c=3  flip=25.7%  drop= +8.57%  CI95=[-8.57%, +25.71%]   p=0.2539
  high-school-chemistry     N=36  b=3   c=3  flip=16.7%  drop= +0.00%  CI95=[-13.89%, +13.89%]  p=0.6562
  high-school-computer-sci  N=35  b=8   c=3  flip=31.4%  drop=+14.29%  CI95=[-2.86%, +31.43%]   p=0.1133
  high-school-geography     N=36  b=7   c=4  flip=30.6%  drop= +8.33%  CI95=[-8.33%, +25.00%]   p=0.2744
  high-school-mathematics   N=36  b=11  c=2  flip=36.1%  drop=+25.00%  CI95=[+8.33%, +41.67%]   p=0.01123
  high-school-physics       N=36  b=3   c=6  flip=25.0%  drop= -8.33%  CI95=[-25.00%, +5.56%]   p=0.9102
  high-school-statistics    N=36  b=6   c=3  flip=25.0%  drop= +8.33%  CI95=[-8.33%, +25.00%]   p=0.2539
  high-school-world-history N=35  b=3   c=3  flip=17.1%  drop= +0.00%  CI95=[-14.29%, +14.29%]  p=0.6562
  logical-fallacies         N=36  b=12  c=2  flip=38.9%  drop=+27.78%  CI95=[+8.33%, +47.22%]   p=0.00647
  world-religions           N=36  b=7   c=4  flip=30.6%  drop= +8.33%  CI95=[-8.33%, +25.00%]   p=0.2744

Plain English:
  The candidate is statistically significantly worse than the baseline on this
  suite. Treat this quantization as a real quality regression.
========================================================================
```

### What this result means

**DEGRADED — unambiguous.** p=0.0002 with a CI entirely above zero ([+4%, +12.8%])
means the result is not a fluke: Q2_K loses 8.4 accuracy points vs Q8_0 on MMLU.
The damage is real and large enough to matter in practice.

**Per-task damage looks concentrated — but does not survive correction.**
Three tasks have raw p < 0.02:

Sign convention (matches the tool): **degradation is reported as a positive
number of percentage points worse**, computed as δ=(b−c)/N. A positive number
means the candidate lost ground; a negative number means it improved.

| Task | Degradation | raw p | BH q |
|------|-------------|-------|------|
| logical-fallacies | +27.8% worse | 0.0065 | 0.073 |
| high-school-mathematics | +25.0% worse | 0.011 | 0.073 |
| college-mathematics | +16.7% worse | 0.016 | 0.073 |

(All three share q=0.0729: Benjamini-Hochberg enforces monotonicity across
ranks, so neighbouring adjusted values collapse together.)

**None of these is significant.** The suite has 14 tasks, so 14 tests were run.
After Benjamini-Hochberg correction, **0 of 14 tasks** clear q<0.05 (Bonferroni
also rejects nothing: α/14 = 0.0036 < 0.0065). At 14 tests you expect ~0.7 to
clear p<0.05 by chance. Every per-task subset is additionally underpowered at
N≈36 for the 2% target drop — all 14 of them.

The honest reading: the *direction* is consistent and plausible (arithmetic and
logical reasoning degrade most under 2-bit quantization), and it agrees with the
pooled result, but **this run cannot establish per-task claims.** Doing so would
need either many more items per task or a pre-registered hypothesis about which
tasks to test. Earlier versions of this document presented these three as
"individually significant"; that was wrong, and the tool now prints BH-adjusted
q-values next to raw p so the mistake is harder to make.

formal-logic actually *improves* (b=1, c=5, δ=−11.1%, p=0.98 one-sided), illustrating why
per-task breakdown matters: the pooled number hides the asymmetry. A 2-bit model
that reliably fails at arithmetic and logical-fallacies reasoning while holding
steady elsewhere is a different risk profile than uniform degradation.

**Sensitivity vs. the verdict — not a contradiction.** The run reports 21%
sensitivity for a *2%* drop while returning DEGRADED. Both are correct: the
sensitivity figure describes a smaller hypothetical effect than the one actually
found. At N=500 and a 27.2% flip rate this design could reliably detect drops of
**7.5% or larger** (the minimum detectable effect); the observed drop was 8.4%,
comfortably above it. Detecting a 2% drop would have needed ~7340 prompts.

That this needed a paragraph of explanation was itself the signal: the report now
leads with the minimum detectable effect rather than a raw power percentage, and
states explicitly that these numbers do not weaken a significant verdict.

### All measured runs

Sign convention: **degradation is positive percentage points worse**, δ=(b−c)/N.
MDE = minimum detectable effect, the smallest drop that design could reliably
detect at 95% power given its N and flip rate.

| Run | Pair | Suite | N | b | c | Drop | p | MDE | Verdict |
|-----|------|-------|---|---|---|------|---|-----|---------|
| 1 | Llama-3.2-3B Q8_0→Q4_K_M | knowledge | 200 | 5 | 13 | −4.00% | 0.985 | 6.07% | INCONCLUSIVE |
| 2 | Gemma-3-4B Q8_0→Q4_K_M | knowledge | 200 | 11 | 15 | −2.00% | 0.837 | 7.60% | INCONCLUSIVE |
| 3 | Gemma-3-4B Q8_0→Q2_K | knowledge | 200 | 32 | 20 | +6.00% | 0.063 | 11.27% | INCONCLUSIVE |
| 4 | Gemma-3-4B Q8_0→Q2_K | knowledge | 500 | 89 | 47 | +8.40% | 0.0002 | 7.52% | **DEGRADED** |
| 5 | Gemma-3-4B Q8_0→Q2_K | reasoning | 200 | 33 | 14 | +9.50% | 0.0040 | 10.66% | **DEGRADED** |
| 6 | Gemma-3-4B Q8_0→Q4_K_M | reasoning | 200 | 7 | 10 | −1.50% | 0.834 | 5.85% | INCONCLUSIVE |
| 7 | Qwen2.5-7B Q8_0→Q4_K_M | reasoning | 500 | 16 | 11 | +1.00% | 0.221 | 3.11% | INCONCLUSIVE |

Runs 1–4 used raw completion prompts; runs 5–6 used `--chat-template gemma`,
run 7 `--chat-template qwen`. Run 7 is the only run over the full 500-item
reasoning suite.

**The headline finding: Q4_K_M looks safe at 3–4B scale, Q2_K does not.** But
read the MDE column before trusting the first half of that sentence — runs 1–2
only rule out drops larger than ~6–7.6%. A 3% regression would have been
invisible. That is what INCONCLUSIVE means here, and it is why the tool refuses
to print "passed".

**Runs 3 and 4 are the same model pair at different N, and they are the clearest
result on this page.** Same baseline, same candidate, same backend; only N
changed, 200 → 500. The verdict moved from INCONCLUSIVE (p=0.063) to DEGRADED
(p=0.0002).

The MDE column explains *why*, and it is not luck. At N=200 the design could
only detect drops ≥11.27%; the true drop is ~8–9%, **below its own detection
threshold**. At N=500 the threshold falls to 7.52%, the 8.40% drop clears it, and
the effect becomes visible. The N=200 run was not a near-miss to be re-rolled —
it was a design that could not see an effect of that size, and the tool said so
instead of reporting a pass.

This is the whole argument for the project in two rows of a table: the same
quantization is "fine" or "broken" depending on whether the measurement was
built to detect it, and only the sensitivity analysis distinguishes those cases.

**Run 5 extends the finding past multiple choice.** GSM8K chain-of-thought,
+9.50% drop, p=0.0040. It survives a truncation robustness check: dropping every
pair where either response looks cut off leaves b=28, c=12, N=193, δ=+8.29%,
p=0.0083 — so the verdict is not an artifact of the candidate running out of
tokens mid-derivation. Caveat: the reasoning suite carries a single task tag
(`arithmetic`), so it cannot demonstrate per-task concentration; run 4 is the
evidence for that.

---

## Runs 5 and 6 — the reasoning dose-response

Runs 5 and 6 share a baseline, a suite, a template, a seed, an N, and a backend.
The **only** difference is the candidate's quantization level. That makes them a
controlled dose-response pair, and it is the most informative comparison on this
page.

| Candidate | Accuracy | Flip rate | Drop | p | MDE | Verdict |
|-----------|----------|-----------|------|---|-----|---------|
| (Q8_0 baseline) | 82.0% | — | — | — | — | — |
| Q4_K_M | 83.5% | **8.5%** | −1.50% | 0.834 | 5.85% | INCONCLUSIVE |
| Q2_K | 72.5% | **23.5%** | +9.50% | 0.0040 | 10.66% | **DEGRADED** |

### The flip rate is the finding, not the p-value

The accuracy numbers say Q4_K_M is fine and Q2_K is broken. The **flip rate**
says something the accuracy numbers cannot: Q4_K_M disagrees with the Q8_0
baseline on 8.5% of prompts, Q2_K on 23.5% — nearly three times as often.

That difference is itself testable, because the baseline is shared and
**byte-identical** across the two runs (see below). Scoring each prompt by
whether the candidate diverged from the baseline, and running the same paired
test across the two candidates:

```
Q2_K diverged where Q4_K_M did not : b = 37
Q4_K_M diverged where Q2_K did not : c =  7
exact one-sided McNemar            : p = 2.65e-06
```

Q2_K's behaviour departs from the baseline far more than Q4_K_M's does, and that
holds independently of whether the departure happened to be an improvement or a
regression. Churn and net accuracy change are different quantities: Q4_K_M's
net −1.50% is built from 17 flips in both directions, not from 17 quiet
agreements.

### What run 6 does and does not license

Run 6 returned INCONCLUSIVE with the candidate *nominally ahead* (83.5% vs
82.0%, δ=−1.50%, p=0.834). The honest reading has two halves:

- **What it rules out:** drops larger than about **5.9%**. A regression of that
  size or bigger would very likely have shown up here, and did not.
- **What it does not rule out:** anything smaller. Reaching the 2% target
  would need ~2236 prompts. Q4_K_M is *not* certified safe by this run — it is
  un-incriminated at a 6-point resolution.

Note a counter-intuitive property of the MDE visible here: run 6 has a **lower**
MDE (5.85%) than run 5 (10.66%) at the same N=200. Fewer flips means fewer
informative trials, but it also means each flip carries more weight, because the
same δ implies a more lopsided split of a smaller disagreement pool. At low flip
rates the second effect dominates. A candidate that rarely disagrees with its
baseline is *easier* to clear at a given margin, not harder.

### Determinism, measured rather than assumed

The README cautions that temp-0 GGUF inference can be slightly
nondeterministic. Runs 5 and 6 let us check instead of hedging: both executed
the same baseline model over the same 200 prompts at `--seed 0`, hours apart, in
separate processes.

**All 200 baseline responses were byte-identical across the two runs.**

So on this CPU backend, with a fixed seed and a fixed context size, generation
was exactly reproducible. The caveat still stands in general — it is a real
llama.cpp effect across differing backends, batch sizes and hardware — but it
did not bite here, and that is now a measurement rather than an assumption. It
is also what makes the cross-run divergence test above legitimate: the two
candidates were compared against literally the same baseline text.

### Length inflation scales with quantization

Published work on quantized reasoning models reports that damaged models fail
by generating longer, less decisive chains of thought. These two runs show that
as a dose-response against a shared baseline:

| Candidate | Mean response | vs baseline | Longer than baseline | Paired Wilcoxon |
|-----------|---------------|-------------|----------------------|-----------------|
| Q8_0 (baseline) | 679 chars | — | — | — |
| Q4_K_M | 738 chars | +59 | 108/200 | p = 7.4e-04 |
| Q2_K | 841 chars | +162 | 155/200 | p = 1.7e-11 |

Monotone in quantization aggressiveness. Two caveats on reading this as
confirmation of that literature: these are instruction-tuned models, not
reasoning-tuned ones, and GGUF k-quants are a different quantization family
from the GPTQ/AWQ methods those papers evaluate.

The practical consequence belongs to this tool rather than to those papers:
under a **fixed token budget** — the consumer condition — length inflation
converts into truncation. Run 5's truncation counts were 17/200 candidate vs
10/200 baseline; run 6's were 12/200 vs 10/200, which the health check correctly
reported as no asymmetry. Dropping every possibly-truncated pair from run 5 left
b=28, c=12, N=193, δ=+8.29%, p=0.0083 — the DEGRADED verdict survives, so it
measures reasoning failure rather than a token budget artifact.

---

## Resolved: instruct models and chat-template wrapping

Instruct/chat-tuned GGUFs expect their prompts wrapped in the model family's
turn format. **`--chat-template` ships** (`gemma`, `qwen`, `deepseek-r1`),
applied identically to both models so the comparison stays paired. Runs 1-4
above predate it and used raw completion prompts.

**A correction to the earlier version of this section.** It claimed raw prompts
depress baseline accuracy, which shrinks the right-to-wrong flip pool, which
lowers power and pushes verdicts toward INCONCLUSIVE. That mechanism sounded
right and was not supported by the data. A single-model spot-check on GSM8K
(`scripts/template_spotcheck.py`, 10 items, Gemma-3-4B-it Q8_0) scored raw and
templated prompts **identically at 9/10 each** once the token budget was large
enough. Raw prompting was not the problem.

The actual defect was in the scorer. `numeric_answer_scorer` used a single regex
with a bare `=` as an answer cue and took the **first** match, so in a
chain-of-thought answer an intermediate arithmetic step won over the final
result (gold 5; text `2*5 = 10 ... 10/2 = 5`; scorer picked 10). It is now
tiered, strongest cue first, taking the **last** match within a tier -- see
`tests/test_scoring.py` for the regression tests that pin this.

Blast radius: all four knowledge runs above used `multiple_choice`, not
`numeric`, so **no published result on this page was affected.** The bug would
have corrupted the first reasoning run, and the spot-check gate caught it before
roughly seven hours of inference were spent on unusable output.

Two lessons kept deliberately in this document: a plausible mechanism is not
evidence, and a cheap precondition gate in front of an expensive run pays for
itself.

### The template's effect is model-dependent

A second gate, same script, **Qwen2.5-7B-Instruct Q8_0**, 10 GSM8K items,
`--max-tokens 768 --n-ctx 2048`:

| prompting | correct | emitted the `####` cue | truncated | mean tokens |
|---|---|---|---|---|
| raw completion | 8/10 | 4/10 | 1/10 | 331 |
| `--chat-template qwen` | **10/10** | **9/10** | 0/10 | 234 |

Paired on the same 10 items: raw-only-correct `b=0`, template-only-correct
`c=2`, exact one-sided `p=0.25`. Ten items cannot establish an accuracy
difference and this one does not. **Format compliance** is a different story --
`b=0`, `c=5`, `p=0.031`. The template reliably gets the answer cue emitted, and
the cue is what the scorer reads.

The mechanism is visible in the raw text. Most raw completions open by
*continuing* the prompt's instruction line instead of answering it: `" (replace
<number> with the actual number)"`, `" (do not include the $ sign)"`,
`" (hours)"`. The model is finishing a document, not taking a turn. Raw
prompting also produced the run's only truncation (hit the 768-token cap) and
ran 42% longer per item on average.

So the Gemma result above -- raw and templated scoring 9/10 each -- **does not
generalize.** It was true of Gemma-3-4B and is false of Qwen2.5-7B. What carries
across models is the *gate*, not its outcome. A tool that assumes either answer
is wrong for some model; measuring costs ten items.

---

## Run 7 — Qwen2.5-7B-Instruct Q8_0 vs Q4_K_M, full reasoning suite

Cross-family, cross-size replication. GSM8K, all 500 items, `--chat-template
qwen --max-tokens 768 --n-ctx 2048 --seed 0`, CPU-only. 14.2 hours of inference
(8.34 h baseline, 5.88 h candidate).

```
N=500  b(degraded)=16  c(improved)=11  flips=27 (5.4%)
accuracy 89.60% -> 88.60%   drop +1.00%  (95% CI [-1.00%, +3.00%])
p = 0.221        verdict: INCONCLUSIVE
MDE = 3.11% at 95% power
```

Baseline 89.60% on GSM8K is in the published range for this model, so the
harness is measuring the model rather than measuring its own prompt handling.

### The lowest flip rate yet, so the sharpest run yet

This run has the **best sensitivity of anything on this page** -- MDE 3.11%
against 5.85% for run 6 and 10.66% for run 5. Partly N, mostly flip rate. Held
at fixed N, the relationship is monotone and counter-intuitive:

| flip rate | MDE at N=200 | N=500 | N=1000 |
|---|---|---|---|
| 4.0% (run 7, first 200) | 3.48% | 2.59% | 1.95% |
| 5.4% (run 7, full) | 4.32% | 3.11% | 2.30% |
| 8.5% (run 6) | 5.85% | 4.03% | 2.94% |
| 23.5% (run 5) | 10.66% | 6.97% | 4.99% |

**A candidate that rarely disagrees with its baseline is cheaper to clear.** Only
discordant pairs carry information, so a smaller flip pool looks like less data
-- but δ=(b−c)/N must be built *out of* that pool, so a given δ forces a more
lopsided split of it, and the lopsidedness is what the sign test detects. The
second effect dominates. Noisy candidates need far more prompts to say anything
about, which is the opposite of the intuition that noise is what you measure.

### Instability is model-specific, not prompt-specific

Runs 6 and 7 cover the same first 200 prompts, so each prompt can be labelled by
whether *that model pair* flipped on it. Paired across the two pairs:

```
Gemma pair flipped, Qwen pair did not : 17
Qwen pair flipped, Gemma pair did not :  8
both pairs flipped                    :  0
neither                               : 175
exact one-sided p                     : 0.054
```

The flip-rate difference (8.5% vs 4.0% on the same prompts) **just misses
significance** and is reported as such. The striking number is `both = 0`:
across 200 prompts, not one destabilised both model pairs. Quantization
instability here is a property of the model, not of inherently borderline
benchmark items. That rules out the convenient explanation that a handful of
ambiguous GSM8K questions drive every flip count on this page.

Confound stated plainly: Qwen2.5-7B differs from Gemma-3-4B in family *and*
parameter count. This is consistent with the literature's claim that larger
models absorb k-quantization better; it is not evidence for it, because the
design cannot separate the two factors. Doing so needs two sizes within one
family.

### Significant and negligible at the same time

Response length, baseline → candidate: 819 → 832 chars, **+13.5 (+1.7%)**.
282 of 500 responses longer, 195 shorter, 23 byte-identical. Wilcoxon signed-rank
**p = 8.4e-04**.

So the inflation is real -- and 1.7% of a response is nothing. Compare run 6's
Q4_K_M at +8.7% and run 5's Q2_K at +23.9%. A p-value this small on an effect
this small is the cleanest illustration on this page of why QuantCheck reports
effect sizes and intervals next to every p-value: at N=500 the test can resolve
a difference far below the threshold where anyone should care.

### A prefix is not a sample, demonstrated

Run 7 is the first run over the full suite, so its first 200 items are exactly
what `--limit 200` would have selected:

| slice | N | b | c | flip rate | drop | p |
|---|---|---|---|---|---|---|
| first 200 (`--limit 200`) | 200 | 4 | 4 | 4.0% | +0.00% | 0.637 |
| items 201–500 | 300 | 12 | 7 | 6.3% | +1.67% | — |
| full suite | 500 | 16 | 11 | 5.4% | +1.00% | 0.221 |

`--limit` takes a prefix rather than drawing a fresh sample. The suite files are
shuffled at fetch time (`datasets.py`), so a prefix *is* an unbiased sample of
the pool -- the 4.0% versus 6.3% gap above is ordinary sampling variation, not
bias, and nothing here is wrong. Two things still follow. The prefix is not
**stratified**, so per-task counts drift apart as N shrinks, which is why the
report prints realized per-task N. And runs at different `--limit` values are
nested rather than independent, so they cannot be treated as replications: runs
3 and 4 share their first 200 items, and run 7's first 200 are run 6's items
exactly. That nesting is what licenses the paired cross-run comparison earlier
in this section, and it is also why the MDE column matters more than the
verdict column when comparing rows of the summary table.

---

## Next runs planned

1. **Two sizes inside one family** — e.g. Qwen2.5-3B vs Qwen2.5-7B, both
   Q8_0 vs Q4_K_M, same suite. This is the run that would separate parameter
   count from model family in the run-6-vs-7 comparison above, which is
   currently confounded.
2. Optional: a DeepSeek-R1 distill, to compare GGUF k-quant behaviour against
   the GPTQ/AWQ results reported for reasoning-tuned models. Needs a much larger
   `--n-ctx` and `--max-tokens`, so the truncation diagnostic matters most here.
3. Re-run a Q2_K pair at N=500 on the reasoning suite. Run 5 was DEGRADED at
   N=200 despite its 10.66% MDE exceeding the observed 9.50% drop; at N=500 the
   MDE drops to ~6.97% and the finding would rest on a design that could see it.
