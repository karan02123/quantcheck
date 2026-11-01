# QuantCheck

**Did quantization actually break your model? A statistically honest degradation
test for local GGUF models.**

You downloaded a `Q4_K_M` of your favorite model because the `F16` won't fit in
6 GB of VRAM. It runs. But is it *quietly worse* than the higher-precision
version — and if so, where, and by how much? QuantCheck runs both models on the
same prompts, compares them **per prompt**, and gives you a verdict you can
trust: a p-value, an effect size with a confidence interval, a per-task
breakdown, and — crucially — an honest **sensitivity check** that tells you the
smallest drop your eval set could actually have detected. A null result that
cannot rule out a 10-point regression is not reassurance, and the report says
so.

---

## What this is, and what it is NOT (read this first)

**This tool's statistical method is not novel and is not "ours."** The core test
— a paired, per-sample disagreement test (an exact one-sided
[McNemar test](https://en.wikipedia.org/wiki/McNemar%27s_test) / binomial sign
test on the off-diagonal counts of a 2×2 agreement table) — is a textbook
procedure. Its application to **LLM quantization degradation** follows prior
work, in particular:

> **Kübler, Budhathoki, Kleindessner, Zhou, Yin, Khetan & Karypis (Amazon).**
> *When LLMs get significantly worse: A statistical approach to detect model
> degradations.* ICLR 2026. [arXiv:2602.10144](https://arxiv.org/abs/2602.10144)
> Code: [amazon-science/LLM-Accuracy-Stats](https://github.com/amazon-science/LLM-Accuracy-Stats)
> — establishes paired per-sample significance testing as the right way to ask
> "did this change hurt the model?". Note their scope is model degradations in
> general (inference optimizations, model updates, serving changes); quantization
> is one application, not the paper's subject.

and the broader llama.cpp / GGUF quantization-evaluation literature (perplexity
and KL-divergence comparisons across quant levels; see *Related work* below).

**QuantCheck's contribution is engineering and accessibility, not statistics.**
It wraps that validated method in a one-command, GGUF/llama.cpp-native CLI aimed
at hobbyists running quantized models on consumer hardware — people who today
have no easy way to know if their Q4 download silently degraded the model. We
describe the **tool** as ours; the **test** is established. We make no novelty
claim about the method, and you should not either when citing this.

---

## Motivation — the 6 GB-VRAM hobbyist

Quantization (F16 → Q8 → Q5 → Q4 → …) trades quality for memory. Usually the
trade is fine. Sometimes a particular quant of a particular model falls off a
cliff — and it tends to fall off *unevenly*: arithmetic/reasoning often degrades
first while trivia stays intact. Aggregate accuracy on a mixed benchmark can hide
this. Most "evals" people run also never tell you whether the result was
statistically meaningful or just noise from a small prompt set. QuantCheck is
built to answer the actual question — *is my quantized model worse, where, by how
much, and am I even sure?* — on a laptop, in one command.

---

## Install

Requires **Python 3.10+**.

```bash
pip install git+https://github.com/karan02123/quantcheck.git
```

The bundled benchmark suites ship inside the package, so that install is
self-contained. To work on the code instead:

```bash
git clone https://github.com/karan02123/quantcheck.git && cd quantcheck
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The core install (numpy, scipy, pyyaml) is light and is all you need for the
statistics, scoring, and CLI. Then add a **backend** to actually run models:

**CPU is the supported default; GPU is an optional accelerator.** QuantCheck is
a batch eval, not a latency-sensitive app: it runs once and you read the verdict,
so hours are tolerable in a way they would not be for an interactive tool.
CPU-default maximizes reach (the GGUF/llama.cpp audience is largely
small-GPU/CPU users) and keeps the install to one clean command.

Be realistic about the cost. Measured on this project's own hardware (i7-13620H,
CPU-only, `--n-gpu-layers 0`), a **paired** run means two full passes:

| Pair | Items | Wall clock |
|---|---|---|
| Gemma-3-4B Q8_0 vs Q4_K_M | 200 | ~3 h |
| Gemma-3-4B Q8_0 vs Q2_K | 500 | ~4.6 h |
| Qwen2.5-7B Q8_0 vs Q4_K_M | 500 | ~14.2 h |

Full timings are in [`examples/runs/README.md`](examples/runs/README.md). Partial
GPU offload (`--n-gpu-layers`) will cut this, but **no GPU run is included in
this repository**, so no speedup factor is claimed here. Use the CUDA build below
only if you already have a working CUDA toolchain.

### GGUF backend (default) — `llama-cpp-python`
```bash
# CPU-only (works everywhere):
pip install llama-cpp-python

# NVIDIA CUDA build (for partial GPU offload on ~6 GB VRAM):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
# (Apple Metal: CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python)
```
If the CUDA build is fiddly, use the CPU build — small models (1B–8B) are
perfectly workable on CPU, on the understanding that a few hundred prompts is
hours rather than minutes (see the table above). Start a run and come back.

**Illegal-instruction crash on load (`Windows 0xc000001d` / `SIGILL` on Linux)?**
Some prebuilt `llama-cpp-python` wheels are compiled with an instruction set your
CPU lacks (e.g. AVX-512 on a CPU that only has AVX2). The model file is fine —
the *binary* uses an opcode your CPU can't run, and it crashes during context
setup. Fix: build from source so it targets your actual CPU:
```bash
pip install llama-cpp-python --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```
On Windows this needs the MSVC C++ build tools; the build auto-detects
AVX2/FMA/F16C. This was hit on an AVX2-only laptop while developing QuantCheck;
building from source resolved it, and every run in
[`examples/runs/`](examples/runs/) was produced by a from-source build.

### OpenAI-compatible HTTP backend (optional) — `requests`
```bash
pip install requests   # or: pip install -e ".[http]"
```
Lets QuantCheck talk to a **local** server (llama.cpp `server`, Ollama, LM
Studio, vLLM). Still fully local; no paid APIs anywhere in this tool.

> **This backend is untested.** It has no test coverage and has never been run
> against a live server, so treat it as a starting point rather than a verified
> feature. Every measurement in
> [`examples/REAL_RUN.md`](examples/REAL_RUN.md) used the GGUF backend. Note too
> that servers which batch requests (vLLM among them) are not bit-deterministic
> even at temperature 0, so the byte-identical reproducibility measured for GGUF
> would not carry over -- the paired design still holds, determinism does not.

---

## Quickstart

```bash
quantcheck \
  --baseline tinyllama-1.1b-f16.gguf \
  --candidate tinyllama-1.1b-q4_k_m.gguf \
  --suite reasoning
```

**Running an instruct/chat model?** Add `--chat-template` (`gemma`, `qwen`,
`deepseek-r1`) and raise `--max-tokens`/`--n-ctx` for chain-of-thought output.
Verify the format on 10 items first with `scripts/template_spotcheck.py` -- see
[`examples/commands.md`](examples/commands.md) section 1. Skipping this can
cost you an entire run.

QuantCheck loads the **baseline**, runs every prompt, **unloads it**, then loads
the **candidate** and runs every prompt — never both at once, to respect your
VRAM budget. It scores each response, compares per prompt, and prints a verdict.
Bundled suites: `reasoning` (arithmetic), `knowledge` (multiple choice),
`instruction` (instruction following). See `quantcheck --list-suites`.

On `~6 GB` VRAM, start with `--n-gpu-layers 0` (CPU) and raise it gradually
(e.g. `20`) until just before you hit an out-of-memory error.

---

## Worked example — and how to read it

> ⚠️ **These numbers are SYNTHETIC, not a real measurement.** The example below
> runs on the built-in `mock` backend (`python examples/worked_example.py`): the
> per-prompt outcomes are *fabricated* with a known structure so we can show what
> the output looks like and how to read it. The `+9.2%`, `p=0.001709`, etc. are
> illustrative of the tool's *logic*, **not** a real result on a real Q4 model.
> For **real measurements**, see [`examples/REAL_RUN.md`](examples/REAL_RUN.md)
> (eight paired runs on real GGUF files, 31.4 h of CPU inference, including
> two distinct DEGRADED comparisons — one of them re-measured after a scorer
> fix — and a cross-family replication) and
> [`examples/runs/`](examples/runs/) for the raw result files behind them.

The simulation mimics a `Q4_K_M` that is fine on knowledge/instruction but
quietly worse on multi-step reasoning:

```
========================================================================
QuantCheck verdict -- suite: demo-mixed
  source: examples/worked_example.py (simulated)
========================================================================
[X] DEGRADED -- significant quality regression

Test used : exact one-sided McNemar (binomial sign test on disagreements)
Pooled    : N=300  b(degraded)=17  c(improved)=1  flips=18 (6.0%)
Accuracy  : baseline 76.00%  -> candidate 70.67%
Effect    : degradation +5.33%  (95% bootstrap CI [+3.00%, +8.01%])
p-value   : 7.248e-05  (one-sided, alpha=0.05)

Design sensitivity (what this run could have detected):
  smallest drop detectable: 4.0% at 95% power (N=300, flip rate 6.0%)
  your target was 2.0%  ->  sensitivity for that target: 40%
  -> too small for your 2.0% target: that would need ~1532 prompts at this flip rate.
  (This run DID detect its effect, so the numbers above do not weaken
   the verdict -- they describe smaller hypothetical drops. A drop can
   also be detected while sitting below the minimum above: that bound
   is set at 95% power, not at certainty.)

Per-task breakdown (3 tasks -- exploratory, read q not p):
  instruction                  N=60   b=1   c=0   flip=  1.7%  drop=+1.67% CI95=[+0.00%, +5.00%]  p=0.5 q=0.5
  knowledge                    N=120  b=4   c=0   flip=  3.3%  drop=+3.33% CI95=[+0.83%, +6.67%]  p=0.0625 q=0.0938
  reasoning                    N=120  b=12  c=1   flip= 10.8%  drop=+9.17% CI95=[+4.15%, +15.00%]  p=0.001709 q=0.00513 *
  q = Benjamini-Hochberg adjusted p across these 3 tests; '*' marks q < 0.05.
  1 of 3 tasks remain significant after correction. Raw p is NOT interpretable here: at 3 tests you expect ~0.2 to clear p<0.05 by chance.
  3 of 3 per-task subsets are individually underpowered for a 2.0% drop; treat task lines as direction, not evidence.

Plain English:
  The candidate is statistically significantly worse than the baseline on this suite. Treat this quantization as a real quality regression.

Caveats: this measures one scorer on one benchmark; it is not proof of equivalence, does not capture style/tone/safety, and temp-0 GGUF inference can be slightly nondeterministic. The p-value is exact and conditional (it decides the verdict); the CI is an unconditional bootstrap (it describes magnitude), so they answer different questions and may disagree at the boundary. See README.
```

**How to read it:**

- **`b` vs `c`** are the only counts that matter. `b` = prompts the baseline got
  right but the candidate got wrong (degradations); `c` = the reverse
  (improvements). Here 17 degraded, 1 improved — lopsided toward worse.
- **p-value `7.2e-05`** ≪ 0.05: the lopsidedness is very unlikely under "no
  degradation." Verdict: **DEGRADED**.
- **Effect + CI:** about a **5.3%** accuracy drop, 95% CI roughly **[3%, 8%]**.
  Always read the CI, not just the point estimate.
- **Per-task is the punchline:** the pooled +5.3% is dominated by **reasoning**
  (+9.2%). Knowledge and instruction are basically fine. A single aggregate
  number would have undersold the reasoning damage. *This is the core value-add:
  aggregates hide concentrated damage.*
- **Read `q`, not `p`, on the task lines.** With one test per task, raw p-values
  are not interpretable — at 3 tests you expect ~0.2 to clear p<0.05 by chance.
  The report prints Benjamini–Hochberg adjusted `q` beside each `p` and marks
  survivors with `*`. Here reasoning survives (q=0.0051) and knowledge does not
  (p=0.0625 → q=0.094). It also warns that all 3 subsets are individually
  underpowered. Task lines show *direction*; the pooled test is the evidence.
- **Design sensitivity, even when significant.** The report says this design
  could only reliably resolve drops of **4.0%** or larger, and had just 40%
  sensitivity for the 2% target (~1532 prompts needed). Note the wording: it
  explicitly says this does **not** weaken a verdict that was already reached —
  that would be the post-hoc power fallacy. The number exists to stop you
  over-trusting a *non*-significant result on a small set. An effect can also be
  detected while sitting below that bound, because the bound is set at 95%
  power, not at certainty.

If the run had come back non-significant **and** underpowered, the verdict would
be **INCONCLUSIVE**, never "passed."

---

## How the statistics work (plain English)

**Both models see the same prompts, so their scores are paired and correlated.**
You must *not* treat the two accuracies as two independent samples — that ignores
the pairing, throws away power, and can mislead. Instead we build a 2×2 table of
per-prompt agreement:

|                        | candidate correct | candidate wrong |
|------------------------|:-----------------:|:---------------:|
| **baseline correct**   | a                 | **b** (degraded)|
| **baseline wrong**     | **c** (improved)  | d               |

- **Agreements `a` and `d` carry no information** about which model is better —
  they cancel. Only the disagreements `b` and `c` tell you anything.
- **The test.** Under the null "quantization caused no degradation," a flipped
  prompt is equally likely to flip either way, so `b ~ Binomial(b+c, 0.5)`. To
  ask "is the candidate *worse*," we run a **one-sided exact binomial test**
  (`scipy.stats.binomtest(b, b+c, 0.5, alternative='greater')`) — this is exactly
  the exact form of McNemar's test. Small p-value ⇒ evidence of degradation.
- **The p-value means:** *if the candidate were not degraded, how likely is a
  split this lopsided toward "worse" by chance?* It is **not** the probability the
  model is broken, and a large p-value is **not** proof the models are equal.
- **Effect size with a CI.** We report the accuracy difference `(b − c)/N` with a
  **paired bootstrap** 95% CI (resample *prompts* with replacement, n=2000 by
  default, so the interval respects the pairing). A point estimate without a CI
  is not reported, ever. Note the p-value and the CI answer different questions:
  the p-value is exact and *conditional* on the flip count and decides the
  verdict, while the bootstrap CI is *unconditional* and describes magnitude, so
  at the boundary they can disagree.
- **Why the power check matters.** A non-significant result can mean "no real
  difference" *or* "too little data to tell." Power depends on the **flip rate**
  `(b+c)/N`, not just `N`: if models rarely disagree, each prompt carries little
  signal and you need many more of them. QuantCheck computes whether your `N` was
  enough to detect the drop you care about (default 2%) at 95% power, and if not,
  approximately how many prompts you'd need. *Most tools never tell you this.*

The statistical core lives in [`quantcheck/stats.py`](quantcheck/stats.py), with
extensive docstrings, and is verified against analytically-known answers and
Monte-Carlo simulation in [`tests/test_stats.py`](tests/test_stats.py) (e.g. a
hand-computed binomial p-value of `56/1024`; bootstrap CI coverage ≈ 95%; the
power formula checked against direct simulation of the exact test).

---

## What this tool does NOT tell you

Honesty matters more than looking good. QuantCheck **cannot**:

- **Prove equivalence.** A non-significant result is *not* "the models are the
  same." It means *this test, on this data, did not detect a difference* — which
  is why the verdict is "no significant degradation (not proof of equivalence)"
  or "inconclusive," never "passed."
- **Escape its scorer.** The verdict is only as good as the `L(response)→{0,1}`
  scorer. Each scorer has documented failure modes (answer parsing, formatting).
  A scorer that mis-parses one model's style more than the other's would
  manufacture a fake difference. We apply the *same* scorer to both models to
  keep any bias symmetric — but garbage in, garbage out.
- **Escape its benchmark.** Results are about *these prompts*. A clean bill on
  `reasoning` says nothing about coding, long-context, multilingual, or your
  specific use case. Bring a custom suite that matches what you actually do.
- **Measure what accuracy misses.** Style, tone, verbosity, refusals, formatting,
  and safety behavior are not captured by a 0/1 correctness score. A model can be
  "not significantly less accurate" yet feel worse.
- **Guarantee determinism.** Generation is temperature 0 with a fixed seed, but
  **even temp-0 GGUF inference can be slightly nondeterministic** due to
  floating-point reduction order across backends/batch sizes — a known llama.cpp
  caveat. Run twice if a borderline result matters.
- **Certify equivalence, at any margin.** This is the deliberate scope boundary.
  The exactness of the test rests on a conditional argument that holds *only* at
  the null δ=0: given the flip count n, b ~ Binomial(n, 0.5) with no nuisance
  parameters. At a non-zero margin that breaks, so "equivalent within m points"
  needs score-based intervals (Tango 1998; Nam 1997) or exact unconditional
  methods (Fagerland et al. 2014) — and note those cover a margin on the
  *difference*; a margin on the *ratio* is a separate literature again (Nam &
  Blackwelder 2002). Rather than ship one CLI whose
  superiority test is exact and whose equivalence test is asymptotic, QuantCheck
  tests δ=0 exactly and reports the **minimum detectable effect**, so a null
  result is quantified instead of merely reassuring.
- **Treat per-task lines as findings.** With one test per task, raw per-task
  p-values are not interpretable: at 14 tasks you expect ~0.7 to clear p<0.05 by
  chance. The report prints Benjamini–Hochberg `q` alongside `p` and flags how
  many per-task subsets are individually underpowered. Per-task lines show
  *direction*; the pooled test is the evidence.
- **Mix hardware between the two models.** CPU and GPU inference of the *same*
  GGUF can produce slightly different outputs (same numerical-nondeterminism
  effect). **Run both the baseline and the candidate on the same backend and the
  same `--n-gpu-layers`** — never baseline-on-GPU vs candidate-on-CPU, or you'll
  contaminate the comparison with a hardware difference masquerading as a
  quantization effect. QuantCheck uses one `--backend` / `--n-gpu-layers` for
  both models by design, so just don't change hardware between runs.

---

## Custom suites

Provide a JSONL file (one JSON object per line):

```json
{"prompt": "What is 12 * 12? Answer with a number.", "answer": "144", "scorer": "numeric", "task": "math"}
{"prompt": "Capital of Japan?\nA) Seoul\nB) Beijing\nC) Bangkok\nD) Tokyo\nAnswer:", "answer": "D", "scorer": "multiple_choice", "task": "geo"}
{"prompt": "Reply with the word 'banana'.", "answer": "banana", "scorer": "instruction", "task": "follow"}
```

- `prompt` (required), `answer` (required), `scorer` (optional; default
  `exact_match`), `task` (optional; for the per-task breakdown).
- Scorers: `numeric`, `multiple_choice`, `instruction` (all keywords present,
  `|`-separated), `exact_match`. See
  [`quantcheck/scoring.py`](quantcheck/scoring.py) for each scorer's parsing rule
  and known failure modes.

```bash
quantcheck --baseline base.gguf --candidate cand.gguf --suite my_prompts.jsonl
```

The bundled `reasoning` and `knowledge` suites hold **500 real items each**,
sampled from the GSM8K and MMLU test splits with a recorded seed (see
[`quantcheck/suites/SOURCE.md`](quantcheck/suites/SOURCE.md)); `instruction` is
20 hand-authored items and is a demo, not a measurement. Every result in
[`examples/REAL_RUN.md`](examples/REAL_RUN.md) was measured on these files.

Even so, prefer a **custom suite** for anything that matters. GSM8K and MMLU are
long-public and therefore contaminated, and a clean verdict on them says nothing
about your workload. Contamination hurts a paired test less than a leaderboard —
both models see identical prompts, so it inflates both sides and largely cancels
in the `b`/`c` counts — but it still makes the absolute accuracies unreliable.

---

## Configuration

`configs/default.yaml` holds thresholds and execution defaults; any CLI flag
overrides the config. Key knobs: `alpha` (significance), `min-drop` (the drop you
care about, for the sensitivity check), `target-power`, `n-gpu-layers`,
`max-tokens`, `n-ctx`, `seed`, `n-boot`. All seeds and generation settings are
logged into every result file (`results/quantcheck_<suite>_<timestamp>.json` +
`.txt`) for reproducibility.

Two flags matter more than the rest:

- **`--chat-template <name>`** wraps every prompt in an instruct model's turn
  format (`gemma`, `qwen`, `deepseek-r1`), applied **identically to both models**
  so the comparison stays paired. Required for instruct models on
  chain-of-thought suites: fed raw completion prompts, an instruct model often
  ignores the output-format instruction, the scorer extracts nothing, and the run
  measures a parsing artifact instead of quantization.
- **`--n-ctx`** sets the context window (default 2048). Raise it for
  long-chain-of-thought models. The CLI refuses to run when `--max-tokens` is not
  smaller than `--n-ctx`, because that silently truncates every response and
  reads as degradation rather than as a misconfiguration.

Note that `--limit N` takes a **prefix** of the suite file, not a fresh sample.
Bundled suites are shuffled at fetch time so a prefix is unbiased overall, but
per-task counts will not stay balanced — read the realized per-task N off the
report.

---

## Development / tests

```bash
pip install -e ".[dev]"
pytest                     # 81 tests; the stats tests are the ones that matter
                           # (+10 cross-validation tests, skipped by default)
```

`tests/test_stats.py` is the file to scrutinize: if the statistics are subtly
wrong, nothing downstream matters. It checks the implementation against values
known *independently* of the implementation (closed-form binomial p-values,
simulated CI coverage, and the power calc against Monte-Carlo of the exact test).
`tests/test_runner.py` asserts the hard invariant that **two models are never
loaded at once**.

**Cross-validated against the reference implementation.** Those checks are all
self-derived, so a shared misreading of the method would pass every one of them.
`tests/test_reference_agreement.py` closes that gap by comparing against the
code released with Kübler et al. (2026) as a **black box**: across 514
contingency tables — 500 randomized, plus boundary shapes, plus all seven
published runs — the two implementations agree with a maximum absolute
difference of **0.0**, exactly rather than within tolerance.

One behavioural difference was found. When two models agree on every prompt
(`b = c = 0`) there are no discordant pairs and the correct answer is `p = 1.0`;
QuantCheck returns that, while the reference raises `ZeroDivisionError`. That
case is what a *lossless* optimization produces, so it is reachable in practice.

The test is skipped unless `QUANTCHECK_REFERENCE_REPO` points at a checkout. No
reference code is vendored here, and none was read for implementation detail —
that repository is CC BY-NC 4.0, and QuantCheck's statistics were implemented
from the paper. See [`NOTICE`](NOTICE).

---

## Related work & citations

- **Kübler, J., Budhathoki, K., Kleindessner, M., Zhou, X., Yin, J., Khetan, A.
  & Karypis, G. (2026).** *When LLMs get significantly worse: A statistical
  approach to detect model degradations.* ICLR 2026.
  [arXiv:2602.10144](https://arxiv.org/abs/2602.10144). Code:
  [amazon-science/LLM-Accuracy-Stats](https://github.com/amazon-science/LLM-Accuracy-Stats)
  — the paired per-sample significance test this tool packages. **The method is
  theirs / the field's; QuantCheck is the packaging.** Two capability gaps worth
  naming rather than hiding: they report attributing degradations as small as
  **0.3%**, which takes far more samples than a laptop run affords (QuantCheck's
  sharpest measured run so far resolves 3.1% — see `examples/REAL_RUN.md`), and
  they propose methods for **aggregating across multiple benchmarks** into a
  single decision, which QuantCheck does not implement: it tests one suite at a
  time. *Provenance: QuantCheck implements the method from the paper's
  description; no code from the reference repository was used. That repository is
  licensed CC BY-NC 4.0, which would be incompatible with this project's MIT
  licence, so the distinction matters.*
- **Tango, T. (1998).** *Equivalence test and confidence interval for the
  difference in proportions for the paired-sample design.* Statistics in
  Medicine 17(8):891–908.
  doi:[10.1002/(SICI)1097-0258(19980430)17:8<891::AID-SIM780>3.0.CO;2-B](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C891::AID-SIM780%3E3.0.CO;2-B)
  — the score-based method a non-inferiority verdict
  would require; see *Limitations* for why QuantCheck does not implement it.
- **Nam, J. (1997).** *Establishing equivalence of two treatments and sample
  size requirements in matched-pairs design.* Biometrics 53(4):1422.
  doi:[10.2307/2533508](https://doi.org/10.2307/2533508) — equivalence at a
  margin on the **difference** of paired proportions, with sample-size results.
- **Nam, J. & Blackwelder, W.C. (2002).** *Analysis of the ratio of marginal
  probabilities in a matched-pair setting.* Statistics in Medicine
  21(5):689–699. doi:[10.1002/sim.1017](https://doi.org/10.1002/sim.1017)
  — the corresponding method for a margin on the **ratio** rather than the
  difference. Cited here for completeness; QuantCheck reports neither.
- **Fagerland, M.W., Lydersen, S. & Laake, P. (2014).** *Recommended tests and
  confidence intervals for paired binomial proportions.* Statistics in Medicine
  33(16):2850–75. doi:[10.1002/sim.6148](https://doi.org/10.1002/sim.6148)
  — the reference review for paired binary
  methods, including exact unconditional alternatives.
- **Benjamini, Y. & Hochberg, Y. (1995).** *Controlling the false discovery
  rate: a practical and powerful approach to multiple testing.* JRSS B
  57(1):289–300.
  doi:[10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
  — the per-task multiplicity correction.
- **McNemar, Q. (1947).** *Note on the sampling error of the difference between
  correlated proportions or percentages.* Psychometrika 12(2):153–157.
  doi:[10.1007/BF02295996](https://doi.org/10.1007/BF02295996) — the
  paired test itself.
- **Efron, B. & Tibshirani, R.J. (1993).** *An Introduction to the Bootstrap.*
  Chapman & Hall, Monographs on Statistics and Applied Probability 57.
  ISBN 0-412-04231-2. — the percentile bootstrap used for the paired
  effect-size CI.

- **Cobbe, K. et al. (2021).** *Training Verifiers to Solve Math Word Problems*
  (GSM8K), arXiv:2110.14168. — the `reasoning` suite is 500 items sampled from
  this benchmark's test split. MIT licensed; see [`NOTICE`](NOTICE).
- **Hendrycks, D. et al. (2021).** *Measuring Massive Multitask Language
  Understanding* (MMLU), arXiv:2009.03300. — the `knowledge` suite is 500 items
  sampled from this benchmark's test split. MIT licensed; see
  [`NOTICE`](NOTICE).
- **Zhou, J. et al. (2023).** *Instruction-Following Evaluation for Large Language
  Models* (IFEval), arXiv:2311.07911. — inspiration for the `instruction` suite.
- **llama.cpp quantization evaluation** — the community practice of comparing
  quant levels via perplexity and KL-divergence against the full-precision model
  (see the llama.cpp repository's quantization docs and discussions). QuantCheck
  complements those continuous metrics with a *task-accuracy* significance test.

Every reference above carries a resolvable identifier (DOI, arXiv ID or ISBN).
That is deliberate: a citation without one cannot be checked in thirty seconds,
and an unverifiable reference is the cheapest possible defect in a project whose
whole claim is honest measurement. All identifiers listed here were checked
against the Crossref, arXiv and OpenLibrary registries rather than copied from
memory or from a search summary.

---

## License

MIT — see [`LICENSE`](LICENSE). Third-party notices are in
[`NOTICE`](NOTICE).

The bundled suites in `quantcheck/suites/` redistribute samples from two public
benchmarks, **GSM8K** and **MMLU**. Both are MIT-licensed and therefore
compatible; attribution is in [`NOTICE`](NOTICE) and the selection details,
seeds and scoring notes in
[`quantcheck/suites/SOURCE.md`](quantcheck/suites/SOURCE.md).

No third-party code is vendored here. Notably, the reference implementation for
Kübler et al. (2026) is CC BY-NC 4.0, which MIT could not absorb; the method in
`quantcheck/stats.py` was implemented from the paper.
