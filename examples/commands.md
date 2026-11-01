# Example commands

All commands assume the package is installed (`pip install -e .`) so the
`quantcheck` entry point is on your PATH. You can equally run
`python -m quantcheck.cli ...`.

> **Instruct/chat models need `--chat-template`.** Without it the model receives
> a bare completion prompt and may continue your instruction text instead of
> answering it. The effect is model-dependent and can be large — see
> *The template's effect is model-dependent* in [`REAL_RUN.md`](REAL_RUN.md).
> Run the spot-check in §1 before spending hours on a full run.

## 1. Check the prompt format first (10 items, one model, minutes)

```bash
python scripts/template_spotcheck.py \
  --model models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  --template qwen --n 10 --n-gpu-layers 0 \
  --max-tokens 768 --n-ctx 2048
```

Scores raw vs templated prompting on the same items and reports accuracy, parse
rate, answer-cue rate and truncations. Proceed only if the templated parse rate
is high and accuracy is plausible for the model. If templated accuracy is near
zero, the template or the answer cue is wrong — fix that first.

Registered templates: `gemma`, `qwen`, `deepseek-r1`.

## 2. A real paired run

This is run 7 from [`REAL_RUN.md`](REAL_RUN.md), verbatim — 500 GSM8K items,
CPU-only, 14.2 h:

```bash
quantcheck \
  --baseline models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  --candidate models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --suite reasoning --chat-template qwen \
  --n-gpu-layers 0 --max-tokens 768 --n-ctx 2048 --seed 0
```

`--n-ctx` must exceed `--max-tokens` plus the longest prompt, or every response
is silently truncated and reads as degradation; the CLI refuses up front if
`--max-tokens >= --n-ctx`. Chain-of-thought models need a larger budget than the
256-token default.

## 3. Offload layers to a GPU

Start at 0 and raise gradually; lower it if you hit an OOM. Both models get the
same setting, so the comparison stays paired.

```bash
quantcheck --baseline base-q8.gguf --candidate cand-q4km.gguf \
  --suite knowledge --n-gpu-layers 20
```

## 4. Use the config file (CLI flags still override it)

```bash
quantcheck --config configs/default.yaml \
  --baseline base.gguf --candidate cand.gguf --suite instruction
```

## 5. A custom suite (your own JSONL)

The bundled suites are public benchmarks and therefore contaminated; your own
task data is the better measurement.

```bash
quantcheck --baseline base.gguf --candidate cand.gguf \
  --suite my_prompts.jsonl --min-drop 0.01
```

`my_prompts.jsonl` (one object per line):

```json
{"prompt": "What is 2+2? Answer with a number.", "answer": "4", "scorer": "numeric", "task": "math"}
{"prompt": "Capital of Japan? Reply A-D...\nAnswer:", "answer": "D", "scorer": "multiple_choice", "task": "geo"}
```

`--limit N` takes the **first N items**, not a random sample, so per-task counts
will not be balanced and two runs at different limits are not nested samples.

## 6. Re-analyse a finished run (no model, no re-scoring)

The 2x2 counts are sufficient statistics, so saved results can be re-rendered
through current analysis code — useful after a reporting improvement:

```bash
python scripts/rerender_report.py examples/runs/*.json --write
```

## 7. Compare against a local OpenAI-compatible server

> ⚠️ **Untested path.** This backend has no test coverage and has never been
> exercised against a live server. The code is in
> `quantcheck/runner.py::OpenAICompatBackend`; treat it as a starting point, not
> a verified feature. Two further caveats if you try it: servers that batch
> requests (e.g. vLLM) are not bit-deterministic even at temperature 0, so the
> byte-identical reproducibility measured for the GGUF backend will not hold —
> the paired design still works, but determinism is no longer a given. And a
> remote endpoint may silently change the model under you mid-run.

Run two servers (llama.cpp `server`, Ollama, LM Studio) sequentially, then:

```bash
quantcheck --backend openai \
  --baseline llama3-f16 --candidate llama3-q4 \
  --suite reasoning
# base_url defaults to http://localhost:8080/v1 (override via config 'extra').
```

## 8. Smoke run with no models at all (mock backend)

```bash
quantcheck --backend mock --baseline a --candidate b --suite knowledge --limit 10
```

Mock outcomes are **fabricated** with a known structure to exercise the plumbing
and show what the report looks like. They are not a measurement.

## 9. Reproduce the README worked example (simulated, no models)

```bash
python examples/worked_example.py
```

## 10. Refresh the bundled suites from the real benchmarks

```bash
quantcheck fetch-suites --n-reasoning 500 --n-knowledge 500 --seed 42
```

Requires `pip install datasets`. Overwrites `quantcheck/suites/reasoning.jsonl`
and `quantcheck/suites/knowledge.jsonl`, recording benchmark, split, n and seed
in each file's header. The committed suites were produced by exactly this command — see
[`../quantcheck/suites/SOURCE.md`](../quantcheck/suites/SOURCE.md).

## Exit codes (handy for CI gating)

- `0` — ran fine, verdict is **not** DEGRADED.
- `1` — usage / load error.
- `2` — ran fine, candidate is significantly **DEGRADED**.

Note that `0` covers both `INCONCLUSIVE` and `NO_SIGNIFICANT_DEGRADATION`. A
gate built on exit codes alone cannot tell "we checked and it is fine" from "our
eval was too small to tell" — read the minimum-detectable-effect line in the
report, which every run prints.
