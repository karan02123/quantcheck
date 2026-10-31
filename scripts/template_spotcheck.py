"""Template precondition spot-check (gate before the GSM8K paired run).

Loads ONE model once and runs the first N reasoning items twice — raw and
chat-template-wrapped — then reports, for each variant, how many answers the
numeric scorer could actually parse and how many were correct. Both passes
reuse the same loaded model, so this is one model-load, not two.

Why this gate exists
--------------------
Instruct models fed raw completion prompts often ignore the "#### <number>"
instruction and ramble, so the numeric scorer extracts nothing and BOTH models
score near zero — the contingency table collapses and the GSM8K run measures a
parsing artifact, not quantization. Run this on the baseline (Q8_0) first and
confirm BEFORE spending hours on the full paired run:

  * templated PARSE rate is high (the scorer finds a number for most items), and
  * templated ACCURACY lands in a plausible range for the model
    (a 3-4B instruct model sitting sub-20% on GSM8K is the template/parsing
    artifact talking, not real reasoning ability).

If raw already parses and scores fine, the template is not strictly needed; if
templated is much better, gate the real run on --chat-template.

Usage (run it yourself; this is a model-inference run):

    python scripts/template_spotcheck.py \
        --model models/google_gemma-3-4b-it-Q8_0.gguf \
        --template gemma --n 10 --n-gpu-layers 0 --max-tokens 384
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantcheck.runner import GenerationConfig, ModelSpec, make_backend  # noqa: E402
from quantcheck.scoring import numeric_answer_scorer  # noqa: E402
from quantcheck.suites import load_suite  # noqa: E402
from quantcheck.templates import get_template  # noqa: E402


def _truncate(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def main(argv: list[str] | None = None) -> int:
    pa = argparse.ArgumentParser(prog="template_spotcheck")
    pa.add_argument("--model", required=True, help="path to a single GGUF")
    pa.add_argument("--template", default="gemma", help="chat template name")
    pa.add_argument("--suite", default="reasoning", help="suite to sample from")
    pa.add_argument("--n", type=int, default=10, help="number of items to check")
    pa.add_argument("--n-gpu-layers", type=int, default=0)
    pa.add_argument("--max-tokens", type=int, default=384)
    pa.add_argument("--n-ctx", type=int, default=2048,
                    help="context window; must exceed --max-tokens plus the prompt")
    args = pa.parse_args(argv)

    if args.max_tokens >= args.n_ctx:
        print(f"error: --max-tokens ({args.max_tokens}) must be smaller than "
              f"--n-ctx ({args.n_ctx})", file=sys.stderr)
        return 1

    tmpl = get_template(args.template)
    suite = load_suite(args.suite, limit=args.n)
    print(f"Spot-check: model={args.model}")
    print(f"  template={tmpl.name}  suite={suite.name}  items={len(suite)}  "
          f"max_tokens={args.max_tokens}")

    spec = ModelSpec(path_or_id=args.model, backend="gguf",
                     n_gpu_layers=args.n_gpu_layers)

    # Raw: no template, no stop. Templated: wrap + stop at end-of-turn.
    gen_raw = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, seed=0,
                               n_ctx=args.n_ctx)
    gen_tpl = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, seed=0,
                               stop=tmpl.stop, n_ctx=args.n_ctx,
                               prompt_template=tmpl.template)

    be = make_backend(spec, gen_raw)
    be.load()

    def _ntok(text: str) -> int:
        """Generated length in tokens, for truncation detection."""
        try:
            return len(be._llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:  # noqa: BLE001 - diagnostic only
            return -1

    raw_parse = raw_correct = tpl_parse = tpl_correct = 0
    raw_hash = tpl_hash = raw_trunc = tpl_trunc = 0
    records = []
    try:
        for i, item in enumerate(suite.items, start=1):
            raw_out = be.generate(gen_raw.render(item.prompt), gen_raw)
            tpl_out = be.generate(gen_tpl.render(item.prompt), gen_tpl)

            raw_s = numeric_answer_scorer(raw_out, item.answer)
            tpl_s = numeric_answer_scorer(tpl_out, item.answer)
            # "parsed" = scorer found ANY number (correct or not).
            raw_p = 1 if _has_number(raw_out) else 0
            tpl_p = 1 if _has_number(tpl_out) else 0
            # Did the model emit the instructed "#### <n>" delimiter?
            raw_h = 1 if "####" in raw_out else 0
            tpl_h = 1 if "####" in tpl_out else 0
            # Truncation: generation likely stopped on the token budget.
            raw_n, tpl_n = _ntok(raw_out), _ntok(tpl_out)
            raw_t = 1 if raw_n >= args.max_tokens - 2 else 0
            tpl_t = 1 if tpl_n >= args.max_tokens - 2 else 0

            raw_parse += raw_p; raw_correct += raw_s
            tpl_parse += tpl_p; tpl_correct += tpl_s
            raw_hash += raw_h; tpl_hash += tpl_h
            raw_trunc += raw_t; tpl_trunc += tpl_t

            records.append({
                "i": i, "gold": item.answer, "prompt": item.prompt,
                "raw": {"text": raw_out, "correct": raw_s, "hash": raw_h,
                        "tokens": raw_n, "truncated": raw_t},
                "tpl": {"text": tpl_out, "correct": tpl_s, "hash": tpl_h,
                        "tokens": tpl_n, "truncated": tpl_t},
            })

            tr = " TRUNC" if raw_t else ""
            tt = " TRUNC" if tpl_t else ""
            print(f"\n[{i}/{len(suite)}] gold={item.answer}")
            print(f"  RAW  ok={raw_s} ####={raw_h} tok={raw_n}{tr} | {_truncate(raw_out)}")
            print(f"  TPL  ok={tpl_s} ####={tpl_h} tok={tpl_n}{tt} | {_truncate(tpl_out)}")
    finally:
        be.unload()

    # Save full outputs so scoring can be re-checked offline without re-running.
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = out_dir / f"spotcheck_{Path(args.model).stem}_{args.template}.jsonl"
    with dump.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(suite)
    print("\n" + "=" * 60)
    print(f"RAW       : correct {raw_correct}/{n} ({100*raw_correct/n:.0f}%)  "
          f"parse {raw_parse}/{n}  '####' {raw_hash}/{n}  trunc {raw_trunc}/{n}")
    print(f"TEMPLATED : correct {tpl_correct}/{n} ({100*tpl_correct/n:.0f}%)  "
          f"parse {tpl_parse}/{n}  '####' {tpl_hash}/{n}  trunc {tpl_trunc}/{n}")
    print("=" * 60)
    print(f"Full outputs saved: {dump}")
    if max(raw_trunc, tpl_trunc) > n // 4:
        print("WARNING: many responses hit the token budget - the final answer may "
              "never be emitted. Raise --max-tokens and re-check before the real run.")
    if tpl_hash < n // 2:
        print("NOTE: the model mostly ignored the '#### <number>' instruction, so "
              "scoring falls back to weaker answer cues.")
    print("Gate: proceed to the full run with --chat-template only if TEMPLATED "
          "parse rate is high and accuracy is plausible for the model.\n"
          "If templated accuracy is still near zero, the template string or the "
          "answer cue is wrong — fix that before running.")
    return 0


def _has_number(text: str) -> bool:
    import re
    return re.search(r"-?\$?\d[\d,]*(?:\.\d+)?", text) is not None


if __name__ == "__main__":
    raise SystemExit(main())
