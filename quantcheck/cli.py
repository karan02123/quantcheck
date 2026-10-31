"""QuantCheck command-line entry point.

One command compares a baseline model against a quantized candidate on a suite
and prints an honest verdict::

    quantcheck --baseline model-f16.gguf --candidate model-q4km.gguf --suite reasoning

Models run **one at a time** (baseline fully, unload, then candidate) to respect
a small VRAM budget — see :mod:`quantcheck.runner`. Config is read from a YAML
file; any CLI flag overrides the config. All seeds and generation settings are
logged into the JSON result file for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .report import evaluate, format_report, score_responses
from .runner import GenerationConfig, ModelSpec, run_model
from .suites import list_bundled_suites, load_suite


def _load_config(path: str | None) -> dict:
    """Load the YAML config file, returning {} if no path is given.

    Args:
        path: path to a YAML config, or None.

    Returns:
        Parsed config dict (empty if path is None).

    Raises:
        FileNotFoundError: if a path is given but missing.
        ImportError: if PyYAML is not installed.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("reading a config file needs PyYAML: pip install pyyaml") from e
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _cmd_fetch_suites(argv: list[str]) -> int:
    """Subcommand: fetch real GSM8K + MMLU benchmark subsets into suites/."""
    pa = argparse.ArgumentParser(
        prog="quantcheck fetch-suites",
        description=(
            "Replace the hand-authored bundled suites with real public benchmark "
            "samples (GSM8K for reasoning, MMLU for knowledge). Requires the "
            "'datasets' package: pip install datasets"
        ),
    )
    pa.add_argument("--n-reasoning", type=int, default=200,
                    help="items to fetch from GSM8K test split (default 200)")
    pa.add_argument("--n-knowledge", type=int, default=200,
                    help="items to fetch from MMLU test split (default 200)")
    pa.add_argument("--seed", type=int, default=42,
                    help="sampling seed (logged in the JSONL header, default 42)")
    pa.add_argument("--out-dir", default=None,
                    help="output directory (default: suites/ in the repo)")
    args = pa.parse_args(argv)

    from .datasets import fetch_and_write_all
    from pathlib import Path
    out = Path(args.out_dir) if args.out_dir else None
    try:
        written = fetch_and_write_all(
            n_reasoning=args.n_reasoning,
            n_knowledge=args.n_knowledge,
            seed=args.seed,
            out_dir=out,
            verbose=True,
        )
    except (ImportError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("Done. Suites now use real benchmark samples:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print(
        "\nNote: instruction suite unchanged (see suites/SOURCE.md for why).\n"
        "Re-run 'quantcheck fetch-suites' to refresh with a different seed or n."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    pa = argparse.ArgumentParser(
        prog="quantcheck",
        description=(
            "Did quantization actually break your model? A statistically honest, "
            "paired per-sample degradation test for local models. The statistical "
            "method (exact one-sided McNemar) is established (Kübler et al. 2026); "
            "this tool packages it for GGUF/local use. It does NOT prove equivalence."
        ),
    )
    pa.add_argument("--baseline", required=False, help="path/id of the higher-precision baseline model")
    pa.add_argument("--candidate", required=False, help="path/id of the quantized candidate model")
    pa.add_argument(
        "--suite",
        default="reasoning",
        help="bundled suite name (reasoning/knowledge/instruction) or path to a custom JSONL",
    )
    pa.add_argument("--config", default=None, help="YAML config file (CLI flags override it)")
    pa.add_argument("--backend", default=None, help="backend: gguf (default) | openai | mock")
    pa.add_argument("--n-gpu-layers", type=int, default=None,
                    help="gguf only: layers to offload to GPU (0=CPU-only, -1=all)")
    pa.add_argument("--limit", type=int, default=None, help="evaluate at most this many items")
    pa.add_argument("--chat-template", default=None,
                    help="wrap each prompt in an instruct model's turn format "
                         "(e.g. 'gemma'); applied identically to both models. "
                         "Required for instruct models on chain-of-thought suites.")
    pa.add_argument("--max-tokens", type=int, default=None, help="max new tokens per prompt")
    pa.add_argument("--n-ctx", type=int, default=None,
                    help="context window for local backends (default 2048). Raise it "
                         "for long chain-of-thought models, whose traces can run to "
                         "thousands of tokens; it must exceed --max-tokens plus the "
                         "longest prompt.")
    pa.add_argument("--seed", type=int, default=None, help="generation + bootstrap seed")
    pa.add_argument("--alpha", type=float, default=None, help="one-sided significance level")
    pa.add_argument("--min-drop", type=float, default=None,
                    help="minimum accuracy drop you care about (fraction, e.g. 0.02)")
    pa.add_argument("--target-power", type=float, default=None, help="target power (default 0.95)")
    pa.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default 2000)")
    pa.add_argument("--out-dir", default=None, help="directory for result files (default ./results)")
    pa.add_argument("--list-suites", action="store_true", help="list bundled suites and exit")
    pa.add_argument("--version", action="version", version=f"quantcheck {__version__}")
    return pa


def _cfg(args, config: dict, key: str, default):
    """Resolve a setting: CLI flag (if set) overrides config, which overrides default."""
    cli_val = getattr(args, key.replace("-", "_"), None)
    if cli_val is not None:
        return cli_val
    return config.get(key, default)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Exit codes:
        0 — ran successfully and the verdict is NOT "DEGRADED".
        1 — a usage / loading error occurred.
        2 — ran successfully but the candidate was found significantly DEGRADED
            (useful for CI gating).
    """
    # Route the fetch-suites subcommand before the main parser sees it.
    if argv and argv[0] == "fetch-suites":
        return _cmd_fetch_suites(argv[1:])
    if len(sys.argv) > 1 and sys.argv[1] == "fetch-suites" and argv is None:
        return _cmd_fetch_suites(sys.argv[2:])

    args = _build_parser().parse_args(argv)

    # Best-effort: make stdout UTF-8 so reports render on consoles (e.g. Windows
    # cp1252) that would otherwise choke on non-ASCII. Harmless if unsupported.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    if args.list_suites:
        names = list_bundled_suites()
        print("Bundled suites:", ", ".join(names) if names else "(none installed)")
        return 0

    try:
        config = _load_config(args.config)
    except (FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not args.baseline or not args.candidate:
        print("error: --baseline and --candidate are required", file=sys.stderr)
        return 1

    # Resolve settings (CLI > config > default).
    backend = _cfg(args, config, "backend", "gguf")
    n_gpu_layers = _cfg(args, config, "n-gpu-layers", 0)
    limit = _cfg(args, config, "limit", None)
    max_tokens = _cfg(args, config, "max-tokens", 256)
    n_ctx = _cfg(args, config, "n-ctx", 2048)
    seed = _cfg(args, config, "seed", 0)
    alpha = _cfg(args, config, "alpha", 0.05)
    min_drop = _cfg(args, config, "min-drop", 0.02)
    target_power = _cfg(args, config, "target-power", 0.95)
    n_boot = _cfg(args, config, "n-boot", 2000)
    out_dir = Path(_cfg(args, config, "out-dir", "results"))
    chat_template = _cfg(args, config, "chat-template", None)

    template_str: str | None = None
    stop: tuple[str, ...] = ()
    if chat_template is not None:
        from .templates import get_template
        try:
            tmpl = get_template(chat_template)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        template_str = tmpl.template
        stop = tmpl.stop
        print(f"Chat template '{tmpl.name}' applied to both models (stop={list(stop)}).")

    # A context window that cannot hold the prompt *plus* the answer budget
    # silently truncates every response, which reads as degradation rather than
    # as a misconfiguration. Refuse up front instead of producing a bad verdict.
    if max_tokens >= n_ctx:
        print(
            f"error: --max-tokens ({max_tokens}) must be smaller than --n-ctx "
            f"({n_ctx}), with room left over for the prompt. Raise --n-ctx.",
            file=sys.stderr,
        )
        return 1

    gen = GenerationConfig(
        max_tokens=max_tokens, temperature=0.0, seed=seed,
        stop=stop, n_ctx=n_ctx, prompt_template=template_str,
    )

    # Load the suite.
    try:
        suite = load_suite(args.suite, limit=limit)
    except (FileNotFoundError, ValueError) as e:
        print(f"error loading suite: {e}", file=sys.stderr)
        return 1
    print(f"Suite '{suite.name}' loaded: {len(suite)} items, tasks={suite.tasks}")

    baseline_spec = ModelSpec(path_or_id=args.baseline, backend=backend, n_gpu_layers=n_gpu_layers)
    candidate_spec = ModelSpec(path_or_id=args.candidate, backend=backend, n_gpu_layers=n_gpu_layers)

    def _progress(label):
        def cb(i, total):
            if i == 1 or i == total or i % 25 == 0:
                print(f"  [{label}] {i}/{total}", flush=True)
        return cb

    # ---- Sequential, VRAM-safe execution: baseline fully, THEN candidate. ----
    try:
        print(f"Running baseline ({args.baseline}) ...")
        base_out = run_model(baseline_spec, suite, gen, progress=_progress("baseline"))
        print(f"Running candidate ({args.candidate}) ...")
        cand_out = run_model(candidate_spec, suite, gen, progress=_progress("candidate"))
    except (FileNotFoundError, ImportError, RuntimeError) as e:
        print(f"error running models: {e}", file=sys.stderr)
        return 1

    # ---- Score + evaluate. ----
    base_scores = score_responses(suite, base_out.responses)
    cand_scores = score_responses(suite, cand_out.responses)

    report = evaluate(
        suite,
        base_scores,
        cand_scores,
        baseline_responses=base_out.responses,
        candidate_responses=cand_out.responses,
        baseline_spec=baseline_spec.as_dict(),
        candidate_spec=candidate_spec.as_dict(),
        generation=gen.as_dict(),
        timings={"baseline_s": base_out.elapsed_s, "candidate_s": cand_out.elapsed_s},
        alpha=alpha,
        min_detectable_drop=min_drop,
        target_power=target_power,
        n_boot=n_boot,
        bootstrap_seed=seed,
    )

    print("\n" + format_report(report))

    # ---- Persist results. ----
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created_utc.replace(":", "").replace("-", "").split(".")[0]
    base = out_dir / f"quantcheck_{suite.name}_{stamp}"
    base.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    base.with_suffix(".txt").write_text(format_report(report), encoding="utf-8")
    print(f"\nSaved: {base.with_suffix('.json')}\n       {base.with_suffix('.txt')}")

    return 2 if report.verdict == "DEGRADED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
