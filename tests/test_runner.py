"""Tests for the runner — the critical invariant: never two models loaded at once."""

from __future__ import annotations

import pytest

from quantcheck.runner import (
    GenerationConfig,
    ModelSpec,
    MockBackend,
    run_model,
    make_backend,
)
from quantcheck.suites import Suite, SuiteItem


def _toy_suite(n=5) -> Suite:
    items = [
        SuiteItem(prompt=f"q{i}", answer="x", scorer="exact_match", task="t")
        for i in range(n)
    ]
    return Suite(name="toy", items=items, source="test")


def test_run_model_returns_one_response_per_item():
    suite = _toy_suite(4)
    gen = GenerationConfig(max_tokens=8, seed=0)
    spec = ModelSpec(path_or_id="m", backend="mock")
    out = run_model(spec, suite, gen)
    assert len(out.responses) == 4
    assert out.responses == ["q0", "q1", "q2", "q3"]  # mock echoes the prompt


def test_runner_unloads_before_returning():
    suite = _toy_suite(3)
    gen = GenerationConfig()
    be = MockBackend(ModelSpec(path_or_id="m", backend="mock"), gen)
    run_model(ModelSpec(path_or_id="m", backend="mock"), suite, gen, backend=be)
    assert be.loaded is False  # unloaded on the way out


def test_never_two_models_loaded_simultaneously():
    """Simulate the CLI's baseline-then-candidate flow against a shared log and
    assert that the candidate is never loaded while the baseline is still live."""
    suite = _toy_suite(3)
    gen = GenerationConfig()
    log: list = []

    base = MockBackend(ModelSpec(path_or_id="baseline", backend="mock"), gen,
                       responder=lambda p: "base:" + p, load_log=log)
    cand = MockBackend(ModelSpec(path_or_id="candidate", backend="mock"), gen,
                       responder=lambda p: "cand:" + p, load_log=log)

    run_model(base.spec, suite, gen, backend=base)
    run_model(cand.spec, suite, gen, backend=cand)

    # Expected lifecycle: load baseline, unload baseline, load candidate, unload candidate.
    assert log == [
        ("load", "baseline"),
        ("unload", "baseline"),
        ("load", "candidate"),
        ("unload", "candidate"),
    ]
    # Stronger invariant: scan the log; loaded count never exceeds 1.
    live = 0
    peak = 0
    for action, _ in log:
        live += 1 if action == "load" else -1
        peak = max(peak, live)
    assert peak == 1


def test_unloads_even_if_generate_raises():
    suite = _toy_suite(3)
    gen = GenerationConfig()

    def boom(_p):
        raise RuntimeError("model exploded")

    be = MockBackend(ModelSpec(path_or_id="m", backend="mock"), gen, responder=boom)
    with pytest.raises(RuntimeError):
        run_model(be.spec, suite, gen, backend=be)
    assert be.loaded is False  # finally-block unload ran despite the error


def test_generate_before_load_raises():
    be = MockBackend(ModelSpec(path_or_id="m", backend="mock"), GenerationConfig())
    with pytest.raises(RuntimeError):
        be.generate("hi", GenerationConfig())


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_backend(ModelSpec(path_or_id="m", backend="nope"), GenerationConfig())


# --- chat-template wrapping (minimal precondition for instruct-model runs) --- #
def test_render_is_identity_without_template():
    """Default (no template) must leave prompts byte-identical to the old path."""
    gen = GenerationConfig()
    assert gen.prompt_template is None
    assert gen.render("2+2?") == "2+2?"


def test_render_wraps_with_template():
    from quantcheck.templates import get_template

    tmpl = get_template("gemma")
    gen = GenerationConfig(prompt_template=tmpl.template, stop=tmpl.stop)
    out = gen.render("2+2?")
    assert out == "<start_of_turn>user\n2+2?<end_of_turn>\n<start_of_turn>model\n"
    assert "<bos>" not in out  # tokenizer adds BOS; template must not double it


def test_run_model_applies_template_to_every_prompt():
    """run_model must wrap every prompt identically (keeps the comparison paired)."""
    from quantcheck.templates import get_template

    tmpl = get_template("gemma")
    suite = _toy_suite(4)
    gen = GenerationConfig(prompt_template=tmpl.template, stop=tmpl.stop)
    spec = ModelSpec(path_or_id="m", backend="mock")
    out = run_model(spec, suite, gen)  # mock echoes the (rendered) prompt
    assert all(r.startswith("<start_of_turn>user\n") for r in out.responses)
    assert all(r.endswith("<start_of_turn>model\n") for r in out.responses)


def test_generation_config_logs_template():
    """The template must be captured in as_dict() so result files are reproducible."""
    from quantcheck.templates import get_template

    tmpl = get_template("gemma")
    d = GenerationConfig(prompt_template=tmpl.template, stop=tmpl.stop).as_dict()
    assert d["prompt_template"] == tmpl.template
    assert d["stop"] == ("<end_of_turn>",)


def test_unknown_template_raises():
    from quantcheck.templates import get_template

    with pytest.raises(ValueError):
        get_template("not-a-real-template")


@pytest.mark.parametrize("name", ["gemma", "qwen", "deepseek-r1"])
def test_registered_templates_are_wellformed(name):
    """Every template must take {prompt}, declare stops, and not double BOS."""
    from quantcheck.templates import get_template, list_templates

    assert name in list_templates()
    tmpl = get_template(name)
    out = tmpl.wrap("2+2?")
    assert "2+2?" in out
    assert "{prompt}" not in out  # placeholder was actually substituted
    assert tmpl.stop, f"{name} declares no stop strings"
    assert "<bos>" not in out  # tokenizer adds BOS; template must not duplicate it


def test_qwen_template_is_chatml_with_system_turn():
    """Qwen2.5 expects ChatML; its own template injects this default system line."""
    from quantcheck.templates import get_template

    out = get_template("qwen").wrap("2+2?")
    assert out.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\n2+2?<|im_end|>\n" in out
    assert out.endswith("<|im_start|>assistant\n")
    assert get_template("qwen").stop == ("<|im_end|>",)


def test_n_ctx_is_configurable_and_logged():
    """A raised context window must reach the backend and the result file."""
    gen = GenerationConfig(max_tokens=4096, n_ctx=8192)
    assert gen.n_ctx == 8192
    assert gen.as_dict()["n_ctx"] == 8192
    assert GenerationConfig().n_ctx == 2048  # default unchanged


def test_gguf_backend_missing_file_errors_clearly():
    from quantcheck.runner import LlamaCppBackend
    be = LlamaCppBackend(ModelSpec(path_or_id="/no/such/model.gguf", backend="gguf"),
                         GenerationConfig())
    with pytest.raises(FileNotFoundError):
        be.load()
