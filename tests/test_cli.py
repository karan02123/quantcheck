"""Tests for CLI argument handling that can fail a run before any model loads."""

from __future__ import annotations

from quantcheck.cli import main


def test_max_tokens_must_fit_inside_n_ctx(capsys):
    """A token budget larger than the context window truncates every response,
    which would read as degradation. The CLI must refuse instead of running."""
    rc = main([
        "--baseline", "b", "--candidate", "c", "--backend", "mock",
        "--suite", "reasoning", "--limit", "2",
        "--max-tokens", "4096", "--n-ctx", "2048",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--max-tokens" in err and "--n-ctx" in err


def test_unknown_chat_template_fails_before_loading(capsys):
    rc = main([
        "--baseline", "b", "--candidate", "c", "--backend", "mock",
        "--chat-template", "not-a-real-template",
    ])
    assert rc == 1
    assert "unknown chat template" in capsys.readouterr().err
