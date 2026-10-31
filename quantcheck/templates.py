"""Minimal chat-template wrapping for instruction-tuned models.

Why this exists
---------------
Instruction/chat-tuned GGUFs expect their prompts wrapped in the model family's
turn format. Fed a *raw* completion prompt, an instruct model often ignores the
task instruction (e.g. "write your final answer as #### <number>") and rambles,
so the scorer extracts nothing — you end up measuring a parsing artifact, not
quantization. This is worst on chain-of-thought tasks like GSM8K.

This module is deliberately **minimal**: it hardcodes the few template strings
needed for the measured runs rather than implementing a general per-model
template engine (that is a later adoption feature — see examples/REAL_RUN.md).
Each entry carries the recommended ``stop`` strings so generation halts at the
end of the model's turn instead of hallucinating a new one.

Template strings use a single ``{prompt}`` placeholder. They intentionally do
**not** include a leading BOS token: ``llama-cpp-python`` adds BOS at tokenize
time, so embedding ``<bos>`` here would double it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ChatTemplate", "CHAT_TEMPLATES", "get_template", "list_templates"]


@dataclass(frozen=True)
class ChatTemplate:
    """A prompt-wrapping template plus the stop strings that close the turn.

    Attributes:
        name: registry key (e.g. "gemma").
        template: format string with a single ``{prompt}`` placeholder.
        stop: strings that should terminate generation (the turn delimiter).
    """

    name: str
    template: str
    stop: tuple[str, ...]

    def wrap(self, prompt: str) -> str:
        """Wrap a raw prompt in this template's turn format."""
        return self.template.format(prompt=prompt)


# Gemma 2 / Gemma 3 instruction format. No system role; the task instruction
# goes in the user turn. BOS is added by the tokenizer, not here.
_GEMMA = ChatTemplate(
    name="gemma",
    template="<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
    stop=("<end_of_turn>",),
)

# Qwen2 / Qwen2.5 "ChatML" format. Unlike Gemma, Qwen *does* have a system role,
# and its own chat template injects this exact default system line when the
# caller supplies none — so reproducing it here keeps the prompt faithful to how
# the model was tuned rather than inventing a bare user-only variant.
_QWEN = ChatTemplate(
    name="qwen",
    template=(
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    stop=("<|im_end|>",),
)

# DeepSeek-R1 distills are ChatML-ish but drop the default system line and open
# the assistant turn with a <think> block. The distill cards specify no system
# prompt, so the task instruction goes in the user turn.
_DEEPSEEK_R1 = ChatTemplate(
    name="deepseek-r1",
    template="<｜User｜>{prompt}<｜Assistant｜>",
    stop=("<｜end▁of▁sentence｜>", "<｜User｜>"),
)

CHAT_TEMPLATES: dict[str, ChatTemplate] = {
    "gemma": _GEMMA,
    "qwen": _QWEN,
    "deepseek-r1": _DEEPSEEK_R1,
}


def get_template(name: str) -> ChatTemplate:
    """Look up a chat template by name.

    Args:
        name: registry key (see :func:`list_templates`).

    Returns:
        The ChatTemplate.

    Raises:
        ValueError: if the name is unknown, listing valid options.
    """
    try:
        return CHAT_TEMPLATES[name]
    except KeyError:
        raise ValueError(
            f"unknown chat template {name!r}; valid: {sorted(CHAT_TEMPLATES)}"
        ) from None


def list_templates() -> list[str]:
    """Return the registered chat-template names."""
    return sorted(CHAT_TEMPLATES)
