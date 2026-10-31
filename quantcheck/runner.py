"""VRAM-safe, model-agnostic sequential execution.

QuantCheck compares two models, but never holds both in memory at once — a hard
requirement for the ~6GB-VRAM consumer hardware this tool targets. The runner:

    1. loads the **baseline** backend,
    2. generates a response for every prompt (deterministically),
    3. **unloads** the baseline (frees the handle, runs gc, frees CUDA cache),
    4. loads the **candidate** backend,
    5. generates for every prompt,
    6. unloads the candidate.

Model-agnostic by design
-------------------------
Execution goes through a small :class:`ModelBackend` protocol. The default
backend runs **GGUF** files via ``llama-cpp-python`` (the project's primary
target), but any local engine can be plugged in by registering a backend:

    - ``gguf``  : local GGUF via llama-cpp-python (CPU or CUDA).
    - ``openai``: any local OpenAI-compatible HTTP server (llama.cpp ``server``,
      Ollama's ``/v1`` endpoint, LM Studio, vLLM) — still fully local, no paid API.
    - ``mock``  : deterministic in-memory backend for tests / dry runs.

This keeps the statistics and reporting completely independent of how a model is
executed: the rest of the tool only ever sees prompt-in / text-out.

Determinism
-----------
Generation is configured for temperature 0 with a fixed seed so the comparison
isolates the quantization effect. All generation settings are captured in
:class:`GenerationConfig` and logged into every result file. NOTE: even at
temperature 0, GGUF inference can be **slightly nondeterministic** due to
floating-point reduction order on some backends/batch sizes; this is a known
llama.cpp caveat documented in the README, not a bug in this tool.
"""

from __future__ import annotations

import gc
import importlib
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol, Sequence, runtime_checkable

from .suites import Suite

__all__ = [
    "GenerationConfig",
    "ModelSpec",
    "ModelBackend",
    "MockBackend",
    "ModelRunOutput",
    "register_backend",
    "make_backend",
    "run_model",
]


# --------------------------------------------------------------------------- #
# Configuration / specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic generation settings, logged into every result file.

    Attributes:
        max_tokens: maximum new tokens to generate per prompt.
        temperature: sampling temperature; 0.0 for greedy/deterministic decoding.
        top_p / top_k: nucleus / top-k params (irrelevant at temperature 0, but
            logged for completeness).
        seed: RNG seed passed to the backend.
        stop: optional stop strings.
        n_ctx: context window size for local backends.
        prompt_template: optional format string with a single ``{prompt}``
            placeholder. When set, each suite prompt is wrapped with it before
            generation (see :mod:`quantcheck.templates`). ``None`` = raw prompt.
            Applied identically to both models, so the comparison stays paired.
    """

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 0
    stop: tuple[str, ...] = ()
    n_ctx: int = 2048
    prompt_template: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self, prompt: str) -> str:
        """Return ``prompt`` wrapped by ``prompt_template`` (or unchanged)."""
        if self.prompt_template is None:
            return prompt
        return self.prompt_template.format(prompt=prompt)


@dataclass(frozen=True)
class ModelSpec:
    """How to instantiate one model's backend.

    Attributes:
        path_or_id: GGUF file path (gguf backend), model name (openai backend),
            or arbitrary id (mock backend).
        backend: backend name ("gguf", "openai", "mock", or a registered name).
        n_gpu_layers: gguf only — layers to offload to GPU. 0 = CPU-only;
            -1 = offload all (will OOM if the model exceeds VRAM).
        extra: backend-specific keyword arguments (e.g. base_url for openai).
    """

    path_or_id: str
    backend: str = "gguf"
    n_gpu_layers: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Backend protocol + registry
# --------------------------------------------------------------------------- #
@runtime_checkable
class ModelBackend(Protocol):
    """A loadable, queryable, unloadable model.

    Lifecycle: ``load()`` -> repeated ``generate()`` -> ``unload()``. The runner
    guarantees only one backend is loaded at a time.
    """

    def load(self) -> None:
        """Acquire model resources (read GGUF into RAM/VRAM, open a session)."""
        ...

    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        """Return the model's text response to ``prompt`` under ``gen``."""
        ...

    def unload(self) -> None:
        """Release all resources so the next model can load within budget."""
        ...


_BACKEND_REGISTRY: dict[str, Callable[[ModelSpec, GenerationConfig], ModelBackend]] = {}


def register_backend(
    name: str, factory: Callable[[ModelSpec, GenerationConfig], ModelBackend]
) -> None:
    """Register a backend factory under ``name`` (overrides any existing)."""
    _BACKEND_REGISTRY[name] = factory


def make_backend(spec: ModelSpec, gen: GenerationConfig) -> ModelBackend:
    """Construct a backend for ``spec``.

    Args:
        spec: the model specification (path/id, backend name, options).
        gen: generation config (needed by some backends at construction, e.g.
            n_ctx for local engines).

    Returns:
        An unloaded ModelBackend.

    Raises:
        ValueError: if the backend name is not registered.
    """
    try:
        factory = _BACKEND_REGISTRY[spec.backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {spec.backend!r}; registered: {sorted(_BACKEND_REGISTRY)}"
        ) from None
    return factory(spec, gen)


# --------------------------------------------------------------------------- #
# GGUF backend (llama-cpp-python) — the primary target
# --------------------------------------------------------------------------- #
class LlamaCppBackend:
    """GGUF backend using ``llama-cpp-python``.

    Imports llama_cpp lazily so the rest of the tool (and the test suite) does
    not require it to be installed. Raises a clear, actionable error if the GGUF
    cannot be loaded or VRAM is exceeded.
    """

    def __init__(self, spec: ModelSpec, gen: GenerationConfig) -> None:
        self.spec = spec
        self.gen = gen
        self._llm = None  # the live llama_cpp.Llama handle, or None when unloaded

    def load(self) -> None:
        import os

        if not os.path.exists(self.spec.path_or_id):
            raise FileNotFoundError(
                f"GGUF not found: {self.spec.path_or_id}\n"
                f"Pass a valid path with --baseline / --candidate."
            )
        try:
            llama_cpp = importlib.import_module("llama_cpp")
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for the 'gguf' backend.\n"
                "Install CPU build:  pip install llama-cpp-python\n"
                "Install CUDA build: CMAKE_ARGS=\"-DGGML_CUDA=on\" "
                "pip install llama-cpp-python\n"
                f"(original import error: {e})"
            ) from e
        try:
            self._llm = llama_cpp.Llama(
                model_path=self.spec.path_or_id,
                n_gpu_layers=self.spec.n_gpu_layers,
                n_ctx=self.gen.n_ctx,
                seed=self.gen.seed,
                logits_all=False,
                verbose=False,
                **self.spec.extra,
            )
        except Exception as e:  # noqa: BLE001 — surface any load failure clearly
            raise RuntimeError(
                f"failed to load GGUF {self.spec.path_or_id!r}: {e}\n"
                f"If this is an out-of-memory error, try fewer GPU layers "
                f"(--n-gpu-layers 0 for CPU-only) or a smaller / more quantized model."
            ) from e

    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        if self._llm is None:
            raise RuntimeError("backend.generate() called before load()")
        out = self._llm(
            prompt,
            max_tokens=gen.max_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            top_k=gen.top_k,
            seed=gen.seed,
            stop=list(gen.stop) or None,
        )
        return out["choices"][0]["text"]

    def unload(self) -> None:
        # Drop the handle, force gc, and free the CUDA cache if torch is present.
        self._llm = None
        gc.collect()
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — torch is optional; never fail unload
            pass


# --------------------------------------------------------------------------- #
# OpenAI-compatible local HTTP backend (llama.cpp server, Ollama, LM Studio...)
# --------------------------------------------------------------------------- #
class OpenAICompatBackend:
    """Backend for a local OpenAI-compatible HTTP server. Still fully local.

    ``spec.path_or_id`` is the model name; ``spec.extra['base_url']`` is the
    endpoint (default ``http://localhost:8080/v1``). Uses the ``requests``
    library lazily. No paid API: this is for self-hosted servers.
    """

    def __init__(self, spec: ModelSpec, gen: GenerationConfig) -> None:
        self.spec = spec
        self.gen = gen
        self.base_url = spec.extra.get("base_url", "http://localhost:8080/v1")
        self.api_key = spec.extra.get("api_key", "not-needed")
        self._requests = None

    def load(self) -> None:
        try:
            self._requests = importlib.import_module("requests")
        except ImportError as e:
            raise ImportError(
                "the 'openai' backend needs the 'requests' package: pip install requests"
            ) from e

    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        if self._requests is None:
            raise RuntimeError("backend.generate() called before load()")
        payload = {
            "model": self.spec.path_or_id,
            "prompt": prompt,
            "max_tokens": gen.max_tokens,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "seed": gen.seed,
            "stop": list(gen.stop) or None,
        }
        try:
            r = self._requests.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=300,
            )
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"request to local server {self.base_url} failed: {e}\n"
                f"Is the server running and serving model {self.spec.path_or_id!r}?"
            ) from e
        return r.json()["choices"][0]["text"]

    def unload(self) -> None:
        # The server owns the weights; nothing to free locally. We do not send a
        # shutdown — the user manages the two servers. (For strict one-at-a-time
        # VRAM, run the servers sequentially yourself; the gguf backend is the
        # default precisely because it enforces this in-process.)
        self._requests = None
        gc.collect()


# --------------------------------------------------------------------------- #
# Mock backend (deterministic, no dependencies) — for tests and dry runs
# --------------------------------------------------------------------------- #
class MockBackend:
    """Deterministic in-memory backend driven by a response function.

    Used by the test suite to verify the runner never holds two models at once,
    and for ``--dry-run`` smoke tests without any model files.

    Args:
        responder: maps a prompt to a canned response string. Defaults to echoing
            the prompt.
        load_log: optional shared list the backend appends ("load"/"unload",
            spec.path_or_id) events to, so tests can assert lifecycle ordering.
    """

    def __init__(
        self,
        spec: ModelSpec,
        gen: GenerationConfig,
        responder: Callable[[str], str] | None = None,
        load_log: list | None = None,
    ) -> None:
        self.spec = spec
        self.gen = gen
        self.responder = responder or (lambda p: p)
        self.load_log = load_log
        self.loaded = False

    def load(self) -> None:
        self.loaded = True
        if self.load_log is not None:
            self.load_log.append(("load", self.spec.path_or_id))

    def generate(self, prompt: str, gen: GenerationConfig) -> str:
        if not self.loaded:
            raise RuntimeError("backend.generate() called before load()")
        return self.responder(prompt)

    def unload(self) -> None:
        self.loaded = False
        if self.load_log is not None:
            self.load_log.append(("unload", self.spec.path_or_id))


# Register the built-in backends.
register_backend("gguf", LlamaCppBackend)
register_backend("openai", OpenAICompatBackend)
register_backend("mock", lambda spec, gen: MockBackend(spec, gen))


# --------------------------------------------------------------------------- #
# Run one model over a suite (load -> generate all -> unload)
# --------------------------------------------------------------------------- #
@dataclass
class ModelRunOutput:
    """Raw responses from one model over a suite.

    Attributes:
        spec: the model spec that produced these responses.
        responses: per-prompt raw text, aligned with ``suite.items``.
        elapsed_s: wall-clock seconds for load + all generations + unload.
    """

    spec: ModelSpec
    responses: list[str]
    elapsed_s: float


def run_model(
    spec: ModelSpec,
    suite: Suite,
    gen: GenerationConfig,
    *,
    backend: ModelBackend | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ModelRunOutput:
    """Load one model, generate a response for every suite item, then unload.

    This function guarantees the model is **unloaded before it returns** (even on
    error), so a caller running baseline then candidate never holds both at once.

    Args:
        spec: the model specification.
        suite: the evaluation suite (defines prompt order = pairing key).
        gen: deterministic generation settings.
        backend: pre-built backend (mainly for tests); otherwise built from spec.
        progress: optional callback(i, total) invoked after each prompt.

    Returns:
        ModelRunOutput with one response per suite item, in suite order.

    Raises:
        Whatever the backend raises on load/generate, after ensuring unload() ran.
    """
    be = backend if backend is not None else make_backend(spec, gen)
    responses: list[str] = []
    start = time.time()
    be.load()
    try:
        total = len(suite.items)
        for i, item in enumerate(suite.items, start=1):
            responses.append(be.generate(gen.render(item.prompt), gen))
            if progress is not None:
                progress(i, total)
    finally:
        # Unload no matter what so the next model fits in the VRAM budget.
        be.unload()
    return ModelRunOutput(spec=spec, responses=responses, elapsed_s=time.time() - start)
