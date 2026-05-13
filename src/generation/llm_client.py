"""LLM client with two backends: Ollama (primary, local) and Gemini (fallback, cloud).

Public API:

- :func:`generate` — primary entry point. Tries Ollama; falls back to Gemini
  on any Ollama failure. Returns the generated text.
- :func:`rewrite_query` — uses the same LLM stack to rewrite a user question
  into a documentation-style retrieval query (Query2Doc).
- :class:`OllamaClient` / :class:`GeminiClient` — usable directly when you
  want to bypass the dispatcher.
- :class:`LLMError`, :class:`OllamaError`, :class:`GeminiError` — exception
  hierarchy so callers can distinguish failure modes.

Every external call wraps the underlying SDK / HTTP request in try/except,
logs context (model, prompt size, endpoint), and re-raises as the appropriate
domain exception. The dispatcher in :func:`generate` catches
:class:`OllamaError` specifically to trigger the Gemini fallback path; any
``GeminiError`` after that propagates.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from config import settings
from src.generation.prompts import (
    build_query_rewrite_prompt,
    combine_for_ollama,
)
from src.utils.logger import setup_logger

log = setup_logger("generation.llm_client")

# Optional dependency — only imported when GEMINI_API_KEY is configured so
# that running Ollama-only does not require the SDK to be importable.
try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore

_OLLAMA_GENERATE_PATH: str = "/api/generate"


class LLMError(RuntimeError):
    """Base class — raised when both backends fail."""


class OllamaError(LLMError):
    """Raised when the Ollama backend fails."""


class GeminiError(LLMError):
    """Raised when the Gemini backend fails."""


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

class OllamaClient:
    """Talk to a local Ollama server over HTTP.

    Uses ``/api/generate`` with ``stream=false`` so the whole response comes
    back in one JSON body — simpler than wiring up a streaming parser and
    good enough for our latency budget on CPU.
    """

    def __init__(
        self,
        host: str = settings.OLLAMA_HOST,
        model: str = settings.OLLAMA_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS,
        timeout_seconds: int = settings.OLLAMA_TIMEOUT_SECONDS,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """Send a single prompt and return the model's text response.

        Raises :class:`OllamaError` with a context-rich message on transport,
        decode, or API-level failure.
        """
        prompt = combine_for_ollama(system, user)
        url = self.host + _OLLAMA_GENERATE_PATH
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.error(
                f"Ollama generate request failed: {exc} "
                f"(url={url}, model={self.model}, prompt_chars={len(prompt)})"
            )
            raise OllamaError(f"Ollama POST failed: {exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            log.error(f"Ollama returned non-JSON body (url={url}): {exc}")
            raise OllamaError(f"Ollama response not JSON: {exc}") from exc

        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            log.error(f"Ollama returned empty/invalid response (model={self.model}, body keys={list(body.keys())})")
            raise OllamaError(f"empty Ollama response: {body}")

        log.debug(
            f"Ollama generated {len(text)} chars in {body.get('total_duration', 0) / 1e9:.1f}s "
            f"(eval_count={body.get('eval_count')}, model={self.model})"
        )
        return text


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

class GeminiClient:
    """Talk to Google Gemini via the ``google-generativeai`` SDK.

    The SDK is imported lazily at module load. If the key isn't configured or
    the SDK isn't installed, calling :meth:`generate` raises
    :class:`GeminiError` — the dispatcher upstream catches that and gives up
    rather than retrying.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.GEMINI_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._configured = False
        self._client = None  # populated on first use

        key = api_key or settings.GEMINI_API_KEY
        if key and genai is not None:
            try:
                genai.configure(api_key=key)
                self._configured = True
            except Exception as exc:  # SDK raises a variety on bad keys
                log.error(f"Gemini configure failed: {exc}")
                # Do not raise here — only on .generate() so callers can
                # construct the client without surprises at import time.

    def _model(self, system: str):
        """Lazy-initialise the underlying ``GenerativeModel`` with a system
        prompt. Cached for the lifetime of this client.
        """
        if self._client is not None:
            return self._client
        if not self._configured or genai is None:
            raise GeminiError(
                "Gemini not configured (missing GEMINI_API_KEY or google-generativeai SDK)"
            )
        try:
            self._client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system,
            )
        except Exception as exc:
            log.error(f"Gemini GenerativeModel init failed (model={self.model}): {exc}")
            raise GeminiError(f"Gemini init failed: {exc}") from exc
        return self._client

    def generate(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        """Send the system + user prompt to Gemini and return the text.

        Raises :class:`GeminiError` on any transport or API failure with a
        context-rich message.
        """
        try:
            model = self._model(system)
        except GeminiError:
            raise

        gen_config = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_tokens,
        }

        try:
            response = model.generate_content(user, generation_config=gen_config)
        except Exception as exc:  # google.api_core errors, network errors, etc.
            log.error(
                f"Gemini generate_content failed: {exc} "
                f"(model={self.model}, user_chars={len(user)})"
            )
            raise GeminiError(f"Gemini generate_content failed: {exc}") from exc

        try:
            text = response.text  # convenience accessor; may raise if blocked
        except Exception as exc:
            # Most common reason: safety filter blocked the response.
            log.error(f"Gemini response.text unavailable (model={self.model}): {exc}")
            raise GeminiError(f"Gemini returned no text (likely safety block): {exc}") from exc

        if not text or not text.strip():
            log.error(f"Gemini returned empty text (model={self.model})")
            raise GeminiError(f"empty Gemini response")

        log.debug(f"Gemini generated {len(text)} chars (model={self.model})")
        return text


# --------------------------------------------------------------------------- #
# Module-level singletons (cheap to construct, no I/O at init time)
# --------------------------------------------------------------------------- #

_ollama: OllamaClient | None = None
_gemini: GeminiClient | None = None


def _ollama_client() -> OllamaClient:
    global _ollama
    if _ollama is None:
        _ollama = OllamaClient()
    return _ollama


def _gemini_client() -> GeminiClient:
    global _gemini
    if _gemini is None:
        _gemini = GeminiClient()
    return _gemini


# --------------------------------------------------------------------------- #
# Dispatcher — primary entry point
# --------------------------------------------------------------------------- #

def generate(
    system: str,
    user: str,
    temperature: Optional[float] = None,
    prefer: str = "ollama",
) -> tuple[str, str]:
    """Generate a completion. Returns ``(response_text, backend_used)``.

    Tries the preferred backend first. On :class:`OllamaError`, automatically
    retries with Gemini if the Gemini client is configured. Any
    :class:`GeminiError` propagates — there's no further fallback.

    ``prefer`` is mostly for debugging / forcing a specific path (e.g., to
    benchmark Gemini in isolation). Default is ``"ollama"`` per project spec.
    """
    if prefer == "gemini":
        text = _gemini_client().generate(system, user, temperature=temperature)
        return text, "gemini"

    # Default: Ollama primary, Gemini fallback.
    t0 = time.time()
    try:
        text = _ollama_client().generate(system, user, temperature=temperature)
        log.info(f"LLM generate (ollama): {len(text)} chars in {time.time() - t0:.1f}s")
        return text, "ollama"
    except OllamaError as exc:
        log.warning(f"Ollama failed ({exc}); falling back to Gemini")
        try:
            text = _gemini_client().generate(system, user, temperature=temperature)
            log.info(f"LLM generate (gemini fallback): {len(text)} chars in {time.time() - t0:.1f}s")
            return text, "gemini"
        except GeminiError as gex:
            log.error(f"Both backends failed. Ollama: {exc}. Gemini: {gex}")
            raise LLMError(f"both backends failed (ollama: {exc}; gemini: {gex})") from gex


# --------------------------------------------------------------------------- #
# Query rewriter — uses the same LLM stack but with a tight system prompt
# --------------------------------------------------------------------------- #

def rewrite_query(query: str) -> str:
    """Return ``query`` rewritten as a documentation-style retrieval query.

    Prefers Gemini specifically — qwen2.5-coder:1.5b is too small to follow
    keyword-extraction instructions reliably (it tends to just rephrase),
    while gemini-2.5-flash surfaces exact FastAPI class/decorator names that
    match what's in the embedded chunks. If Gemini is unavailable we fall
    back to Ollama; if both fail we return the original query unchanged so
    retrieval still works.
    """
    system, user = build_query_rewrite_prompt(query)
    try:
        text, backend = generate(system, user, temperature=0.1, prefer="gemini")
    except LLMError:
        # Gemini path failed — try Ollama as a last resort.
        try:
            text, backend = generate(system, user, temperature=0.1, prefer="ollama")
        except LLMError as exc:
            log.warning(f"rewrite_query: both backends failed ({exc}); returning original")
            return query

    rewritten = text.strip().splitlines()[0].strip()
    # Strip wrapping quotes / leading punctuation if the model added them.
    rewritten = rewritten.strip("\"'`").lstrip("- ").strip()
    if not rewritten or len(rewritten) > 200:
        log.warning(f"rewrite_query: unusable output {rewritten!r}; returning original")
        return query

    log.info(f"rewrite_query[{backend}]: {query!r} -> {rewritten!r}")
    return rewritten
