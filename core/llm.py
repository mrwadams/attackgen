"""Unified LLM interface. All providers routed through LiteLLM.

Public entrypoint: `call_llm(config, messages) -> str`.

Provider-specific quirks (Gemini safety, Anthropic max_tokens, OpenAI reasoning
models, Custom api_base) are handled in `_build_litellm_kwargs` and driven off
the registry in `core/models.py`, not by name-matching at the call site.

LangSmith tracing wraps `call_llm` via `@traceable` when a `Client` is available;
the active run id is stashed in `st.session_state['run_id']` so the existing
feedback widget on each page continues to work without changes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

# Silence LiteLLM's import-time WARNINGs about optional AWS deps (Bedrock /
# SageMaker pre-load) — we don't use them, and they fire before
# `litellm.suppress_debug_info = True` would take effect.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402  (logger setup above must run first)
import streamlit as st  # noqa: E402

from core.models import (
    get_litellm_prefix,
    get_provider,
    model_uses_completion_tokens,
)
from core.schemas import LLMConfig

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


# Gemini safety settings — allow security-related content generation
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ---------------------------------------------------------------------------
# LangSmith setup — optional. Kept for the existing feedback widget.
# ---------------------------------------------------------------------------

# Configure LangSmith for any page that calls `call_llm`. Previously each page
# set these at the top of its script; we centralise so a page can't forget to
# (page 4 didn't), and so adding a fifth page doesn't mean copying the block
# again.
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "AttackGen")

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable

    _langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    _langsmith_client: Client | None = Client() if _langsmith_api_key else None
except Exception:
    _langsmith_client = None
    traceable = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LiteLLM kwargs assembly
# ---------------------------------------------------------------------------


def _build_litellm_kwargs(config: LLMConfig) -> dict:
    provider = get_provider(config.provider)
    prefix = get_litellm_prefix(config.provider)
    model = prefix + config.model_name

    kwargs: dict = {
        "model": model,
        "temperature": config.temperature,
        # Retry transient errors (429s, timeouts, 5xx). LiteLLM defers to the
        # provider SDK, which uses exponential backoff and respects Retry-After.
        "num_retries": 3,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key

    if config.api_base:
        kwargs["api_base"] = config.api_base

    # OpenAI reasoning models (gpt-5.x) use max_completion_tokens
    if model_uses_completion_tokens(config.provider, config.model_name):
        if config.max_tokens:
            kwargs["max_completion_tokens"] = config.max_tokens
    elif config.provider == "Anthropic API":
        # Anthropic requires max_tokens; default generously for full scenarios.
        kwargs["max_tokens"] = config.max_tokens or 16000
    elif config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens

    if config.provider == "Google AI API":
        kwargs["safety_settings"] = GEMINI_SAFETY_SETTINGS

    if provider and provider.provider_key == "Custom":
        # Force the OpenAI-compatible path explicitly. LiteLLM's model-string
        # auto-detection misroutes when the user-typed model name contains a
        # slash (e.g. "qwen/qwen3-32b"), interpreting the part before the slash
        # as the provider. Setting custom_llm_provider bypasses that, and we
        # pass the bare model name so the endpoint receives it unchanged.
        kwargs["model"] = config.model_name
        kwargs["custom_llm_provider"] = "openai"
        # litellm needs an api_key even when the endpoint doesn't require one.
        kwargs.setdefault("api_key", "not-required")

    return kwargs


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def _raw_call(config: LLMConfig, messages: list[dict]) -> str:
    kwargs = _build_litellm_kwargs(config)
    response = litellm.completion(messages=messages, **kwargs)
    return response.choices[0].message.content or ""


def _raw_stream(config: LLMConfig, messages: list[dict]) -> Iterator[str]:
    kwargs = _build_litellm_kwargs(config)
    for chunk in litellm.completion(messages=messages, stream=True, **kwargs):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def call_llm(config: LLMConfig, messages: list[dict]) -> str:
    """Generate a single text response from any supported provider.

    Returns the assistant's content as a plain string. Raises on transport
    errors after LiteLLM has exhausted its retries — callers handle that with
    a try/except + st.error, the same pattern the old per-provider wrappers used.
    """
    if _langsmith_client is not None and traceable is not None:
        @traceable(
            run_type="llm",
            name=config.trace_name,
            tags=list(config.trace_tags),
            client=_langsmith_client,
        )
        def _traced(messages: list[dict], *, run_tree) -> str:
            content = _raw_call(config, messages)
            st.session_state["run_id"] = str(run_tree.id)
            return content

        return _traced(messages)

    return _raw_call(config, messages)


def call_llm_stream(config: LLMConfig, messages: list[dict]) -> Iterator[str]:
    """Stream text deltas from any supported provider.

    Yields raw assistant content as it arrives. The run_id is published into
    session_state at the start so the LangSmith feedback widget can pick it
    up even if the user clicks thumbs-up mid-stream.
    """
    if _langsmith_client is not None and traceable is not None:
        @traceable(
            run_type="llm",
            name=config.trace_name,
            tags=list(config.trace_tags),
            client=_langsmith_client,
        )
        def _traced(messages: list[dict], *, run_tree) -> Iterator[str]:
            st.session_state["run_id"] = str(run_tree.id)
            yield from _raw_stream(config, messages)

        return _traced(messages)

    return _raw_stream(config, messages)
