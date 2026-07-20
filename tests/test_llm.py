"""Tests for `core.llm._build_litellm_kwargs` and `call_llm`."""

from __future__ import annotations

import pytest

import core.llm as llm_module
from core.schemas import LLMConfig

_build_litellm_kwargs = llm_module._build_litellm_kwargs
call_llm = llm_module.call_llm
GEMINI_SAFETY_SETTINGS = llm_module.GEMINI_SAFETY_SETTINGS


def test_kwargs_always_set_num_retries_to_three() -> None:
    config = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="k")
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["num_retries"] == 3


def test_openai_provider_uses_max_completion_tokens() -> None:
    """Routing is keyed off the OpenAI provider, not a per-model flag: every
    current OpenAI chat model takes max_completion_tokens, not max_tokens."""
    config = LLMConfig(
        provider="OpenAI API",
        model_name="gpt-5.5",
        api_key="k",
        max_tokens=4096,
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["max_completion_tokens"] == 4096
    assert "max_tokens" not in kwargs
    # OpenAI keeps the empty litellm_prefix, so the model string is bare.
    assert kwargs["model"] == "gpt-5.5"


def test_openai_provider_without_max_tokens_omits_completion_tokens() -> None:
    config = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="k")
    kwargs = _build_litellm_kwargs(config)
    assert "max_completion_tokens" not in kwargs
    assert "max_tokens" not in kwargs


def test_openai_provider_omits_temperature() -> None:
    """gpt-5.x reject any temperature but the default (1), so we must not send
    one — otherwise litellm raises BadRequestError (regression: v0.13). The
    whole OpenAI family is the reasoning family, so we gate on the provider."""
    config = LLMConfig(
        provider="OpenAI API", model_name="gpt-5.5", api_key="k", temperature=0.7
    )
    kwargs = _build_litellm_kwargs(config)
    assert "temperature" not in kwargs


def test_non_reasoning_models_still_send_temperature() -> None:
    config = LLMConfig(
        provider="Anthropic API",
        model_name="claude-sonnet-5",
        api_key="k",
        temperature=0.7,
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["temperature"] == 0.7


def test_anthropic_defaults_max_tokens_to_16000_when_unset() -> None:
    config = LLMConfig(
        provider="Anthropic API", model_name="claude-sonnet-5", api_key="k"
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["max_tokens"] == 16000
    assert kwargs["model"] == "anthropic/claude-sonnet-5"


def test_anthropic_respects_explicit_max_tokens() -> None:
    config = LLMConfig(
        provider="Anthropic API",
        model_name="claude-sonnet-5",
        api_key="k",
        max_tokens=8000,
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["max_tokens"] == 8000


def test_google_ai_attaches_safety_settings() -> None:
    config = LLMConfig(
        provider="Google AI API", model_name="gemini-3.1-pro-preview", api_key="k"
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["safety_settings"] == GEMINI_SAFETY_SETTINGS
    assert kwargs["model"] == "gemini/gemini-3.1-pro-preview"


def test_custom_provider_strips_prefix_and_forces_openai_routing() -> None:
    config = LLMConfig(
        provider="Custom",
        model_name="my-local-model",
        api_base="http://127.0.0.1:1234/v1",
    )
    kwargs = _build_litellm_kwargs(config)
    # Bare model name — no openai/ prefix even though the provider has one.
    assert kwargs["model"] == "my-local-model"
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["api_base"] == "http://127.0.0.1:1234/v1"
    # Falls back to a placeholder when none provided.
    assert kwargs["api_key"] == "not-required"


def test_custom_provider_with_slash_in_model_name_regression() -> None:
    """Regression test: LiteLLM's auto-detection misroutes "qwen/qwen3-32b"
    by reading "qwen" as a provider. Setting custom_llm_provider="openai" and
    sending the bare model name bypasses that. Fixed in commit 479d578."""
    config = LLMConfig(
        provider="Custom",
        model_name="qwen/qwen3-32b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="actual-key",
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["model"] == "qwen/qwen3-32b"
    assert kwargs["custom_llm_provider"] == "openai"
    # User-provided key wins over the "not-required" fallback.
    assert kwargs["api_key"] == "actual-key"


@pytest.fixture
def disable_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the LangSmith-bypass path so call_llm hits _raw_call directly."""
    monkeypatch.setattr(llm_module, "_langsmith_client", None)


def test_call_llm_returns_completion_content(
    disable_langsmith, mock_litellm_completion
) -> None:
    mock_litellm_completion.content = "hello world"
    config = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="k")

    result = call_llm(config, [{"role": "user", "content": "hi"}])

    assert result == "hello world"
    assert len(mock_litellm_completion.calls) == 1


def test_call_llm_passes_built_kwargs_to_litellm(
    disable_langsmith, mock_litellm_completion
) -> None:
    config = LLMConfig(
        provider="Anthropic API", model_name="claude-sonnet-5", api_key="k"
    )
    messages = [{"role": "user", "content": "hi"}]

    call_llm(config, messages)

    _args, kwargs = mock_litellm_completion.calls[0]
    assert kwargs["messages"] == messages
    assert kwargs["model"] == "anthropic/claude-sonnet-5"
    assert kwargs["max_tokens"] == 16000
    assert kwargs["api_key"] == "k"
    assert kwargs["num_retries"] == 3


def test_stash_run_id_swallows_missing_streamlit_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Headless callers hit the traced path without a Streamlit ScriptRunContext.

    Writing to ``st.session_state`` then raises; ``_stash_run_id`` must swallow it
    so an MCP-server generate call doesn't crash when LANGCHAIN_API_KEY is set.
    """
    class _Exploding:
        def __setitem__(self, key, value):
            raise RuntimeError("no ScriptRunContext")

    monkeypatch.setattr(llm_module.st, "session_state", _Exploding())
    # Must not raise.
    llm_module._stash_run_id("some-run-id")
