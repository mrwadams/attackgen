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


def test_openai_reasoning_model_uses_max_completion_tokens() -> None:
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


def test_openai_reasoning_model_without_max_tokens_omits_completion_tokens() -> None:
    config = LLMConfig(provider="OpenAI API", model_name="gpt-5.5", api_key="k")
    kwargs = _build_litellm_kwargs(config)
    assert "max_completion_tokens" not in kwargs
    assert "max_tokens" not in kwargs


def test_anthropic_defaults_max_tokens_to_16000_when_unset() -> None:
    config = LLMConfig(
        provider="Anthropic API", model_name="claude-sonnet-4-6", api_key="k"
    )
    kwargs = _build_litellm_kwargs(config)
    assert kwargs["max_tokens"] == 16000
    assert kwargs["model"] == "anthropic/claude-sonnet-4-6"


def test_anthropic_respects_explicit_max_tokens() -> None:
    config = LLMConfig(
        provider="Anthropic API",
        model_name="claude-sonnet-4-6",
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
        provider="Anthropic API", model_name="claude-sonnet-4-6", api_key="k"
    )
    messages = [{"role": "user", "content": "hi"}]

    call_llm(config, messages)

    _args, kwargs = mock_litellm_completion.calls[0]
    assert kwargs["messages"] == messages
    assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 16000
    assert kwargs["api_key"] == "k"
    assert kwargs["num_retries"] == 3
