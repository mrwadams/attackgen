"""Tests for `core.schemas.LLMConfig`."""

from __future__ import annotations

from core.schemas import LLMConfig


def test_from_session_state_defaults_when_empty(fake_session_state) -> None:
    config = LLMConfig.from_session_state()
    assert config.provider == "OpenAI API"
    assert config.model_name == ""
    assert config.api_key is None
    assert config.api_base is None
    assert config.trace_name == "AttackGen LLM call"
    assert config.trace_tags == ()


def test_from_session_state_reads_populated_keys(fake_session_state) -> None:
    fake_session_state.update(
        {
            "chosen_model_provider": "Anthropic API",
            "llm_model_name": "claude-sonnet-5",
            "llm_api_key": "sk-test",
            "llm_api_base": "https://example.invalid/v1",
        }
    )

    config = LLMConfig.from_session_state()

    assert config.provider == "Anthropic API"
    assert config.model_name == "claude-sonnet-5"
    assert config.api_key == "sk-test"
    assert config.api_base == "https://example.invalid/v1"


def test_from_session_state_empty_string_api_key_collapses_to_none(
    fake_session_state,
) -> None:
    fake_session_state["llm_api_key"] = ""
    fake_session_state["llm_api_base"] = ""

    config = LLMConfig.from_session_state()

    assert config.api_key is None
    assert config.api_base is None
