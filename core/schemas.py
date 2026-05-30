"""Data models for LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class LLMConfig:
    """Configuration for a single LLM call.

    Constructed by the UI layer from session state via `from_session_state`.
    """

    provider: str           # "OpenAI API", "Anthropic API", "Google AI API", "Mistral API", "Groq API", "Custom"
    model_name: str         # Bare model id, e.g. "gpt-5.5", "claude-sonnet-4-6"
    api_key: str | None = None
    api_base: str | None = None      # For Custom (OpenAI-compatible) endpoints
    temperature: float = 0.7
    max_tokens: int | None = None
    trace_name: str = "AttackGen LLM call"
    trace_tags: tuple[str, ...] = ()

    @classmethod
    def from_session_state(
        cls,
        *,
        trace_name: str = "AttackGen LLM call",
        trace_tags: tuple[str, ...] = (),
    ) -> "LLMConfig":
        """Build an LLMConfig from the keys the Welcome sidebar populates."""
        return cls(
            provider=st.session_state.get("chosen_model_provider", "OpenAI API"),
            model_name=st.session_state.get("llm_model_name", ""),
            api_key=st.session_state.get("llm_api_key") or None,
            api_base=st.session_state.get("llm_api_base") or None,
            trace_name=trace_name,
            trace_tags=trace_tags,
        )
