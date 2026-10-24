"""Data models for LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

# Session-state keys the shared Setup sidebar writes and LLMConfig reads.
PROVIDER_KEY = "chosen_model_provider"
MODEL_KEY = "llm_model_name"
API_BASE_KEY = "llm_api_base"
API_KEY_KEY = "llm_api_key"


@dataclass
class LLMConfig:
    """Configuration for a single LLM call.

    Constructed by the UI layer from session state via `from_session_state`.
    """

    provider: str           # "OpenAI API", "Anthropic API", "Google AI API", "Mistral API", "Groq API", "Custom"
    model_name: str         # Bare model id, e.g. "gpt-6-astra", "claude-sonnet-5"
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
        """Build an LLMConfig from the keys the shared Setup sidebar populates."""
        return cls(
            provider=st.session_state.get(PROVIDER_KEY, "OpenAI API"),
            model_name=st.session_state.get(MODEL_KEY, ""),
            api_key=st.session_state.get(API_KEY_KEY) or None,
            api_base=st.session_state.get(API_BASE_KEY) or None,
            trace_name=trace_name,
            trace_tags=trace_tags,
        )
