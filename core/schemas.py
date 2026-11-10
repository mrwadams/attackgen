"""Data models for LLM configuration, plus the session-state keys for setup and per-page results."""

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


@dataclass(frozen=True)
class ResultKeys:
    """The session-state keys one scenario page owns.

    ``page_id`` namespaces every key so the three pages can coexist in one
    Streamlit session.

    None of these are widget keys — ``core.scenario_page`` sets and clears
    them directly, and ``core.assistant`` writes revised results back to
    ``text`` and ``defense``, which Streamlit only permits for keys no widget
    owns.
    """

    page_id: str

    @property
    def generated(self) -> str:
        return f"{self.page_id}_scenario_generated"

    @property
    def text(self) -> str:
        return f"{self.page_id}_scenario_text"

    @property
    def layer(self) -> str:
        return f"{self.page_id}_scenario_layer"

    @property
    def filename(self) -> str:
        return f"{self.page_id}_scenario_filename"

    @property
    def defense(self) -> str:
        return f"{self.page_id}_scenario_defense"

    @property
    def defense_report(self) -> str:
        """The structured report a narrative retry needs to rebuild its prompt."""
        return f"{self.page_id}_scenario_defense_report"

    @property
    def request(self) -> str:
        """The :class:`ScenarioRequest` frozen when this result was generated."""
        return f"{self.page_id}_scenario_request"

    @property
    def status(self) -> str:
        """The current phase failure, if any (see ``core.scenario_page.Status``)."""
        return f"{self.page_id}_generation_status"

    @property
    def narrative_running(self) -> str:
        """Set while the optional narrative streams; survives a torn-down run."""
        return f"{self.page_id}_narrative_running"

    @property
    def narrative_stop(self) -> str:
        return f"{self.page_id}_narrative_stop_requested"

    @property
    def retry_base(self) -> str:
        return f"{self.page_id}_retry_base_requested"

    @property
    def retry_narrative(self) -> str:
        return f"{self.page_id}_retry_narrative_requested"

    @property
    def run_id(self) -> str:
        """The LangSmith run behind *this* result, so feedback can't drift."""
        return f"{self.page_id}_scenario_run_id"

    @property
    def regenerate(self) -> str:
        return f"{self.page_id}_regenerate_requested"

    @property
    def clear(self) -> str:
        return f"{self.page_id}_clear_requested"

    @property
    def pre_apply(self) -> str:
        """The scenario + narrative stashed before the *first* Assistant apply."""
        return f"{self.page_id}_scenario_pre_apply"

    @property
    def applied_at(self) -> str:
        """When the Assistant last applied changes to this result, if ever."""
        return f"{self.page_id}_scenario_applied_at"

    @property
    def revert(self) -> str:
        return f"{self.page_id}_revert_requested"

    def result_keys(self) -> tuple[str, ...]:
        """Every key holding this page's latest result (what Clear removes)."""
        return (
            self.generated,
            self.text,
            self.layer,
            self.filename,
            self.defense,
            self.defense_report,
            self.request,
            self.status,
            self.run_id,
            self.pre_apply,
            self.applied_at,
        )
