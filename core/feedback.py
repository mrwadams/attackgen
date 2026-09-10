"""LangSmith feedback widget for scenario pages.

Pages don't see the LangSmith client — `render_feedback_widget` initialises one
lazily from ``st.secrets['LANGCHAIN_API_KEY']``. The run it rates is the one the
caller passes in: ``core.llm.call_llm`` sets ``st.session_state['run_id']`` when
a LangSmith client is configured for tracing, and ``core.scenario_page`` copies
it into the page's own result namespace so it stays with that result.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _get_secret(name: str) -> Any | None:
    """Return a Streamlit secret, including when no secrets file exists."""
    try:
        return st.secrets[name]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


def _get_client() -> Any | None:
    api_key = _get_secret("LANGCHAIN_API_KEY")
    if api_key is None:
        return None
    try:
        from langsmith import Client

        return Client(api_key=api_key)
    except Exception:
        return None


def render_feedback_widget(
    *, key_prefix: str, scenario_generated: bool, run_id: str | None = None
) -> None:
    """Render the thumbs-up/down feedback widget for one scenario result.

    The notice + horizontal rule render unconditionally so the page layout
    matches the pre-refactor look even before any scenario has been generated.
    `key_prefix` namespaces the Streamlit button keys so the widget can render
    on multiple pages within one app session without collision.

    `run_id` pins the feedback to the run that produced the scenario on screen.
    Pages persist it alongside the result, so rating a result after navigating
    away and back — or after another page has generated its own scenario —
    cannot submit feedback against a different run. There is deliberately no
    fallback to the session's most recent run: that is whatever any page last
    generated, so a caller with no run id has no run to rate and gets a warning.
    """
    if _get_secret("LANGCHAIN_API_KEY") is None:
        st.info(
            "ℹ️ No LangChain API key has been set. "
            "This run will not be logged to LangSmith."
        )

    placeholder = st.empty()
    st.markdown("---")

    if not scenario_generated:
        return

    client = _get_client()
    if client is None:
        return

    st.markdown("Rate the scenario to help improve this tool.")
    col1, col2, _ = st.columns([0.5, 0.5, 5])
    with col1:
        # Streamlit's accessible name for a button is its label text, not its
        # `help` tooltip. An emoji-only label leaves screen readers to guess
        # (or stay silent), so the label itself carries the meaning.
        if st.button("👍 Helpful", key=f"thumbs_up_{key_prefix}"):
            _submit(client, placeholder, kind="positive", score=1, run_id=run_id)
    with col2:
        if st.button("👎 Not helpful", key=f"thumbs_down_{key_prefix}"):
            _submit(client, placeholder, kind="negative", score=0, run_id=run_id)


def _submit(
    client: Any, placeholder: Any, *, kind: str, score: int, run_id: str | None
) -> None:
    if not run_id:
        placeholder.warning("No run ID found. Please generate a scenario first.")
        return
    try:
        record = client.create_feedback(run_id, kind, score=score, comment="")
        st.session_state["feedback"] = {"feedback_id": str(record.id), "score": score}
        placeholder.success("Feedback submitted. Thank you.")
    except Exception as e:
        placeholder.error(f"An error occurred while creating feedback: {e}")
