"""LangSmith feedback widget for scenario pages.

Pages don't see the LangSmith client — `render_feedback_widget` initialises one
lazily from ``st.secrets['LANGCHAIN_API_KEY']`` and reads the active run id
from ``st.session_state['run_id']`` (set by ``core.llm.call_llm`` when a
LangSmith client is configured for tracing).
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def _get_client() -> Any | None:
    if "LANGCHAIN_API_KEY" not in st.secrets:
        return None
    try:
        from langsmith import Client

        return Client(api_key=st.secrets["LANGCHAIN_API_KEY"])
    except Exception:
        return None


def render_feedback_widget(*, key_prefix: str, scenario_generated: bool) -> None:
    """Render the thumbs-up/down feedback widget for the last scenario run.

    The notice + horizontal rule render unconditionally so the page layout
    matches the pre-refactor look even before any scenario has been generated.
    `key_prefix` namespaces the Streamlit button keys so the widget can render
    on multiple pages within one app session without collision.
    """
    if "LANGCHAIN_API_KEY" not in st.secrets:
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
        if st.button("👍", key=f"thumbs_up_{key_prefix}"):
            _submit(client, placeholder, kind="positive", score=1)
    with col2:
        if st.button("👎", key=f"thumbs_down_{key_prefix}"):
            _submit(client, placeholder, kind="negative", score=0)


def _submit(client: Any, placeholder: Any, *, kind: str, score: int) -> None:
    run_id = st.session_state.get("run_id")
    if not run_id:
        placeholder.warning("No run ID found. Please generate a scenario first.")
        return
    try:
        record = client.create_feedback(run_id, kind, score=score, comment="")
        st.session_state["feedback"] = {"feedback_id": str(record.id), "score": score}
        placeholder.success("Feedback submitted. Thank you.")
    except Exception as e:
        placeholder.error(f"An error occurred while creating feedback: {e}")
