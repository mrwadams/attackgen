"""Deepened entry-point for the three scenario-generating pages.

Each scenario page owns its own widgets, prompt assembly and readiness check.
The shared control flow — generate button, LLM call, response cleaning,
download, render, feedback widget — lives here. Page-specific behaviour comes
in via the ``build_messages`` and ``is_ready`` callbacks; identity (session
state keys, widget keys) comes in via ``page_id``.

Adding a new scenario page is now: write the widgets and prompt builder, then
``run_scenario_page(page_id=..., build_messages=..., is_ready=..., ...)``.
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from core.feedback import render_feedback_widget
from core.llm import call_llm
from core.response import clean_model_response
from core.schemas import LLMConfig

Message = dict
"""A single chat message: ``{"role": "...", "content": "..."}``."""


def run_scenario_page(
    *,
    page_id: str,
    build_messages: Callable[[], list[Message] | None],
    is_ready: Callable[[], bool],
    download_name: str,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str = "Generating scenario...",
    button_label: str = "Generate Scenario",
) -> None:
    """Render the generate-button + scenario lifecycle for one scenario page.

    ``page_id`` namespaces the persisted scenario keys and Streamlit widget
    keys so the three pages can coexist in one Streamlit session without
    colliding. ``build_messages`` may return ``None`` to indicate "nothing to
    send yet" — in that case ``is_ready`` should also be returning ``False``,
    but we double-check before calling the model.
    """
    generated_key = f"{page_id}_scenario_generated"
    text_key = f"{page_id}_scenario_text"

    st.session_state.setdefault(generated_key, False)

    if st.button(button_label, key=f"{page_id}_generate"):
        if is_ready():
            messages = build_messages()
            if messages is not None:
                _generate_and_render(
                    messages=messages,
                    page_id=page_id,
                    trace_name=trace_name,
                    trace_tags=trace_tags,
                    status_text=status_text,
                    download_name=download_name,
                    generated_key=generated_key,
                    text_key=text_key,
                )

    render_feedback_widget(
        key_prefix=page_id,
        scenario_generated=st.session_state.get(generated_key, False),
    )


def _generate_and_render(
    *,
    messages: list[Message],
    page_id: str,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str,
    download_name: str,
    generated_key: str,
    text_key: str,
) -> None:
    config = LLMConfig.from_session_state(
        trace_name=trace_name,
        trace_tags=trace_tags,
    )
    scenario_text: str | None = None
    try:
        with st.status(status_text, expanded=True):
            st.write("Calling the model.")
            scenario_text = call_llm(config, messages)
            st.write("Scenario generated successfully.")
    except Exception as e:
        st.error(f"An error occurred while generating the scenario: {e}")

    st.markdown("---")
    if scenario_text:
        thinking, cleaned = clean_model_response(scenario_text)
        if thinking:
            with st.expander("View Model's Reasoning"):
                st.markdown(thinking)
        _persist_and_render(
            cleaned=cleaned,
            page_id=page_id,
            download_name=download_name,
            generated_key=generated_key,
            text_key=text_key,
        )
    elif st.session_state.get(generated_key) and st.session_state.get(text_key):
        _render_previous(
            page_id=page_id,
            text_key=text_key,
            download_name=download_name,
        )


def _persist_and_render(
    *,
    cleaned: str,
    page_id: str,
    download_name: str,
    generated_key: str,
    text_key: str,
) -> None:
    st.session_state[generated_key] = True
    st.session_state[text_key] = cleaned
    # Cross-page handoff for the AttackGen Assistant chat page.
    st.session_state["last_scenario"] = True
    st.session_state["last_scenario_text"] = cleaned

    st.markdown(cleaned)
    st.download_button(
        label="Download Scenario",
        data=cleaned,
        file_name=download_name,
        mime="text/markdown",
        key=f"{page_id}_download",
    )


def _render_previous(
    *,
    page_id: str,
    text_key: str,
    download_name: str,
) -> None:
    text = st.session_state.get(text_key, "")
    st.markdown("Displaying previously generated scenario:")
    st.markdown(text)
    st.download_button(
        label="Download Scenario",
        data=text,
        file_name=download_name,
        mime="text/markdown",
        key=f"{page_id}_download_previous",
    )
