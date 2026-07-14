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
from core.llm import call_llm_stream
from core.response import clean_model_response, stream_filter_thinking
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
    inline_control: Callable[[], None] | None = None,
    build_layer: Callable[[], tuple[str, str] | None] | None = None,
) -> None:
    """Render the generate-button + scenario lifecycle for one scenario page.

    ``page_id`` namespaces the persisted scenario keys and Streamlit widget
    keys so the three pages can coexist in one Streamlit session without
    colliding. ``build_messages`` may return ``None`` to indicate "nothing to
    send yet" — in that case ``is_ready`` should also be returning ``False``,
    but we double-check before calling the model.

    ``inline_control`` is an optional callback rendered on the same row as the
    generate button (e.g. the AI-enhanced adversary toggle) so page-specific
    controls sit alongside the button rather than being lost above it.

    ``build_layer`` is an optional callback returning ``(layer_json, filename)``
    for an ATT&CK Navigator layer covering the scenario's techniques, or
    ``None`` when the page/matrix has no Navigator representation. It is called
    once at generation time and its result persisted alongside the scenario, so
    the downloaded layer matches the scenario the user is reading even though a
    page may resample techniques on rerun.
    """
    generated_key = f"{page_id}_scenario_generated"
    text_key = f"{page_id}_scenario_text"
    layer_key = f"{page_id}_scenario_layer"

    st.session_state.setdefault(generated_key, False)

    if inline_control is not None:
        button_col, control_col = st.columns([1, 2], vertical_alignment="center")
        with button_col:
            clicked = st.button(button_label, key=f"{page_id}_generate")
        with control_col:
            inline_control()
    else:
        clicked = st.button(button_label, key=f"{page_id}_generate")

    rendered = False
    if clicked and is_ready():
        messages = build_messages()
        if messages is not None:
            # Capture the Navigator layer now, from the same technique set the
            # model is about to see, and persist it below so a later rerun
            # (which may resample techniques) can't drift from it.
            layer_payload = build_layer() if build_layer is not None else None
            _generate_and_render(
                messages=messages,
                page_id=page_id,
                trace_name=trace_name,
                trace_tags=trace_tags,
                status_text=status_text,
                download_name=download_name,
                generated_key=generated_key,
                text_key=text_key,
                layer_key=layer_key,
                layer_payload=layer_payload,
            )
            rendered = True

    # Re-render the persisted scenario on a plain rerun (e.g. after clicking a
    # download button, which reruns the script with the generate button
    # unpressed). Without this the scenario and its downloads would vanish.
    if not rendered and st.session_state.get(generated_key) and st.session_state.get(text_key):
        st.markdown("---")
        _render_previous(
            page_id=page_id,
            text_key=text_key,
            download_name=download_name,
            layer_key=layer_key,
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
    layer_key: str,
    layer_payload: tuple[str, str] | None,
) -> None:
    config = LLMConfig.from_session_state(
        trace_name=trace_name,
        trace_tags=trace_tags,
    )
    scenario_text: str | None = None
    stream_placeholder = st.empty()
    raw_chunks: list[str] = []

    def _tee(chunks):
        for chunk in chunks:
            raw_chunks.append(chunk)
            yield chunk

    try:
        with st.status(status_text, expanded=True) as status:
            st.write("Streaming response from the model.")
            with stream_placeholder.container():
                st.write_stream(
                    stream_filter_thinking(_tee(call_llm_stream(config, messages)))
                )
            scenario_text = "".join(raw_chunks)
            status.update(label="Scenario generated.", state="complete")
    except Exception as e:
        st.error(f"An error occurred while generating the scenario: {e}")

    st.markdown("---")
    if scenario_text:
        thinking, cleaned = clean_model_response(scenario_text)
        # Replace the live-streamed view with the canonical cleaned render so
        # any code-fence stripping (or other tidy-ups) takes effect.
        stream_placeholder.empty()
        if thinking:
            with st.expander("View Model's Reasoning"):
                st.markdown(thinking)
        _persist_and_render(
            cleaned=cleaned,
            page_id=page_id,
            download_name=download_name,
            generated_key=generated_key,
            text_key=text_key,
            layer_key=layer_key,
            layer_payload=layer_payload,
        )
    elif st.session_state.get(generated_key) and st.session_state.get(text_key):
        _render_previous(
            page_id=page_id,
            text_key=text_key,
            download_name=download_name,
            layer_key=layer_key,
        )


def _persist_and_render(
    *,
    cleaned: str,
    page_id: str,
    download_name: str,
    generated_key: str,
    text_key: str,
    layer_key: str,
    layer_payload: tuple[str, str] | None,
) -> None:
    st.session_state[generated_key] = True
    st.session_state[text_key] = cleaned
    st.session_state[layer_key] = layer_payload
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
    _render_layer_download(layer_payload, key=f"{page_id}_download_layer")


def _render_previous(
    *,
    page_id: str,
    text_key: str,
    download_name: str,
    layer_key: str,
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
    _render_layer_download(
        st.session_state.get(layer_key), key=f"{page_id}_download_layer_previous"
    )


def _render_layer_download(
    layer_payload: tuple[str, str] | None, *, key: str
) -> None:
    """Render the ATT&CK Navigator layer download, if one was produced."""
    if not layer_payload:
        return
    layer_json, filename = layer_payload
    st.download_button(
        label="Download ATT&CK Navigator Layer",
        data=layer_json,
        file_name=filename,
        mime="application/json",
        key=key,
    )
