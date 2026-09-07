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

import contextlib
import copy
import inspect
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from core.detections import (
    assemble_defense_document,
    build_narrative_messages,
    defense_download_name,
    defense_to_markdown,
)
from core.feedback import render_feedback_widget
from core.llm import call_llm_stream
from core.navigator import layer_filename, navigator_for_domain
from core.response import clean_model_response, stream_filter_thinking
from core.schemas import LLMConfig

Message = dict
"""A single chat message: ``{"role": "...", "content": "..."}``."""

Snapshot = dict[str, Any]


def _invoke_with_snapshot(callback: Callable, snapshot: Snapshot):
    """Call a generation callback with its snapshot when it accepts one.

    The no-argument form remains supported for callers outside the three main
    pages, but snapshot-aware callbacks are what prevent mutable widgets from
    changing an in-flight generation's prompt or exports.
    """
    try:
        inspect.signature(callback).bind(snapshot)
    except (TypeError, ValueError):
        return callback()
    return callback(snapshot)


# Bound once, module-locally, so a test can patch *this* name instead of
# reaching into the stdlib `time` module object — which patches it for every
# thread in the process, including the stream worker running concurrently.
_monotonic = time.monotonic


def _elapsed_label(phase: str, started: float) -> str:
    elapsed = max(0, int(_monotonic() - started))
    minutes, seconds = divmod(elapsed, 60)
    return f"{phase} · elapsed {minutes}:{seconds:02d}"


_STREAM_POLL_INTERVAL = 0.2
"""Seconds between elapsed-label refreshes while waiting for the next chunk."""

_STREAM_QUEUE_MAX = 64
"""Chunks the worker may run ahead of the consumer before it blocks."""


def _attach_script_run_ctx(thread: threading.Thread) -> None:
    """Give a worker thread the calling script's Streamlit context, if any.

    Guarded: this reaches into `streamlit.runtime.scriptrunner`, which is not a
    stability-guaranteed API, and there is no context at all off the UI (the
    MCP server, tests). Failing here must not stop the stream — the cost of
    losing it is a missing run id, not a broken generation."""
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    except Exception:  # noqa: BLE001 - older/newer Streamlit, or no runtime
        return
    with contextlib.suppress(Exception):
        ctx = get_script_run_ctx()
        if ctx is not None:
            add_script_run_ctx(thread, ctx)


def _stream_on_worker(
    chunks: Iterable[str], *, on_progress: Callable[[], None] | None = None
) -> Iterator[str]:
    """Relay a blocking chunk iterator from a worker thread, ticking while idle.

    Streamlit runs the script on a single thread that blocks inside
    ``st.write_stream``, so a call that stays silent before its first token
    (reasoning models, a large prompt) freezes any progress label driven only
    by chunk arrival. Running ``chunks`` on a worker thread lets the main
    thread poll a queue with a short timeout, calling ``on_progress`` on every
    poll — including the empty ones — so the elapsed label keeps advancing
    during that wait as well as while chunks stream in. An exception raised
    while producing ``chunks`` is relayed and re-raised on the main thread,
    from the same call site a synchronous iteration would have raised it.
    """
    # Bounded: if the consumer stops reading, the worker must block rather than
    # accumulate the rest of the response in memory.
    items: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=_STREAM_QUEUE_MAX)
    stop = threading.Event()

    def _worker() -> None:
        try:
            for chunk in chunks:
                # Re-checked every iteration AND around the blocking put, so a
                # cancelled stream stops pulling from the model promptly instead
                # of draining the whole response into a queue nobody reads.
                if stop.is_set():
                    break
                while not stop.is_set():
                    try:
                        items.put(("chunk", chunk), timeout=_STREAM_POLL_INTERVAL)
                        break
                    except queue.Full:
                        continue
                # Checked again here, not just at the top: if the stop landed
                # while we were blocked on the put, going round the `for` would
                # pull another chunk from the model first — a network read that
                # can block for seconds on exactly the slow models this exists
                # for.
                if stop.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 - relayed to the main thread
            if not stop.is_set():
                with contextlib.suppress(queue.Full):
                    items.put(("error", exc), timeout=_STREAM_POLL_INTERVAL)
        finally:
            # Close the source so the underlying HTTP response is released
            # rather than left open until the daemon thread is collected.
            with contextlib.suppress(Exception):
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
            with contextlib.suppress(queue.Full):
                items.put(("done", None), timeout=_STREAM_POLL_INTERVAL)

    thread = threading.Thread(target=_worker, daemon=True)
    # Carry the Streamlit script run context onto the worker. Without it the
    # thread has no session: `call_llm_stream`'s @traceable body runs here on
    # the first `next()`, and its `_stash_run_id` write to `st.session_state`
    # lands nowhere, so the LangSmith feedback widget never sees a run id. It
    # also silences the per-call "missing ScriptRunContext!" warning.
    _attach_script_run_ctx(thread)
    thread.start()

    try:
        while True:
            try:
                kind, payload = items.get(timeout=_STREAM_POLL_INTERVAL)
            except queue.Empty:
                if on_progress:
                    on_progress()
                continue

            if kind == "chunk":
                if on_progress:
                    on_progress()
                yield payload
            elif kind == "error":
                raise payload
            else:  # "done"
                return
    finally:
        # Reached on GeneratorExit too — Streamlit raises RerunException from
        # status.update() when the user touches a widget mid-generation, which
        # closes this generator without exhausting it.
        stop.set()


def _unique_filenames(download_name: str) -> tuple[str, str, str]:
    """Turn a human base label into unique, filesystem-safe download names.

    ``"AttackGen APT29 Enterprise.md"`` ->
    ``("AttackGen_APT29_Enterprise_20260714-153045.md",
       "AttackGen_APT29_Enterprise_20260714-153045_layer.json",
       "AttackGen_APT29_Enterprise_20260714-153045_detection.md")``.

    Non-alphanumeric runs collapse to ``_`` and the stem is capped so long
    ATLAS case-study titles can't produce an unwieldy filename. A
    generation-time timestamp makes each download distinct. The Navigator layer
    and Detection & Response names are derived from the same stem so the three
    downloads always match.
    """
    base = download_name[:-3] if download_name.endswith(".md") else download_name
    stem = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:80] or "scenario"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_name = f"{stem}_{stamp}.md"
    return md_name, layer_filename(md_name), defense_download_name(md_name)


def run_scenario_page(
    *,
    page_id: str,
    build_messages: Callable[..., list[Message] | None],
    is_ready: Callable[[], bool],
    download_name: str,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str = "Generating scenario...",
    button_label: str = "Generate Scenario",
    inline_control: Callable[[], None] | None = None,
    build_layer: Callable[..., str | None] | None = None,
    build_defense: Callable[..., dict | None] | None = None,
    defense_narrative: bool = False,
    capture_inputs: Callable[[], Snapshot] | None = None,
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

    ``build_layer`` is an optional callback returning the ATT&CK Navigator
    layer JSON for the scenario's techniques, or ``None`` when the page/matrix
    has no Navigator representation. It is called once at generation time and
    its result persisted alongside the scenario, so the downloaded layer
    matches the scenario the user is reading even though a page may resample
    techniques on rerun.

    ``build_defense`` is an optional callback returning the structured
    "Detection & Response" report (from ``core.detections.build_defense_report``)
    for the scenario's techniques, or ``None`` when there's no defensive data.
    Like ``build_layer`` it is captured at generation time so it can't drift
    from the scenario. When ``defense_narrative`` is true, a second model call
    weaves those detections/mitigations into a stage-by-stage defender's
    walkthrough; the flag is read at generation time from the page's toggle.

    ``capture_inputs`` returns JSON-native metadata describing the inputs at
    the instant Generate is pressed. Snapshot-aware build callbacks receive a
    deep copy of that mapping; no-argument callbacks remain supported for
    compatibility. The snapshot is persisted with the result.

    ``download_name`` is a human base label (e.g. ``"AttackGen APT29
    Enterprise.md"``); the markdown and layer downloads get a sanitised,
    timestamped variant so files are meaningful and unique across scenarios.
    """
    generated_key = f"{page_id}_scenario_generated"
    text_key = f"{page_id}_scenario_text"
    layer_key = f"{page_id}_scenario_layer"
    filename_key = f"{page_id}_scenario_filename"
    defense_key = f"{page_id}_scenario_defense"
    snapshot_key = f"{page_id}_scenario_input_snapshot"

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
        # Freeze every user-controlled value before any slow work starts. Page
        # callbacks receive this snapshot rather than consulting live widgets.
        snapshot = copy.deepcopy(capture_inputs() if capture_inputs else {})
        snapshot.setdefault("scenario_type", page_id)
        snapshot.setdefault("captured_at", datetime.now(timezone.utc).isoformat())
        identity = snapshot.setdefault("identity", {})
        identity.setdefault("page_id", page_id)
        identity.setdefault("trace_name", trace_name)
        identity.setdefault("trace_tags", list(trace_tags))
        identity.setdefault("provider", st.session_state.get("chosen_model_provider"))
        identity.setdefault("model", st.session_state.get("llm_model_name"))
        identity.setdefault("download_name", download_name)
        st.session_state[snapshot_key] = snapshot

        _generate_and_render(
            snapshot=snapshot,
            build_messages=build_messages,
            build_layer=build_layer,
            build_defense=build_defense,
            page_id=page_id,
            trace_name=trace_name,
            trace_tags=trace_tags,
            status_text=status_text,
            human_name=download_name,
            generated_key=generated_key,
            text_key=text_key,
            layer_key=layer_key,
            filename_key=filename_key,
            defense_key=defense_key,
            defense_narrative=defense_narrative,
        )
        rendered = bool(st.session_state.get(generated_key))

    # Re-render the persisted scenario on a plain rerun (e.g. after clicking a
    # download button, which reruns the script with the generate button
    # unpressed). Without this the scenario and its downloads would vanish.
    if not rendered and st.session_state.get(generated_key) and st.session_state.get(text_key):
        st.markdown("---")
        _render_previous(
            page_id=page_id,
            text_key=text_key,
            filename_key=filename_key,
            download_name=download_name,
            layer_key=layer_key,
            defense_key=defense_key,
        )

    render_feedback_widget(
        key_prefix=page_id,
        scenario_generated=st.session_state.get(generated_key, False),
    )


def _generate_and_render(
    *,
    snapshot: Snapshot,
    build_messages: Callable[..., list[Message] | None],
    build_layer: Callable[..., str | None] | None,
    build_defense: Callable[..., dict | None] | None,
    page_id: str,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str,
    human_name: str,
    generated_key: str,
    text_key: str,
    layer_key: str,
    filename_key: str,
    defense_key: str,
    defense_narrative: bool,
) -> None:
    """Coordinate and expose each phase of a scenario generation."""
    started = _monotonic()
    stream_placeholder = st.empty()
    result_placeholder = st.empty()
    raw_chunks: list[str] = []
    scenario_text: str | None = None

    def set_phase(status, phase: str, *, state: str = "running") -> None:
        status.update(label=_elapsed_label(phase, started), state=state)

    try:
        with st.status(_elapsed_label("Preparing inputs", started), expanded=True) as status:
            st.write(
                "Generation often takes 30–50 seconds; reasoning and local models may "
                "take several minutes. You can follow each phase here."
            )
            messages = _invoke_with_snapshot(build_messages, snapshot)
            if messages is None:
                status.update(label="No scenario inputs were available.", state="error")
                return
            config = LLMConfig.from_session_state(
                trace_name=trace_name,
                trace_tags=trace_tags,
            )

            set_phase(status, "Generating base scenario")
            st.write(status_text)

            def _tee(chunks):
                for chunk in chunks:
                    raw_chunks.append(chunk)
                    yield chunk

            with stream_placeholder.container():
                st.write_stream(
                    stream_filter_thinking(
                        _tee(
                            _stream_on_worker(
                                call_llm_stream(config, messages),
                                on_progress=lambda: set_phase(
                                    status, "Generating base scenario"
                                ),
                            )
                        )
                    )
                )
            scenario_text = "".join(raw_chunks)
            set_phase(status, "Base scenario available")

            thinking, cleaned = clean_model_response(scenario_text)
            if not cleaned:
                status.update(label="The model returned no scenario.", state="error")
                return

            # Deterministic artifacts are built only after the base model has
            # completed, and exclusively from the frozen input snapshot.
            set_phase(status, "Building deterministic exports")
            snapshot_human_name = (
                snapshot.get("identity", {}).get("download_name") or human_name
            )
            md_name, layer_name, defense_name = _unique_filenames(snapshot_human_name)
            layer_json = (
                _invoke_with_snapshot(build_layer, snapshot) if build_layer else None
            )
            layer_payload = (layer_json, layer_name) if layer_json else None
            defense_report = (
                _invoke_with_snapshot(build_defense, snapshot) if build_defense else None
            )
            defense_state = _build_defense_state(
                report=defense_report,
                defense_name=defense_name,
                human_name=snapshot_human_name,
            )

            # This is the key phase boundary: persist the base result, exports,
            # and Assistant handoff before starting optional model enrichment.
            stream_placeholder.empty()
            st.markdown("---")
            if thinking:
                with st.expander("View Model's Reasoning"):
                    st.markdown(thinking)
            with result_placeholder.container():
                _persist_and_render(
                    cleaned=cleaned,
                    page_id=page_id,
                    download_name=md_name,
                    generated_key=generated_key,
                    text_key=text_key,
                    layer_key=layer_key,
                    filename_key=filename_key,
                    defense_key=defense_key,
                    layer_payload=layer_payload,
                    defense_state=defense_state,
                )

            run_narrative = snapshot.get("modifiers", {}).get(
                "purple_team_narrative", defense_narrative
            )
            if defense_report and run_narrative:
                set_phase(status, "Generating purple-team narrative")
                narrative_md = _stream_defense_narrative(
                    report=defense_report,
                    scenario_text=cleaned,
                    trace_name=trace_name,
                    on_progress=lambda: set_phase(
                        status, "Generating purple-team narrative"
                    ),
                )
                if narrative_md:
                    defense_state = _add_narrative_to_defense_state(
                        defense_state=defense_state,
                        narrative_md=narrative_md,
                        human_name=snapshot_human_name,
                    )
                    st.session_state[defense_key] = defense_state
                    st.session_state["last_defense_narrative"] = narrative_md
                    # Replace, rather than duplicate, the already available base
                    # render now that its optional companion is complete.
                    result_placeholder.empty()
                    with result_placeholder.container():
                        _render_result(
                            page_id=page_id,
                            cleaned=cleaned,
                            file_name=md_name,
                            layer_payload=layer_payload,
                            defense_state=defense_state,
                            variant="current_enriched",
                        )

            set_phase(status, "Complete", state="complete")
    except Exception as e:
        st.error(f"An error occurred while generating the scenario: {e}")

    if not scenario_text and st.session_state.get(generated_key) and st.session_state.get(text_key):
        _render_previous(
            page_id=page_id,
            text_key=text_key,
            filename_key=filename_key,
            download_name=human_name,
            layer_key=layer_key,
            defense_key=defense_key,
        )


def _build_defense_state(
    *,
    report: dict | None,
    defense_name: str,
    human_name: str,
) -> dict | None:
    """Build the deterministic companion state without making an LLM call."""
    if not report:
        return None
    deterministic_md = defense_to_markdown(report)
    title = human_name[:-3] if human_name.endswith(".md") else human_name
    download_md = assemble_defense_document(deterministic_md, None, title=title)
    return {
        "deterministic_md": deterministic_md,
        "narrative_md": None,
        "download_md": download_md,
        "filename": defense_name,
    }


def _add_narrative_to_defense_state(
    *, defense_state: dict | None, narrative_md: str, human_name: str
) -> dict:
    """Return a companion state enriched with the completed narrative."""
    state = dict(defense_state or {})
    deterministic_md = state.get("deterministic_md", "")
    title = human_name[:-3] if human_name.endswith(".md") else human_name
    state["narrative_md"] = narrative_md
    state["download_md"] = assemble_defense_document(
        deterministic_md, narrative_md, title=title
    )
    return state


def _stream_defense_narrative(
    *,
    report: dict,
    scenario_text: str,
    trace_name: str,
    on_progress: Callable[[], None] | None = None,
) -> str | None:
    """Stream the optional purple-team narrative pass; return its cleaned text."""
    config = LLMConfig.from_session_state(
        trace_name=f"{trace_name} — Detection & Response",
        trace_tags=("purple_team_narrative",),
    )
    messages = build_narrative_messages(scenario_text, report)
    placeholder = st.empty()
    chunks: list[str] = []

    def _tee(gen):
        for chunk in gen:
            chunks.append(chunk)
            yield chunk

    try:
        # Reasoning models can spend minutes on this second call before emitting
        # any text; running it on a worker thread (see _stream_on_worker) lets
        # the elapsed label keep advancing during that silent wait.
        st.write(
            "Walking the scenario from the defender's side. Reasoning models can "
            "take several minutes here; the elapsed timer keeps ticking while "
            "the model thinks."
        )
        with placeholder.container():
            st.write_stream(
                stream_filter_thinking(
                    _tee(
                        _stream_on_worker(
                            call_llm_stream(config, messages),
                            on_progress=on_progress,
                        )
                    )
                )
            )
    except Exception as e:
        st.error(f"An error occurred while generating the purple-team narrative: {e}")
        return None

    # Replace the live stream with the canonical cleaned render (done by the
    # caller via _render_defense_body), mirroring the main scenario's handling.
    placeholder.empty()
    _, cleaned = clean_model_response("".join(chunks))
    return cleaned or None


def _persist_and_render(
    *,
    cleaned: str,
    page_id: str,
    download_name: str,
    generated_key: str,
    text_key: str,
    layer_key: str,
    filename_key: str,
    defense_key: str,
    layer_payload: tuple[str, str] | None,
    defense_state: dict | None,
) -> None:
    st.session_state[generated_key] = True
    st.session_state[text_key] = cleaned
    st.session_state[layer_key] = layer_payload
    st.session_state[filename_key] = download_name
    st.session_state[defense_key] = defense_state
    # Cross-page handoff for the AttackGen Assistant chat page. The defense
    # narrative rides along so the Assistant can refine it too; set it
    # unconditionally (None when there's no narrative) so a stale one from an
    # earlier generation can't linger after a plain-scenario regen.
    st.session_state["last_scenario"] = True
    st.session_state["last_scenario_text"] = cleaned
    st.session_state["last_defense_narrative"] = (
        defense_state.get("narrative_md") if defense_state else None
    )

    _render_result(
        page_id=page_id,
        cleaned=cleaned,
        file_name=download_name,
        layer_payload=layer_payload,
        defense_state=defense_state,
        variant="current",
    )


def _render_previous(
    *,
    page_id: str,
    text_key: str,
    filename_key: str,
    download_name: str,
    layer_key: str,
    defense_key: str,
) -> None:
    text = st.session_state.get(text_key, "")
    # Prefer the name fixed at generation time so it stays stable (and matches
    # the layer) across the reruns a download click triggers.
    file_name = st.session_state.get(filename_key) or download_name
    st.markdown("Displaying previously generated scenario:")
    _render_result(
        page_id=page_id,
        cleaned=text,
        file_name=file_name,
        layer_payload=st.session_state.get(layer_key),
        defense_state=st.session_state.get(defense_key),
        variant="previous",
    )


def _render_result(
    *,
    page_id: str,
    cleaned: str,
    file_name: str,
    layer_payload: tuple[str, str] | None,
    defense_state: dict | None,
    variant: str,
) -> None:
    """Render the finished scenario and its Detection & Response companion.

    When a companion exists, the two long outputs go in side-by-side tabs so the
    reader switches rather than scrolls; otherwise the scenario renders plainly.
    ``variant`` ("current" / "previous") namespaces the download-button keys so a
    generation run and a plain rerun can't collide on a Streamlit widget key.
    """
    if defense_state:
        scenario_tab, defense_tab = st.tabs(["📄 Scenario", "🛡️ Detection & Response"])
        with scenario_tab:
            _render_scenario(page_id, cleaned, file_name, layer_payload, variant)
        with defense_tab:
            _render_defense_body(page_id, defense_state, variant)
    else:
        _render_scenario(page_id, cleaned, file_name, layer_payload, variant)


def _render_scenario(
    page_id: str,
    cleaned: str,
    file_name: str,
    layer_payload: tuple[str, str] | None,
    variant: str,
) -> None:
    st.markdown(cleaned)
    st.download_button(
        label="Download Scenario",
        data=cleaned,
        file_name=file_name,
        mime="text/markdown",
        key=f"{page_id}_download_{variant}",
    )
    _render_layer_download(layer_payload, key=f"{page_id}_download_layer_{variant}")


def _render_defense_body(page_id: str, defense_state: dict, variant: str) -> None:
    """Render the Detection & Response tab body.

    The optional narrative reads inline (it's the digestible walkthrough); the
    deterministic STIX join sits in an expander as reference — expanded when
    there's no narrative so the tab is never empty. A single download bundles
    both into one Markdown file.
    """
    narrative_md = defense_state.get("narrative_md")
    if narrative_md:
        st.markdown(narrative_md)
    if defense_state.get("deterministic_md"):
        with st.expander(
            "🛡️ Detection & Response reference (MITRE detection strategies & mitigations)",
            expanded=not narrative_md,
        ):
            st.markdown(defense_state["deterministic_md"])
    st.download_button(
        label="Download Detection & Response",
        data=defense_state["download_md"],
        file_name=defense_state["filename"],
        mime="text/markdown",
        key=f"{page_id}_download_defense_{variant}",
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
    # Point at the Navigator that actually loads this layer's domain — an ATLAS
    # layer won't parse in the ATT&CK Navigator, or vice versa.
    try:
        domain = json.loads(layer_json).get("domain", "")
    except (ValueError, TypeError):
        domain = ""
    nav_name, nav_url = navigator_for_domain(domain)
    st.caption(
        f"Upload to the [{nav_name}]({nav_url}) via "
        "**Open Existing Layer → Upload from local**."
    )
