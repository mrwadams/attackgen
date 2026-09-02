"""Deepened entry-point for the three scenario-generating pages.

Each scenario page owns its own widgets, prompt assembly and readiness check.
The shared control flow — generate button, LLM call, response cleaning,
download, render, feedback widget — lives here. Page-specific behaviour comes
in via the ``build_messages`` and ``is_ready`` callbacks; identity (session
state keys, widget keys) comes in via ``page_id``.

Adding a new scenario page is now: write the widgets and prompt builder, then
``run_scenario_page(page_id=..., build_messages=..., is_ready=..., ...)``.

Generation runs in phases, and only the *base* phase is allowed to fail the
run. Once the base scenario and its deterministic exports are persisted, the
optional purple-team narrative is enrichment: if it errors, is skipped, or
never finishes, the run still ends as a degraded success — everything already
produced stays on screen and downloadable, a notice explains what is missing,
and a focused **Retry purple-team narrative** action re-runs *only* the
narrative against the persisted base. A base-phase failure is attributed to the
base phase instead, and its retry replays the inputs captured when Generate was
pressed.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
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

try:  # Streamlit >= 1.37
    from streamlit.runtime.scriptrunner_utils.exceptions import (
        RerunException,
        StopException,
    )
except ImportError:  # pragma: no cover - older Streamlit module layout
    try:
        from streamlit.runtime.scriptrunner.exceptions import (
            RerunException,
            StopException,
        )
    except ImportError:
        RerunException = StopException = None

_SCRIPT_CONTROL = tuple(
    exc for exc in (RerunException, StopException) if exc is not None
)
"""Streamlit's rerun/stop signals, which travel as ordinary exceptions."""

Message = dict
"""A single chat message: ``{"role": "...", "content": "..."}``."""

Snapshot = dict[str, Any]

Status = dict[str, str]
"""Why a phase produced no output: ``{"phase": ..., "reason": ..., "detail": ...}``."""


@dataclass(frozen=True)
class _Keys:
    """The session-state keys one scenario page owns.

    ``page_id`` namespaces every key so the three pages can coexist in one
    Streamlit session; grouping them here keeps the phase helpers' signatures
    readable now that a run also tracks progress and failure state.

    None of these are widget keys — the phase helpers set and clear them
    directly, which Streamlit only permits for keys no widget owns.
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
    def snapshot(self) -> str:
        return f"{self.page_id}_scenario_input_snapshot"

    @property
    def status(self) -> str:
        """The current phase failure, if any (see ``Status``)."""
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


BASE_PHASE = "base"
NARRATIVE_PHASE = "narrative"

_BASE_KEPT = (
    "The scenario, its downloads, the ATT&CK Navigator layer and the "
    "deterministic Detection & Response reference are complete and usable."
)

_NARRATIVE_REASONS = {
    "error": "The purple-team narrative failed: {detail}.",
    "stopped": "You skipped the purple-team narrative before it finished.",
    "interrupted": "The purple-team narrative didn't finish.",
}


def _is_script_control(exc: BaseException) -> bool:
    """Is this Streamlit tearing the run down rather than a phase failing?

    A queued rerun — which is exactly what pressing Skip, or any other widget,
    does mid-stream — surfaces as an exception inside the running script. A
    bare ``except Exception`` would swallow it, dropping the rerun the click
    asked for and reporting a control signal as a generation error. Every broad
    handler below re-raises these instead, which also leaves the in-flight
    marker set so the next run can report the phase as skipped.
    """
    return bool(_SCRIPT_CONTROL) and isinstance(exc, _SCRIPT_CONTROL)


def _failure(phase: str, reason: str, detail: str = "") -> Status:
    """Record why a phase produced no output."""
    return {"phase": phase, "reason": reason, "detail": detail}


def _degraded_message(status: Status) -> str:
    """Explain a missing optional phase without implying the run failed."""
    template = _NARRATIVE_REASONS.get(
        status.get("reason", ""), _NARRATIVE_REASONS["interrupted"]
    )
    reason = template.format(detail=status.get("detail") or "no output was returned")
    return f"{reason} {_BASE_KEPT}"


def _base_failure_message(status: Status) -> str:
    detail = status.get("detail") or "the model returned nothing"
    return (
        f"The base scenario failed to generate: {detail}. Nothing downstream "
        "ran. The inputs captured when you pressed Generate are preserved — "
        "retry to run them again unchanged."
    )


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


def _elapsed_label(phase: str, started: float) -> str:
    elapsed = max(0, int(time.monotonic() - started))
    minutes, seconds = divmod(elapsed, 60)
    return f"{phase} · elapsed {minutes}:{seconds:02d}"


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
    That second call is optional enrichment — see the module docstring for how
    a failed, skipped or unfinished narrative degrades rather than blocks.

    ``capture_inputs`` returns JSON-native metadata describing the inputs at
    the instant Generate is pressed. Snapshot-aware build callbacks receive a
    deep copy of that mapping; no-argument callbacks remain supported for
    compatibility. The snapshot is persisted with the result, which is also
    what lets a base-phase retry replay the run unchanged.

    ``download_name`` is a human base label (e.g. ``"AttackGen APT29
    Enterprise.md"``); the markdown and layer downloads get a sanitised,
    timestamped variant so files are meaningful and unique across scenarios.
    """
    keys = _Keys(page_id)
    st.session_state.setdefault(keys.generated, False)

    # A narrative still flagged in-flight belongs to a previous script run that
    # never finished it; settle that into a degraded status before rendering.
    _settle_interrupted_narrative(keys)

    # Retry buttons use on_click callbacks, which Streamlit fires before this
    # script runs — so the request is known here, whatever the notice's position.
    retry_base = bool(st.session_state.pop(keys.retry_base, False))
    retry_narrative = bool(st.session_state.pop(keys.retry_narrative, False))

    if inline_control is not None:
        button_col, control_col = st.columns([1, 2], vertical_alignment="center")
        with button_col:
            clicked = st.button(button_label, key=f"{page_id}_generate")
        with control_col:
            inline_control()
    else:
        clicked = st.button(button_label, key=f"{page_id}_generate")

    # Reserve the notice's position now; it's written at the end of the run,
    # once we know whether this run's phases produced everything they should.
    notice_slot = st.empty()

    def run_generation(snapshot: Snapshot) -> bool:
        """Run every phase for one snapshot; report whether it rendered."""
        st.session_state.pop(keys.status, None)
        _generate_and_render(
            snapshot=snapshot,
            keys=keys,
            build_messages=build_messages,
            build_layer=build_layer,
            build_defense=build_defense,
            trace_name=trace_name,
            trace_tags=trace_tags,
            status_text=status_text,
            human_name=download_name,
            defense_narrative=defense_narrative,
        )
        return bool(st.session_state.get(keys.generated))

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
        st.session_state[keys.snapshot] = snapshot
        rendered = run_generation(snapshot)
    elif retry_base:
        # Replay the captured inputs rather than the live widgets, so a retry
        # regenerates the run the user actually asked for.
        rendered = run_generation(
            copy.deepcopy(st.session_state.get(keys.snapshot) or {})
        )
    elif retry_narrative:
        rendered = _retry_narrative(
            keys=keys, trace_name=trace_name, download_name=download_name
        )

    # Re-render the persisted scenario on a plain rerun (e.g. after clicking a
    # download button, which reruns the script with the generate button
    # unpressed). Without this the scenario and its downloads would vanish.
    if not rendered and st.session_state.get(keys.generated) and st.session_state.get(keys.text):
        st.markdown("---")
        _render_previous(keys=keys, download_name=download_name)

    _render_recovery_notice(keys, slot=notice_slot)

    render_feedback_widget(
        key_prefix=page_id,
        scenario_generated=st.session_state.get(keys.generated, False),
    )


def _generate_and_render(
    *,
    snapshot: Snapshot,
    keys: _Keys,
    build_messages: Callable[..., list[Message] | None],
    build_layer: Callable[..., str | None] | None,
    build_defense: Callable[..., dict | None] | None,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str,
    human_name: str,
    defense_narrative: bool,
) -> None:
    """Coordinate and expose each phase of a scenario generation."""
    started = time.monotonic()
    stream_placeholder = st.empty()
    result_placeholder = st.empty()
    raw_chunks: list[str] = []
    scenario_text: str | None = None
    base_persisted = False

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
                st.session_state[keys.status] = _failure(
                    BASE_PHASE, "error", "no scenario inputs were available"
                )
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
                    # Streamlit renders the yielded delta immediately; refreshing
                    # the label here keeps elapsed time visible as output arrives.
                    set_phase(status, "Generating base scenario")
                    yield chunk

            with stream_placeholder.container():
                st.write_stream(
                    stream_filter_thinking(_tee(call_llm_stream(config, messages)))
                )
            scenario_text = "".join(raw_chunks)
            set_phase(status, "Base scenario available")

            thinking, cleaned = clean_model_response(scenario_text)
            if not cleaned:
                status.update(label="The model returned no scenario.", state="error")
                st.session_state[keys.status] = _failure(
                    BASE_PHASE, "error", "the model returned no scenario"
                )
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
            # Everything after this point can fail without costing the user the
            # scenario they are already reading.
            stream_placeholder.empty()
            st.markdown("---")
            if thinking:
                with st.expander("View Model's Reasoning"):
                    st.markdown(thinking)
            with result_placeholder.container():
                _persist_and_render(
                    cleaned=cleaned,
                    keys=keys,
                    download_name=md_name,
                    layer_payload=layer_payload,
                    defense_state=defense_state,
                    defense_report=defense_report,
                )
            base_persisted = True

            run_narrative = snapshot.get("modifiers", {}).get(
                "purple_team_narrative", defense_narrative
            )
            if defense_report and run_narrative:
                set_phase(status, "Generating purple-team narrative")
                enriched = _run_narrative_phase(
                    keys=keys,
                    report=defense_report,
                    scenario_text=cleaned,
                    defense_state=defense_state,
                    human_name=snapshot_human_name,
                    trace_name=trace_name,
                    on_progress=lambda: set_phase(
                        status, "Generating purple-team narrative"
                    ),
                )
                if enriched:
                    # Replace, rather than duplicate, the already available base
                    # render now that its optional companion is complete.
                    result_placeholder.empty()
                    with result_placeholder.container():
                        _render_result(
                            page_id=keys.page_id,
                            cleaned=cleaned,
                            file_name=md_name,
                            layer_payload=layer_payload,
                            defense_state=enriched,
                            variant="current_enriched",
                        )

            # A missing optional phase is a degraded success, not a failed run:
            # the status closes complete and the notice explains the gap.
            if st.session_state.get(keys.status):
                set_phase(
                    status, "Complete without purple-team narrative", state="complete"
                )
            else:
                set_phase(status, "Complete", state="complete")
    except Exception as e:
        if _is_script_control(e):
            raise
        st.error(f"An error occurred while generating the scenario: {e}")
        # Past the phase boundary everything is optional enrichment, so blame
        # that phase — never tell the user to regenerate a base scenario that
        # is already sitting on the page, intact.
        st.session_state[keys.status] = _failure(
            NARRATIVE_PHASE if base_persisted else BASE_PHASE, "error", str(e)
        )

    if (
        not scenario_text
        and st.session_state.get(keys.generated)
        and st.session_state.get(keys.text)
    ):
        _render_previous(keys=keys, download_name=human_name)


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


# --- Optional narrative phase ------------------------------------------------


def _request_narrative_stop(stop_key: str) -> None:
    st.session_state[stop_key] = True


def _request_retry(request_key: str) -> None:
    st.session_state[request_key] = True


def _settle_interrupted_narrative(keys: _Keys) -> None:
    """Turn a narrative left in flight by a torn-down run into a status record.

    Streamlit aborts the running script as soon as a rerun is queued — which is
    exactly what pressing Skip (or refreshing, or stopping the run) does while
    the narrative streams. The base scenario is already persisted at that
    point, but so is the in-progress marker, so on the next run we can tell the
    optional phase never finished and offer a retry instead of silently
    dropping it.
    """
    if not st.session_state.pop(keys.narrative_running, False):
        st.session_state.pop(keys.narrative_stop, None)
        return
    stopped = bool(st.session_state.pop(keys.narrative_stop, False))
    st.session_state[keys.status] = _failure(
        NARRATIVE_PHASE, "stopped" if stopped else "interrupted"
    )


def _render_skip_control(keys: _Keys) -> None:
    """Offer a way out of a slow or stalled optional phase.

    Clicking this queues a rerun, which tears down the in-flight narrative
    stream; ``_settle_interrupted_narrative`` then reports it as skipped on the
    next run. The base scenario is already persisted, so the rerun re-renders
    it with its downloads intact.
    """
    st.button(
        "Skip purple-team narrative",
        key=f"{keys.page_id}_skip_narrative",
        on_click=_request_narrative_stop,
        args=(keys.narrative_stop,),
        help=(
            "Stop waiting for the optional narrative. The scenario and its "
            "downloads are already saved, and you can retry the narrative later."
        ),
    )


def _run_narrative_phase(
    *,
    keys: _Keys,
    report: dict,
    scenario_text: str,
    defense_state: dict | None,
    human_name: str,
    trace_name: str,
    on_progress: Callable[[], None] | None = None,
) -> dict | None:
    """Run the optional narrative and merge it into the persisted companion.

    Returns the enriched Detection & Response state on success. On failure —
    error, skip, or no output — it returns ``None``, leaves the persisted base
    result and deterministic companion exactly as they were, and records the
    degraded status that drives the notice and its Retry action.
    """
    st.session_state.pop(keys.narrative_stop, None)
    # Flag the phase in flight *before* the call, so a run torn down mid-stream
    # is recognisable on the next one.
    st.session_state[keys.narrative_running] = True
    _render_skip_control(keys)
    # Deliberately not in a `finally`: if the run is torn down here the marker
    # must survive, so the next run knows this phase never finished.
    narrative_md, failure = _stream_defense_narrative(
        report=report,
        scenario_text=scenario_text,
        trace_name=trace_name,
        stop_key=keys.narrative_stop,
        on_chunk=on_progress,
    )
    st.session_state.pop(keys.narrative_running, None)

    if failure:
        st.session_state[keys.status] = failure
        return None

    enriched = _add_narrative_to_defense_state(
        defense_state=defense_state,
        narrative_md=narrative_md,
        human_name=human_name,
    )
    st.session_state[keys.defense] = enriched
    st.session_state["last_defense_narrative"] = narrative_md
    st.session_state.pop(keys.status, None)
    return enriched


def _stream_defense_narrative(
    *,
    report: dict,
    scenario_text: str,
    trace_name: str,
    stop_key: str,
    on_chunk: Callable[[], None] | None = None,
) -> tuple[str | None, Status | None]:
    """Stream the optional purple-team narrative pass.

    Returns ``(narrative_md, failure)`` with exactly one side set: the cleaned
    narrative, or a status record saying why there isn't one. Errors are
    contained here — the caller's base result must survive them.
    """
    placeholder = st.empty()
    chunks: list[str] = []

    def _tee(gen):
        for chunk in gen:
            # A skip requested mid-stream ends the phase here and discards the
            # partial narrative. In a browser this rarely fires (the rerun tears
            # the run down first), but it also stops a stalled stream cleanly.
            if st.session_state.get(stop_key):
                return
            chunks.append(chunk)
            if on_chunk:
                on_chunk()
            yield chunk

    try:
        config = LLMConfig.from_session_state(
            trace_name=f"{trace_name} — Detection & Response",
            trace_tags=("purple_team_narrative",),
        )
        messages = build_narrative_messages(scenario_text, report)
        # Reasoning models can spend minutes on this second call before emitting
        # any text; the elapsed label only advances as chunks arrive, so it looks
        # stalled during that wait. Say so, so a still spinner reads as "working".
        st.write(
            "Walking the scenario from the defender's side. Reasoning models can "
            "take several minutes here and may show no progress until the first "
            "text arrives."
        )
        with placeholder.container():
            st.write_stream(
                stream_filter_thinking(_tee(call_llm_stream(config, messages)))
            )
    except Exception as e:
        if _is_script_control(e):
            raise
        st.error(f"An error occurred while generating the purple-team narrative: {e}")
        return None, _failure(NARRATIVE_PHASE, "error", str(e))

    # Replace the live stream with the canonical cleaned render (done by the
    # caller via _render_defense_body), mirroring the main scenario's handling.
    placeholder.empty()
    if st.session_state.pop(stop_key, False):
        return None, _failure(NARRATIVE_PHASE, "stopped")
    _, cleaned = clean_model_response("".join(chunks))
    if not cleaned:
        return None, _failure(
            NARRATIVE_PHASE, "error", "the model returned no narrative"
        )
    return cleaned, None


def _retry_narrative(*, keys: _Keys, trace_name: str, download_name: str) -> bool:
    """Re-run only the narrative against the already-persisted base scenario.

    Nothing about the base is regenerated — no second base-scenario model call,
    no resampled techniques, no new filenames. Returns whether this rendered
    the (now enriched) result, so the caller doesn't render it twice.
    """
    report = st.session_state.get(keys.defense_report)
    scenario_text = st.session_state.get(keys.text)
    defense_state = st.session_state.get(keys.defense)
    if not (report and scenario_text and defense_state):
        # Nothing to enrich — e.g. state was cleared since the notice rendered.
        st.session_state.pop(keys.status, None)
        return False

    snapshot = st.session_state.get(keys.snapshot) or {}
    human_name = snapshot.get("identity", {}).get("download_name") or download_name
    started = time.monotonic()
    phase = "Retrying purple-team narrative"
    try:
        with st.status(_elapsed_label(phase, started), expanded=True) as status:
            enriched = _run_narrative_phase(
                keys=keys,
                report=report,
                scenario_text=scenario_text,
                defense_state=defense_state,
                human_name=human_name,
                trace_name=trace_name,
                on_progress=lambda: status.update(label=_elapsed_label(phase, started)),
            )
            status.update(
                label=_elapsed_label(
                    "Complete" if enriched else "Purple-team narrative still unavailable",
                    started,
                ),
                state="complete",
            )

        if not enriched:
            return False

        _render_result(
            page_id=keys.page_id,
            cleaned=scenario_text,
            file_name=st.session_state.get(keys.filename) or download_name,
            layer_payload=st.session_state.get(keys.layer),
            defense_state=enriched,
            variant="retry_enriched",
        )
        return True
    except Exception as e:
        if _is_script_control(e):
            raise
        # A retry of optional enrichment must never cost the base result: fall
        # through to the caller's re-render of the persisted scenario.
        st.error(f"An error occurred while retrying the purple-team narrative: {e}")
        st.session_state[keys.status] = _failure(NARRATIVE_PHASE, "error", str(e))
        return False


def _render_recovery_notice(keys: _Keys, *, slot) -> None:
    """Explain the current phase failure and offer its focused retry.

    A missing optional narrative reads as a degraded success (warning, base
    intact); a base-phase failure reads as an error against that phase. Both
    retries go through ``on_click`` so the request is handled at the top of the
    next run, wherever this notice sits on the page.
    """
    status = st.session_state.get(keys.status)
    if not status:
        return
    with slot.container():
        if status.get("phase") == NARRATIVE_PHASE:
            st.warning(_degraded_message(status))
            st.button(
                "Retry purple-team narrative",
                key=f"{keys.page_id}_retry_narrative",
                on_click=_request_retry,
                args=(keys.retry_narrative,),
                help=(
                    "Re-runs only the narrative against the scenario above — "
                    "the scenario itself is not generated again."
                ),
            )
        else:
            st.error(_base_failure_message(status))
            st.button(
                "Retry base scenario",
                key=f"{keys.page_id}_retry_base",
                on_click=_request_retry,
                args=(keys.retry_base,),
                help=(
                    "Re-runs the scenario with the inputs captured when you "
                    "pressed Generate."
                ),
            )


# --- Rendering ---------------------------------------------------------------


def _persist_and_render(
    *,
    cleaned: str,
    keys: _Keys,
    download_name: str,
    layer_payload: tuple[str, str] | None,
    defense_state: dict | None,
    defense_report: dict | None,
) -> None:
    st.session_state[keys.generated] = True
    st.session_state[keys.text] = cleaned
    st.session_state[keys.layer] = layer_payload
    st.session_state[keys.filename] = download_name
    st.session_state[keys.defense] = defense_state
    # The structured report is kept so a narrative retry can rebuild its prompt
    # from the same data, without re-deriving it from (possibly changed) widgets.
    st.session_state[keys.defense_report] = defense_report
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
        page_id=keys.page_id,
        cleaned=cleaned,
        file_name=download_name,
        layer_payload=layer_payload,
        defense_state=defense_state,
        variant="current",
    )


def _render_previous(*, keys: _Keys, download_name: str) -> None:
    text = st.session_state.get(keys.text, "")
    # Prefer the name fixed at generation time so it stays stable (and matches
    # the layer) across the reruns a download click triggers.
    file_name = st.session_state.get(keys.filename) or download_name
    st.markdown("Displaying previously generated scenario:")
    _render_result(
        page_id=keys.page_id,
        cleaned=text,
        file_name=file_name,
        layer_payload=st.session_state.get(keys.layer),
        defense_state=st.session_state.get(keys.defense),
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
