"""Deepened entry-point for the three scenario-generating pages.

Each scenario page owns its own widgets, prompt assembly and its own readiness
*requirements*. The shared control flow — readiness summary, generate button,
LLM call, response cleaning, result summary, downloads, result actions,
cross-page handoff and feedback widget — lives here. Page-specific behaviour
comes in via the ``build_messages`` callback and the ``requirements``/``setup``
readiness data; identity (session state keys, widget keys) comes in via
``page_id``.

Adding a new scenario page is now: write the widgets and prompt builder, then
``run_scenario_page(page_id=..., build_messages=..., requirements=...,
setup=...)``.

Readiness is data, not a side effect: a page lists what it still needs, the
coordinator merges that with the shared Setup blockers, shows them all in one
summary and disables Generate until none remain — so a click is never spent
discovering a predictable validation error, and no model call can start from an
incomplete form.

A persisted result belongs to its page for the lifetime of the session. Its
keys are not widget keys, so navigating to the Assistant and back re-renders
the same scenario, the same captured inputs and the same stable filenames; the
result also stays put when the form is edited, and only an explicit Regenerate
or Clear replaces or removes it.

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

import contextlib
import copy
import inspect
import json
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from core.assistant import (
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_FLAG_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
    render_assistant_link,
)
from core.detections import (
    assemble_defense_document,
    build_narrative_messages,
    defense_download_name,
    defense_to_markdown,
)
from core.feedback import render_feedback_widget
from core.llm import call_llm_stream
from core.navigator import layer_filename, navigator_for_domain
from core.readiness import Readiness, Requirements, resolve_readiness
from core.response import clean_model_response, stream_filter_thinking
from core.schemas import LLMConfig
from core.state import setup_was_restored_from_link
from core.summary import (
    describe_inputs,
    inputs_changed,
    summarise_scenario,
    summary_line,
)

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

    def result_keys(self) -> tuple[str, ...]:
        """Every key holding this page's latest result (what Clear removes)."""
        return (
            self.generated,
            self.text,
            self.layer,
            self.filename,
            self.defense,
            self.defense_report,
            self.snapshot,
            self.status,
            self.run_id,
        )


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


class _PhaseAborted(Exception):
    """A phase gave up after recording why, with nothing left to render.

    Raised instead of returning so the caller's tail still runs: a run that
    produces no scenario of its own must put the page's persisted result back
    on screen, not leave the page blank while the session still holds one.
    """


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


def _detail_phrase(detail: str) -> str:
    """Trim a failure detail so the sentence around it ends in one full stop.

    Client errors are quoted verbatim into these notices, and they routinely
    arrive already punctuated — litellm's connection errors end in ".", giving
    "Connection error.." once the template adds its own. Strip whatever
    sentence-ending punctuation the detail brought and let the template supply it.
    """
    return detail.rstrip().rstrip(".!?") or detail


def _degraded_message(status: Status) -> str:
    """Explain a missing optional phase without implying the run failed."""
    template = _NARRATIVE_REASONS.get(
        status.get("reason", ""), _NARRATIVE_REASONS["interrupted"]
    )
    detail = _detail_phrase(status.get("detail", "")) or "no output was returned"
    reason = template.format(detail=detail)
    return f"{reason} {_BASE_KEPT}"


def _base_failure_message(status: Status) -> str:
    detail = _detail_phrase(status.get("detail", "")) or "the model returned nothing"
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
    is_ready: Callable[[], bool] | None = None,
    download_name: str,
    trace_name: str,
    trace_tags: tuple[str, ...],
    status_text: str = "Generating scenario...",
    button_label: str = "Generate Scenario",
    render_modifiers: Callable[[], None] | None = None,
    requirements: Requirements = None,
    setup: Any | None = None,
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

    ``render_modifiers`` is an optional callback rendering the page's
    generation modifiers (the AI-enhanced adversary and purple-team toggles).
    It is drawn *above* the Generate button, so a modifier that changes what is
    generated — or adds a second model call — is a decision the user makes
    before starting the request rather than one they notice afterwards.

    ``requirements`` and ``setup`` are the page's readiness inputs, as data:
    ``requirements`` yields the page's own blockers ("Select a threat actor
    group…") and ``setup`` is the shared :class:`core.sidebar.SetupState`.
    Together they drive one visible readiness summary and the Generate button's
    disabled state, so no click is ever spent discovering predictable
    validation. ``is_ready`` remains supported for callers that don't supply
    readiness data.

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

    # Result actions and retries use on_click callbacks, which Streamlit fires
    # before this script runs — so every request is known here, whatever the
    # position of the control that raised it.
    retry_base = bool(st.session_state.pop(keys.retry_base, False))
    retry_narrative = bool(st.session_state.pop(keys.retry_narrative, False))
    regenerate = bool(st.session_state.pop(keys.regenerate, False))
    if st.session_state.pop(keys.clear, False):
        _clear_result(keys)

    # Capture the current form once. The same snapshot describes what *will* be
    # generated (the pre-flight summary), is frozen as the run's inputs when
    # Generate is pressed, and is compared with the shown result's inputs to
    # tell the user when the form has moved on.
    current_inputs = copy.deepcopy(capture_inputs()) if capture_inputs else {}

    readiness = resolve_readiness(requirements=requirements, setup=setup)
    ready = readiness.ready and (is_ready() if is_ready is not None else True)

    if render_modifiers is not None:
        _render_modifier_controls(render_modifiers)

    _render_readiness(readiness, snapshot=current_inputs, ready=ready)

    clicked = st.button(
        button_label,
        key=f"{page_id}_generate",
        disabled=not ready,
        type="primary",
        help=(
            None
            if ready
            else "Complete the requirements listed above to enable generation."
        ),
    )

    # Regenerate is deliberately never disabled: it lives with the result,
    # which outlives the widget state that produced it -- navigating away and
    # back resets a multiselect but keeps the scenario. Say why the click did
    # nothing rather than swallowing it.
    if regenerate and not ready:
        st.warning(
            "Regenerate needs the requirements above to be met. Nothing was "
            "regenerated, and the result below is unchanged."
        )

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
    if (clicked or regenerate) and ready:
        # Freeze every user-controlled value before any slow work starts. Page
        # callbacks receive this snapshot rather than consulting live widgets.
        snapshot = copy.deepcopy(current_inputs)
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

    # Re-render the persisted scenario on a plain rerun — after a download
    # click, after a widget change, and (because none of these keys are widget
    # keys) after navigating away to the Assistant and back. Without this the
    # scenario and its downloads would vanish from a page that still holds them.
    has_result = bool(
        st.session_state.get(keys.generated) and st.session_state.get(keys.text)
    )
    if not rendered and has_result:
        st.markdown("---")
        _render_previous(
            keys=keys, download_name=download_name, current_inputs=current_inputs
        )
    elif not rendered and not has_result:
        _render_no_result_note()

    _render_recovery_notice(keys, slot=notice_slot)

    render_feedback_widget(
        key_prefix=page_id,
        scenario_generated=st.session_state.get(keys.generated, False),
        run_id=st.session_state.get(keys.run_id),
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
    started = _monotonic()
    stream_placeholder = st.empty()
    result_placeholder = st.empty()
    raw_chunks: list[str] = []
    scenario_text: str | None = None
    base_persisted = False

    def set_phase(status, phase: str, *, state: str = "running") -> None:
        status.update(label=_elapsed_label(phase, started), state=state)

    try:
        run_narrative = snapshot.get("modifiers", {}).get(
            "purple_team_narrative", defense_narrative
        )
        with st.status(_elapsed_label("Preparing inputs", started), expanded=True) as status:
            st.write(
                "Generation runs in phases, and each one's elapsed time is shown "
                "here. The base scenario often takes 30–50 seconds; reasoning and "
                "local models routinely take several minutes."
                + (
                    " The purple-team narrative then makes a second model call — "
                    "the scenario, its downloads and the Navigator layer are saved "
                    "before it starts."
                    if run_narrative
                    else ""
                )
            )
            messages = _invoke_with_snapshot(build_messages, snapshot)
            if messages is None:
                status.update(label="No scenario inputs were available.", state="error")
                st.session_state[keys.status] = _failure(
                    BASE_PHASE, "error", "no scenario inputs were available"
                )
                raise _PhaseAborted
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
                st.session_state[keys.status] = _failure(
                    BASE_PHASE, "error", "the model returned no scenario"
                )
                raise _PhaseAborted

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
                            snapshot=snapshot,
                        )

            # A missing optional phase is a degraded success, not a failed run:
            # the status closes complete and the notice explains the gap.
            if st.session_state.get(keys.status):
                set_phase(
                    status, "Complete without purple-team narrative", state="complete"
                )
            else:
                set_phase(status, "Complete", state="complete")
    except _PhaseAborted:
        # The status is already recorded; the tail below restores the previous
        # result, if the page has one.
        pass
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

    # Whether this run rendered anything is what matters, not whether the model
    # returned text: a stream that produced only reasoning tags, or failed after
    # partial output, leaves `scenario_text` set with nothing on the page.
    if (
        not base_persisted
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


def _render_skip_control(keys: _Keys):
    """Offer a way out of a slow or stalled optional phase.

    Clicking this queues a rerun, which tears down the in-flight narrative
    stream; ``_settle_interrupted_narrative`` then reports it as skipped on the
    next run. The base scenario is already persisted, so the rerun re-renders
    it with its downloads intact.

    Returns the placeholder holding the button so the caller can clear it once
    the phase is over — a Skip control left on screen after the narrative has
    settled offers to interrupt something that is no longer running.
    """
    slot = st.empty()
    with slot.container():
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
    return slot


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
    skip_slot = _render_skip_control(keys)
    # Deliberately not in a `finally`: if the run is torn down here the marker
    # must survive, so the next run knows this phase never finished. The Skip
    # control is cleared on the same terms — only once the phase has settled,
    # so a run torn down mid-stream leaves it on screen for the rerun to replace.
    narrative_md, failure = _stream_defense_narrative(
        report=report,
        scenario_text=scenario_text,
        trace_name=trace_name,
        stop_key=keys.narrative_stop,
        on_progress=on_progress,
    )
    st.session_state.pop(keys.narrative_running, None)
    skip_slot.empty()

    if failure:
        st.session_state[keys.status] = failure
        return None

    enriched = _add_narrative_to_defense_state(
        defense_state=defense_state,
        narrative_md=narrative_md,
        human_name=human_name,
    )
    st.session_state[keys.defense] = enriched
    # Only refresh the Assistant's narrative while the handoff still points at
    # this page -- the same ownership rule _persist_and_render and _clear_result
    # use. A narrative retry can land after another page has generated, and must
    # not pair its narrative with that page's scenario.
    meta = st.session_state.get(SCENARIO_META_KEY) or {}
    if meta.get("page_id") == keys.page_id:
        st.session_state[DEFENSE_NARRATIVE_KEY] = narrative_md
    st.session_state.pop(keys.status, None)
    return enriched


def _stream_defense_narrative(
    *,
    report: dict,
    scenario_text: str,
    trace_name: str,
    stop_key: str,
    on_progress: Callable[[], None] | None = None,
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
            yield chunk

    try:
        config = LLMConfig.from_session_state(
            trace_name=f"{trace_name} — Detection & Response",
            trace_tags=("purple_team_narrative",),
        )
        messages = build_narrative_messages(scenario_text, report)
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
    started = _monotonic()
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
            snapshot=snapshot,
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


def _request_flag(request_key: str) -> None:
    """Record a result-action request for the top of the next script run."""
    st.session_state[request_key] = True


def _clear_result(keys: _Keys) -> None:
    """Drop this page's result on an explicit request, leaving Setup intact.

    Only the page's own result namespace (and the Assistant handoff, when it
    still points here) is removed — provider, model, matrix, industry and size
    are shared setup and survive, so starting a new scenario never means
    reconfiguring the provider.
    """
    for key in keys.result_keys():
        st.session_state.pop(key, None)
    st.session_state[keys.generated] = False
    meta = st.session_state.get(SCENARIO_META_KEY) or {}
    if meta.get("page_id") == keys.page_id:
        for key in (
            SCENARIO_FLAG_KEY,
            SCENARIO_TEXT_KEY,
            DEFENSE_NARRATIVE_KEY,
            SCENARIO_META_KEY,
        ):
            st.session_state.pop(key, None)


def _render_modifier_controls(render_modifiers: Callable[[], None]) -> None:
    """Draw the page's generation modifiers above the Generate button."""
    st.markdown("**Generation options**")
    render_modifiers()
    st.caption(
        "The purple-team narrative adds a second model call, so the run takes "
        "longer."
    )


def _render_fact_table(facts: Iterable[tuple[str, str]]) -> None:
    lines = [f"- **{label}:** {value}" for label, value in facts]
    if lines:
        st.markdown("\n".join(lines))


def _render_readiness(
    readiness: Readiness, *, snapshot: Snapshot, ready: bool
) -> None:
    """Show every outstanding requirement, or confirm what is about to run.

    Page requirements come first (see ``core.readiness``) so the missing
    selection the user is looking at isn't reported behind the shared Setup
    fields. When everything is satisfied, the same snapshot that will be frozen
    by Generate is summarised instead — a last chance to catch a wrong matrix or
    organisation profile before any provider usage begins.
    """
    if readiness.blockers:
        st.info(
            "Complete these before generating:\n\n"
            + "\n".join(f"- {blocker}" for blocker in readiness.blockers)
        )
        return
    if not ready:
        return
    line = summary_line(snapshot)
    if not line:
        return
    st.caption(f"Ready to generate — {line}")


def _render_no_result_note() -> None:
    """Answer "where did my scenario go?" after a reload, rather than say nothing.

    The setup that came back is mirrored through the query string, but that is
    our plumbing, not something the user asked for or can see — so this says
    what was lost and how to get it back, not how the restore works.
    """
    if setup_was_restored_from_link():
        st.caption(
            "Scenarios aren't kept when the page reloads — generate again to "
            "recreate one."
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
    # Pin the LangSmith run to *this* result, so rating a scenario after
    # navigating (or after another page has generated) can't submit feedback
    # against a different run.
    st.session_state[keys.run_id] = st.session_state.get("run_id")

    snapshot = st.session_state.get(keys.snapshot) or {}
    # Cross-page handoff for the AttackGen Assistant chat page. The defense
    # narrative rides along so the Assistant can refine it too; set it
    # unconditionally (None when there's no narrative) so a stale one from an
    # earlier generation can't linger after a plain-scenario regen. The metadata
    # is what lets the Assistant name the scenario it is discussing and offer
    # the route back to the page that produced it.
    st.session_state[SCENARIO_FLAG_KEY] = True
    st.session_state[SCENARIO_TEXT_KEY] = cleaned
    st.session_state[DEFENSE_NARRATIVE_KEY] = (
        defense_state.get("narrative_md") if defense_state else None
    )
    st.session_state[SCENARIO_META_KEY] = {
        "page_id": keys.page_id,
        "filename": download_name,
        "generated_at": snapshot.get("captured_at"),
        "snapshot": copy.deepcopy(snapshot),
    }

    _render_result(
        page_id=keys.page_id,
        cleaned=cleaned,
        file_name=download_name,
        layer_payload=layer_payload,
        defense_state=defense_state,
        variant="current",
        snapshot=snapshot,
    )


def _render_previous(
    *, keys: _Keys, download_name: str, current_inputs: Snapshot | None = None
) -> None:
    text = st.session_state.get(keys.text, "")
    # Prefer the name fixed at generation time so it stays stable (and matches
    # the layer) across the reruns a download click triggers.
    file_name = st.session_state.get(keys.filename) or download_name
    snapshot = st.session_state.get(keys.snapshot) or {}
    st.markdown("Displaying previously generated scenario:")
    _render_result(
        page_id=keys.page_id,
        cleaned=text,
        file_name=file_name,
        layer_payload=st.session_state.get(keys.layer),
        defense_state=st.session_state.get(keys.defense),
        variant="previous",
        snapshot=snapshot,
        stale=inputs_changed(snapshot, current_inputs),
    )


def _render_result(
    *,
    page_id: str,
    cleaned: str,
    file_name: str,
    layer_payload: tuple[str, str] | None,
    defense_state: dict | None,
    variant: str,
    snapshot: Snapshot | None = None,
    stale: bool = False,
) -> None:
    """Render the finished scenario, its companion, and the result actions.

    The order is fixed on purpose: what produced this result, a compact summary
    and section index for a long document, the full Markdown (in side-by-side
    tabs when a Detection & Response companion exists, so the reader switches
    rather than scrolls), then one action area holding every next step.
    ``variant`` ("current" / "previous" / …) namespaces the widget keys so a
    generation run and a plain rerun can't collide on a Streamlit widget key.
    """
    _render_result_meta(snapshot, stale=stale)
    _render_summary_surface(cleaned)
    if defense_state:
        scenario_tab, defense_tab = st.tabs(["📄 Scenario", "🛡️ Detection & Response"])
        with scenario_tab:
            st.markdown(cleaned)
        with defense_tab:
            _render_defense_body(defense_state)
    else:
        st.markdown(cleaned)
    _render_result_actions(
        page_id=page_id,
        cleaned=cleaned,
        file_name=file_name,
        layer_payload=layer_payload,
        defense_state=defense_state,
        variant=variant,
    )


def _render_result_meta(snapshot: Snapshot | None, *, stale: bool) -> None:
    """Say what produced this result, and whether the form has moved on."""
    if not snapshot:
        return
    line = summary_line(snapshot)
    if line:
        st.caption(f"Generated from — {line}")
    if stale:
        st.info("Your selections have changed since this scenario was generated.")
    with st.expander("Full inputs"):
        _render_fact_table(describe_inputs(snapshot))


def _render_summary_surface(cleaned: str) -> None:
    """A compact overview plus section navigation for a long scenario.

    The generated Markdown is never truncated or rewritten — this sits above it
    so a facilitator can orient themselves, and jump straight to the parts an
    exercise is actually run from, without scrolling the whole document.
    """
    summary = summarise_scenario(cleaned)
    if not summary.is_useful:
        return
    if summary.overview:
        st.markdown(f"**Summary:** {summary.overview}")
    if summary.sections:
        st.caption(
            f"{len(summary.sections)} sections · ~{summary.word_count:,} words · "
            f"~{summary.reading_minutes} min read"
        )
        with st.expander("🧭 Jump to a section"):
            if summary.quick_links:
                st.markdown(
                    "**Facilitator shortcuts:** "
                    + " · ".join(
                        f"[{label}](#{section.anchor})"
                        for label, section in summary.quick_links
                    )
                )
            st.markdown(
                "\n".join(
                    f"{'  ' * max(0, section.level - 1)}- [{section.title}]"
                    f"(#{section.anchor})"
                    for section in summary.sections
                )
            )


def _render_result_actions(
    *,
    page_id: str,
    cleaned: str,
    file_name: str,
    layer_payload: tuple[str, str] | None,
    defense_state: dict | None,
    variant: str,
) -> None:
    """One consistent home for every next step available on a result.

    Assistant handoff, all three downloads, regenerate and clear live together
    here rather than being scattered between tabs, so the same actions are in
    the same place whether the result was just generated or re-rendered after
    navigating back to the page.
    """
    keys = _Keys(page_id)
    st.markdown("**Next steps**")
    assistant_col, regenerate_col, clear_col = st.columns(3)
    with assistant_col:
        render_assistant_link()
    with regenerate_col:
        st.button(
            "Regenerate",
            key=f"{page_id}_regenerate_{variant}",
            on_click=_request_flag,
            args=(keys.regenerate,),
            help=(
                "Generate again with your current selections. This result stays "
                "on screen until the new one replaces it."
            ),
        )
    with clear_col:
        st.button(
            "Clear result",
            key=f"{page_id}_clear_{variant}",
            on_click=_request_flag,
            args=(keys.clear,),
            help=(
                "Remove this scenario from the session. Your Setup selections are "
                "kept."
            ),
        )

    downloads: list[Callable[[], None]] = [
        lambda: st.download_button(
            label="Download Scenario",
            data=cleaned,
            file_name=file_name,
            mime="text/markdown",
            key=f"{page_id}_download_{variant}",
        )
    ]
    if layer_payload:
        downloads.append(
            lambda: _render_layer_download(
                layer_payload, key=f"{page_id}_download_layer_{variant}"
            )
        )
    if defense_state:
        downloads.append(
            lambda: st.download_button(
                label="Download Detection & Response",
                data=defense_state["download_md"],
                file_name=defense_state["filename"],
                mime="text/markdown",
                key=f"{page_id}_download_defense_{variant}",
            )
        )
    for column, render in zip(st.columns(len(downloads)), downloads):
        with column:
            render()


def _render_defense_body(defense_state: dict) -> None:
    """Render the Detection & Response tab body.

    The optional narrative reads inline (it's the digestible walkthrough); the
    deterministic STIX join sits in an expander as reference — expanded when
    there's no narrative so the tab is never empty. The combined download lives
    in the shared result-action area below.
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
