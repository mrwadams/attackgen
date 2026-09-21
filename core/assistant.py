"""The AttackGen Assistant's state contract, prompts and recovery surfaces.

The chat page itself is thin: it renders what this module resolves. Keeping the
handoff, the prompt assembly and the model seam here means they can be tested
without a Streamlit runtime, and means the Assistant's two navigation
affordances — the empty state that offers the three ways to *create* a
scenario, and the "Back to scenario" route to the page that produced the one
being discussed — are derived from the same registry the scenario coordinator
links with (:mod:`core.routes`).

The handoff itself is written by ``core.scenario_page`` when a base scenario is
persisted, so the Assistant always has the scenario the user was reading, plus
enough metadata to say *which* scenario that is.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import streamlit as st

from core.detections import assemble_defense_document, defense_title
from core.llm import call_llm, call_llm_stream
from core.response import clean_model_response, stream_filter_thinking
from core.routes import ASSISTANT_PAGE, SCENARIO_PAGES, PageInfo, page_info
from core.schemas import LLMConfig
from core.summary import summary_line

# Session-state keys of the cross-page handoff.
SCENARIO_FLAG_KEY = "last_scenario"
SCENARIO_TEXT_KEY = "last_scenario_text"
DEFENSE_NARRATIVE_KEY = "last_defense_narrative"
SCENARIO_META_KEY = "last_scenario_meta"
CLEANED_REPLY_KEY = "_last_assistant_cleaned"


SCENARIO_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users update and ask questions about their incident "
    "response scenario. Only respond to questions or requests relating to the scenario, or "
    "incident response testing in general. Format your responses using proper Markdown syntax "
    "with headers, bullet points, and formatting for readability."
)

DEFENSE_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users refine the purple-team Detection & Response "
    "narrative that accompanies their incident response scenario. The narrative walks the "
    "scenario from the defender's side — detection opportunities, log sources, and response "
    "actions, stage by stage. Only respond to questions or requests relating to the detection "
    "and response of this scenario, or purple-team testing in general. Keep your suggestions "
    "grounded in the scenario provided for reference. Format your responses using proper "
    "Markdown syntax with headers, bullet points, and formatting for readability."
)

BOTH_SYSTEM_PROMPT = (
    "You are an AI assistant that helps users refine an incident response scenario and its "
    "accompanying purple-team Detection & Response narrative together. When a requested change "
    "affects both — a different threat actor, industry, technique, or timeline — apply it "
    "consistently across the two so the attacker's scenario and the defender's walkthrough stay "
    "aligned, and make clear which output each part of your response applies to. Only respond to "
    "questions or requests relating to the scenario, its detection and response, or incident "
    "response testing in general. Format your responses using proper Markdown syntax with "
    "headers, bullet points, and formatting for readability."
)

# The editing modes, keyed by the value the page's radio resolves to.
TARGETS: dict[str, dict[str, Any]] = {
    "scenario": {
        "system_prompt": SCENARIO_SYSTEM_PROMPT,
        "greeting": (
            "Hi, I can help you update and ask questions about your incident response scenario."
        ),
        "trace_name": "AttackGen Assistant",
        "trace_tags": ("assistant",),
    },
    "defense": {
        "system_prompt": DEFENSE_SYSTEM_PROMPT,
        "greeting": (
            "Hi, I can help you refine the Detection & Response narrative for your scenario."
        ),
        "trace_name": "AttackGen Assistant — Detection & Response",
        "trace_tags": ("assistant", "purple_team_narrative"),
    },
    "both": {
        "system_prompt": BOTH_SYSTEM_PROMPT,
        "greeting": (
            "Hi, I can help you refine the scenario and its Detection & Response narrative "
            "together, keeping changes consistent across both."
        ),
        "trace_name": "AttackGen Assistant — Scenario + Detection & Response",
        "trace_tags": ("assistant", "purple_team_narrative"),
    },
}


@dataclass(frozen=True)
class AssistantScenario:
    """The scenario the Assistant is working on, and where it came from."""

    text: str
    defense_narrative: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def origin(self) -> PageInfo | None:
        """The page that generated this scenario, when it is still known."""
        meta = self.meta or {}
        return page_info(str(meta.get("page_id", "")))

    @property
    def title(self) -> str:
        """A short human identity for the result, e.g. ``APT29 · Enterprise ATT&CK``."""
        meta = self.meta or {}
        line = summary_line(meta.get("snapshot"))
        if line:
            return line
        origin = self.origin
        return origin.label if origin else "Generated scenario"


def scenario_handoff(session_state: Any | None = None) -> AssistantScenario | None:
    """Resolve the scenario handed off by the last scenario generation.

    Returns ``None`` when no scenario has been generated in this session — the
    Assistant's empty state, which offers the three ways to create one.
    """
    state = st.session_state if session_state is None else session_state
    text = state.get(SCENARIO_TEXT_KEY) if state.get(SCENARIO_FLAG_KEY) else None
    if not text:
        return None
    return AssistantScenario(
        text=text,
        defense_narrative=state.get(DEFENSE_NARRATIVE_KEY),
        meta=state.get(SCENARIO_META_KEY) or {},
    )


def build_assistant_messages(
    *,
    target: str,
    scenario_text: str,
    defense_narrative: str | None,
    chat_history: str,
    user_input: str,
) -> list[dict[str, str]]:
    """Assemble the system + user messages for one Assistant turn."""
    if target == "both":
        context = (
            f"Here is the incident response scenario:\n\n{scenario_text}\n\n"
            f"Here is the accompanying Detection & Response narrative:\n\n{defense_narrative}\n\n"
            f"The user wants to refine both together. When a requested change affects both, "
            f"apply it consistently across them and show the update to each.\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    elif target == "defense":
        context = (
            f"Here is the scenario, for reference:\n\n{scenario_text}\n\n"
            f"Here is the current Detection & Response narrative the user wants to refine:"
            f"\n\n{defense_narrative}\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    else:
        context = (
            f"Here is the scenario that the user previously generated:\n\n{scenario_text}\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    return [
        {"role": "system", "content": TARGETS[target]["system_prompt"]},
        {"role": "user", "content": context},
    ]


def stream_assistant_reply(
    *,
    target: str,
    scenario: AssistantScenario,
    chat_history: str,
    user_input: str,
) -> Iterator[str]:
    """Stream one Assistant reply, stashing the cleaned text for the history.

    The cleaned (thinking-stripped) reply lands in
    ``st.session_state[CLEANED_REPLY_KEY]`` so the caller can append exactly
    what the user saw to the conversation, rather than the raw stream.
    """
    messages = build_assistant_messages(
        target=target,
        scenario_text=scenario.text,
        defense_narrative=scenario.defense_narrative,
        chat_history=chat_history,
        user_input=user_input,
    )
    config = LLMConfig.from_session_state(
        trace_name=TARGETS[target]["trace_name"],
        trace_tags=TARGETS[target]["trace_tags"],
    )
    raw_chunks: list[str] = []

    def _tee(chunks):
        for chunk in chunks:
            raw_chunks.append(chunk)
            yield chunk

    try:
        yield from stream_filter_thinking(_tee(call_llm_stream(config, messages)))
    except Exception as e:  # noqa: BLE001 - surfaced to the user in the chat
        message = f"An error occurred while calling the model: {e}"
        yield f"\n\n{message}"
        st.session_state[CLEANED_REPLY_KEY] = message
        return

    thinking, cleaned = clean_model_response("".join(raw_chunks))
    if thinking:
        with st.expander("View Model's Reasoning"):
            st.markdown(thinking)
    st.session_state[CLEANED_REPLY_KEY] = cleaned


# --- Apply: rewriting the downloadable artifact from the chat ---------------
#
# The chat stream refines a conversation; "apply" is a separate, dedicated
# model call that takes the whole of that mode's history and returns the
# complete revised artifact as raw Markdown, which is then written back to the
# page state and downloads the scenario coordinator renders. `both` mode makes
# two such calls — one per artifact — each given the other artifact's current
# text so a change that affects both stays aligned across them.

APPLY_TRACE_NAME = "AttackGen Assistant — Apply"
APPLY_TRACE_TAGS = ("assistant", "assistant_apply")

APPLY_SYSTEM_PROMPT = (
    "You are revising a document, given its current text and a chat history of "
    "requested changes. Return the complete, revised document as raw Markdown. "
    "Do not include any conversational framing, preamble, acknowledgement, or "
    "explanation of what changed — reply with only the finished document."
)

ARTIFACT_LABELS = {
    "scenario": "the scenario",
    "defense": "the Detection & Response narrative",
}


@dataclass(frozen=True)
class ApplyCall:
    """One dedicated apply call: which artifact it revises, and its messages."""

    artifact: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class AppliedArtifact:
    """One artifact's guarded, revised text from a completed apply call."""

    artifact: str
    text: str


def build_apply_messages(
    *,
    target: str,
    scenario_text: str,
    defense_narrative: str | None,
    chat_history: str,
) -> list[ApplyCall]:
    """Build the dedicated apply call(s) for one Assistant mode.

    ``scenario``/``defense`` modes need one call; ``both`` needs two, one per
    artifact, each also given the *other* artifact's current text so the pair
    can be kept consistent without either call revising both at once.
    """
    calls: list[ApplyCall] = []

    if target in ("scenario", "both"):
        reference = (
            "\n\nFor reference, here is the current Detection & Response "
            f"narrative (do not include or revise it in your reply):\n\n{defense_narrative}"
            if target == "both" and defense_narrative
            else ""
        )
        user_content = (
            f"Here is the current incident response scenario:\n\n{scenario_text}\n\n"
            f"Here is the chat history of requested refinements:\n{chat_history}"
            f"{reference}\n\n"
            "Return the complete revised scenario as raw Markdown, and nothing else."
        )
        calls.append(
            ApplyCall(
                artifact="scenario",
                messages=[
                    {"role": "system", "content": APPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        )

    if target in ("defense", "both"):
        reference = (
            "\n\nFor reference, here is the current incident response scenario "
            f"(do not include or revise it in your reply):\n\n{scenario_text}"
            if target == "both"
            else ""
        )
        user_content = (
            f"Here is the current Detection & Response narrative:\n\n{defense_narrative}\n\n"
            f"Here is the chat history of requested refinements:\n{chat_history}"
            f"{reference}\n\n"
            "Return the complete revised narrative as raw Markdown, and nothing else."
        )
        calls.append(
            ApplyCall(
                artifact="defense",
                messages=[
                    {"role": "system", "content": APPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        )

    return calls


def guard_applied_artifact(raw_text: str) -> str | None:
    """Refuse an empty or whitespace-only apply return.

    Thinking tags are stripped the same way a chat reply's are; there is no
    length-based check — a short-but-real revision is not an error.
    """
    if not raw_text:
        return None
    _, cleaned = clean_model_response(raw_text)
    cleaned = cleaned.strip()
    return cleaned or None


def apply_write_back(
    *,
    page_id: str,
    artifact: str,
    revised_text: str,
    session_state: Any | None = None,
) -> None:
    """Write one revised artifact back to its page state and the Assistant handoff.

    Mirrors the keys ``core.scenario_page`` persists: ``{page_id}_scenario_text``
    for the scenario; ``{page_id}_scenario_defense`` for the Detection &
    Response companion, whose ``narrative_md`` is replaced and whose
    ``download_md`` is rebuilt from it — the deterministic STIX reference
    (``deterministic_md``) is carried over byte-for-byte, never regenerated.
    The cross-page handoff keys are updated too, so the Assistant's own panels
    and a later chat turn's base move with the artifact.
    """
    state = st.session_state if session_state is None else session_state
    if artifact == "scenario":
        state[f"{page_id}_scenario_text"] = revised_text
        state[SCENARIO_TEXT_KEY] = revised_text
    elif artifact == "defense":
        defense_state = dict(state.get(f"{page_id}_scenario_defense") or {})
        title = defense_title(defense_state.get("download_md", ""))
        deterministic_md = defense_state.get("deterministic_md", "")
        defense_state["narrative_md"] = revised_text
        defense_state["download_md"] = assemble_defense_document(
            deterministic_md, revised_text, title=title
        )
        state[f"{page_id}_scenario_defense"] = defense_state
        state[DEFENSE_NARRATIVE_KEY] = revised_text
    else:
        raise ValueError(f"unknown artifact: {artifact!r}")


def run_apply(
    *, target: str, scenario: AssistantScenario, chat_history: str
) -> list[AppliedArtifact] | None:
    """Make the dedicated apply call(s) for ``target`` and guard each result.

    Returns ``None`` — refusing the whole apply rather than writing back a
    partial, out-of-alignment pair — if any call's return is empty or
    whitespace-only. Raises whatever the model call raises; the caller (the
    Assistant page) decides how to surface that.
    """
    calls = build_apply_messages(
        target=target,
        scenario_text=scenario.text,
        defense_narrative=scenario.defense_narrative,
        chat_history=chat_history,
    )
    config = LLMConfig.from_session_state(
        trace_name=APPLY_TRACE_NAME, trace_tags=APPLY_TRACE_TAGS
    )
    results: list[AppliedArtifact] = []
    for call in calls:
        raw = call_llm(config, call.messages)
        guarded = guard_applied_artifact(raw)
        if guarded is None:
            return None
        results.append(AppliedArtifact(artifact=call.artifact, text=guarded))
    return results


def apply_summary_note(applied: list[AppliedArtifact]) -> str:
    """The chat message marking a successful apply point in the history."""
    labels = [ARTIFACT_LABELS.get(a.artifact, a.artifact) for a in applied]
    return f"✅ Applied changes to {' and '.join(labels)}."


def resolve_download_artifacts(
    scenario: AssistantScenario, session_state: Any | None = None
) -> dict[str, tuple[str, str]]:
    """Filenames + data for the downloads the Assistant page offers.

    Reads the page-scoped keys ``core.scenario_page`` persists for the
    scenario's originating page, so a download offered here is always the same
    file (same name, same content) the scenario page itself would offer.
    Returns only the artifacts that exist — e.g. no ``"defense"`` entry when
    no purple-team narrative was generated.
    """
    state = st.session_state if session_state is None else session_state
    page_id = (scenario.meta or {}).get("page_id")
    artifacts: dict[str, tuple[str, str]] = {}
    if not page_id:
        return artifacts

    filename = state.get(f"{page_id}_scenario_filename")
    text = state.get(f"{page_id}_scenario_text")
    if filename and text:
        artifacts["scenario"] = (filename, text)

    defense_state = state.get(f"{page_id}_scenario_defense")
    if defense_state and defense_state.get("download_md"):
        artifacts["defense"] = (defense_state["filename"], defense_state["download_md"])

    return artifacts


# --- Navigation surfaces -----------------------------------------------------


def render_empty_state() -> None:
    """Offer the three ways to create a scenario instead of a dead end."""
    st.info(
        "No scenario yet. Generate one and it will be available here for "
        "questions and refinement — pick a starting point below."
    )
    for page in SCENARIO_PAGES:
        st.page_link(page.path, label=page.label, icon=page.icon)
        st.caption(page.description)


def render_scenario_identity(scenario: AssistantScenario) -> None:
    """Say which result the chat is using, and offer the way back to it."""
    origin = scenario.origin
    where = f" · generated on {origin.label}" if origin else ""
    st.caption(f"Discussing: **{scenario.title}**{where}")
    if origin:
        st.page_link(
            origin.path,
            label="Back to this scenario",
            icon="⬅️",
            help=(
                f"Return to {origin.label}, where the scenario and its downloads "
                "are still available."
            ),
        )


def render_assistant_link(*, label: str = "Open in Assistant") -> None:
    """Link to the Assistant from a scenario page's result actions."""
    st.page_link(ASSISTANT_PAGE.path, label=label, icon=ASSISTANT_PAGE.icon)
