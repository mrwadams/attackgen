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

from core.llm import call_llm_stream
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
