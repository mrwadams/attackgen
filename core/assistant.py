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
from enum import Enum
from typing import Any

import streamlit as st

from core.detections import assemble_defense_document, defense_title
from core.llm import call_llm, call_llm_stream
from core.response import clean_model_response, stream_filter_thinking
from core.routes import ASSISTANT_PAGE, SCENARIO_PAGES, PageInfo, page_info
from core.schemas import LLMConfig, ResultKeys
from core.summary import summary_line

# Session-state keys of the cross-page handoff.
SCENARIO_FLAG_KEY = "last_scenario"
SCENARIO_TEXT_KEY = "last_scenario_text"
DEFENSE_NARRATIVE_KEY = "last_defense_narrative"
SCENARIO_META_KEY = "last_scenario_meta"
CLEANED_REPLY_KEY = "_last_assistant_cleaned"

# Session-state key of the Assistant's conversations. Owned by
# :class:`Conversation`; nothing else reads or writes it.
CONVERSATIONS_KEY = "assistant_conversations"


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

# Widget keys of the Assistant page.
TARGET_WIDGET_KEY = "assistant_target"
CLEAR_BUTTON_KEY = "clear_button"


class Artifact(Enum):
    """A document the Assistant revises and offers for download."""

    # (panel title, download label, how the apply note names it)
    SCENARIO = ("Generated Scenario", "Download Scenario", "the scenario")
    DEFENSE = (
        "Detection & Response Narrative",
        "Download Detection & Response",
        "the Detection & Response narrative",
    )

    def __init__(self, panel_title: str, download_label: str, noun: str) -> None:
        self.panel_title = panel_title
        self.download_label = download_label
        self.noun = noun

    def text_in(self, scenario: AssistantScenario) -> str | None:
        """This artifact's current text in the handed-off scenario."""
        if self is Artifact.SCENARIO:
            return scenario.text
        return scenario.defense_narrative


class Target(Enum):
    """What the Assistant is editing, as picked on the page's "Editing:" radio.

    Each target owns its radio label, the artifacts it shows and revises, its
    prompt, greeting and trace, and the key of its Apply button.
    """

    # (radio label, artifacts, system prompt, greeting, trace name, trace tags)
    SCENARIO = (
        "Scenario",
        (Artifact.SCENARIO,),
        SCENARIO_SYSTEM_PROMPT,
        "Hi, I can help you update and ask questions about your incident response scenario.",
        "AttackGen Assistant",
        ("assistant",),
    )
    DEFENSE = (
        "Detection & Response",
        (Artifact.DEFENSE,),
        DEFENSE_SYSTEM_PROMPT,
        "Hi, I can help you refine the Detection & Response narrative for your scenario.",
        "AttackGen Assistant — Detection & Response",
        ("assistant", "purple_team_narrative"),
    )
    BOTH = (
        "Scenario + Detection & Response",
        (Artifact.SCENARIO, Artifact.DEFENSE),
        BOTH_SYSTEM_PROMPT,
        "Hi, I can help you refine the scenario and its Detection & Response narrative "
        "together, keeping changes consistent across both.",
        "AttackGen Assistant — Scenario + Detection & Response",
        ("assistant", "purple_team_narrative"),
    )

    def __init__(
        self,
        label: str,
        artifacts: tuple[Artifact, ...],
        system_prompt: str,
        greeting: str,
        trace_name: str,
        trace_tags: tuple[str, ...],
    ) -> None:
        self.label = label
        self.artifacts = artifacts
        self.system_prompt = system_prompt
        self.greeting = greeting
        self.trace_name = trace_name
        self.trace_tags = trace_tags

    @classmethod
    def from_label(cls, label: str) -> Target:
        """The target whose radio label is ``label``."""
        for target in cls:
            if target.label == label:
                return target
        raise ValueError(f"unknown editing target: {label!r}")

    @property
    def apply_key(self) -> str:
        """The widget key of this target's Apply button."""
        return f"assistant_apply_{self.name.lower()}"

    def panels(self, scenario: AssistantScenario) -> list[tuple[str, str | None]]:
        """The (title, text) panels this target shows above the chat."""
        return [(artifact.panel_title, artifact.text_in(scenario)) for artifact in self.artifacts]


@dataclass(frozen=True)
class AssistantScenario:
    """The scenario the Assistant is working on, and where it came from.

    ``meta`` is the handoff metadata the coordinator writes: the originating
    ``page_id``, the ``filename`` and ``generated_at`` of the result, and the
    :class:`core.request.ScenarioRequest` that produced it under ``request``.
    """

    text: str
    defense_narrative: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def origin(self) -> PageInfo | None:
        """The page that generated this scenario, when it is still known."""
        meta = self.meta or {}
        return page_info(str(meta.get("page_id", "")))

    @property
    def identity(self) -> tuple[str, ...] | None:
        """Which generated result this is, from the handoff metadata alone.

        The coordinator stamps every persisted result with the page that
        produced it and when it was generated, so two generations — on
        different pages, or a regenerate on the same one — never share an
        identity. ``None`` when the metadata carries neither.
        """
        meta = self.meta or {}
        parts = tuple(
            str(meta.get(field) or "") for field in ("page_id", "generated_at", "filename")
        )
        return parts if any(parts) else None

    @property
    def title(self) -> str:
        """A short human identity for the result, e.g. ``APT29 · Enterprise ATT&CK``."""
        meta = self.meta or {}
        line = summary_line(meta.get("request"))
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


class Conversation:
    """The Assistant's chat about one scenario, for one editing target.

    A conversation belongs to the scenario it is about as well as the target
    being edited: switching between Scenario, Detection & Response and the
    combined mode keeps a separate conversation per mode, while a newly
    generated scenario starts every mode afresh. Only the current scenario's
    conversations are kept — the first access for a different scenario drops
    the superseded ones, so they don't accumulate for the life of the session.

    Works against any mapping (``st.session_state`` by default), so the rule can
    be tested without a Streamlit runtime.
    """

    def __init__(
        self,
        scenario: AssistantScenario,
        target: Target,
        session_state: Any | None = None,
    ) -> None:
        self.target = target
        self._state = st.session_state if session_state is None else session_state
        store = self._state.get(CONVERSATIONS_KEY)
        if not store or store.get("identity") != scenario.identity:
            store = {"identity": scenario.identity, "targets": {}}
            self._state[CONVERSATIONS_KEY] = store
        targets = store["targets"]
        if target not in targets:
            targets[target] = self._seed()
        self._messages: list[dict[str, str]] = targets[target]

    def _seed(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.target.greeting}]

    @property
    def messages(self) -> list[dict[str, str]]:
        """Every turn so far, starting with the greeting."""
        return list(self._messages)

    def chat_history(self) -> str:
        """The turns so far, formatted as the chat history sent to the model."""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self._messages)

    def append(self, role: str, content: str) -> None:
        """Add one turn to this conversation."""
        self._messages.append({"role": role, "content": content})

    def reset(self) -> None:
        """Start this scenario's conversation for this target over."""
        self._messages[:] = self._seed()


def build_assistant_messages(
    *,
    target: Target,
    scenario_text: str,
    defense_narrative: str | None,
    chat_history: str,
    user_input: str,
) -> list[dict[str, str]]:
    """Assemble the system + user messages for one Assistant turn."""
    if target is Target.BOTH:
        context = (
            f"Here is the incident response scenario:\n\n{scenario_text}\n\n"
            f"Here is the accompanying Detection & Response narrative:\n\n{defense_narrative}\n\n"
            f"The user wants to refine both together. When a requested change affects both, "
            f"apply it consistently across them and show the update to each.\n\n"
            f"Chat history:\n{chat_history}\n\nUser: {user_input}"
        )
    elif target is Target.DEFENSE:
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
        {"role": "system", "content": target.system_prompt},
        {"role": "user", "content": context},
    ]


def stream_assistant_reply(
    *,
    target: Target,
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
        trace_name=target.trace_name,
        trace_tags=target.trace_tags,
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

@dataclass(frozen=True)
class ApplyCall:
    """One dedicated apply call: which artifact it revises, and its messages."""

    artifact: Artifact
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class AppliedArtifact:
    """One artifact's guarded, revised text from a completed apply call."""

    artifact: Artifact
    text: str


def build_apply_messages(
    *,
    target: Target,
    scenario_text: str,
    defense_narrative: str | None,
    chat_history: str,
) -> list[ApplyCall]:
    """Build the dedicated apply call(s) for one Assistant mode.

    The single-artifact targets need one call; ``Target.BOTH`` needs two, one per
    artifact, each also given the *other* artifact's current text so the pair
    can be kept consistent without either call revising both at once.
    """
    calls: list[ApplyCall] = []

    if Artifact.SCENARIO in target.artifacts:
        reference = (
            "\n\nFor reference, here is the current Detection & Response "
            f"narrative (do not include or revise it in your reply):\n\n{defense_narrative}"
            if target is Target.BOTH and defense_narrative
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
                artifact=Artifact.SCENARIO,
                messages=[
                    {"role": "system", "content": APPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
        )

    if Artifact.DEFENSE in target.artifacts:
        reference = (
            "\n\nFor reference, here is the current incident response scenario "
            f"(do not include or revise it in your reply):\n\n{scenario_text}"
            if target is Target.BOTH
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
                artifact=Artifact.DEFENSE,
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
    artifact: Artifact,
    revised_text: str,
    session_state: Any | None = None,
) -> None:
    """Write one revised artifact back to its page state and the Assistant handoff.

    Writes through :class:`core.schemas.ResultKeys`, the keys
    ``core.scenario_page`` persists: ``text`` for the scenario; ``defense`` for
    the Detection & Response companion, whose ``narrative_md`` is replaced and
    whose ``download_md`` is rebuilt from it — the deterministic STIX reference
    (``deterministic_md``) is carried over byte-for-byte, never regenerated.
    The cross-page handoff keys are updated too, so the Assistant's own panels
    and a later chat turn's base move with the artifact.
    """
    state = st.session_state if session_state is None else session_state
    keys = ResultKeys(page_id)
    if artifact is Artifact.SCENARIO:
        state[keys.text] = revised_text
        state[SCENARIO_TEXT_KEY] = revised_text
    elif artifact is Artifact.DEFENSE:
        defense_state = dict(state.get(keys.defense) or {})
        title = defense_title(defense_state.get("download_md", ""))
        deterministic_md = defense_state.get("deterministic_md", "")
        defense_state["narrative_md"] = revised_text
        defense_state["download_md"] = assemble_defense_document(
            deterministic_md, revised_text, title=title
        )
        state[keys.defense] = defense_state
        state[DEFENSE_NARRATIVE_KEY] = revised_text
    else:
        raise ValueError(f"unknown artifact: {artifact!r}")


def run_apply(
    *, target: Target, scenario: AssistantScenario, chat_history: str
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
    labels = [a.artifact.noun for a in applied]
    return f"✅ Applied changes to {' and '.join(labels)}."


def resolve_download_artifacts(
    scenario: AssistantScenario, session_state: Any | None = None
) -> dict[Artifact, tuple[str, str]]:
    """Filenames + data for the downloads the Assistant page offers.

    Reads the page-scoped keys ``core.scenario_page`` persists for the
    scenario's originating page, so a download offered here is always the same
    file (same name, same content) the scenario page itself would offer.
    Returns only the artifacts that exist — e.g. no ``Artifact.DEFENSE`` entry
    when no purple-team narrative was generated.
    """
    state = st.session_state if session_state is None else session_state
    page_id = (scenario.meta or {}).get("page_id")
    artifacts: dict[Artifact, tuple[str, str]] = {}
    if not page_id:
        return artifacts

    keys = ResultKeys(page_id)
    filename = state.get(keys.filename)
    text = state.get(keys.text)
    if filename and text:
        artifacts[Artifact.SCENARIO] = (filename, text)

    defense_state = state.get(keys.defense)
    if defense_state and defense_state.get("download_md"):
        artifacts[Artifact.DEFENSE] = (defense_state["filename"], defense_state["download_md"])

    return artifacts


@dataclass(frozen=True)
class AssistantDownload:
    """One download button the Assistant page offers."""

    label: str
    data: str
    filename: str

    @property
    def key(self) -> str:
        """The widget key of this download button."""
        return f"assistant_download_{self.filename}"


def assistant_downloads(
    scenario: AssistantScenario, target: Target, session_state: Any | None = None
) -> list[AssistantDownload]:
    """The downloads for ``target``'s artifacts that exist, in its order."""
    artifacts = resolve_download_artifacts(scenario, session_state)
    downloads = []
    for artifact in target.artifacts:
        if artifact in artifacts:
            filename, data = artifacts[artifact]
            downloads.append(AssistantDownload(artifact.download_label, data, filename))
    return downloads


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


def choose_target(scenario: AssistantScenario) -> Target:
    """Render the "Editing:" choice and return the target picked.

    The Detection & Response and combined targets only exist when a
    purple-team narrative was generated alongside the scenario (page 1/2
    toggle). The combined target refines both together so a change made to one
    can be carried consistently into the other.
    """
    if not scenario.defense_narrative:
        return Target.SCENARIO
    label = st.radio(
        "Editing:",
        [target.label for target in Target],
        horizontal=True,
        key=TARGET_WIDGET_KEY,
    )
    return Target.from_label(label)


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
