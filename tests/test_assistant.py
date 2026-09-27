"""Tests for `core.assistant` — the Assistant's handoff, routes and model seam.

The Assistant page is a thin renderer over this module, so these tests assert
the behaviour the page promises: an empty state that offers the three ways to
create a scenario, a visible identity for the scenario being discussed, a route
back to the page that generated it, and that the captured scenario plus the
user's message reach the model seam with the reply retained.

No assertions are made about model wording — only about what is sent and what
is kept.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
import streamlit as st

from core.assistant import (
    CLEANED_REPLY_KEY,
    CONVERSATIONS_KEY,
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_FLAG_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
    TARGETS,
    Conversation,
    build_assistant_messages,
    render_empty_state,
    render_scenario_identity,
    scenario_handoff,
    stream_assistant_reply,
)
from core.routes import SCENARIO_PAGES, THREAT_GROUP_PAGE


def test_every_registered_page_path_exists() -> None:
    """`st.page_link` raises if a path doesn't match a real page file."""
    from pathlib import Path

    from core.routes import ASSISTANT_PAGE

    repo_root = Path(__file__).resolve().parent.parent
    for page in SCENARIO_PAGES + (ASSISTANT_PAGE,):
        assert (repo_root / page.path).is_file(), page.path


@pytest.fixture
def stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record the Streamlit surface `core.assistant` renders into."""
    controls: dict[str, Any] = {"page_links": [], "captions": [], "infos": [], "markdown": []}

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(
        st, "page_link", lambda path, **kwargs: controls["page_links"].append({"path": path, **kwargs})
    )
    monkeypatch.setattr(st, "caption", lambda body, *a, **k: controls["captions"].append(body))
    monkeypatch.setattr(st, "info", lambda body, *a, **k: controls["infos"].append(body))
    monkeypatch.setattr(st, "markdown", lambda body="", *a, **k: controls["markdown"].append(body))
    monkeypatch.setattr(st, "expander", _expander)
    return controls


def _handoff(state: dict[str, Any]) -> None:
    """Write the handoff a threat-group generation leaves behind."""
    state[SCENARIO_FLAG_KEY] = True
    state[SCENARIO_TEXT_KEY] = "# APT29 Scenario\n\nBody."
    state[DEFENSE_NARRATIVE_KEY] = None
    state[SCENARIO_META_KEY] = {
        "page_id": "threat_group",
        "filename": "AttackGen_APT29_Enterprise_20260908-093000.md",
        "generated_at": "2026-09-08T09:30:00+00:00",
        "snapshot": {
            "matrix": "Enterprise",
            "organisation": {"industry": "Finance / Banking", "company_size": "Medium"},
            "selected_entity": {"type": "threat actor group", "name": "APT29"},
            "modifiers": {"purple_team_narrative": True},
        },
    }


class TestHandoff:
    def test_no_scenario_yet(self, fake_session_state) -> None:
        assert scenario_handoff() is None

    def test_scenario_identity_and_origin_survive_the_handoff(
        self, fake_session_state
    ) -> None:
        _handoff(fake_session_state)

        scenario = scenario_handoff()

        assert scenario is not None
        assert scenario.text == "# APT29 Scenario\n\nBody."
        assert scenario.origin == THREAT_GROUP_PAGE
        assert "APT29" in scenario.title
        assert "Enterprise ATT&CK" in scenario.title

    def test_a_cleared_scenario_is_gone(self, fake_session_state) -> None:
        _handoff(fake_session_state)
        fake_session_state[SCENARIO_FLAG_KEY] = False

        assert scenario_handoff() is None


def _second_handoff(state: dict[str, Any], *, page_id: str = "custom") -> None:
    """Write the handoff a later generation leaves behind, replacing the first."""
    state[SCENARIO_TEXT_KEY] = "# Second Scenario\n\nBody."
    state[SCENARIO_META_KEY] = {
        "page_id": page_id,
        "filename": "AttackGen_Custom_Enterprise_20260908-101500.md",
        "generated_at": "2026-09-08T10:15:00+00:00",
        "snapshot": {"matrix": "Enterprise"},
    }


class TestConversation:
    def test_identity_comes_from_the_handoff_metadata(self, fake_session_state) -> None:
        _handoff(fake_session_state)
        first = scenario_handoff().identity
        _second_handoff(fake_session_state, page_id="threat_group")

        assert first is not None
        assert scenario_handoff().identity != first

    def test_a_new_conversation_is_seeded_with_the_target_greeting(
        self, fake_session_state
    ) -> None:
        _handoff(fake_session_state)

        conversation = Conversation(scenario_handoff(), "defense")

        assert conversation.messages == [
            {"role": "assistant", "content": TARGETS["defense"]["greeting"]}
        ]

    def test_a_second_scenario_starts_a_fresh_conversation(self, fake_session_state) -> None:
        _handoff(fake_session_state)
        first = Conversation(scenario_handoff(), "scenario")
        first.append("user", "Make it about ransomware.")
        first.append("assistant", "Done.")

        _second_handoff(fake_session_state)
        second = Conversation(scenario_handoff(), "scenario")

        assert second.messages == [
            {"role": "assistant", "content": TARGETS["scenario"]["greeting"]}
        ]
        assert "ransomware" not in second.chat_history()

    def test_regenerating_on_the_same_page_starts_a_fresh_conversation(
        self, fake_session_state
    ) -> None:
        _handoff(fake_session_state)
        Conversation(scenario_handoff(), "scenario").append("user", "old turn")

        _second_handoff(fake_session_state, page_id="threat_group")

        assert "old turn" not in Conversation(scenario_handoff(), "scenario").chat_history()

    def test_returning_to_the_same_scenario_keeps_its_history(
        self, fake_session_state
    ) -> None:
        _handoff(fake_session_state)
        Conversation(scenario_handoff(), "scenario").append("user", "Add an inject.")

        # A later visit (a rerun, or navigating away and back) resolves the
        # handoff afresh; the result it names is unchanged.
        again = Conversation(scenario_handoff(), "scenario")

        assert again.messages[-1] == {"role": "user", "content": "Add an inject."}

    def test_each_target_keeps_its_own_conversation(self, fake_session_state) -> None:
        _handoff(fake_session_state)
        Conversation(scenario_handoff(), "scenario").append("user", "scenario turn")
        Conversation(scenario_handoff(), "defense").append("user", "defense turn")

        scenario_chat = Conversation(scenario_handoff(), "scenario").chat_history()
        defense_chat = Conversation(scenario_handoff(), "defense").chat_history()
        both_chat = Conversation(scenario_handoff(), "both").chat_history()

        assert "scenario turn" in scenario_chat and "defense turn" not in scenario_chat
        assert "defense turn" in defense_chat and "scenario turn" not in defense_chat
        assert "turn" not in both_chat

    def test_clear_resets_only_the_current_scenario_and_target(
        self, fake_session_state
    ) -> None:
        _handoff(fake_session_state)
        scenario_chat = Conversation(scenario_handoff(), "scenario")
        scenario_chat.append("user", "scenario turn")
        Conversation(scenario_handoff(), "defense").append("user", "defense turn")

        scenario_chat.reset()

        assert Conversation(scenario_handoff(), "scenario").messages == [
            {"role": "assistant", "content": TARGETS["scenario"]["greeting"]}
        ]
        assert "defense turn" in Conversation(scenario_handoff(), "defense").chat_history()

    def test_superseded_conversations_do_not_accumulate(self, fake_session_state) -> None:
        _handoff(fake_session_state)
        for target in TARGETS:
            Conversation(scenario_handoff(), target).append("user", "first scenario")

        _second_handoff(fake_session_state)
        Conversation(scenario_handoff(), "scenario")

        store = fake_session_state[CONVERSATIONS_KEY]
        assert list(store["targets"]) == ["scenario"]
        assert "first scenario" not in repr(store)

    def test_only_the_current_scenario_turns_reach_the_model(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _handoff(fake_session_state)
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"
        fake_session_state["llm_api_key"] = "k"
        stale = Conversation(scenario_handoff(), "scenario")
        stale.append("user", "Question about APT29.")
        stale.append("assistant", "Answer about APT29.")

        _second_handoff(fake_session_state)
        scenario = scenario_handoff()
        conversation = Conversation(scenario, "scenario")
        conversation.append("user", "Earlier question.")
        conversation.append("assistant", "Earlier answer.")
        sent: list[Any] = []

        def _stream(config, messages):
            sent.append(messages)
            yield "Reply."

        monkeypatch.setattr("core.assistant.call_llm_stream", _stream)

        list(
            stream_assistant_reply(
                target="scenario",
                scenario=scenario,
                chat_history=conversation.chat_history(),
                user_input="New question.",
            )
        )

        content = sent[0][1]["content"]
        assert "Earlier question." in content and "Earlier answer." in content
        assert TARGETS["scenario"]["greeting"] in content
        assert "APT29" not in content


class TestNavigationSurfaces:
    def test_empty_state_offers_all_three_generation_entry_points(
        self, fake_session_state, stub_streamlit
    ) -> None:
        render_empty_state()

        linked = [link["path"] for link in stub_streamlit["page_links"]]
        assert linked == [page.path for page in SCENARIO_PAGES]
        assert stub_streamlit["infos"], "the empty state should explain itself"

    def test_identity_names_the_scenario_and_offers_the_way_back(
        self, fake_session_state, stub_streamlit
    ) -> None:
        _handoff(fake_session_state)

        render_scenario_identity(scenario_handoff())

        caption = " ".join(stub_streamlit["captions"])
        assert "APT29" in caption
        assert THREAT_GROUP_PAGE.label in caption
        back = stub_streamlit["page_links"][0]
        assert back["path"] == THREAT_GROUP_PAGE.path
        assert back["label"] == "Back to this scenario"

    def test_identity_without_a_known_origin_still_renders(
        self, fake_session_state, stub_streamlit
    ) -> None:
        _handoff(fake_session_state)
        fake_session_state[SCENARIO_META_KEY]["page_id"] = "gone"

        render_scenario_identity(scenario_handoff())

        assert stub_streamlit["captions"]
        assert stub_streamlit["page_links"] == []


class TestModelSeam:
    def test_scenario_and_user_message_reach_the_model(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _handoff(fake_session_state)
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"
        fake_session_state["llm_api_key"] = "k"
        sent: list[Any] = []

        def _stream(config, messages):
            sent.append((config, messages))
            yield "<think>plan</think>"
            yield "Here is the answer."

        monkeypatch.setattr("core.assistant.call_llm_stream", _stream)

        chunks = list(
            stream_assistant_reply(
                target="scenario",
                scenario=scenario_handoff(),
                chat_history="assistant: greeting",
                user_input="Which inject tests containment?",
            )
        )

        config, messages = sent[0]
        assert config.trace_tags == ("assistant",)
        assert "# APT29 Scenario" in messages[1]["content"]
        assert "Which inject tests containment?" in messages[1]["content"]
        # The user sees the reply, with the model's thinking filtered out...
        assert "Here is the answer." in "".join(chunks)
        # ...and the cleaned reply is what the chat history keeps.
        assert fake_session_state[CLEANED_REPLY_KEY] == "Here is the answer."

    def test_defense_target_carries_the_narrative_for_reference(self) -> None:
        messages = build_assistant_messages(
            target="defense",
            scenario_text="# Scenario",
            defense_narrative="## Defender walkthrough",
            chat_history="",
            user_input="Add a log source.",
        )

        assert "# Scenario" in messages[1]["content"]
        assert "## Defender walkthrough" in messages[1]["content"]
        assert "Add a log source." in messages[1]["content"]

    def test_both_target_carries_scenario_and_narrative(self) -> None:
        messages = build_assistant_messages(
            target="both",
            scenario_text="# Scenario",
            defense_narrative="## Defender walkthrough",
            chat_history="",
            user_input="Change the industry.",
        )

        assert "# Scenario" in messages[1]["content"]
        assert "## Defender walkthrough" in messages[1]["content"]

    def test_model_error_is_reported_in_the_chat_not_raised(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _handoff(fake_session_state)
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"

        def _stream(_config, _messages):
            raise RuntimeError("upstream 503")
            yield  # pragma: no cover - generator marker

        monkeypatch.setattr("core.assistant.call_llm_stream", _stream)

        chunks = list(
            stream_assistant_reply(
                target="scenario",
                scenario=scenario_handoff(),
                chat_history="",
                user_input="hi",
            )
        )

        assert "upstream 503" in "".join(chunks)
        assert "upstream 503" in fake_session_state[CLEANED_REPLY_KEY]
