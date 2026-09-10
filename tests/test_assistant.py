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
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_FLAG_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
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
        fake_session_state["llm_model_name"] = "gpt-5.6-sol"
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
        fake_session_state["llm_model_name"] = "gpt-5.6-sol"

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
