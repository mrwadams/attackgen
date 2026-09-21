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
    APPLY_TRACE_NAME,
    APPLY_TRACE_TAGS,
    CLEANED_REPLY_KEY,
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_FLAG_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
    AppliedArtifact,
    AssistantScenario,
    apply_summary_note,
    apply_write_back,
    build_apply_messages,
    build_assistant_messages,
    guard_applied_artifact,
    render_empty_state,
    render_scenario_identity,
    resolve_download_artifacts,
    run_apply,
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


# --- Apply: rewriting the downloadable artifact from the chat (issue #58) ----


class TestBuildApplyMessages:
    def test_scenario_mode_makes_one_call_with_no_cross_reference(self) -> None:
        calls = build_apply_messages(
            target="scenario",
            scenario_text="# Scenario",
            defense_narrative="## Narrative",
            chat_history="user: shorten it",
        )

        assert [c.artifact for c in calls] == ["scenario"]
        content = calls[0].messages[1]["content"]
        assert "# Scenario" in content
        assert "shorten it" in content
        # Not in `both` mode: the other artifact isn't pulled in for reference.
        assert "## Narrative" not in content

    def test_defense_mode_makes_one_call_with_no_cross_reference(self) -> None:
        calls = build_apply_messages(
            target="defense",
            scenario_text="# Scenario",
            defense_narrative="## Narrative",
            chat_history="user: add a log source",
        )

        assert [c.artifact for c in calls] == ["defense"]
        content = calls[0].messages[1]["content"]
        assert "## Narrative" in content
        assert "add a log source" in content
        assert "# Scenario" not in content

    def test_both_mode_makes_two_calls_each_given_the_others_current_text(self) -> None:
        calls = build_apply_messages(
            target="both",
            scenario_text="# Scenario",
            defense_narrative="## Narrative",
            chat_history="user: change the industry",
        )

        assert sorted(c.artifact for c in calls) == ["defense", "scenario"]
        scenario_call = next(c for c in calls if c.artifact == "scenario")
        defense_call = next(c for c in calls if c.artifact == "defense")

        # Each call revises its own artifact but is given the other's current
        # text so the pair can be kept aligned.
        assert "# Scenario" in scenario_call.messages[1]["content"]
        assert "## Narrative" in scenario_call.messages[1]["content"]
        assert "# Scenario" in defense_call.messages[1]["content"]
        assert "## Narrative" in defense_call.messages[1]["content"]

    def test_both_mode_omits_the_narrative_reference_when_there_is_none(self) -> None:
        """A `both`-mode call that has no narrative yet must not print "None"."""
        calls = build_apply_messages(
            target="both",
            scenario_text="# Scenario",
            defense_narrative=None,
            chat_history="",
        )

        scenario_call = next(c for c in calls if c.artifact == "scenario")
        assert "None" not in scenario_call.messages[1]["content"]


class TestGuardAppliedArtifact:
    def test_empty_string_is_refused(self) -> None:
        assert guard_applied_artifact("") is None

    def test_whitespace_only_is_refused(self) -> None:
        assert guard_applied_artifact("   \n\t  ") is None

    def test_thinking_only_is_refused(self) -> None:
        assert guard_applied_artifact("<think>reasoning about it</think>") is None

    def test_thinking_is_stripped_from_a_real_return(self) -> None:
        raw = "<think>plan</think>\n# Revised scenario\n\nBody."
        assert guard_applied_artifact(raw) == "# Revised scenario\n\nBody."

    def test_short_return_is_not_refused(self) -> None:
        """No length-based check — a short-but-real revision must pass."""
        assert guard_applied_artifact("Ok.") == "Ok."


class TestApplyWriteBack:
    def test_scenario_artifact_updates_page_state_and_handoff(self) -> None:
        state = {"threat_group_scenario_text": "# Old", "last_scenario_text": "# Old"}

        apply_write_back(
            page_id="threat_group",
            artifact="scenario",
            revised_text="# New scenario",
            session_state=state,
        )

        assert state["threat_group_scenario_text"] == "# New scenario"
        assert state["last_scenario_text"] == "# New scenario"

    def test_defense_artifact_replaces_narrative_and_rebuilds_download(self) -> None:
        deterministic_md = "## 🛡️ Detection & Response\n\nSome reference."
        state = {
            "threat_group_scenario_defense": {
                "deterministic_md": deterministic_md,
                "narrative_md": "## Old walkthrough",
                "download_md": (
                    "# Detection & Response — AttackGen APT29 Enterprise\n\n"
                    "## Old walkthrough\n\n---\n\n## Detection & Response Reference\n\n"
                    + deterministic_md
                ),
                "filename": "scn_detection.md",
            },
            "last_defense_narrative": "## Old walkthrough",
        }

        apply_write_back(
            page_id="threat_group",
            artifact="defense",
            revised_text="## New walkthrough",
            session_state=state,
        )

        defense = state["threat_group_scenario_defense"]
        assert defense["narrative_md"] == "## New walkthrough"
        assert "## New walkthrough" in defense["download_md"]
        assert "## Old walkthrough" not in defense["download_md"]
        # The deterministic STIX reference is carried over byte-for-byte.
        assert deterministic_md in defense["download_md"]
        assert defense["deterministic_md"] == deterministic_md
        # The title recovered from the old document is preserved in the new one.
        assert "AttackGen APT29 Enterprise" in defense["download_md"]
        # Filenames are untouched by an apply.
        assert defense["filename"] == "scn_detection.md"
        assert state["last_defense_narrative"] == "## New walkthrough"

    def test_unknown_artifact_raises(self) -> None:
        with pytest.raises(ValueError):
            apply_write_back(
                page_id="threat_group",
                artifact="nope",
                revised_text="x",
                session_state={},
            )


class TestRunApply:
    def _scenario(self, **overrides) -> AssistantScenario:
        defaults = dict(
            text="# Scenario", defense_narrative="## Narrative", meta={"page_id": "threat_group"}
        )
        defaults.update(overrides)
        return AssistantScenario(**defaults)

    def test_scenario_mode_makes_one_traced_call(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"
        calls: list[Any] = []

        def _call_llm(config, messages):
            calls.append((config, messages))
            return "# Revised scenario"

        monkeypatch.setattr("core.assistant.call_llm", _call_llm)

        result = run_apply(
            target="scenario", scenario=self._scenario(), chat_history="user: shorten it"
        )

        assert result == [AppliedArtifact(artifact="scenario", text="# Revised scenario")]
        assert len(calls) == 1
        config, _messages = calls[0]
        assert config.trace_name == APPLY_TRACE_NAME
        assert config.trace_tags == APPLY_TRACE_TAGS

    def test_both_mode_makes_two_calls_and_returns_both_artifacts(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"

        def _call_llm(_config, messages):
            # `build_apply_messages` always yields scenario before defense.
            content = messages[1]["content"]
            return (
                "# Revised scenario"
                if content.startswith("Here is the current incident response scenario")
                else "## Revised narrative"
            )

        monkeypatch.setattr("core.assistant.call_llm", _call_llm)

        result = run_apply(
            target="both", scenario=self._scenario(), chat_history="user: change the industry"
        )

        assert {a.artifact for a in result} == {"scenario", "defense"}
        texts = {a.artifact: a.text for a in result}
        assert texts["scenario"] == "# Revised scenario"
        assert texts["defense"] == "## Revised narrative"

    def test_any_empty_return_refuses_the_whole_apply(
        self, fake_session_state, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_session_state["chosen_model_provider"] = "OpenAI API"
        fake_session_state["llm_model_name"] = "gpt-5.5"
        seen = []

        def _call_llm(_config, messages):
            seen.append(messages)
            return "   " if len(seen) == 1 else "## Revised narrative"

        monkeypatch.setattr("core.assistant.call_llm", _call_llm)

        result = run_apply(
            target="both", scenario=self._scenario(), chat_history=""
        )

        assert result is None


class TestApplySummaryNote:
    def test_scenario_only(self) -> None:
        note = apply_summary_note([AppliedArtifact(artifact="scenario", text="x")])
        assert note == "✅ Applied changes to the scenario."

    def test_both_artifacts(self) -> None:
        note = apply_summary_note(
            [
                AppliedArtifact(artifact="scenario", text="x"),
                AppliedArtifact(artifact="defense", text="y"),
            ]
        )
        assert note == "✅ Applied changes to the scenario and the Detection & Response narrative."


class TestResolveDownloadArtifacts:
    def test_returns_scenario_and_defense_when_both_persisted(self) -> None:
        state = {
            "threat_group_scenario_filename": "scn_20260908.md",
            "threat_group_scenario_text": "# Scenario",
            "threat_group_scenario_defense": {
                "filename": "scn_20260908_detection.md",
                "download_md": "# Detection & Response — scn\n\n...",
            },
        }
        scenario = AssistantScenario(text="# Scenario", meta={"page_id": "threat_group"})

        artifacts = resolve_download_artifacts(scenario, session_state=state)

        assert artifacts["scenario"] == ("scn_20260908.md", "# Scenario")
        assert artifacts["defense"] == (
            "scn_20260908_detection.md",
            "# Detection & Response — scn\n\n...",
        )

    def test_no_page_id_returns_nothing(self) -> None:
        scenario = AssistantScenario(text="# Scenario", meta={})
        assert resolve_download_artifacts(scenario, session_state={}) == {}

    def test_no_defense_state_omits_it(self) -> None:
        state = {
            "threat_group_scenario_filename": "scn.md",
            "threat_group_scenario_text": "# Scenario",
        }
        scenario = AssistantScenario(text="# Scenario", meta={"page_id": "threat_group"})

        artifacts = resolve_download_artifacts(scenario, session_state=state)

        assert artifacts == {"scenario": ("scn.md", "# Scenario")}
