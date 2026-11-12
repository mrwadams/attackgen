"""Page-level tests for the AttackGen Assistant page, driven by Streamlit's AppTest.

Each editing target is driven end to end: its panels, chat, apply and
downloads. The model calls are replaced, so no assertion depends on wording a
model would choose.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core.assistant import (
    DEFENSE_NARRATIVE_KEY,
    SCENARIO_FLAG_KEY,
    SCENARIO_META_KEY,
    SCENARIO_TEXT_KEY,
)
from core.routes import ASSISTANT_PAGE
from core.schemas import MODEL_KEY, PROVIDER_KEY, ResultKeys
from core.sidebar import COMPANY_SIZES

_PAGE = str(Path(__file__).resolve().parent.parent / ASSISTANT_PAGE.path)
_KEYS = ResultKeys("threat_group")

_SCENARIO = "# APT29 Scenario\n\nBody."
_NARRATIVE = "## Detection walkthrough\n\nWatch the logs."
_SCENARIO_FILE = "AttackGen_APT29_Enterprise_20260908-093000.md"
_DEFENSE_FILE = "AttackGen_APT29_Enterprise_20260908-093000_detection.md"


def _page(monkeypatch: pytest.MonkeyPatch, *, narrative: bool = True) -> AppTest:
    """The Assistant page after a threat-group generation, with setup complete."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # AppTest runs this page alone, so links to its sibling pages cannot resolve.
    monkeypatch.setattr(st, "page_link", lambda *_a, **_k: None)
    at = AppTest.from_file(_PAGE, default_timeout=60)
    at.session_state[PROVIDER_KEY] = "OpenAI API"
    at.session_state[MODEL_KEY] = "gpt-5.5"
    at.session_state["matrix"] = "Enterprise"
    at.session_state["industry"] = "Finance / Banking"
    at.session_state["company_size"] = COMPANY_SIZES[1]

    at.session_state[SCENARIO_FLAG_KEY] = True
    at.session_state[SCENARIO_TEXT_KEY] = _SCENARIO
    at.session_state[DEFENSE_NARRATIVE_KEY] = _NARRATIVE if narrative else None
    at.session_state[SCENARIO_META_KEY] = {
        "page_id": "threat_group",
        "filename": _SCENARIO_FILE,
        "generated_at": "2026-09-08T09:30:00+00:00",
    }
    at.session_state[_KEYS.text] = _SCENARIO
    at.session_state[_KEYS.filename] = _SCENARIO_FILE
    if narrative:
        at.session_state[_KEYS.defense] = {
            "filename": _DEFENSE_FILE,
            "narrative_md": _NARRATIVE,
            "deterministic_md": "## Reference\n\nT1566",
            "download_md": "# Detection & Response — APT29\n\n## Reference\n\nT1566",
        }
    at.run()
    assert not at.exception
    return at


def _pick(at: AppTest, label: str) -> None:
    next(r for r in at.radio if r.label == "Editing:").set_value(label).run()
    assert not at.exception


def _download_labels(at: AppTest) -> list[str]:
    return [d.proto.label for d in at.get("download_button")]


def _chat_text(at: AppTest) -> str:
    return "\n".join(m.value for msg in at.chat_message for m in msg.markdown)


class TargetCase(NamedTuple):
    label: str
    panels: list[str]
    downloads: list[str]
    prompt: str  # a phrase of the target's system prompt
    note: str  # the chat note after a successful apply
    text_after_apply: str
    narrative_after_apply: str


TARGET_CASES = [
    TargetCase(
        label="Scenario",
        panels=["Generated Scenario"],
        downloads=["Download Scenario"],
        prompt="helps users update and ask questions about their incident response scenario",
        note="✅ Applied changes to the scenario.",
        text_after_apply="# Revised scenario",
        narrative_after_apply=_NARRATIVE,
    ),
    TargetCase(
        label="Detection & Response",
        panels=["Detection & Response Narrative"],
        downloads=["Download Detection & Response"],
        prompt="helps users refine the purple-team Detection & Response narrative",
        note="✅ Applied changes to the Detection & Response narrative.",
        text_after_apply=_SCENARIO,
        narrative_after_apply="## Revised narrative",
    ),
    TargetCase(
        label="Scenario + Detection & Response",
        panels=["Generated Scenario", "Detection & Response Narrative"],
        downloads=["Download Scenario", "Download Detection & Response"],
        prompt="helps users refine an incident response scenario and its accompanying",
        note="✅ Applied changes to the scenario and the Detection & Response narrative.",
        text_after_apply="# Revised scenario",
        narrative_after_apply="## Revised narrative",
    ),
]


@pytest.mark.parametrize("case", TARGET_CASES, ids=lambda c: c.label)
def test_each_target_shows_its_panels_and_downloads(monkeypatch, case):
    at = _page(monkeypatch)
    _pick(at, case.label)

    assert [e.label for e in at.expander] == case.panels
    assert _download_labels(at) == case.downloads


@pytest.mark.parametrize("case", TARGET_CASES, ids=lambda c: c.label)
def test_each_target_chats_then_applies_and_stashes_the_original(monkeypatch, case):
    sent: list[list[dict[str, str]]] = []

    def _stream(_config, messages):
        sent.append(messages)
        yield "Sure, shortened."

    def _call_llm(_config, messages):
        content = messages[1]["content"]
        if content.startswith("Here is the current incident response scenario"):
            return "# Revised scenario"
        return "## Revised narrative"

    monkeypatch.setattr("core.assistant.call_llm_stream", _stream)
    monkeypatch.setattr("core.assistant.call_llm", _call_llm)

    at = _page(monkeypatch)
    _pick(at, case.label)
    at.chat_input[0].set_value("Shorten it.").run()

    assert not at.exception
    assert case.prompt in sent[0][0]["content"]
    assert "Shorten it." in _chat_text(at)
    assert "Sure, shortened." in _chat_text(at)

    next(b for b in at.button if b.label == "Apply changes to downloads").click().run()

    assert not at.exception
    assert case.note in _chat_text(at)
    # The stash `revert_to_original` restores holds what the page generated.
    assert at.session_state[_KEYS.pre_apply] == {"text": _SCENARIO, "narrative": _NARRATIVE}
    assert at.session_state[_KEYS.text] == case.text_after_apply
    assert at.session_state[_KEYS.defense]["narrative_md"] == case.narrative_after_apply


def test_without_a_narrative_only_the_scenario_is_offered(monkeypatch):
    at = _page(monkeypatch, narrative=False)

    assert not [r for r in at.radio if r.label == "Editing:"]
    assert [e.label for e in at.expander] == ["Generated Scenario"]
    assert _download_labels(at) == ["Download Scenario"]


def test_each_target_keeps_its_own_conversation(monkeypatch):
    monkeypatch.setattr("core.assistant.call_llm_stream", lambda _c, _m: iter(["Done."]))
    at = _page(monkeypatch)
    at.chat_input[0].set_value("Scenario turn.").run()

    _pick(at, "Detection & Response")

    assert "Scenario turn." not in _chat_text(at)
    _pick(at, "Scenario")
    assert "Scenario turn." in _chat_text(at)


_TARGET_NAMES = {"scenario", "defense", "both"}


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


@pytest.mark.parametrize("path", [Path(_PAGE), Path(_PAGE).parent.parent / "core" / "assistant.py"])
def test_no_bare_target_strings(path):
    assert not _TARGET_NAMES & set(_string_constants(path))


def test_page_defines_no_state_or_widget_keys():
    tree = ast.parse(Path(_PAGE).read_text(encoding="utf-8"))
    widget_keys = [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "key" and isinstance(keyword.value, (ast.Constant, ast.JoinedStr))
    ]
    state_subscripts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "session_state"
        and isinstance(node.slice, (ast.Constant, ast.JoinedStr))
    ]
    state_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "session_state"
        and node.args
        and isinstance(node.args[0], (ast.Constant, ast.JoinedStr))
    ]
    assert not widget_keys
    assert not state_subscripts
    assert not state_calls
