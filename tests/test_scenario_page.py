"""Tests for `core.scenario_page.run_scenario_page`.

The interface is the test surface: given a build_messages callback and a
readiness predicate, we assert what reaches `call_llm_stream` and what lands
in session_state. Streamlit's UI calls are stubbed to no-ops; we don't render
anything — we only care about the control flow at the seam.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
import streamlit as st

import core.llm as llm_module
from core.scenario_page import run_scenario_page


@pytest.fixture
def stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """No-op out the Streamlit UI surface that `run_scenario_page` touches.

    Returns a dict the test can mutate to control widget return values:
      - `button_returns`: bool returned by `st.button`
    """
    controls: dict[str, Any] = {"button_returns": False}

    def _button(*_args, **_kwargs):
        return controls["button_returns"]

    @contextmanager
    def _status(*_args, **_kwargs):
        yield None

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    def _noop(*_args, **_kwargs):
        return None

    def _write_stream(stream):
        # Drain the generator so the production code's `_tee` captures chunks.
        for _ in stream:
            pass

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "status", _status)
    monkeypatch.setattr(st, "expander", _expander)
    monkeypatch.setattr(st, "markdown", _noop)
    monkeypatch.setattr(st, "write", _noop)
    monkeypatch.setattr(st, "write_stream", _write_stream)
    monkeypatch.setattr(st, "download_button", _noop)
    monkeypatch.setattr(st, "info", _noop)
    monkeypatch.setattr(st, "warning", _noop)
    monkeypatch.setattr(st, "error", _noop)
    # `render_feedback_widget` calls `st.empty()` then `st.markdown('---')`.
    monkeypatch.setattr(st, "empty", lambda: _FakePlaceholder())
    # `st.secrets` membership tests — pretend no LangSmith key is configured
    # so the feedback widget short-circuits cleanly during tests.
    monkeypatch.setattr(st, "secrets", {})

    return controls


class _FakePlaceholder:
    def success(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def error(self, *_a, **_k): pass
    def empty(self, *_a, **_k): pass

    @contextmanager
    def container(self, *_a, **_k):
        yield None


@pytest.fixture
def disable_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force call_llm_stream to hit `_raw_stream` directly so litellm sees the messages."""
    monkeypatch.setattr(llm_module, "_langsmith_client", None)


def test_does_nothing_when_button_not_pressed(
    stub_streamlit, fake_session_state, mock_litellm_completion
) -> None:
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert mock_litellm_completion.calls == []
    assert "threat_group_scenario_generated" in fake_session_state
    assert fake_session_state["threat_group_scenario_generated"] is False


def test_skips_llm_when_not_ready(
    stub_streamlit, fake_session_state, mock_litellm_completion
) -> None:
    stub_streamlit["button_returns"] = True
    build_calls: list[None] = []

    def build():
        build_calls.append(None)
        return [{"role": "user", "content": "x"}]

    run_scenario_page(
        page_id="threat_group",
        build_messages=build,
        is_ready=lambda: False,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert mock_litellm_completion.calls == []
    assert build_calls == []
    assert fake_session_state["threat_group_scenario_generated"] is False


def test_happy_path_calls_llm_cleans_response_and_persists(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
) -> None:
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "<think>plan</think>\n# Scenario\n\nBody."

    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    messages = [{"role": "user", "content": "build me a scenario"}]

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: messages,
        is_ready=lambda: True,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    # The seam: call_llm_stream got the page's messages.
    assert len(mock_litellm_completion.calls) == 1
    _args, kwargs = mock_litellm_completion.calls[0]
    assert kwargs["messages"] == messages
    assert kwargs["model"] == "gpt-5.5"

    # The cleaned response — not the raw one — is what gets persisted.
    cleaned = fake_session_state["threat_group_scenario_text"]
    assert "<think>" not in cleaned
    assert cleaned.startswith("# Scenario")

    # Cross-page handoff for the Assistant page.
    assert fake_session_state["last_scenario"] is True
    assert fake_session_state["last_scenario_text"] == cleaned

    # The artifact flag is set.
    assert fake_session_state["threat_group_scenario_generated"] is True


def test_page_id_namespaces_session_state(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
) -> None:
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "scenario A"

    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    run_scenario_page(
        page_id="custom",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="custom_scenario.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
    )

    assert "custom_scenario_text" in fake_session_state
    assert "custom_scenario_generated" in fake_session_state
    # The "threat_group_*" namespace is untouched by a "custom" page invocation.
    assert "threat_group_scenario_text" not in fake_session_state


def test_trace_name_and_tags_reach_llm_config(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True

    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    captured: dict[str, Any] = {}

    def _fake_call_llm_stream(config, msgs):
        captured["config"] = config
        captured["messages"] = msgs
        yield "ok"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _fake_call_llm_stream)

    run_scenario_page(
        page_id="ai_insider",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="ai_insider_threat_scenario.md",
        trace_name="AI Insider Threat Scenario",
        trace_tags=("ai_insider_scenario",),
    )

    cfg = captured["config"]
    assert cfg.trace_name == "AI Insider Threat Scenario"
    assert cfg.trace_tags == ("ai_insider_scenario",)


def _capture_downloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every `st.download_button` call's kwargs."""
    calls: list[dict[str, Any]] = []

    def _record(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(st, "download_button", _record)
    return calls


def test_layer_persisted_and_offered_for_download(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "# Scenario"
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    downloads = _capture_downloads(monkeypatch)
    payload = ('{"domain": "enterprise-attack"}', "threat_group_scenario_layer.json")

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda: payload,
    )

    # The captured layer is persisted verbatim for later reruns.
    assert fake_session_state["threat_group_scenario_layer"] == payload

    # Both the markdown scenario and the Navigator layer are offered.
    layer_downloads = [d for d in downloads if d.get("mime") == "application/json"]
    assert len(layer_downloads) == 1
    assert layer_downloads[0]["data"] == payload[0]
    assert layer_downloads[0]["file_name"] == "threat_group_scenario_layer.json"


def test_no_layer_download_when_build_layer_returns_none(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "# Scenario"
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="custom",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="custom_scenario.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
        build_layer=lambda: None,  # e.g. an unsupported matrix
    )

    assert fake_session_state["custom_scenario_layer"] is None
    # Only the markdown download — no JSON layer button.
    assert all(d.get("mime") != "application/json" for d in downloads)


def test_persisted_scenario_and_downloads_survive_rerun(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain rerun (e.g. after a download click) must keep the scenario and
    both download buttons — not blank the page because Generate is unpressed."""
    stub_streamlit["button_returns"] = False  # Generate not clicked this run.
    fake_session_state["threat_group_scenario_generated"] = True
    fake_session_state["threat_group_scenario_text"] = "# Prior scenario"
    fake_session_state["threat_group_scenario_layer"] = (
        '{"domain": "enterprise-attack"}',
        "threat_group_scenario_layer.json",
    )

    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: None,
        is_ready=lambda: False,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda: None,
    )

    # Both the markdown scenario and the persisted layer are re-offered.
    assert any(d.get("mime") == "text/markdown" for d in downloads)
    layer_downloads = [d for d in downloads if d.get("mime") == "application/json"]
    assert len(layer_downloads) == 1
    assert layer_downloads[0]["file_name"] == "threat_group_scenario_layer.json"


def test_no_layer_download_when_build_layer_absent(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page 3 passes no build_layer at all — the lifecycle must not break."""
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "# Scenario"
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="ai_insider",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="ai_insider_threat_scenario.md",
        trace_name="AI Insider Threat Scenario",
        trace_tags=("ai_insider_scenario",),
    )

    assert fake_session_state["ai_insider_scenario_layer"] is None
    assert all(d.get("mime") != "application/json" for d in downloads)
