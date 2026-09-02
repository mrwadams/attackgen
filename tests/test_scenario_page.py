"""Tests for `core.scenario_page.run_scenario_page`.

The interface is the test surface: given a build_messages callback and a
readiness predicate, we assert what reaches `call_llm_stream` and what lands
in session_state. Streamlit's UI calls are stubbed to no-ops; we don't render
anything — we only care about the control flow at the seam.
"""

from __future__ import annotations

import itertools
import re
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

import core.llm as llm_module
import core.scenario_page as scenario_page_module
from core.scenario_page import _unique_filenames, run_scenario_page


class TestUniqueFilenames:
    def test_meaningful_sanitised_and_timestamped(self):
        md, layer, detection = _unique_filenames("AttackGen APT29 Enterprise.md")
        assert re.fullmatch(r"AttackGen_APT29_Enterprise_\d{8}-\d{6}\.md", md)
        # The layer and detection downloads always share the markdown's stem.
        assert layer == md[:-3] + "_layer.json"
        assert detection == md[:-3] + "_detection.md"

    def test_special_characters_collapse(self):
        md, _layer, _detection = _unique_filenames("Weird / Name & C&C.md")
        assert re.fullmatch(r"Weird_Name_C_C_\d{8}-\d{6}\.md", md)

    def test_long_title_is_capped(self):
        md, _layer, _detection = _unique_filenames("A" * 200 + ".md")
        stem = md[: -len("_20260714-153045.md")]  # strip the "_<timestamp>.md" suffix
        assert len(stem) <= 80

    def test_empty_base_falls_back(self):
        md, _layer, _detection = _unique_filenames(".md")
        assert md.startswith("scenario_")


@pytest.fixture
def stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """No-op out the Streamlit UI surface that `run_scenario_page` touches.

    Returns a dict the test can mutate to control widget return values:
      - `button_returns`: bool returned by `st.button`
    """
    controls: dict[str, Any] = {
        "button_returns": False,
        "status_labels": [],
        "stream_chunks": [],
        "on_stream_chunk": None,
    }

    def _button(*_args, **_kwargs):
        return controls["button_returns"]

    @contextmanager
    def _status(*args, **_kwargs):
        if args:
            controls["status_labels"].append(args[0])

        def _update(*_args, **kwargs):
            if "label" in kwargs:
                controls["status_labels"].append(kwargs["label"])

        yield SimpleNamespace(update=_update)

    @contextmanager
    def _expander(*_args, **_kwargs):
        yield None

    def _tabs(labels, *_args, **_kwargs):
        return [_FakeTab() for _ in labels]

    def _noop(*_args, **_kwargs):
        return None

    def _write_stream(stream):
        # Drain the generator while recording what became visible incrementally.
        for chunk in stream:
            controls["stream_chunks"].append(chunk)
            if controls["on_stream_chunk"]:
                controls["on_stream_chunk"](chunk, list(controls["stream_chunks"]))

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "status", _status)
    monkeypatch.setattr(st, "expander", _expander)
    monkeypatch.setattr(st, "tabs", _tabs)
    monkeypatch.setattr(st, "markdown", _noop)
    monkeypatch.setattr(st, "caption", _noop)
    monkeypatch.setattr(st, "write", _noop)
    monkeypatch.setattr(st, "write_stream", _write_stream)
    monkeypatch.setattr(st, "download_button", _noop)
    monkeypatch.setattr(st, "info", _noop)
    monkeypatch.setattr(st, "warning", _noop)
    monkeypatch.setattr(st, "error", _noop)
    # `render_feedback_widget` calls `st.empty()` then `st.markdown('---')`.
    monkeypatch.setattr(st, "empty", _FakePlaceholder)
    # Pretend no LangSmith key is configured so the feedback widget
    # short-circuits cleanly during these general scenario-page tests.
    monkeypatch.setattr(st, "secrets", {})

    return controls


class _FakeTab:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


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
    # No defense companion here, so nothing for the Assistant to refine there.
    assert fake_session_state["last_defense_narrative"] is None

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
    layer_json = '{"domain": "enterprise-attack"}'

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda: layer_json,
    )

    # The layer is persisted as (json, generated_filename) for later reruns.
    stored_json, stored_layer_name = fake_session_state["threat_group_scenario_layer"]
    assert stored_json == layer_json

    md_name = fake_session_state["threat_group_scenario_filename"]
    # Meaningful, sanitised, timestamped, and the layer shares the md's stem.
    assert re.fullmatch(r"AttackGen_APT29_Enterprise_\d{8}-\d{6}\.md", md_name)
    assert stored_layer_name == md_name[:-3] + "_layer.json"

    # Both the markdown scenario and the Navigator layer are offered, named to match.
    md_downloads = [d for d in downloads if d.get("mime") == "text/markdown"]
    layer_downloads = [d for d in downloads if d.get("mime") == "application/json"]
    assert md_downloads[0]["file_name"] == md_name
    assert len(layer_downloads) == 1
    assert layer_downloads[0]["data"] == layer_json
    assert layer_downloads[0]["file_name"] == stored_layer_name


def _run_and_capture_caption(
    monkeypatch: pytest.MonkeyPatch, fake_session_state, stub_streamlit, layer_json: str
) -> str:
    """Generate a scenario whose layer is `layer_json`; return the layer caption."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    captions: list[str] = []
    monkeypatch.setattr(st, "caption", lambda text, *a, **k: captions.append(text))

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen Group Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda: layer_json,
    )
    return "\n".join(captions)


def test_layer_caption_targets_attack_navigator_for_attack_domains(
    stub_streamlit, fake_session_state, mock_litellm_completion,
    disable_langsmith_tracing, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_litellm_completion.content = "# Scenario"
    caption = _run_and_capture_caption(
        monkeypatch, fake_session_state, stub_streamlit, '{"domain": "enterprise-attack"}'
    )
    assert "ATT&CK Navigator" in caption
    assert "ATLAS Navigator" not in caption


def test_layer_caption_targets_atlas_navigator_for_atlas_domain(
    stub_streamlit, fake_session_state, mock_litellm_completion,
    disable_langsmith_tracing, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_litellm_completion.content = "# Scenario"
    caption = _run_and_capture_caption(
        monkeypatch, fake_session_state, stub_streamlit, '{"domain": "atlas-atlas"}'
    )
    assert "ATLAS Navigator" in caption


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
    md_name = "AttackGen_APT29_Enterprise_20260714-153045.md"
    layer_name = "AttackGen_APT29_Enterprise_20260714-153045_layer.json"
    fake_session_state["threat_group_scenario_generated"] = True
    fake_session_state["threat_group_scenario_text"] = "# Prior scenario"
    fake_session_state["threat_group_scenario_filename"] = md_name
    fake_session_state["threat_group_scenario_layer"] = (
        '{"domain": "enterprise-attack"}',
        layer_name,
    )

    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: None,
        is_ready=lambda: False,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda: None,
    )

    # Both downloads re-offered with the names fixed at generation time — not
    # re-timestamped by this rerun.
    md_downloads = [d for d in downloads if d.get("mime") == "text/markdown"]
    assert md_downloads[0]["file_name"] == md_name
    layer_downloads = [d for d in downloads if d.get("mime") == "application/json"]
    assert len(layer_downloads) == 1
    assert layer_downloads[0]["file_name"] == layer_name


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


# --- Detection & Response (purple-team) companion ----------------------------

# A minimal report shaped like core.detections.build_defense_report output.
_DEFENSE_REPORT = {
    "matrix": "Enterprise",
    "techniques": [
        {
            "id": "T1059",
            "name": "Command and Scripting Interpreter",
            "detection_strategies": [
                {"id": "DET0516", "name": "Behavioral Detection", "analytics": []}
            ],
            "mitigations": [{"id": "M1042", "name": "Disable or Remove Feature", "description": ""}],
        }
    ],
    "log_sources": ["WinEventLog:Security (EventCode=4624)"],
}


def test_defense_persisted_and_offered_for_download(
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
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_defense=lambda: _DEFENSE_REPORT,
        defense_narrative=False,
    )

    # Deterministic-only (no narrative): one model call, defense state persisted.
    assert len(mock_litellm_completion.calls) == 1
    state = fake_session_state["threat_group_scenario_defense"]
    assert state["narrative_md"] is None
    assert "Command and Scripting Interpreter (T1059)" in state["deterministic_md"]
    # Deterministic-only: no narrative for the Assistant to refine.
    assert fake_session_state["last_defense_narrative"] is None

    md_name = fake_session_state["threat_group_scenario_filename"]
    detection_downloads = [
        d for d in downloads if d.get("file_name", "").endswith("_detection.md")
    ]
    assert len(detection_downloads) == 1
    assert detection_downloads[0]["file_name"] == md_name[:-3] + "_detection.md"
    # The download bundles the deterministic reference.
    assert "Detection & Response Reference" not in detection_downloads[0]["data"]  # no narrative section
    assert "## 🛡️ Detection & Response" in detection_downloads[0]["data"]


def test_defense_narrative_makes_second_llm_call_and_persists(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "## Detection walkthrough\n\nStage 1."
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_defense=lambda: _DEFENSE_REPORT,
        defense_narrative=True,
    )

    # Two model calls: the scenario, then the purple-team narrative.
    assert len(mock_litellm_completion.calls) == 2
    state = fake_session_state["threat_group_scenario_defense"]
    assert state["narrative_md"] and "Detection walkthrough" in state["narrative_md"]
    # The combined download carries both the narrative and the reference section.
    assert "Detection & Response Reference" in state["download_md"]
    # The narrative is handed to the Assistant so it can be refined there too.
    assert fake_session_state["last_defense_narrative"] == state["narrative_md"]


def test_no_defense_download_when_build_defense_returns_none(
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
        build_defense=lambda: None,  # e.g. ATLAS technique with no mitigations
        defense_narrative=True,  # even requested, nothing to narrate
    )

    # No defensive data -> no narrative call, no detection download.
    assert len(mock_litellm_completion.calls) == 1
    assert fake_session_state["custom_scenario_defense"] is None
    assert all(not d.get("file_name", "").endswith("_detection.md") for d in downloads)


def test_result_is_tabbed_when_defense_present(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a Detection & Response companion, scenario + defence go in tabs so
    the reader switches rather than scrolls through both stacked outputs."""
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "# Scenario"
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    tab_calls: list[list[str]] = []
    monkeypatch.setattr(st, "tabs", lambda labels, *a, **k: (tab_calls.append(labels) or [_FakeTab() for _ in labels]))

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_defense=lambda: _DEFENSE_REPORT,
        defense_narrative=False,
    )

    assert tab_calls == [["📄 Scenario", "🛡️ Detection & Response"]]


def test_result_not_tabbed_without_defense(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    disable_langsmith_tracing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No companion (e.g. page 3) -> plain single-column render, no tabs."""
    stub_streamlit["button_returns"] = True
    mock_litellm_completion.content = "# Scenario"
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    tab_calls: list[list[str]] = []
    monkeypatch.setattr(st, "tabs", lambda labels, *a, **k: (tab_calls.append(labels) or [_FakeTab() for _ in labels]))

    run_scenario_page(
        page_id="ai_insider",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="ai_insider_threat_scenario.md",
        trace_name="AI Insider Threat Scenario",
        trace_tags=("ai_insider_scenario",),
    )

    assert tab_calls == []


def test_persisted_defense_survives_rerun(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain rerun must re-offer the Detection & Response download."""
    stub_streamlit["button_returns"] = False
    fake_session_state["threat_group_scenario_generated"] = True
    fake_session_state["threat_group_scenario_text"] = "# Prior scenario"
    fake_session_state["threat_group_scenario_filename"] = "scn_20260714-153045.md"
    fake_session_state["threat_group_scenario_layer"] = None
    fake_session_state["threat_group_scenario_defense"] = {
        "deterministic_md": "## 🛡️ Detection & Response",
        "narrative_md": None,
        "download_md": "# Detection & Response — scn\n\n## 🛡️ Detection & Response",
        "filename": "scn_20260714-153045_detection.md",
    }

    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: None,
        is_ready=lambda: False,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_defense=lambda: None,
    )

    detection_downloads = [
        d for d in downloads if d.get("file_name", "").endswith("_detection.md")
    ]
    assert len(detection_downloads) == 1
    assert detection_downloads[0]["file_name"] == "scn_20260714-153045_detection.md"


# --- Phased generation coordinator ------------------------------------------


def test_phase_sequence_and_base_is_persisted_before_optional_enrichment(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"
    downloads = _capture_downloads(monkeypatch)
    calls = 0

    def controlled_stream(_config, _messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "# Base "
            yield "scenario"
            return

        # The base and every deterministic export must be usable before the
        # optional second model call starts yielding.
        assert fake_session_state["threat_group_scenario_generated"] is True
        assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
        assert fake_session_state["last_scenario_text"] == "# Base scenario"
        assert fake_session_state["last_defense_narrative"] is None
        assert fake_session_state["threat_group_scenario_layer"][0] == (
            '{"domain": "enterprise-attack"}'
        )
        assert any(d.get("label") == "Download Scenario" for d in downloads)
        assert any(d.get("mime") == "application/json" for d in downloads)
        yield "## Defender walkthrough"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", controlled_stream)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _snapshot: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda _snapshot: '{"domain": "enterprise-attack"}',
        build_defense=lambda _snapshot: _DEFENSE_REPORT,
        defense_narrative=True,
        capture_inputs=lambda: {"matrix": "Enterprise"},
    )

    assert calls == 2
    phase_names = [
        "Preparing inputs",
        "Generating base scenario",
        "Base scenario available",
        "Building deterministic exports",
        "Generating purple-team narrative",
        "Complete",
    ]
    cursor = 0
    for phase in phase_names:
        cursor = next(
            i + 1
            for i, label in enumerate(stub_streamlit["status_labels"][cursor:], cursor)
            if label.startswith(phase) and "elapsed" in label
        )
    assert fake_session_state["last_defense_narrative"] == "## Defender walkthrough"


def test_generate_captures_input_and_identity_metadata(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "Anthropic API"
    fake_session_state["llm_model_name"] = "claude-sonnet-4-6"
    source = {
        "scenario_type": "custom",
        "matrix": "Enterprise",
        "organisation": {"industry": "Finance", "company_size": "Large"},
        "selected_techniques": ["PowerShell (T1059.001)"],
        "sampled_techniques": ["PowerShell (T1059.001)"],
        "modifiers": {"ai_uplift": True},
    }
    callback_snapshots = []

    def build_messages(snapshot):
        callback_snapshots.append(snapshot)
        # Mutating the original widget-backed mapping after capture must not
        # affect prompt/export callbacks or persisted metadata.
        source["matrix"] = "ICS"
        source["selected_techniques"].append("Changed later")
        return [{"role": "user", "content": snapshot["matrix"]}]

    def stream(_config, messages):
        assert messages[0]["content"] == "Enterprise"
        yield "# Scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", stream)

    run_scenario_page(
        page_id="custom",
        build_messages=build_messages,
        is_ready=lambda: True,
        download_name="AttackGen Custom Enterprise.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario", "ai_enhanced"),
        build_layer=lambda snapshot: (
            '{"domain": "enterprise-attack"}'
            if snapshot["matrix"] == "Enterprise"
            else None
        ),
        capture_inputs=lambda: source,
    )

    captured = fake_session_state["custom_scenario_input_snapshot"]
    assert captured["matrix"] == "Enterprise"
    assert captured["selected_techniques"] == ["PowerShell (T1059.001)"]
    assert captured["identity"] == {
        "page_id": "custom",
        "trace_name": "Custom Scenario",
        "trace_tags": ["custom_scenario", "ai_enhanced"],
        "provider": "Anthropic API",
        "model": "claude-sonnet-4-6",
        "download_name": "AttackGen Custom Enterprise.md",
    }
    assert captured["captured_at"]
    assert callback_snapshots[0]["matrix"] == "Enterprise"
    assert fake_session_state["custom_scenario_layer"] is not None


def test_streamed_base_text_is_visible_chunk_by_chunk(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    visible_steps: list[str] = []
    stub_streamlit["on_stream_chunk"] = (
        lambda _chunk, chunks: visible_steps.append("".join(chunks))
    )

    def controlled_stream(_config, _messages):
        yield "# Partial"
        yield " scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", controlled_stream)

    run_scenario_page(
        page_id="custom",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="custom.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
    )

    visible = stub_streamlit["stream_chunks"]
    assert len(visible) > 1
    assert "".join(visible) == "# Partial scenario"
    assert visible_steps[0] != visible_steps[-1]
    assert visible_steps[-1] == "# Partial scenario"
    assert fake_session_state["custom_scenario_text"] == "# Partial scenario"


# --- Ticking elapsed timer during pre-first-token waits (issue #89) --------
#
# `call_llm_stream` now runs on a worker thread (`_stream_on_worker`) so the
# main thread can poll a queue with a short timeout and refresh the elapsed
# label even before any chunk has arrived. These tests use a real (short)
# `time.sleep` inside a fixture stream to force that idle-polling window,
# with `_STREAM_POLL_INTERVAL` shrunk so it resolves quickly, and replace
# `time.monotonic` with a strictly-increasing counter so every poll produces
# a distinct, deterministic elapsed label regardless of real-time jitter.


def _phase_labels(labels: list[str], phase: str) -> list[str]:
    return [label for label in labels if label.startswith(f"{phase} ·")]


def test_elapsed_label_advances_while_base_scenario_call_is_silent(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the base call's first token, the elapsed label must keep
    ticking rather than freezing -- the core behaviour issue #89 asks for."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"

    clock = itertools.count()
    monkeypatch.setattr(scenario_page_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(scenario_page_module, "_STREAM_POLL_INTERVAL", 0.005)

    def silent_then_stream(_config, _messages):
        # Real sleep on the worker thread: gives the main thread's polling
        # loop room to tick several times before the first token arrives.
        time.sleep(0.1)
        yield "# Scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", silent_then_stream)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    ticks = _phase_labels(stub_streamlit["status_labels"], "Generating base scenario")
    # The fake clock advances by one on every tick, so distinct labels prove
    # the wait produced repeated, advancing refreshes -- not a frozen one.
    assert len(set(ticks)) >= 3


def test_elapsed_label_advances_while_narrative_call_is_silent(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same ticking behaviour applies to the second (purple-team narrative)
    streamed call, the case the issue calls out as most visible."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"

    clock = itertools.count()
    monkeypatch.setattr(scenario_page_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(scenario_page_module, "_STREAM_POLL_INTERVAL", 0.005)

    calls = 0

    def controlled_stream(_config, _messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "# Base scenario"
            return
        time.sleep(0.1)
        yield "## Defender walkthrough"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", controlled_stream)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _snapshot: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_defense=lambda _snapshot: _DEFENSE_REPORT,
        defense_narrative=True,
        capture_inputs=lambda: {},
    )

    assert calls == 2
    ticks = _phase_labels(
        stub_streamlit["status_labels"], "Generating purple-team narrative"
    )
    assert len(set(ticks)) >= 3


def test_streamed_output_still_renders_incrementally_via_worker_thread(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No regression: chunks relayed through the worker thread still arrive
    at `st.write_stream` incrementally, not batched after the call ends."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    visible_steps: list[str] = []
    stub_streamlit["on_stream_chunk"] = (
        lambda _chunk, chunks: visible_steps.append("".join(chunks))
    )

    def controlled_stream(_config, _messages):
        yield "# Part one "
        yield "part two "
        yield "part three"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", controlled_stream)

    run_scenario_page(
        page_id="custom",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="custom.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
    )

    visible = stub_streamlit["stream_chunks"]
    full_text = "# Part one part two part three"
    assert len(visible) > 1
    assert "".join(visible) == full_text
    # At least one intermediate step was visible before the full text -- i.e.
    # chunks rendered as they arrived rather than only once, after the call.
    assert any(step != full_text for step in visible_steps[:-1])
    assert visible_steps[-1] == full_text
    assert fake_session_state["custom_scenario_text"] == full_text


def test_threaded_stream_error_surfaces_same_message_and_no_completion(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model error raised mid-stream on the worker thread must reach the
    user the same way a synchronous error would, and the status must never
    reach "Complete" -- i.e. no spinner is left running as if still working."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"

    errors: list[str] = []
    monkeypatch.setattr(st, "error", lambda msg, *a, **k: errors.append(msg))

    def failing_stream(_config, _messages):
        yield "partial output"
        raise RuntimeError("boom")

    monkeypatch.setattr("core.scenario_page.call_llm_stream", failing_stream)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda: [{"role": "user", "content": "x"}],
        is_ready=lambda: True,
        download_name="threat_group_scenario.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert errors == ["An error occurred while generating the scenario: boom"]
    assert fake_session_state["threat_group_scenario_generated"] is False
    assert not any(
        label.startswith("Complete") for label in stub_streamlit["status_labels"]
    )
