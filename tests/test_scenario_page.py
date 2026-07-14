"""Tests for `core.scenario_page.run_scenario_page`.

The interface is the test surface: given a build_messages callback and a
readiness predicate, we assert what reaches `call_llm_stream` and what lands
in session_state. Streamlit's UI calls are stubbed to no-ops; we don't render
anything — we only care about the control flow at the seam.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

import core.llm as llm_module
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
    controls: dict[str, Any] = {"button_returns": False}

    def _button(*_args, **_kwargs):
        return controls["button_returns"]

    @contextmanager
    def _status(*_args, **_kwargs):
        # Yield a real object: production code calls `status.update(...)`.
        yield SimpleNamespace(update=lambda *_a, **_k: None)

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
    monkeypatch.setattr(st, "caption", _noop)
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
