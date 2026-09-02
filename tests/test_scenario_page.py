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
from core.scenario_page import _SCRIPT_CONTROL, _unique_filenames, run_scenario_page


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

    and read to see what was rendered: `buttons` (every `st.button` call's
    kwargs, so a test can press one via its `on_click`), `warnings`, `errors`
    and `status_labels`.
    """
    controls: dict[str, Any] = {
        "button_returns": False,
        "status_labels": [],
        "stream_chunks": [],
        "on_stream_chunk": None,
        "buttons": [],
        "warnings": [],
        "errors": [],
    }

    def _button(*args, **kwargs):
        controls["buttons"].append(
            {"label": args[0] if args else kwargs.get("label"), **kwargs}
        )
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
    monkeypatch.setattr(st, "warning", lambda msg, *a, **k: controls["warnings"].append(msg))
    monkeypatch.setattr(st, "error", lambda msg, *a, **k: controls["errors"].append(msg))
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


# --- Degraded success and retry for optional enrichment ----------------------


@pytest.fixture
def controllable_stream(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A scripted stand-in for `call_llm_stream`, one script per model call.

    Set `.scripts` to a list of step-lists — the first is played by the base
    scenario call, the second by the purple-team narrative call, and so on
    (calls past the end yield a single default chunk). Each step is either:

      - a string, yielded as a stream chunk;
      - a callable, invoked *between* chunks (to land a click mid-stream);
      - an exception instance, raised from inside the stream.

    Every call's config and messages are recorded in `.calls`, so a test can
    prove which phase ran — and, for a retry, which phase *didn't*.
    """
    scripted = SimpleNamespace(scripts=[], calls=[])

    def _stream(config, messages):
        scripted.calls.append(SimpleNamespace(config=config, messages=messages))
        index = len(scripted.calls) - 1
        steps = scripted.scripts[index] if index < len(scripted.scripts) else ["# Scenario"]
        for step in steps:
            if isinstance(step, BaseException):
                raise step
            if callable(step):
                step()
                continue
            yield step

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _stream)
    return scripted


def _is_narrative_call(call) -> bool:
    """The purple-team pass is the one tagged for it in LangSmith."""
    return call.config.trace_tags == ("purple_team_narrative",)


def _click(stub_streamlit: dict[str, Any], key: str) -> None:
    """Fire the `on_click` callback Streamlit runs when that button is pressed.

    Streamlit runs the callback before the next script run, which is exactly
    how the page picks the request up — so a test presses a button by calling
    its callback and then running the page again.
    """
    for record in stub_streamlit["buttons"]:
        if record.get("key") == key:
            record["on_click"](*record.get("args", ()))
            return
    raise AssertionError(f"no button was rendered with key {key!r}")


def _run_page(**overrides) -> None:
    """Run the threat-group page with a defence report and narrative enabled."""
    kwargs: dict[str, Any] = {
        "page_id": "threat_group",
        "build_messages": lambda _snapshot: [{"role": "user", "content": "x"}],
        "is_ready": lambda: True,
        "download_name": "AttackGen APT29 Enterprise.md",
        "trace_name": "Threat Group Scenario",
        "trace_tags": ("threat_group_scenario",),
        "build_layer": lambda _snapshot: '{"domain": "enterprise-attack"}',
        "build_defense": lambda _snapshot: _DEFENSE_REPORT,
        "defense_narrative": True,
        "capture_inputs": lambda: {"matrix": "Enterprise"},
    }
    kwargs.update(overrides)
    run_scenario_page(**kwargs)


def _generate_with_narrative(
    stub_streamlit, fake_session_state, controllable_stream, narrative_script
) -> None:
    """Generate a scenario, playing `narrative_script` for the optional phase."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"
    controllable_stream.scripts = [["# Base scenario"], narrative_script]
    _run_page()


def test_failed_narrative_leaves_base_scenario_and_exports_usable(
    stub_streamlit,
    fake_session_state,
    controllable_stream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads = _capture_downloads(monkeypatch)
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )

    # The base phase's output survives the optional phase's failure, whole.
    assert fake_session_state["threat_group_scenario_generated"] is True
    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["last_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_scenario_layer"][0] == (
        '{"domain": "enterprise-attack"}'
    )
    defense = fake_session_state["threat_group_scenario_defense"]
    assert "Command and Scripting Interpreter (T1059)" in defense["deterministic_md"]
    assert defense["narrative_md"] is None
    assert fake_session_state["last_defense_narrative"] is None

    # All three downloads are offered: scenario, Navigator layer, defence.
    assert any(d.get("label") == "Download Scenario" for d in downloads)
    assert any(d.get("mime") == "application/json" for d in downloads)
    assert any(d.get("file_name", "").endswith("_detection.md") for d in downloads)

    # The run still closes as a (degraded) success, not an error.
    assert any(
        label.startswith("Complete without purple-team narrative")
        for label in stub_streamlit["status_labels"]
    )


def test_degraded_message_explains_failure_and_offers_narrative_retry(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )

    status = fake_session_state["threat_group_generation_status"]
    assert status["phase"] == "narrative"
    assert status["reason"] == "error"

    # A warning (degraded success), not an error — and it says what failed.
    notice = "\n".join(stub_streamlit["warnings"])
    assert "purple-team narrative failed" in notice
    assert "upstream 503" in notice
    assert "complete and usable" in notice

    retry = [
        b for b in stub_streamlit["buttons"] if b.get("key") == "threat_group_retry_narrative"
    ]
    assert len(retry) == 1
    assert retry[0]["label"] == "Retry purple-team narrative"


def test_narrative_retry_reruns_only_the_narrative_and_merges_it(
    stub_streamlit,
    fake_session_state,
    controllable_stream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )
    md_name = fake_session_state["threat_group_scenario_filename"]

    _click(stub_streamlit, "threat_group_retry_narrative")

    # The rerun the click triggers: Generate is not pressed, and the page's
    # readiness has since lapsed — the retry must not depend on either.
    stub_streamlit["button_returns"] = False
    stub_streamlit["warnings"].clear()
    controllable_stream.scripts = [[], [], ["## Defender walkthrough"]]
    downloads = _capture_downloads(monkeypatch)
    _run_page(is_ready=lambda: False)

    # Three model calls in total, and the retry's is the narrative — the base
    # scenario is never generated a second time.
    assert len(controllable_stream.calls) == 3
    assert _is_narrative_call(controllable_stream.calls[2])
    assert sum(1 for c in controllable_stream.calls if not _is_narrative_call(c)) == 1

    # The narrative is merged into the Detection & Response view and download.
    defense = fake_session_state["threat_group_scenario_defense"]
    assert defense["narrative_md"] == "## Defender walkthrough"
    assert "Defender walkthrough" in defense["download_md"]
    assert "Detection & Response Reference" in defense["download_md"]
    assert fake_session_state["last_defense_narrative"] == "## Defender walkthrough"
    detection_downloads = [
        d for d in downloads if d.get("file_name", "").endswith("_detection.md")
    ]
    assert detection_downloads[-1]["data"] == defense["download_md"]

    # The base result is untouched — same text, same download names.
    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_scenario_filename"] == md_name

    # No degraded state left behind, so no notice on the next rerun.
    assert "threat_group_generation_status" not in fake_session_state
    assert stub_streamlit["warnings"] == []


def test_skip_requested_mid_stream_abandons_narrative_and_keeps_base(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    def request_skip():
        st.session_state["threat_group_narrative_stop_requested"] = True

    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        ["## Half a walk", request_skip, "through"],
    )

    # The partial narrative is discarded; the base result stands on its own.
    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_scenario_defense"]["narrative_md"] is None
    assert fake_session_state["last_defense_narrative"] is None
    status = fake_session_state["threat_group_generation_status"]
    assert status == {"phase": "narrative", "reason": "stopped", "detail": ""}
    assert "You skipped the purple-team narrative" in "\n".join(stub_streamlit["warnings"])
    assert any(
        b.get("key") == "threat_group_retry_narrative" for b in stub_streamlit["buttons"]
    )


def test_skip_button_requests_the_stop(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    """The Skip control is what sets the flag the narrative phase watches."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    controllable_stream.scripts = [["# Base scenario"], ["## Walkthrough"]]
    _run_page()

    skip = [
        b for b in stub_streamlit["buttons"] if b.get("key") == "threat_group_skip_narrative"
    ]
    assert len(skip) == 1
    _click(stub_streamlit, "threat_group_skip_narrative")
    assert fake_session_state["threat_group_narrative_stop_requested"] is True


def test_narrative_left_in_flight_by_a_torn_down_run_is_reported_next_run(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    """Pressing Skip reruns the script, which kills the in-flight stream.

    The next run finds the in-progress marker with no narrative and must report
    the optional phase as skipped — with the base result still on the page —
    rather than pretend it completed.
    """
    stub_streamlit["button_returns"] = False
    fake_session_state.update(
        {
            "threat_group_scenario_generated": True,
            "threat_group_scenario_text": "# Base scenario",
            "threat_group_scenario_filename": "scn_20260714-153045.md",
            "threat_group_scenario_layer": None,
            "threat_group_scenario_defense": {
                "deterministic_md": "## 🛡️ Detection & Response",
                "narrative_md": None,
                "download_md": "# Detection & Response — scn",
                "filename": "scn_20260714-153045_detection.md",
            },
            # Left behind by the run the rerun tore down.
            "threat_group_narrative_running": True,
            "threat_group_narrative_stop_requested": True,
        }
    )

    _run_page(is_ready=lambda: False)

    assert controllable_stream.calls == []  # nothing is regenerated
    assert fake_session_state["threat_group_generation_status"] == {
        "phase": "narrative",
        "reason": "stopped",
        "detail": "",
    }
    # The markers are consumed, so the next rerun doesn't re-report it.
    assert "threat_group_narrative_running" not in fake_session_state
    assert "threat_group_narrative_stop_requested" not in fake_session_state
    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert any(
        b.get("key") == "threat_group_retry_narrative" for b in stub_streamlit["buttons"]
    )


def test_degraded_state_renders_base_and_notice_on_a_plain_rerun(
    stub_streamlit,
    fake_session_state,
    controllable_stream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded run stays degraded-but-complete across reruns: every base
    download is re-offered, and the notice keeps offering the retry."""
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )

    stub_streamlit["button_returns"] = False
    stub_streamlit["buttons"].clear()
    stub_streamlit["warnings"].clear()
    downloads = _capture_downloads(monkeypatch)
    _run_page(is_ready=lambda: False)

    assert len(controllable_stream.calls) == 2  # no phase re-ran on this rerun
    assert any(d.get("label") == "Download Scenario" for d in downloads)
    assert any(d.get("mime") == "application/json" for d in downloads)
    assert any(d.get("file_name", "").endswith("_detection.md") for d in downloads)
    assert "purple-team narrative failed" in "\n".join(stub_streamlit["warnings"])
    assert any(
        b.get("key") == "threat_group_retry_narrative" for b in stub_streamlit["buttons"]
    )


def test_base_failure_is_attributed_to_the_base_phase(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    controllable_stream.scripts = [[RuntimeError("rate limited")]]

    _run_page()

    status = fake_session_state["threat_group_generation_status"]
    assert status["phase"] == "base"
    assert status["detail"] == "rate limited"
    # Nothing usable was produced, so this reads as an error, not degraded.
    assert stub_streamlit["warnings"] == []
    assert fake_session_state["threat_group_scenario_generated"] is False
    notice = "\n".join(stub_streamlit["errors"])
    assert "base scenario failed to generate" in notice.lower()
    assert "rate limited" in notice
    retry = [
        b for b in stub_streamlit["buttons"] if b.get("key") == "threat_group_retry_base"
    ]
    assert len(retry) == 1
    assert retry[0]["label"] == "Retry base scenario"


def test_base_retry_replays_the_captured_inputs(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    controllable_stream.scripts = [[RuntimeError("rate limited")]]
    source = {"matrix": "Enterprise"}

    _run_page(
        defense_narrative=False,
        build_messages=lambda snapshot: [
            {"role": "user", "content": snapshot["matrix"]}
        ],
        capture_inputs=lambda: source,
    )

    # The widgets move on after the failure; the retry must not follow them.
    source["matrix"] = "ICS"
    _click(stub_streamlit, "threat_group_retry_base")
    stub_streamlit["button_returns"] = False
    controllable_stream.scripts = [[], ["# Base scenario"]]

    _run_page(
        is_ready=lambda: False,
        defense_narrative=False,
        build_messages=lambda snapshot: [
            {"role": "user", "content": snapshot["matrix"]}
        ],
        capture_inputs=lambda: source,
    )

    assert len(controllable_stream.calls) == 2
    assert controllable_stream.calls[1].messages == [
        {"role": "user", "content": "Enterprise"}
    ]
    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_scenario_generated"] is True
    assert "threat_group_generation_status" not in fake_session_state


@pytest.mark.skipif(
    not _SCRIPT_CONTROL, reason="Streamlit exposes no script-control exceptions"
)
def test_rerun_signal_during_the_narrative_is_not_swallowed(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    """Pressing Skip queues a rerun, which Streamlit raises inside the running
    script. Swallowing it would drop the rerun and report a control signal as a
    generation failure, so it propagates — with the in-flight marker left set,
    which is how the next run knows the optional phase never finished."""
    rerun = _SCRIPT_CONTROL[0](None)

    with pytest.raises(type(rerun)):
        _generate_with_narrative(
            stub_streamlit, fake_session_state, controllable_stream, [rerun]
        )

    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_narrative_running"] is True
    assert "threat_group_generation_status" not in fake_session_state
