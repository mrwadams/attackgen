"""Tests for `core.scenario_page.run_scenario_page`.

The interface is the test surface: given a build_messages callback and a
readiness predicate, we assert what reaches `call_llm_stream` and what lands
in session_state. Streamlit's UI calls are stubbed to no-ops; we don't render
anything — we only care about the control flow at the seam.
"""

from __future__ import annotations

import copy
import itertools
import re
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st

import core.llm as llm_module
from core.scenario_page import (
    BASE_PHASE,
    _SCRIPT_CONTROL,
    _stream_on_worker,
    _unique_filenames,
    run_scenario_page,
)


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
    kwargs, so a test can press one via its `on_click`), `page_links`,
    `markdown`, `captions`, `infos`, `warnings`, `errors` and `status_labels`.
    """
    controls: dict[str, Any] = {
        "button_returns": False,
        "status_labels": [],
        "stream_chunks": [],
        "on_stream_chunk": None,
        "buttons": [],
        "page_links": [],
        "markdown": [],
        "captions": [],
        "infos": [],
        "warnings": [],
        "errors": [],
        # Every placeholder `st.empty()` handed out, so a test can ask whether
        # the one holding a transient control was cleared before the run ended.
        "placeholders": [],
    }

    def _button(*args, **kwargs):
        label = args[0] if args else kwargs.get("label")
        controls["buttons"].append({"label": label, **kwargs})
        if _OPEN_PLACEHOLDERS:
            _OPEN_PLACEHOLDERS[-1].rendered.append(label)
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

    def _page_link(path, *_args, **kwargs):
        controls["page_links"].append({"path": path, **kwargs})

    def _columns(spec, *_args, **_kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeTab() for _ in range(count)]

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "page_link", _page_link)
    monkeypatch.setattr(st, "columns", _columns)
    monkeypatch.setattr(st, "status", _status)
    monkeypatch.setattr(st, "expander", _expander)
    monkeypatch.setattr(st, "tabs", _tabs)
    monkeypatch.setattr(
        st, "markdown", lambda body="", *a, **k: controls["markdown"].append(str(body))
    )
    monkeypatch.setattr(
        st, "caption", lambda body="", *a, **k: controls["captions"].append(str(body))
    )
    monkeypatch.setattr(st, "write", _noop)
    monkeypatch.setattr(st, "write_stream", _write_stream)
    monkeypatch.setattr(st, "download_button", _noop)
    monkeypatch.setattr(st, "info", lambda msg, *a, **k: controls["infos"].append(msg))
    monkeypatch.setattr(st, "warning", lambda msg, *a, **k: controls["warnings"].append(msg))
    monkeypatch.setattr(st, "error", lambda msg, *a, **k: controls["errors"].append(msg))
    def _empty():
        slot = _FakePlaceholder()
        controls["placeholders"].append(slot)
        return slot

    # `render_feedback_widget` calls `st.empty()` then `st.markdown('---')`.
    monkeypatch.setattr(st, "empty", _empty)
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
    """A stand-in for `st.empty()` that remembers what was put in it.

    `rendered` collects the buttons written through `container()`, and
    `cleared` records whether `empty()` was called — together they let a test
    say "this control was rendered into a slot, and that slot was cleared".
    """

    def __init__(self):
        self.rendered: list[str] = []
        self.cleared = False

    def success(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def error(self, *_a, **_k): pass

    def empty(self, *_a, **_k):
        self.cleared = True

    @contextmanager
    def container(self, *_a, **_k):
        _OPEN_PLACEHOLDERS.append(self)
        try:
            yield None
        finally:
            _OPEN_PLACEHOLDERS.pop()


_OPEN_PLACEHOLDERS: list[_FakePlaceholder] = []
"""The `st.empty()` containers currently open, innermost last."""


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


_BASE_KEPT_PREFIX = "The scenario, its downloads,"
"""Opening of the reassurance clause the degraded notice appends."""


def _skip_slot(stub_streamlit: dict[str, Any]):
    """The placeholder the Skip control was rendered into, if any."""
    for slot in stub_streamlit["placeholders"]:
        if "Skip purple-team narrative" in slot.rendered:
            return slot
    return None


def test_skip_control_is_cleared_once_the_narrative_settles(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    """A Skip button outliving its phase offers to interrupt nothing.

    The control belongs to the window where the narrative is actually in
    flight, so once the phase settles — here, by failing — the slot holding it
    must be cleared rather than left on screen beside the retry.
    """
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )

    slot = _skip_slot(stub_streamlit)
    assert slot is not None, "the Skip control was never rendered"
    assert slot.cleared, "the Skip control outlived the phase it interrupts"


@pytest.mark.skipif(
    not _SCRIPT_CONTROL, reason="Streamlit exposes no script-control exceptions"
)
def test_skip_control_survives_a_run_torn_down_mid_stream(
    stub_streamlit, fake_session_state, controllable_stream
) -> None:
    """Clearing it is conditional on the phase settling, not on leaving the call.

    A rerun tears the script down mid-stream; the phase has not settled, so the
    control must still be standing for the next run to replace — the same terms
    on which the in-flight marker survives.
    """
    rerun = _SCRIPT_CONTROL[0](None)

    def _tear_down_on_narrative_chunk(chunk: str, _seen: list[str]) -> None:
        if chunk.startswith("##"):
            raise rerun

    stub_streamlit["on_stream_chunk"] = _tear_down_on_narrative_chunk

    with pytest.raises(type(rerun)):
        _generate_with_narrative(
            stub_streamlit,
            fake_session_state,
            controllable_stream,
            ["## Defender walkthrough"],
        )

    slot = _skip_slot(stub_streamlit)
    assert slot is not None, "the Skip control was never rendered"
    assert not slot.cleared
    assert fake_session_state["threat_group_narrative_running"] is True


_PUNCTUATED_DETAILS = ["Connection error.", "Connection error", "rate limited!"]
"""Client error strings as they actually arrive: punctuated, bare, emphatic."""


@pytest.mark.parametrize("detail", _PUNCTUATED_DETAILS)
def test_degraded_notice_ends_in_a_single_full_stop(
    stub_streamlit, fake_session_state, controllable_stream, detail: str
) -> None:
    """Client errors are quoted verbatim and often arrive already punctuated.

    litellm's connection errors end in "."; appending the template's own stop
    gave users "Connection error..". Whatever punctuation the detail brought,
    the sentence around it must close exactly once.
    """
    _generate_with_narrative(
        stub_streamlit, fake_session_state, controllable_stream, [RuntimeError(detail)]
    )

    notice = "\n".join(stub_streamlit["warnings"])
    assert notice.startswith("The purple-team narrative failed: ")
    assert f"{detail.rstrip('.!')}. {_BASE_KEPT_PREFIX}" in notice
    assert ".." not in notice
    assert "!." not in notice


@pytest.mark.parametrize("detail", _PUNCTUATED_DETAILS)
def test_base_failure_notice_ends_in_a_single_full_stop(
    stub_streamlit, fake_session_state, controllable_stream, detail: str
) -> None:
    """The base-phase notice quotes the same details and must read the same."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    controllable_stream.scripts = [[RuntimeError(detail)]]

    _run_page()

    notice = "\n".join(stub_streamlit["errors"])
    assert f"{detail.rstrip('.!')}. Nothing downstream ran." in notice
    assert ".." not in notice
    assert "!." not in notice


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
    """Pressing Skip queues a rerun, which Streamlit raises on the main thread
    from the next Streamlit call the running script makes — here, rendering a
    narrative chunk. Swallowing it would drop the rerun and report a control
    signal as a generation failure, so it propagates — with the in-flight marker
    left set, which is how the next run knows the optional phase never finished.

    The signal is injected at the render call rather than from inside the model
    stream because the stream runs on a worker thread (`_stream_on_worker`);
    Streamlit only ever raises this on the thread running the script."""
    rerun = _SCRIPT_CONTROL[0](None)

    def _tear_down_on_narrative_chunk(chunk: str, _seen: list[str]) -> None:
        if chunk.startswith("##"):
            raise rerun

    stub_streamlit["on_stream_chunk"] = _tear_down_on_narrative_chunk

    with pytest.raises(type(rerun)):
        _generate_with_narrative(
            stub_streamlit,
            fake_session_state,
            controllable_stream,
            ["## Defender walkthrough"],
        )

    assert fake_session_state["threat_group_scenario_text"] == "# Base scenario"
    assert fake_session_state["threat_group_narrative_running"] is True
    assert "threat_group_generation_status" not in fake_session_state


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
    monkeypatch.setattr("core.scenario_page._monotonic", lambda: next(clock))
    monkeypatch.setattr("core.scenario_page._STREAM_POLL_INTERVAL", 0.005)

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
    monkeypatch.setattr("core.scenario_page._monotonic", lambda: next(clock))
    monkeypatch.setattr("core.scenario_page._STREAM_POLL_INTERVAL", 0.005)

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

    # The source BLOCKS after the first chunk until the consumer has actually
    # seen it. If the relay buffered the response and only handed it over at
    # the end, `seen_first` would never be set and this deadlocks -- the wait
    # times out and the test fails. An assertion over the accumulated prefixes
    # cannot do that job: the stub builds those prefixes itself, so they differ
    # from the full text by construction whether or not anything was buffered.
    seen_first = threading.Event()
    stub_streamlit["on_stream_chunk"] = lambda _chunk, chunks: (
        visible_steps.append("".join(chunks)),
        seen_first.set(),
    )

    # Recorded, not asserted in-generator: the page wraps generation in a broad
    # `except Exception`, so an assert raised here would be swallowed and the
    # test would fail somewhere unrelated with a message about the wrong thing.
    rendered_before_second: list[bool] = []

    def controlled_stream(_config, _messages):
        yield "# Part one "
        rendered_before_second.append(seen_first.wait(timeout=5))
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

    assert rendered_before_second == [True], (
        "the first chunk was not rendered before the second was produced — the "
        "relay buffered the response instead of streaming it"
    )
    visible = stub_streamlit["stream_chunks"]
    full_text = "# Part one part two part three"
    assert len(visible) > 1
    assert "".join(visible) == full_text
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

    # The base-phase notice follows it; the model's own message must still be
    # the first thing the user is told.
    assert errors[0] == "An error occurred while generating the scenario: boom"
    assert fake_session_state["threat_group_scenario_generated"] is False
    assert not any(
        label.startswith("Complete") for label in stub_streamlit["status_labels"]
    )


def test_worker_thread_is_given_the_script_run_context_before_it_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relay must attach the Streamlit script run context to the worker
    BEFORE starting it.

    `call_llm_stream` returns a @traceable generator whose body runs on the
    first `next()` — which now happens on the worker. That body stashes the
    LangSmith run id into `st.session_state`, and without a script run context
    the write goes nowhere and the feedback widget reports "No run ID found".

    This asserts the wiring, not the Streamlit behaviour, and deliberately so:
    `fake_session_state` replaces session state with a plain dict, which every
    thread can see, so the suite CANNOT reproduce the production failure. That
    is how the original change passed 257 tests with this bug in it.
    """
    seen: dict[str, object] = {}

    def fake_attach(thread: threading.Thread) -> None:
        seen["thread"] = thread
        seen["alive_at_attach"] = thread.is_alive()

    monkeypatch.setattr("core.scenario_page._attach_script_run_ctx", fake_attach)

    assert list(_stream_on_worker(iter(["a", "b"]))) == ["a", "b"]
    assert isinstance(seen.get("thread"), threading.Thread)
    # Attaching after start() is a race the context may lose, so ordering is
    # the whole point of the assertion.
    assert seen["alive_at_attach"] is False


def test_closing_the_stream_stops_the_worker_pulling_from_the_model() -> None:
    """A cancelled generation must stop draining the model, not run to
    completion into a queue nobody is reading.

    Streamlit raises RerunException from `status.update()` when the user
    touches a widget mid-generation, which closes this generator without
    exhausting it. Before the stop event existed, the worker kept pulling and
    held the HTTP response open.
    """
    produced: list[int] = []
    closed = threading.Event()

    def source():
        try:
            for i in range(100_000):
                produced.append(i)
                yield f"chunk-{i} "
        finally:
            closed.set()

    gen = _stream_on_worker(source())
    assert next(gen).startswith("chunk-0")
    gen.close()

    assert closed.wait(timeout=5), "the source generator was never closed"
    settled = len(produced)
    time.sleep(0.3)
    assert len(produced) == settled, "the worker kept pulling after cancellation"
    # Bounded by the queue, so it stops early rather than draining the response.
    assert settled < 1_000


# --- Readiness, result actions and cross-page navigation (issue #45) ---------


def _setup_state(**overrides):
    """A `SetupState` for the shared Setup fields, complete unless overridden."""
    from core.sidebar import get_setup_state

    state = {
        "chosen_model_provider": "OpenAI API",
        "llm_model_name": "gpt-5.5",
        "matrix": "Enterprise",
        "industry": "Finance / Banking",
        "company_size": "Medium (51-200 employees)",
    }
    state.update(overrides)
    return get_setup_state(state, {"OPENAI_API_KEY": "k"})


PAGE_CASES = [
    ("threat_group", "Select a threat actor group for the scenario."),
    ("custom", "Select at least one ATT&CK technique for the scenario."),
    ("ai_insider", "Select at least one threat category for the scenario."),
]


@pytest.mark.parametrize("page_id, page_blocker", PAGE_CASES)
def test_generate_is_disabled_and_every_blocker_is_listed(
    stub_streamlit,
    fake_session_state,
    mock_litellm_completion,
    page_id: str,
    page_blocker: str,
) -> None:
    """The global requirements behave the same on every page; the page-specific
    one is listed first, and Generate can't be spent finding out."""
    stub_streamlit["button_returns"] = True  # a click on a disabled button
    build_calls: list[None] = []

    run_scenario_page(
        page_id=page_id,
        build_messages=lambda: build_calls.append(None),
        requirements=[page_blocker],
        setup=_setup_state(industry=None, company_size=None),
        download_name="scenario.md",
        trace_name="Scenario",
        trace_tags=("scenario",),
    )

    assert mock_litellm_completion.calls == []
    assert build_calls == []
    generate = next(b for b in stub_streamlit["buttons"] if b.get("key") == f"{page_id}_generate")
    assert generate["disabled"] is True

    summary = "\n".join(stub_streamlit["infos"])
    assert summary.index(page_blocker) < summary.index("industry")
    assert "Select your company's industry in the Setup sidebar." in summary
    assert "Select your company's size in the Setup sidebar." in summary


@pytest.mark.parametrize("page_id, page_blocker", PAGE_CASES)
def test_generate_is_enabled_and_confirms_the_inputs_when_ready(
    stub_streamlit,
    fake_session_state,
    page_id: str,
    page_blocker: str,
) -> None:
    stub_streamlit["button_returns"] = False

    run_scenario_page(
        page_id=page_id,
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="scenario.md",
        trace_name="Scenario",
        trace_tags=("scenario",),
        capture_inputs=lambda: {
            "matrix": "Enterprise",
            "organisation": {
                "industry": "Finance / Banking",
                "company_size": "Medium (51-200 employees)",
            },
            "selected_entity": {"type": "threat actor group", "name": "APT29"},
            "modifiers": {"purple_team_narrative": True},
        },
    )

    generate = next(b for b in stub_streamlit["buttons"] if b.get("key") == f"{page_id}_generate")
    assert generate["disabled"] is False
    # No readiness blockers — the only notice is the feedback widget's own.
    assert not any("Complete these before generating" in info for info in stub_streamlit["infos"])

    # Story 16: what is about to be generated is confirmed before any usage.
    confirmation = "\n".join(stub_streamlit["captions"])
    assert "Ready to generate" in confirmation
    assert "Enterprise ATT&CK" in confirmation
    assert "APT29" in confirmation
    assert "Purple-team narrative" in confirmation


def test_modifiers_are_rendered_before_the_generate_button(
    stub_streamlit, fake_session_state, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modifier that adds a model call must be a decision made before the
    request starts, not one noticed beside or after the button."""
    order: list[str] = []

    def _modifiers():
        order.append("modifiers")

    def _button(*_args, **kwargs):
        if kwargs.get("key") == "threat_group_generate":
            order.append("generate")
        return False

    monkeypatch.setattr(st, "button", _button)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=[],
        setup=_setup_state(),
        download_name="scenario.md",
        trace_name="Scenario",
        trace_tags=("scenario",),
        render_modifiers=_modifiers,
    )

    assert order == ["modifiers", "generate"]


def _generate_threat_group_result(
    stub_streamlit, fake_session_state, monkeypatch, *, inputs=None
) -> None:
    """Generate one complete threat-group result (scenario + layer + defence)."""
    stub_streamlit["button_returns"] = True
    fake_session_state["chosen_model_provider"] = "OpenAI API"
    fake_session_state["llm_model_name"] = "gpt-5.5"
    fake_session_state["llm_api_key"] = "k"

    def _stream(_config, _messages):
        fake_session_state["run_id"] = "run-abc"
        yield "# APT29 Scenario\n\nA phased intrusion.\n\n## Injects\n\nInject 1."

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _stream)
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda _s: '{"domain": "enterprise-attack"}',
        build_defense=lambda _s: _DEFENSE_REPORT,
        capture_inputs=lambda: copy.deepcopy(inputs or _THREAT_GROUP_INPUTS),
    )


_THREAT_GROUP_INPUTS = {
    "scenario_type": "threat_group",
    "matrix": "Enterprise",
    "organisation": {
        "industry": "Finance / Banking",
        "company_size": "Medium (51-200 employees)",
    },
    "selected_entity": {"type": "threat actor group", "name": "APT29"},
    "selected_techniques": ["T1566"],
    "sampled_techniques": ["T1566"],
    "modifiers": {"ai_uplift": False, "purple_team_narrative": False},
}


def test_result_survives_visiting_the_assistant_and_returning(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance case from issue #45's manual testing: generate, open the
    Assistant, come back to a page whose selector has reset — and still find the
    scenario, its captured inputs, its stable filenames and its actions."""
    from core.assistant import scenario_handoff
    from core.routes import ASSISTANT_PAGE, THREAT_GROUP_PAGE

    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)
    md_name = fake_session_state["threat_group_scenario_filename"]
    layer_name = fake_session_state["threat_group_scenario_layer"][1]
    detection_name = fake_session_state["threat_group_scenario_defense"]["filename"]

    # ...the Assistant knows which scenario it is discussing, and how to get back.
    scenario = scenario_handoff()
    assert scenario is not None
    assert scenario.text.startswith("# APT29 Scenario")
    assert scenario.origin == THREAT_GROUP_PAGE
    assert "APT29" in scenario.title

    # ...and returning re-runs the page from scratch: Generate unpressed, and
    # the group selector back at its neutral default.
    stub_streamlit["button_returns"] = False
    stub_streamlit["buttons"].clear()
    stub_streamlit["page_links"].clear()
    stub_streamlit["captions"].clear()
    downloads = _capture_downloads(monkeypatch)

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=["Select a threat actor group for the scenario."],
        setup=_setup_state(),
        download_name="AttackGen None Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        build_layer=lambda _s: None,
        build_defense=lambda _s: None,
    )

    # Every artefact from that one generation is still offered, under the names
    # it was generated with.
    assert [d["file_name"] for d in downloads] == [md_name, layer_name, detection_name]
    assert downloads[0]["data"].startswith("# APT29 Scenario")

    # The result's own inputs are shown with it, not the page's current form.
    assert any("APT29" in caption for caption in stub_streamlit["captions"])

    # The action area is intact.
    assert any(link["path"] == ASSISTANT_PAGE.path for link in stub_streamlit["page_links"])
    keys = {b.get("key") for b in stub_streamlit["buttons"]}
    assert "threat_group_regenerate_previous" in keys
    assert "threat_group_clear_previous" in keys


def test_each_page_keeps_its_own_latest_result(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)

    def _stream(_config, _messages):
        yield "# Custom scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _stream)
    run_scenario_page(
        page_id="custom",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen Custom Enterprise.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
        capture_inputs=lambda: {"matrix": "Enterprise", "modifiers": {}},
    )

    assert fake_session_state["threat_group_scenario_text"].startswith("# APT29")
    assert fake_session_state["custom_scenario_text"] == "# Custom scenario"
    # The Assistant follows the most recent generation.
    assert fake_session_state["last_scenario_meta"]["page_id"] == "custom"


def test_editing_the_form_keeps_the_result_and_says_it_is_out_of_date(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)
    stub_streamlit["button_returns"] = False
    stub_streamlit["infos"].clear()
    downloads = _capture_downloads(monkeypatch)

    edited = copy.deepcopy(_THREAT_GROUP_INPUTS)
    edited["selected_entity"]["name"] = "APT28"

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT28 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        capture_inputs=lambda: edited,
    )

    # The old result is not destroyed by editing the form...
    assert fake_session_state["threat_group_scenario_text"].startswith("# APT29")
    assert downloads
    # ...but it is not passed off as matching the new selection either.
    assert any("differ from the inputs" in info for info in stub_streamlit["infos"])


def test_regenerate_action_runs_again_with_the_current_inputs(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)
    first_text = fake_session_state["threat_group_scenario_text"]

    _click(stub_streamlit, "threat_group_regenerate_current")

    # The rerun the click triggers: Generate itself is not pressed.
    stub_streamlit["button_returns"] = False
    edited = copy.deepcopy(_THREAT_GROUP_INPUTS)
    edited["selected_entity"]["name"] = "APT28"

    def _stream(_config, _messages):
        yield "# APT28 Scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _stream)
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT28 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        capture_inputs=lambda: edited,
    )

    assert first_text.startswith("# APT29")
    assert fake_session_state["threat_group_scenario_text"] == "# APT28 Scenario"
    assert fake_session_state["threat_group_scenario_input_snapshot"]["selected_entity"][
        "name"
    ] == "APT28"


def test_clear_result_removes_only_the_result(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)

    _click(stub_streamlit, "threat_group_clear_current")

    stub_streamlit["button_returns"] = False
    downloads = _capture_downloads(monkeypatch)
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert fake_session_state["threat_group_scenario_generated"] is False
    assert "threat_group_scenario_text" not in fake_session_state
    assert downloads == []
    # The Assistant's handoff went with it...
    assert "last_scenario_text" not in fake_session_state
    # ...but Setup did not.
    assert fake_session_state["chosen_model_provider"] == "OpenAI API"
    assert fake_session_state["llm_model_name"] == "gpt-5.5"
    assert fake_session_state["llm_api_key"] == "k"


def test_clearing_one_page_leaves_another_pages_result_alone(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)
    fake_session_state["custom_scenario_generated"] = True
    fake_session_state["custom_scenario_text"] = "# Custom scenario"

    _click(stub_streamlit, "threat_group_clear_current")
    stub_streamlit["button_returns"] = False
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert fake_session_state["custom_scenario_text"] == "# Custom scenario"


def test_long_result_gets_a_summary_and_section_navigation(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)

    rendered = "\n".join(stub_streamlit["markdown"])
    # The complete Markdown is still rendered, unmodified...
    assert "# APT29 Scenario\n\nA phased intrusion.\n\n## Injects\n\nInject 1." in rendered
    # ...with a compact summary and jump links above it.
    assert "**Summary:** A phased intrusion." in rendered
    assert "[Injects](#injects)" in rendered


def test_feedback_is_pinned_to_the_run_that_produced_the_result(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)

    # A later generation elsewhere moves the session's "current" run id on.
    def _stream(_config, _messages):
        fake_session_state["run_id"] = "run-xyz"
        yield "# Custom scenario"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _stream)
    run_scenario_page(
        page_id="custom",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen Custom Enterprise.md",
        trace_name="Custom Scenario",
        trace_tags=("custom_scenario",),
    )

    assert fake_session_state["threat_group_scenario_run_id"] == "run-abc"
    assert fake_session_state["custom_scenario_run_id"] == "run-xyz"


def test_deep_link_without_a_result_explains_that_scenarios_are_session_only(
    stub_streamlit, fake_session_state
) -> None:
    from core.state import RESTORED_KEY

    fake_session_state[RESTORED_KEY] = True

    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: None,
        requirements=["Select a threat actor group for the scenario."],
        setup=_setup_state(),
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    note = "\n".join(stub_streamlit["captions"])
    assert "current browser session only" in note


def test_regenerate_says_why_nothing_happened_when_the_form_is_not_ready(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate sits with the result, which outlives the widgets that made it.

    Navigating away and back resets the page's selector but keeps the scenario,
    so the button is reachable while the form is incomplete. It must say why it
    did nothing rather than consuming the click in silence.
    """
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)

    _click(stub_streamlit, "threat_group_regenerate_current")

    # The rerun the click triggers, on a page whose selector has reset.
    stub_streamlit["button_returns"] = False
    stub_streamlit["warnings"].clear()

    def _unexpected(_config, _messages):
        raise AssertionError("no model call may run while the form is incomplete")

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _unexpected)
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=["Select a threat actor group for the scenario."],
        setup=_setup_state(),
        download_name="AttackGen None Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
    )

    assert any("Regenerate needs" in w for w in stub_streamlit["warnings"])
    # ...and the result the user already has is left alone.
    assert fake_session_state["threat_group_scenario_text"].startswith("# APT29")


def test_a_run_that_produces_no_scenario_still_shows_the_persisted_result(
    stub_streamlit,
    fake_session_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed generation must not blank a page that still holds a result.

    The model answering with reasoning and nothing else ends the base phase
    early. That records the failure, but the page has an earlier scenario in
    session state, so it is put back on screen with its downloads intact.
    """
    _generate_threat_group_result(stub_streamlit, fake_session_state, monkeypatch)
    md_name = fake_session_state["threat_group_scenario_filename"]

    stub_streamlit["button_returns"] = True
    stub_streamlit["markdown"].clear()

    def _thinking_only(_config, _messages):
        yield "<think>weighing the options</think>"

    monkeypatch.setattr("core.scenario_page.call_llm_stream", _thinking_only)
    downloads = _capture_downloads(monkeypatch)
    run_scenario_page(
        page_id="threat_group",
        build_messages=lambda _s: [{"role": "user", "content": "x"}],
        requirements=[],
        setup=_setup_state(),
        download_name="AttackGen APT29 Enterprise.md",
        trace_name="Threat Group Scenario",
        trace_tags=("threat_group_scenario",),
        capture_inputs=lambda: copy.deepcopy(_THREAT_GROUP_INPUTS),
    )

    # The failure is recorded against the base phase...
    status = fake_session_state["threat_group_generation_status"]
    assert status["phase"] == BASE_PHASE
    # ...and the scenario the page already had is still rendered and downloadable.
    assert any("previously generated" in body for body in stub_streamlit["markdown"])
    assert [d["file_name"] for d in downloads][0] == md_name
    assert downloads[0]["data"].startswith("# APT29 Scenario")


def test_narrative_retry_leaves_another_pages_assistant_handoff_alone(
    stub_streamlit,
    fake_session_state,
    controllable_stream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The narrative handoff follows the same ownership rule as the scenario.

    A retry can land after a different page has generated and taken over the
    Assistant. Its narrative belongs to this page's scenario, so it must not be
    paired with the scenario the Assistant is now showing.
    """
    _generate_with_narrative(
        stub_streamlit,
        fake_session_state,
        controllable_stream,
        [RuntimeError("upstream 503")],
    )

    _click(stub_streamlit, "threat_group_retry_narrative")

    # Another page generates before the retry runs, taking the handoff with it.
    stub_streamlit["button_returns"] = False
    fake_session_state["last_scenario_meta"] = {"page_id": "custom"}
    fake_session_state["last_scenario_text"] = "# Custom scenario"
    fake_session_state.pop("last_defense_narrative", None)

    controllable_stream.scripts = [[], [], ["## Defender walkthrough"]]
    _run_page(is_ready=lambda: False)

    # The retry enriched its own page's result...
    defense = fake_session_state["threat_group_scenario_defense"]
    assert defense["narrative_md"] == "## Defender walkthrough"
    # ...without attaching that narrative to the custom page's scenario.
    assert "last_defense_narrative" not in fake_session_state
    assert fake_session_state["last_scenario_text"] == "# Custom scenario"
