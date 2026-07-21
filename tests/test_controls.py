"""Tests for `core.controls` — the defensive control overlay.

The overlay appends a control-assessment fragment to a scenario's prompt and
adds a `control_overlay` LangSmith tag when the user supplies a control
description. It is a no-op when the description is empty or whitespace, and is
namespaced per page so two scenario pages keep independent state.
"""

from __future__ import annotations

from core.controls import (
    apply_controls,
    append_controls,
    controls_trace_tags,
    get_controls,
)


def test_append_is_noop_when_empty() -> None:
    assert append_controls("base content", "") == "base content"
    assert append_controls("base content", "   \n  ") == "base content"


def test_append_adds_fragment_and_interpolates_description() -> None:
    result = append_controls("base content", "  EDR everywhere, no egress logging  ")

    assert result != "base content"
    assert "Defensive control overlay" in result
    # The description is stripped and interpolated verbatim.
    assert "EDR everywhere, no egress logging" in result
    assert result.startswith("base content")


def test_defaults_to_empty_before_input_rendered(fake_session_state) -> None:
    assert get_controls("threat_group") == ""
    assert apply_controls("base content", "threat_group") == "base content"
    assert controls_trace_tags(("threat_group_scenario",), "threat_group") == (
        "threat_group_scenario",
    )


def test_apply_appends_from_session_state(fake_session_state) -> None:
    fake_session_state["threat_group_controls"] = "MFA on all remote access"

    result = apply_controls("base content", "threat_group")

    assert "Defensive control overlay" in result
    assert "MFA on all remote access" in result


def test_trace_tags_gains_control_overlay_when_set(fake_session_state) -> None:
    fake_session_state["custom_controls"] = "segmentation"

    tags = controls_trace_tags(("custom_scenario",), "custom")

    assert tags == ("custom_scenario", "control_overlay")


def test_whitespace_only_description_is_off(fake_session_state) -> None:
    fake_session_state["custom_controls"] = "   "

    assert apply_controls("x", "custom") == "x"
    assert controls_trace_tags(("custom_scenario",), "custom") == ("custom_scenario",)


def test_state_is_namespaced_per_page(fake_session_state) -> None:
    fake_session_state["threat_group_controls"] = "EDR"

    assert get_controls("threat_group") == "EDR"
    assert get_controls("custom") == ""
    assert apply_controls("x", "custom") == "x"
