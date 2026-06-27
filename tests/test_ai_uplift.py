"""Tests for `core.ai_uplift` — the AI-enhanced adversary toggle.

The toggle reframes a scenario's prompt and adds an `ai_enhanced` LangSmith tag
when on, and is namespaced per page so two scenario pages keep independent
state. These cover the on/off branches and the namespacing.
"""

from __future__ import annotations

from core.ai_uplift import (
    AI_UPLIFT_PROMPT,
    apply_ai_uplift,
    is_ai_uplift_on,
    uplift_trace_tags,
)


def test_defaults_to_off_before_toggle_rendered(fake_session_state) -> None:
    assert is_ai_uplift_on("threat_group") is False
    assert apply_ai_uplift("base content", "threat_group") == "base content"
    assert uplift_trace_tags(("threat_group_scenario",), "threat_group") == (
        "threat_group_scenario",
    )


def test_apply_appends_prompt_when_on(fake_session_state) -> None:
    fake_session_state["threat_group_ai_uplift"] = True

    result = apply_ai_uplift("base content", "threat_group")

    assert result == "base content" + AI_UPLIFT_PROMPT


def test_trace_tags_gains_ai_enhanced_when_on(fake_session_state) -> None:
    fake_session_state["custom_ai_uplift"] = True

    tags = uplift_trace_tags(("custom_scenario",), "custom")

    assert tags == ("custom_scenario", "ai_enhanced")


def test_toggle_state_is_namespaced_per_page(fake_session_state) -> None:
    fake_session_state["threat_group_ai_uplift"] = True

    # The "custom" page must not inherit the threat-group page's toggle.
    assert is_ai_uplift_on("threat_group") is True
    assert is_ai_uplift_on("custom") is False
    assert apply_ai_uplift("x", "custom") == "x"
